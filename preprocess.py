"""
preprocess.py
─────────────
PURPOSE:
  Transform raw zone CSVs into PyTorch-ready sliding window datasets.

PIPELINE:
  Raw hourly CSV
    → Feature engineering (lag, rolling, cyclical time encoding)
    → StandardScaler on demand-derived columns only (fit on train only)
    → Sliding window sequences  [lookback=48h input → 24h target]
    → Train / Val / Test split  (70 / 15 / 15)
    → Save scaler for inverse transform at inference time

KEY DECISIONS:
  1. WHY StandardScaler over MinMaxScaler:
     MinMaxScaler compresses extreme spikes to near-1.0 and quiet nights
     to near-0.0, making gradients tiny at both ends. StandardScaler keeps
     the distribution mean-centered — gradients flow more uniformly.

  2. WHY fit scaler on train ONLY:
     Fitting on full data leaks future statistics. The model would implicitly
     know the global mean/std of the test period — that's cheating.

  3. WHY lookback=48h:
     48h captures two full daily cycles. The model sees yesterday + day before,
     giving it enough context to distinguish a Tuesday morning from a Saturday.

  4. WHY scale only demand-derived columns (not sin/cos time features):
     sin/cos are already in [-1, 1] by construction — scaling them again
     just adds numerical noise. Lag and rolling features share the same
     kWh units as demand, so they DO need scaling for consistent gradient
     magnitude across all 10 input channels.

  5. WHY separate X_demand for the autoencoder:
     The autoencoder's job is to reconstruct the demand signal and flag
     anomalies in it. If we feed it all 10 features, the sin/cos time
     channels are trivially easy to reconstruct (they're deterministic),
     which dilutes the reconstruction error signal and hurts detection.
     So we export both: X_full (forecaster) and X_demand (autoencoder).
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data/zones")
OUTPUT_DIR = Path("data/processed")

LOOKBACK   = 48    # hours of history in each input window
HORIZON    = 24    # hours ahead to forecast
STEP       = 1     # sliding window stride (1 = maximum overlap)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15  # TEST = remaining 0.15

BATCH_SIZE = 32
ZONES      = [1, 2, 3, 4]

# Columns that share kWh units → need StandardScaler
# sin/cos time encodings are excluded (already in [-1,1])
SCALE_COLS = [
    "demand_kwh",
    "demand_lag24",
    "demand_lag168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
]

# Full feature set fed to the LSTM forecaster
FEATURE_COLS = [
    "demand_kwh",      # target signal
    "demand_lag24",    # same hour yesterday  — highest-signal feature
    "demand_lag168",   # same hour last week  — captures weekly seasonality
    "hour_sin",        # cyclical hour encoding (avoids 23→0 discontinuity)
    "hour_cos",
    "dow_sin",         # cyclical day-of-week encoding
    "dow_cos",
    "rolling_mean_24", # local trend baseline
    "rolling_std_24",  # local volatility
    "rolling_mean_168" # weekly baseline
]

INPUT_SIZE = len(FEATURE_COLS)  # 10 — single source of truth for model init


# ─── DATASET CLASS ────────────────────────────────────────────────────────────

class DemandDataset(Dataset):
    """
    PyTorch Dataset wrapping sliding window arrays.

    Each sample returns:
      X_full   : (LOOKBACK, INPUT_SIZE) — all features, for the forecaster
      X_demand : (LOOKBACK, 1)          — demand column only, for the autoencoder
      y        : (HORIZON, 1)           — demand forecast target
      anomaly  : (HORIZON,)             — ground truth anomaly flags in target window
    """

    def __init__(
        self,
        X_full: np.ndarray,
        X_demand: np.ndarray,
        y: np.ndarray,
        anomaly_labels: np.ndarray,
    ):
        self.X_full   = torch.FloatTensor(X_full)      # (N, LOOKBACK, INPUT_SIZE)
        self.X_demand = torch.FloatTensor(X_demand)    # (N, LOOKBACK, 1)
        self.y        = torch.FloatTensor(y)           # (N, HORIZON, 1)
        self.anomaly  = torch.FloatTensor(anomaly_labels)  # (N, HORIZON)

    def __len__(self):
        return len(self.X_full)

    def __getitem__(self, idx):
        return (
            self.X_full[idx],
            self.X_demand[idx],
            self.y[idx],
            self.anomaly[idx],
        )


# ─── SLIDING WINDOW ───────────────────────────────────────────────────────────

def create_windows(
    full_vals: np.ndarray,    # (N, INPUT_SIZE) — all scaled features
    demand_vals: np.ndarray,  # (N, 1)          — demand column only, scaled
    anomalies: np.ndarray,    # (N,)             — binary anomaly flags
    lookback: int,
    horizon: int,
    step: int,
):
    """
    Slide a window across the time series to produce training samples.

    Visual (lookback=3, horizon=2, step=1):
      t: [a, b, c, d, e, f, ...]
      window 0: X=[a,b,c]  y=[d,e]
      window 1: X=[b,c,d]  y=[e,f]
      window 2: X=[c,d,e]  y=[f,g]

    Returns:
      X_full   : (W, lookback, INPUT_SIZE)
      X_demand : (W, lookback, 1)
      y        : (W, horizon, 1)
      y_anomaly: (W, horizon)
    """
    X_full_list, X_demand_list, y_list, y_anom_list = [], [], [], []

    for start in range(0, len(full_vals) - lookback - horizon + 1, step):
        end = start + lookback
        X_full_list.append(full_vals[start:end])           # (lookback, INPUT_SIZE)
        X_demand_list.append(demand_vals[start:end])       # (lookback, 1)
        y_list.append(demand_vals[end:end + horizon])      # (horizon, 1)
        y_anom_list.append(anomalies[end:end + horizon])   # (horizon,)

    return (
        np.array(X_full_list),    # (W, lookback, INPUT_SIZE)
        np.array(X_demand_list),  # (W, lookback, 1)
        np.array(y_list),         # (W, horizon, 1)
        np.array(y_anom_list),    # (W, horizon)
    )


# ─── MAIN PREPROCESSING ───────────────────────────────────────────────────────

def preprocess_zone(zone_id: int):
    """
    Full preprocessing pipeline for a single zone.

    Returns:
      train_loader, val_loader, test_loader : DataLoaders with DemandDataset
      scaler                                : fitted StandardScaler (demand-derived cols)
    """
    print(f"\nPreprocessing Zone {zone_id}...")

    # ── LOAD & INDEX ───────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / f"zone_{zone_id}.csv", parse_dates=["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    # ── FEATURE ENGINEERING ────────────────────────────────────────────────
    # Cyclical time encodings — no scaling needed, already in [-1, 1]
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Lag features — shift creates NaNs at the start (dropped below)
    df["demand_lag24"]  = df["demand_kwh"].shift(24)   # same hour yesterday
    df["demand_lag168"] = df["demand_kwh"].shift(168)  # same hour last week

    # Rolling statistics — local trend and volatility context
    df["rolling_mean_24"]  = df["demand_kwh"].rolling(24).mean()
    df["rolling_std_24"]   = df["demand_kwh"].rolling(24).std()
    df["rolling_mean_168"] = df["demand_kwh"].rolling(168).mean()

    # Drop the first 168 rows where any lag/rolling feature is NaN
    df = df.dropna()

    # ── SPLIT (before scaling — no leakage) ───────────────────────────────
    n         = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end]
    df_test  = df.iloc[val_end:]

    # ── SCALE DEMAND-DERIVED COLUMNS ONLY ─────────────────────────────────
    # Fit on train only to avoid leaking test statistics
    scaler = StandardScaler()

    def scale_split(df_split, fit: bool) -> pd.DataFrame:
        out = df_split.copy()
        if fit:
            out[SCALE_COLS] = scaler.fit_transform(df_split[SCALE_COLS])
        else:
            out[SCALE_COLS] = scaler.transform(df_split[SCALE_COLS])
        return out

    df_train_s = scale_split(df_train, fit=True)
    df_val_s   = scale_split(df_val,   fit=False)
    df_test_s  = scale_split(df_test,  fit=False)

    # ── EXTRACT ARRAYS ─────────────────────────────────────────────────────
    def to_arrays(df_s, df_raw):
        full   = df_s[FEATURE_COLS].values.astype(np.float32)   # (N, 10)
        demand = df_s[["demand_kwh"]].values.astype(np.float32) # (N, 1)
        anom   = df_raw["is_anomaly"].values.astype(np.float32) # (N,)
        return full, demand, anom

    full_tr, dem_tr, anom_tr = to_arrays(df_train_s, df_train)
    full_v,  dem_v,  anom_v  = to_arrays(df_val_s,   df_val)
    full_te, dem_te, anom_te = to_arrays(df_test_s,  df_test)

    # ── SLIDING WINDOWS ────────────────────────────────────────────────────
    Xf_tr, Xd_tr, y_tr, a_tr = create_windows(full_tr, dem_tr, anom_tr, LOOKBACK, HORIZON, STEP)
    Xf_v,  Xd_v,  y_v,  a_v  = create_windows(full_v,  dem_v,  anom_v,  LOOKBACK, HORIZON, STEP)
    Xf_te, Xd_te, y_te, a_te = create_windows(full_te, dem_te, anom_te, LOOKBACK, HORIZON, STEP)

    print(f"  Train: {Xf_tr.shape[0]} windows | "
          f"Val: {Xf_v.shape[0]} windows | "
          f"Test: {Xf_te.shape[0]} windows")
    print(f"  X_full shape: {Xf_tr.shape} | X_demand shape: {Xd_tr.shape} | y shape: {y_tr.shape}")

    # ── BUILD DATALOADERS ──────────────────────────────────────────────────
    def make_loader(Xf, Xd, y, a, shuffle):
        return DataLoader(
            DemandDataset(Xf, Xd, y, a),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            pin_memory=True,   # speeds up CPU→GPU transfer if GPU is available
            num_workers=0,     # set >0 on Linux with multiple cores
        )

    train_loader = make_loader(Xf_tr, Xd_tr, y_tr, a_tr, shuffle=True)
    val_loader   = make_loader(Xf_v,  Xd_v,  y_v,  a_v,  shuffle=False)
    test_loader  = make_loader(Xf_te, Xd_te, y_te, a_te, shuffle=False)

    # ── SAVE SCALER ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"scaler_zone_{zone_id}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return train_loader, val_loader, test_loader, scaler


def load_scaler(zone_id: int) -> StandardScaler:
    with open(OUTPUT_DIR / f"scaler_zone_{zone_id}.pkl", "rb") as f:
        return pickle.load(f)


# ─── SANITY CHECK ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for z in ZONES:
        train_loader, val_loader, test_loader, scaler = preprocess_zone(z)

        X_full, X_demand, y, anomaly = next(iter(train_loader))
        print(f"\nZone {z} batch shapes:")
        print(f"  X_full   : {X_full.shape}")    # (32, 48, 10)
        print(f"  X_demand : {X_demand.shape}")  # (32, 48, 1)
        print(f"  y        : {y.shape}")          # (32, 24, 1)
        print(f"  anomaly  : {anomaly.shape}")    # (32, 24)