"""
preprocess.py
─────────────
PURPOSE:
  Transform raw zone CSVs into PyTorch-ready sliding window datasets.

PIPELINE:
  Raw hourly CSV
    → MinMax normalize (per zone, fit on train only)
    → Sliding window sequences  [lookback=48h input → 24h target]
    → Train / Val / Test split  (70 / 15 / 15)
    → Save scalers for inverse transform at inference time

KEY CONCEPTS:
  1. WHY normalize: LSTMs use sigmoid/tanh — saturate outside [-1, 1].
     Raw kWh values (0–5000) kill gradients. Normalizing to [0,1] fixes this.

  2. WHY fit scaler on train ONLY:
     Fitting on full data leaks future statistics into training.
     This is "data leakage" — the model would be cheating.

  3. WHY lookback=48h:
     48h captures two full daily cycles. The model sees yesterday + day before,
     giving it enough context to distinguish a Tuesday from a Saturday pattern.

  4. Sliding window:
     Input : [t-47, t-46, ..., t-0]  (48 timesteps)
     Target: [t+1,  t+2,  ..., t+24] (24 timesteps)
     Step by 1 hour → maximum data utilization.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR    = Path("data/zones")
OUTPUT_DIR  = Path("data/processed")
LOOKBACK    = 48   # hours of history fed to model
HORIZON     = 24   # hours ahead to forecast
STEP        = 1    # stride of sliding window
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# TEST = remaining 15%

BATCH_SIZE  = 32
ZONES       = [1, 2, 3, 4]


# ─── DATASET CLASS ────────────────────────────────────────────────────────────

class DemandDataset(Dataset):
    """
    PyTorch Dataset for time-series sliding windows.

    Each sample:
      X : (LOOKBACK, 1)  — normalized past demand
      y : (HORIZON, 1)   — normalized future demand to predict
      anomaly: (HORIZON,) — ground truth anomaly labels (for autoencoder eval)
    
    WHY a custom Dataset:
      DataLoader needs to batch samples. A custom Dataset lets us define
      exactly what one "sample" is — here it's one sliding window pair.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, anomaly_labels: np.ndarray = None):
        self.X = torch.FloatTensor(X)          # (N, LOOKBACK, 1)
        self.y = torch.FloatTensor(y)          # (N, HORIZON, 1)
        self.anomaly_labels = anomaly_labels   # (N, HORIZON) or None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.anomaly_labels is not None:
            return self.X[idx], self.y[idx], self.anomaly_labels[idx]
        return self.X[idx], self.y[idx]


# ─── SLIDING WINDOW ───────────────────────────────────────────────────────────

def create_windows(values: np.ndarray, anomalies: np.ndarray,
                   lookback: int, horizon: int, step: int):
    """
    Convert flat 1D time series → (N, lookback, 1) inputs, (N, horizon, 1) targets.

    Visual:
      values = [a, b, c, d, e, f, g, h, ...]
      lookback=3, horizon=2, step=1:
        window 0: X=[a,b,c]  y=[d,e]
        window 1: X=[b,c,d]  y=[e,f]
        window 2: X=[c,d,e]  y=[f,g]
        ...
    """
    X, y, y_anomaly = [], [], []
    total = len(values)

    for start in range(0, total - lookback - horizon + 1, step):
        end = start + lookback
        X.append(values[start:end])
        y.append(values[end:end + horizon])
        y_anomaly.append(anomalies[end:end + horizon])

    X = np.array(X)[..., np.newaxis]        # add feature dim → (N, lookback, 1)
    y = np.array(y)[..., np.newaxis]        # (N, horizon, 1)
    y_anomaly = np.array(y_anomaly)         # (N, horizon)

    return X, y, y_anomaly


# ─── MAIN PREPROCESSING ───────────────────────────────────────────────────────

def preprocess_zone(zone_id: int):
    """
    Full preprocessing pipeline for a single zone.
    Returns train/val/test DataLoaders + fitted scaler.
    """
    print(f"\nPreprocessing Zone {zone_id}...")

    df = pd.read_csv(DATA_DIR / f"zone_{zone_id}.csv", parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    values   = df["demand_kwh"].values.astype(np.float32)
    anomaly  = df["is_anomaly"].values.astype(np.float32)

    # ── 1. TRAIN/VAL/TEST SPLIT (before scaling — no leakage) ──────────────
    n = len(values)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    train_vals  = values[:train_end]
    val_vals    = values[train_end:val_end]
    test_vals   = values[val_end:]

    train_anom  = anomaly[:train_end]
    val_anom    = anomaly[train_end:val_end]
    test_anom   = anomaly[val_end:]

    # ── 2. FIT SCALER ON TRAIN ONLY ────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
    val_scaled   = scaler.transform(val_vals.reshape(-1, 1)).flatten()
    test_scaled  = scaler.transform(test_vals.reshape(-1, 1)).flatten()

    # ── 3. SLIDING WINDOWS ─────────────────────────────────────────────────
    X_tr, y_tr, a_tr = create_windows(train_scaled, train_anom, LOOKBACK, HORIZON, STEP)
    X_v,  y_v,  a_v  = create_windows(val_scaled,   val_anom,   LOOKBACK, HORIZON, STEP)
    X_te, y_te, a_te = create_windows(test_scaled,  test_anom,  LOOKBACK, HORIZON, STEP)

    print(f"  Train: {X_tr.shape[0]} windows | "
          f"Val: {X_v.shape[0]} windows | "
          f"Test: {X_te.shape[0]} windows")

    # ── 4. DATALOADERS ─────────────────────────────────────────────────────
    train_loader = DataLoader(DemandDataset(X_tr, y_tr, a_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(DemandDataset(X_v,  y_v,  a_v),  batch_size=BATCH_SIZE)
    test_loader  = DataLoader(DemandDataset(X_te, y_te, a_te), batch_size=BATCH_SIZE)

    # ── 5. SAVE SCALER ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"scaler_zone_{zone_id}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return train_loader, val_loader, test_loader, scaler


def load_scaler(zone_id: int) -> MinMaxScaler:
    with open(OUTPUT_DIR / f"scaler_zone_{zone_id}.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    for z in ZONES:
        train_loader, val_loader, test_loader, scaler = preprocess_zone(z)
        # Quick sanity check
        X_batch, y_batch, _ = next(iter(train_loader))
        print(f"  Zone {z} batch — X: {X_batch.shape}, y: {y_batch.shape}")