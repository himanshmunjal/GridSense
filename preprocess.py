"""
preprocess.py
─────────────
PURPOSE:
  Transform raw zone CSVs into PyTorch-ready sliding window datasets.

PIPELINE:
  Raw hourly zone CSV + hourly weather + public holidays
    → Clean labelled anomalies out of the TRAIN history (interpolate)
    → log1p transform of demand
    → Feature engineering (lag, rolling, cyclical time, holidays, temperature)
    → StandardScaler on demand-derived + temperature columns (fit on train only)
    → Sliding windows: 48h of PAST features + 24h of KNOWN-FUTURE covariates
                       → 24h target
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
     magnitude across all input channels.

  5. WHY log1p(demand):
     Household load is heavily right-skewed (std ≈ mean). In log space the
     injected anomalies are multiplicative (spike ×3–5, dropout ×0–0.1), so
     they become a constant-size shift regardless of the load level. In
     linear space a night-time dropout (15 → 1 kWh) is almost invisible
     after standard scaling. Log space also makes the L1-trained forecaster
     predict the conditional median, which lowers MAPE.

  6. WHY clean anomalies from the training history only:
     A forecaster should learn normal behaviour; spikes in its training
     targets pull predictions up. Val/test stay raw so evaluation is honest.

  7. WHY known-future covariates (X_future):
     At forecast time t we already know the calendar of the next 24 hours,
     whether they are public holidays, the demand at the same hour
     yesterday / last week (t+k−24 and t+k−168 are both ≤ t), and a weather
     forecast. The encoder–decoder forecaster feeds these into its decoder
     step by step, which is where most of the accuracy on real load comes
     from (holidays and temperature swings are invisible in the past window).

  8. WHY add noise to future temperature:
     We only have observed weather, not archived forecasts. Feeding the
     true future temperature would be a "perfect forecast" leak, so every
     window gets Gaussian noise (σ = TEMP_FORECAST_NOISE °C, typical of a
     day-ahead forecast). Val/test noise is seeded, so all models compared
     in train.py see identical inputs.

  9. WHY a separate demand-only series for the autoencoder:
     The autoencoder's job is to reconstruct the demand signal and flag
     anomalies in it. If we feed it all features, the sin/cos time
     channels are trivially easy to reconstruct (they're deterministic),
     which dilutes the reconstruction error signal and hurts detection.
     So we export X_full for the forecaster and `hourly` raw demand arrays
     (uncleaned, so real anomalies remain) for the autoencoder.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import holidays
import torch
from torch.utils.data import Dataset, DataLoader

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR    = Path("data/zones")
WEATHER_CSV = Path("data/weather/lisbon_hourly_utc.csv")
OUTPUT_DIR  = Path("data/processed")

LOOKBACK   = 48    # hours of history in each input window
HORIZON    = 24    # hours ahead to forecast
STEP       = 1     # sliding window stride (1 = maximum overlap)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15  # TEST = remaining 0.15

BATCH_SIZE = 64
ZONES      = [1, 2, 3, 4]

TEMP_FORECAST_NOISE = 1.5   # °C, std of simulated day-ahead forecast error
NOISE_SEED          = 1234

# Columns that share kWh units → need StandardScaler
# sin/cos time encodings are excluded (already in [-1,1])
SCALE_COLS = [
    "demand_kwh",
    "demand_lag24",
    "demand_lag168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "temperature",
]

# Past features fed to the forecaster's encoder (one row per lookback hour)
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
    "rolling_mean_168",# weekly baseline
    "doy_sin",         # yearly seasonality (heating/cooling season)
    "doy_cos",
    "is_weekend",
    "is_holiday",      # Portuguese public holiday
    "temperature",     # observed air temperature (scaled)
]

# Known-future covariates fed to the decoder (one row per forecast hour).
# All of them are available at forecast time — see header note 7.
FUTURE_COLS = [
    "demand_lag24",
    "demand_lag168",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "is_weekend",
    "is_holiday",
    "temperature",     # + simulated forecast noise
]

INPUT_SIZE  = len(FEATURE_COLS)  # 15 — single source of truth for model init
FUTURE_SIZE = len(FUTURE_COLS)   # 11
TEMP_FUTURE_IDX = FUTURE_COLS.index("temperature")


# ─── FEATURE ENGINEERING (shared by training + API) ──────────────────────────

def to_model_space(demand_kwh):
    """kWh → log1p space. Every model in GridSense works in this space."""
    return np.log1p(np.clip(demand_kwh, 0, None))


def from_model_space(values):
    """log1p space → kWh (clipped at 0)."""
    return np.clip(np.expm1(values), 0, None)


def clean_anomalies(demand: pd.Series, is_anomaly: np.ndarray) -> pd.Series:
    """Replace flagged hours with linear interpolation of their neighbours."""
    out = demand.copy()
    out[np.asarray(is_anomaly, dtype=bool)] = np.nan
    return out.interpolate(method="linear").bfill().ffill()


def load_weather() -> pd.Series:
    """
    Hourly Lisbon temperature (°C) on the load data's clock.

    The load data is labelled in Portuguese local time (WET/WEST) while the
    Open-Meteo archive is UTC, so convert, drop the repeated hour at the
    October DST change and fill the skipped March hour.
    """
    w = pd.read_csv(WEATHER_CSV, parse_dates=["datetime_utc"])
    idx = (pd.DatetimeIndex(w["datetime_utc"]).tz_localize("UTC")
           .tz_convert("Europe/Lisbon").tz_localize(None))
    temp = pd.Series(w["temperature_2m"].values, index=idx)
    temp = temp[~temp.index.duplicated(keep="first")]
    full = pd.date_range(temp.index[0], temp.index[-1], freq="h")
    return temp.reindex(full).interpolate()


def holiday_flags(idx: pd.DatetimeIndex) -> np.ndarray:
    """1.0 for hours that fall on a Portuguese national public holiday."""
    pt = holidays.Portugal(years=sorted(set(idx.year)))
    return np.array([d in pt for d in idx.date], dtype=float)


def build_features(demand_kwh: pd.Series, temperature: pd.Series) -> pd.DataFrame:
    """
    Build the unscaled feature frame (FEATURE_COLS ∪ FUTURE_COLS) from an
    hourly kWh series with a DatetimeIndex and an hourly temperature series.

    Rows with NaN lags/rolling stats (first 168h) are dropped. Rows with NaN
    demand are kept as long as their lags exist — the API appends 24 empty
    future hours this way to get the decoder's known-future covariates.
    """
    idx = demand_kwh.index
    log_d = pd.Series(to_model_space(demand_kwh.values), index=idx)

    df = pd.DataFrame(index=idx)
    df["demand_kwh"] = log_d

    # Cyclical time encodings — no scaling needed, already in [-1, 1]
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["doy_sin"]  = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    df["is_holiday"] = holiday_flags(idx)
    df["temperature"] = temperature.reindex(idx).values

    # Lag features — shift creates NaNs at the start (dropped below)
    df["demand_lag24"]  = log_d.shift(24)   # same hour yesterday
    df["demand_lag168"] = log_d.shift(168)  # same hour last week

    # Rolling statistics — local trend and volatility context
    df["rolling_mean_24"]  = log_d.rolling(24).mean()
    df["rolling_std_24"]   = log_d.rolling(24).std()
    df["rolling_mean_168"] = log_d.rolling(168).mean()

    # lag168 is the last column to become available (first 168h)
    return df[df[["demand_lag168", "temperature"]].notna().all(axis=1)]


def scale_demand(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale log-space demand with the demand column (index 0) of the scaler."""
    return (values - scaler.mean_[0]) / scaler.scale_[0]


def unscale_demand(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse of scale_demand, returning kWh."""
    return from_model_space(values * scaler.scale_[0] + scaler.mean_[0])


# ─── DATASET CLASS ────────────────────────────────────────────────────────────

class DemandDataset(Dataset):
    """
    PyTorch Dataset wrapping sliding window arrays.

    Each sample returns:
      X_past   : (LOOKBACK, INPUT_SIZE)  — past features, for the encoder
      X_future : (HORIZON, FUTURE_SIZE)  — known-future covariates, for the decoder
      y        : (HORIZON, 1)            — demand forecast target
      anomaly  : (HORIZON,)              — ground truth anomaly flags in target window
    """

    def __init__(self, X_past, X_future, y, anomaly_labels):
        self.X_past   = torch.FloatTensor(X_past)      # (N, LOOKBACK, INPUT_SIZE)
        self.X_future = torch.FloatTensor(X_future)    # (N, HORIZON, FUTURE_SIZE)
        self.y        = torch.FloatTensor(y)           # (N, HORIZON, 1)
        self.anomaly  = torch.FloatTensor(anomaly_labels)  # (N, HORIZON)

    def __len__(self):
        return len(self.X_past)

    def __getitem__(self, idx):
        return self.X_past[idx], self.X_future[idx], self.y[idx], self.anomaly[idx]


# ─── SLIDING WINDOW ───────────────────────────────────────────────────────────

def create_windows(
    past_vals: np.ndarray,    # (N, INPUT_SIZE)  — scaled past features
    future_vals: np.ndarray,  # (N, FUTURE_SIZE) — scaled future covariates
    demand_vals: np.ndarray,  # (N, 1)           — target demand, scaled
    anomalies: np.ndarray,    # (N,)             — binary anomaly flags
    lookback: int,
    horizon: int,
    step: int,
    temp_noise: np.ndarray = None,  # (W, horizon) noise added to future temperature
):
    """
    Slide a window across the time series to produce training samples.

    Visual (lookback=3, horizon=2, step=1):
      t: [a, b, c, d, e, f, ...]
      window 0: X_past=[a,b,c]  X_future=[d,e]  y=[d,e]
      window 1: X_past=[b,c,d]  X_future=[e,f]  y=[e,f]

    X_future only holds covariates known at forecast time (calendar, lags
    ≥ 24h, temperature forecast) — never the target itself.

    Returns:
      X_past   : (W, lookback, INPUT_SIZE)
      X_future : (W, horizon, FUTURE_SIZE)
      y        : (W, horizon, 1)
      y_anomaly: (W, horizon)
    """
    starts = np.arange(0, len(past_vals) - lookback - horizon + 1, step)
    past_idx   = starts[:, None] + np.arange(lookback)[None, :]
    future_idx = starts[:, None] + lookback + np.arange(horizon)[None, :]

    X_future = future_vals[future_idx].copy()
    if temp_noise is not None:
        X_future[:, :, TEMP_FUTURE_IDX] += temp_noise

    return (
        past_vals[past_idx],
        X_future,
        demand_vals[future_idx],
        anomalies[future_idx],
    )


# ─── MAIN PREPROCESSING ───────────────────────────────────────────────────────

def preprocess_zone(zone_id: int, return_arrays: bool = False):
    """
    Full preprocessing pipeline for a single zone.

    Returns:
      train_loader, val_loader, test_loader : DataLoaders with DemandDataset
      scaler                                : fitted StandardScaler (SCALE_COLS)
      hourly                                : {split: {"demand", "labels", "clean_kwh"}}
                                              raw hourly arrays (scaled log demand) for the
                                              autoencoder + pre-injection load in kWh
      arrays (only if return_arrays)        : {split: (X_past, X_future, y, anomaly)}
                                              — same windows, for LightGBM
    """
    print(f"\nPreprocessing Zone {zone_id}...")

    # ── LOAD & INDEX ───────────────────────────────────────────────────────
    raw = pd.read_csv(DATA_DIR / f"zone_{zone_id}.csv", parse_dates=["datetime"])
    raw = raw.sort_values("datetime").set_index("datetime")
    temperature = load_weather()

    # ── CLEAN TRAIN HISTORY ────────────────────────────────────────────────
    # Only hours inside the training split are cleaned; val/test stay raw.
    train_cut   = int(len(raw) * TRAIN_FRAC)
    train_anoms = raw["is_anomaly"].values.astype(bool).copy()
    train_anoms[train_cut:] = False
    clean_demand = clean_anomalies(raw["demand_kwh"], train_anoms)

    # ── FEATURE ENGINEERING ────────────────────────────────────────────────
    df = build_features(clean_demand, temperature)
    raw = raw.loc[df.index]
    # Raw (uncleaned) demand for the autoencoder — it must see real anomalies
    df["raw_demand"] = to_model_space(raw["demand_kwh"].values)
    df["is_anomaly"] = raw["is_anomaly"].values
    df["clean_kwh"]  = raw["demand_clean_kwh"].values

    # ── SPLIT (before scaling — no leakage) ───────────────────────────────
    n         = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end]
    df_test  = df.iloc[val_end:]

    # ── SCALE DEMAND-DERIVED + TEMPERATURE COLUMNS ────────────────────────
    # Fit on train only to avoid leaking test statistics
    scaler = StandardScaler()
    scaler.fit(df_train[SCALE_COLS])
    temp_std = scaler.scale_[SCALE_COLS.index("temperature")]

    def scale_split(df_split) -> pd.DataFrame:
        out = df_split.copy()
        out[SCALE_COLS] = scaler.transform(df_split[SCALE_COLS])
        out["raw_demand"] = scale_demand(df_split["raw_demand"].values, scaler)
        return out

    rng = np.random.default_rng(NOISE_SEED + zone_id)
    loaders, arrays, hourly = {}, {}, {}
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        s = scale_split(d)
        n_windows = len(s) - LOOKBACK - HORIZON + 1
        # Simulated temperature forecast error, in scaled units
        noise = rng.normal(0, TEMP_FORECAST_NOISE / temp_std, size=(n_windows, HORIZON))
        windows = create_windows(
            s[FEATURE_COLS].values.astype(np.float32),
            s[FUTURE_COLS].values.astype(np.float32),
            s[["demand_kwh"]].values.astype(np.float32),
            s["is_anomaly"].values.astype(np.float32),
            LOOKBACK, HORIZON, STEP, temp_noise=noise.astype(np.float32),
        )
        arrays[name] = windows
        loaders[name] = DataLoader(
            DemandDataset(*windows),
            batch_size=BATCH_SIZE,
            shuffle=(name == "train"),
            pin_memory=torch.cuda.is_available(),  # only helps for CUDA
            num_workers=0,
        )
        # Hourly raw demand per split, used by the autoencoder (point-level
        # anomaly scoring)
        hourly[name] = {
            "demand": s["raw_demand"].values.astype(np.float32),  # scaled log
            "labels": s["is_anomaly"].values.astype(int),
            "clean_kwh": s["clean_kwh"].values,   # pre-injection load, kWh
        }

    Xp, Xf, y, _ = arrays["train"]
    print(f"  Train: {len(Xp)} windows | Val: {len(arrays['val'][0])} windows | "
          f"Test: {len(arrays['test'][0])} windows")
    print(f"  X_past: {Xp.shape} | X_future: {Xf.shape} | y: {y.shape}")

    # ── SAVE SCALER ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"scaler_zone_{zone_id}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    out = (loaders["train"], loaders["val"], loaders["test"], scaler, hourly)
    return out + (arrays,) if return_arrays else out


def load_scaler(zone_id: int) -> StandardScaler:
    with open(OUTPUT_DIR / f"scaler_zone_{zone_id}.pkl", "rb") as f:
        return pickle.load(f)


# ─── SANITY CHECK ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for z in ZONES:
        train_loader, val_loader, test_loader, scaler, _ = preprocess_zone(z)

        X_past, X_future, y, anomaly = next(iter(train_loader))
        print(f"\nZone {z} batch shapes:")
        print(f"  X_past   : {X_past.shape}")    # (64, 48, 15)
        print(f"  X_future : {X_future.shape}")  # (64, 24, 11)
        print(f"  y        : {y.shape}")         # (64, 24, 1)
        print(f"  anomaly  : {anomaly.shape}")   # (64, 24)
