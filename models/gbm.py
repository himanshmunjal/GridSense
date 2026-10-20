"""
models/gbm.py
─────────────
PURPOSE:
  LightGBM forecaster — the strong tabular baseline the LSTM is compared
  against (and blended with) in train.py.

HOW IT FORECASTS 24 HOURS:
  One model for all horizons ("direct, horizon-as-feature"). Every
  (window, step k) pair becomes one row: the known-future covariates of
  target hour t+k, the step k itself, and a summary of the past window.
  This shares data across horizons (24× more rows than one model per step)
  while still letting trees learn that step 1 trusts the last reading more
  than step 24 does.

WHY build rows from the LSTM's windows:
  Same split, same scaling, same noisy temperature forecast — so the model
  comparison in results/metrics.json is apples to apples, and the API can
  build LightGBM inputs from exactly the window it already feeds the LSTM.
"""

import numpy as np
import lightgbm as lgb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocess import FEATURE_COLS, FUTURE_COLS, LOOKBACK, HORIZON

PARAMS = {
    "objective": "l1",          # conditional median in log space, like the LSTM
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}

PAST_SUMMARY = [
    "last_demand", "demand_t-1", "demand_t-2", "mean_last_24",
    "rolling_std_24", "rolling_mean_168", "last_temperature", "demand_lag48",
]
FEATURE_NAMES = ["step"] + [f"fut_{c}" for c in FUTURE_COLS] + PAST_SUMMARY


def tabular_features(X_past: np.ndarray, X_future: np.ndarray) -> np.ndarray:
    """
    (W, LOOKBACK, INPUT_SIZE), (W, HORIZON, FUTURE_SIZE) → (W * HORIZON, n_features)

    Rows are ordered window-major: row w*HORIZON + k is step k of window w.
    """
    W = len(X_past)
    d = FEATURE_COLS.index("demand_kwh")
    demand = X_past[:, :, d]

    steps = np.arange(HORIZON)
    past = np.stack([
        demand[:, -1],
        demand[:, -2],
        demand[:, -3],
        demand[:, -24:].mean(axis=1),
        X_past[:, -1, FEATURE_COLS.index("rolling_std_24")],
        X_past[:, -1, FEATURE_COLS.index("rolling_mean_168")],
        X_past[:, -1, FEATURE_COLS.index("temperature")],
    ], axis=1)                                                    # (W, 7)
    past = np.repeat(past, HORIZON, axis=0)                       # (W*H, 7)
    # Same hour two days before the target: t+k−48 → window position k
    lag48 = demand[:, LOOKBACK - 48 + steps].reshape(-1, 1)       # (W*H, 1)

    return np.hstack([
        np.tile(steps, W).reshape(-1, 1).astype(np.float32),
        X_future.reshape(W * HORIZON, -1),
        past,
        lag48,
    ]).astype(np.float32)


def _columns(exclude) -> list:
    return [i for i, name in enumerate(FEATURE_NAMES) if name not in set(exclude)]


def train_gbm(train_arrays, val_arrays, max_rounds: int = 4000, exclude=()) -> lgb.Booster:
    """
    Fit on train windows with early stopping on validation windows.
    `exclude` drops named features — used for the weather/holiday ablations.
    """
    Xp, Xf, y, _ = train_arrays
    Xvp, Xvf, yv, _ = val_arrays
    cols = _columns(exclude)
    dtrain = lgb.Dataset(tabular_features(Xp, Xf)[:, cols], y.reshape(-1),
                         feature_name=[FEATURE_NAMES[i] for i in cols], free_raw_data=True)
    dval = lgb.Dataset(tabular_features(Xvp, Xvf)[:, cols], yv.reshape(-1), reference=dtrain)
    booster = lgb.train(
        PARAMS, dtrain, num_boost_round=max_rounds, valid_sets=[dval],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
    )
    print(f"  LightGBM best iteration: {booster.best_iteration}")
    return booster


def predict_gbm(booster: lgb.Booster, X_past: np.ndarray, X_future: np.ndarray,
                exclude=()) -> np.ndarray:
    """Forecast in scaled log space, shape (W, HORIZON)."""
    # num_iteration=None → best iteration when trained here, all trees when
    # loaded from a file saved at the best iteration
    pred = booster.predict(tabular_features(X_past, X_future)[:, _columns(exclude)])
    return pred.reshape(len(X_past), HORIZON)
