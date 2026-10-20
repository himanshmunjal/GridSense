"""
api/main.py
───────────
PURPOSE:
  FastAPI server exposing two endpoints:
    GET /forecast?zone=1   → 24-hour demand forecast + confidence bands
    GET /anomalies?zone=1  → recent anomaly detections for the zone

  Also serves the latest actual demand for dashboard comparison.

HOW INFERENCE WORKS AT RUNTIME:
  1. At startup, load all 4 zone models + scalers into memory
  2. Each request: grab the last 48 hours of "live" data (simulated from the
     end of the zone CSV) + the next 24 hours of known covariates (calendar,
     holidays, temperature — the archive's observed weather stands in for a
     weather forecast here)
  3. LSTM (MC Dropout) + LightGBM → validation-weighted blend → kWh → JSON

WHY load models at startup (not per request):
  Loading a PyTorch model takes ~0.5s. At 30 requests/min that's 15s/min wasted.
  Startup loading means models live in RAM — inference is <10ms per request.

RUN:
  uvicorn api.main:app --reload --port 8000
"""

import os
import sys
# On macOS torch and LightGBM each bundle their own libomp. Two OpenMP
# runtimes in one process segfault (in LightGBM fits or CPU LSTM passes)
# unless OpenMP is single-threaded, and this must be set before either
# library is imported. Training runs the networks on MPS, so it costs little.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import torch
import numpy as np
import json
import pickle
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).parent.parent))

from models.forecaster import DemandForecaster, predict_with_uncertainty
from models.autoencoder import LSTMAutoencoder, detect_anomalies
from models.gbm import predict_gbm
from preprocess import (
    build_features, load_weather, scale_demand, unscale_demand, to_model_space,
    SCALE_COLS, FEATURE_COLS, FUTURE_COLS, LOOKBACK, HORIZON, ZONES,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAVED_DIR    = Path("models/saved")
DATA_DIR     = Path("data/zones")
DEVICE       = "cpu"   # API server usually CPU; GPU stays for training

# ─── STARTUP: LOAD EVERYTHING ─────────────────────────────────────────────────

# These will be populated on startup
forecasters   = {}
boosters      = {}   # LightGBM forecasters
lstm_weights  = {}   # LSTM share of the blended forecast
autoencoders  = {}
scalers       = {}
thresholds    = {}
interval_scales = {}  # calibrated band width (in MC stds) per horizon step
zone_data     = {}   # in-memory "live" data buffer per zone
weather       = None # hourly temperature, local time


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan handler (replaces deprecated @app.on_event).
    Code before `yield` runs on startup, code after runs on shutdown.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    _load_models()
    yield
    # ── SHUTDOWN (optional cleanup) ──────────────────────────────────────────
    forecasters.clear()
    autoencoders.clear()


app = FastAPI(title="GridSense API", version="1.0", lifespan=lifespan)

# Allow React frontend to call API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_models():
    """
    Load all models, scalers, thresholds, and data into module-level dicts.
    """
    global weather
    print("Loading models...")
    weather = load_weather()

    with open(SAVED_DIR / "thresholds.json") as f:
        all_thresholds = json.load(f)

    # Written by train.py — keeps architecture in sync with the saved weights
    with open(SAVED_DIR / "model_config.json") as f:
        config = json.load(f)
    fc_cfg, ae_cfg = config["forecaster"], config["autoencoder"]

    for z in ZONES:
        # Forecaster
        fc = DemandForecaster(input_size=fc_cfg["input_size"],
                               future_size=fc_cfg["future_size"],
                               hidden_size=fc_cfg["hidden_size"],
                               num_layers=fc_cfg["num_layers"],
                               output_steps=HORIZON, dropout=fc_cfg["dropout"])
        fc.load_state_dict(torch.load(SAVED_DIR / f"forecaster_zone_{z}.pt",
                                       map_location=DEVICE))
        fc.eval()
        forecasters[z] = fc
        boosters[z] = lgb.Booster(model_file=str(SAVED_DIR / f"lgbm_zone_{z}.txt"))
        lstm_weights[z] = config["lstm_weight"][f"zone_{z}"]

        # Autoencoder
        ae = LSTMAutoencoder(seq_len=LOOKBACK, input_size=1,
                              hidden_size=ae_cfg["hidden_size"],
                              latent_dim=ae_cfg["latent_dim"], num_layers=1)
        ae.load_state_dict(torch.load(SAVED_DIR / f"autoencoder_zone_{z}.pt",
                                       map_location=DEVICE))
        ae.eval()
        autoencoders[z] = ae

        # Scaler
        with open(Path("data/processed") / f"scaler_zone_{z}.pkl", "rb") as f:
            scalers[z] = pickle.load(f)

        # Threshold + calibrated forecast band width
        thresholds[z] = all_thresholds[f"zone_{z}"]["threshold"]
        interval_scales[z] = np.array(config["interval_scale"][f"zone_{z}"], dtype=np.float32)

        # Load zone data (we use this as our "live" data stream)
        df = pd.read_csv(DATA_DIR / f"zone_{z}.csv", parse_dates=["datetime"])
        zone_data[z] = df.sort_values("datetime").reset_index(drop=True)

    print(f"✓ All models loaded for zones {ZONES}")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_latest_window(zone_id: int):
    """
    Scaled model inputs for a forecast issued at the last known hour:
      X_past   : (LOOKBACK, INPUT_SIZE)  — last 48 hours of features
      X_future : (HORIZON, FUTURE_SIZE)  — known covariates of the next 24 hours
    In production this would query a real-time database and a weather
    forecast; here we use the tail of the zone CSV. Features come from
    preprocess.build_features so they always match what the models saw.
    """
    demand = zone_data[zone_id].set_index("datetime")["demand_kwh"]
    # Append the 24 future hours with unknown demand: build_features still
    # fills their calendar, holiday, lag-24/168 and temperature columns
    future_idx = pd.date_range(demand.index[-1] + pd.Timedelta(hours=1),
                               periods=HORIZON, freq="h")
    demand = pd.concat([demand, pd.Series(np.nan, index=future_idx)])

    features = build_features(demand, weather).iloc[-(LOOKBACK + HORIZON):].copy()
    features[SCALE_COLS] = scalers[zone_id].transform(features[SCALE_COLS])
    X_past   = features[FEATURE_COLS].iloc[:LOOKBACK].values.astype(np.float32)
    X_future = features[FUTURE_COLS].iloc[LOOKBACK:].values.astype(np.float32)
    return X_past, X_future


# ─── RESPONSE MODELS ──────────────────────────────────────────────────────────

class ForecastResponse(BaseModel):
    zone: int
    timestamps: List[str]           # ISO datetime strings for next 24h
    forecast: List[float]           # mean prediction (kWh)
    lower_bound: List[float]        # 95% CI lower
    upper_bound: List[float]        # 95% CI upper
    actual_history: List[float]     # last 48h actuals for chart context
    history_timestamps: List[str]


class AnomalyEvent(BaseModel):
    timestamp: str
    demand_kwh: float
    reconstruction_error: float
    anomaly_type: str
    severity: str                   # "low" | "medium" | "high"


class AnomalyResponse(BaseModel):
    zone: int
    anomalies: List[AnomalyEvent]
    threshold: float


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(zone: int = 1):
    """
    Returns 24-hour demand forecast with confidence bands for a zone.
    """
    if zone not in ZONES:
        raise HTTPException(400, f"Zone must be one of {ZONES}")

    model  = forecasters[zone]
    scaler = scalers[zone]
    df     = zone_data[zone]

    # Get input windows
    X_past, X_future = get_latest_window(zone)

    # MC Dropout inference — interval_scale=1 makes (mean − lower) the std
    lstm_mean, lstm_lower, _ = predict_with_uncertainty(
        model, torch.FloatTensor(X_past).unsqueeze(0),
        torch.FloatTensor(X_future).unsqueeze(0),
        n_samples=50, device=DEVICE, interval_scale=1.0,
    )
    lstm_mean = lstm_mean.squeeze().numpy()
    std = lstm_mean - lstm_lower.squeeze().numpy()

    # Validation-weighted blend with LightGBM, validation-calibrated 95% bands
    w = lstm_weights[zone]
    gbm_mean = predict_gbm(boosters[zone], X_past[None], X_future[None])[0]
    mean = w * lstm_mean + (1 - w) * gbm_mean
    half = interval_scales[zone] * std

    # Inverse transform (scaled log space → kWh). Bands are asymmetric in
    # kWh, which suits right-skewed demand.
    mean_kwh  = unscale_demand(mean, scaler)
    lower_kwh = unscale_demand(mean - half, scaler)
    upper_kwh = unscale_demand(mean + half, scaler)

    # Generate future timestamps (next 24h from last known timestamp)
    last_ts = pd.to_datetime(df.iloc[-1]["datetime"])
    future_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=HORIZON, freq="h")

    # History: last 48h actual
    history_raw = df.iloc[-LOOKBACK:]["demand_kwh"].values
    history_ts  = df.iloc[-LOOKBACK:]["datetime"].tolist()

    return ForecastResponse(
        zone=zone,
        timestamps=[t.isoformat() for t in future_ts],
        forecast=mean_kwh.tolist(),
        lower_bound=lower_kwh.tolist(),
        upper_bound=upper_kwh.tolist(),
        actual_history=history_raw.tolist(),
        history_timestamps=[str(t) for t in history_ts],
    )


@app.get("/anomalies", response_model=AnomalyResponse)
def get_anomalies(zone: int = 1, lookback_hours: int = 168):
    """
    Scan the last `lookback_hours` of data for anomalies.
    Returns flagged hours with severity classification.

    Severity based on how far the hourly score exceeds the threshold:
      < 20% above threshold  → low
      20–50% above threshold → medium
      > 50% above threshold  → high
    """
    if zone not in ZONES:
        raise HTTPException(400, f"Zone must be one of {ZONES}")
    if lookback_hours < 1:
        raise HTTPException(400, "lookback_hours must be >= 1")

    model     = autoencoders[zone]
    scaler    = scalers[zone]
    threshold = thresholds[zone]
    df        = zone_data[zone]

    # Score the requested hours plus LOOKBACK hours of extra context, so the
    # earliest requested hours are covered by full windows too
    context = df.iloc[-(lookback_hours + LOOKBACK):].reset_index(drop=True)
    demand  = context["demand_kwh"].values
    values  = scale_demand(to_model_space(demand), scaler).astype(np.float32)
    scores, is_anomaly = detect_anomalies(model, values, threshold, DEVICE)

    # Typical level around each hour, to tell spikes from dropouts
    local_median = pd.Series(demand).rolling(25, center=True, min_periods=1).median().values

    anomaly_events = []
    for i in range(max(0, len(context) - lookback_hours), len(context)):
        if not is_anomaly[i]:
            continue
        error  = float(scores[i])
        excess = (error - threshold) / threshold
        severity = "high" if excess > 0.5 else ("medium" if excess > 0.2 else "low")

        anomaly_events.append(AnomalyEvent(
            timestamp=str(context.iloc[i]["datetime"]),
            demand_kwh=float(demand[i]),
            reconstruction_error=round(error, 6),
            anomaly_type="spike" if demand[i] > local_median[i] else "dropout",
            severity=severity,
        ))

    return AnomalyResponse(zone=zone, anomalies=anomaly_events, threshold=threshold)


@app.get("/zones")
def get_zones():
    """Returns available zone metadata (written by generate_zones.py)."""
    with open(DATA_DIR / "zone_clients.json") as f:
        meta = json.load(f)
    return [{"id": z, "label": meta[f"zone_{z}"]["label"],
             "n_clients": len(meta[f"zone_{z}"]["clients"])} for z in ZONES]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(forecasters.keys())}