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
  2. Each request: grab the last 48 hours of "live" data (simulated from test set)
  3. Run model inference → inverse transform → return JSON

WHY load models at startup (not per request):
  Loading a PyTorch model takes ~0.5s. At 30 requests/min that's 15s/min wasted.
  Startup loading means models live in RAM — inference is <10ms per request.

RUN:
  uvicorn api.main:app --reload --port 8000
"""

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

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.forecaster import DemandForecaster, predict_with_uncertainty
from models.autoencoder import LSTMAutoencoder, detect_anomalies

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAVED_DIR    = Path("models/saved")
DATA_DIR     = Path("data/zones")
LOOKBACK     = 48
HORIZON      = 24
DEVICE       = "cpu"   # API server usually CPU; GPU stays for training
ZONES        = [1, 2, 3, 4]

# ─── STARTUP: LOAD EVERYTHING ─────────────────────────────────────────────────

# These will be populated on startup
forecasters   = {}
autoencoders  = {}
scalers       = {}
thresholds    = {}
zone_data     = {}   # in-memory "live" data buffer per zone


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan handler (replaces deprecated @app.on_event).
    Code before `yield` runs on startup, code after runs on shutdown.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    print("Loading models...")
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
    print("Loading models...")

    with open(SAVED_DIR / "thresholds.json") as f:
        all_thresholds = json.load(f)

    for z in ZONES:
        # Forecaster
        fc = DemandForecaster(input_size=1, hidden_size=128, num_layers=2,
                               output_steps=HORIZON, dropout=0.2)
        fc.load_state_dict(torch.load(SAVED_DIR / f"forecaster_zone_{z}.pt",
                                       map_location=DEVICE))
        fc.eval()
        forecasters[z] = fc

        # Autoencoder
        ae = LSTMAutoencoder(seq_len=LOOKBACK, input_size=1,
                              hidden_size=64, latent_dim=32, num_layers=1)
        ae.load_state_dict(torch.load(SAVED_DIR / f"autoencoder_zone_{z}.pt",
                                       map_location=DEVICE))
        ae.eval()
        autoencoders[z] = ae

        # Scaler
        with open(Path("data/processed") / f"scaler_zone_{z}.pkl", "rb") as f:
            scalers[z] = pickle.load(f)

        # Threshold
        thresholds[z] = all_thresholds[f"zone_{z}"]["threshold"]

        # Load zone data (we use this as our "live" data stream)
        df = pd.read_csv(DATA_DIR / f"zone_{z}.csv", parse_dates=["datetime"])
        zone_data[z] = df.sort_values("datetime").reset_index(drop=True)

    print(f"✓ All models loaded for zones {ZONES}")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_latest_window(zone_id: int) -> np.ndarray:
    """
    Get the last 48 hours of normalized demand for a zone.
    In production this would query a real-time database.
    Here we take the last LOOKBACK rows of the test set.
    """
    df = zone_data[zone_id]
    scaler = scalers[zone_id]

    # Take last LOOKBACK + HORIZON rows from test portion (last 15%)
    n = len(df)
    test_start = int(n * 0.85)
    recent = df.iloc[-(LOOKBACK + HORIZON):-HORIZON]["demand_kwh"].values
    
    if len(recent) < LOOKBACK:
        recent = df.iloc[-LOOKBACK:]["demand_kwh"].values

    normalized = scaler.transform(recent.reshape(-1, 1)).flatten()
    return normalized  # (LOOKBACK,)


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

    # Get input window
    window = get_latest_window(zone)
    X = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)  # (1, 48, 1)

    # MC Dropout inference
    mean, lower, upper = predict_with_uncertainty(model, X, n_samples=50, device=DEVICE)

    # Inverse transform to kWh
    mean_kwh  = scaler.inverse_transform(mean.squeeze().numpy().reshape(-1, 1)).flatten()
    lower_kwh = scaler.inverse_transform(lower.squeeze().numpy().reshape(-1, 1)).flatten()
    upper_kwh = scaler.inverse_transform(upper.squeeze().numpy().reshape(-1, 1)).flatten()

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
    Returns flagged events with severity classification.

    Severity based on how many σ above threshold:
      < 1σ beyond threshold → low
      1–2σ beyond threshold → medium
      > 2σ beyond threshold → high
    """
    if zone not in ZONES:
        raise HTTPException(400, f"Zone must be one of {ZONES}")

    model     = autoencoders[zone]
    scaler    = scalers[zone]
    threshold = thresholds[zone]
    df        = zone_data[zone]

    # Get recent data
    recent = df.iloc[-lookback_hours:].reset_index(drop=True)
    values = scaler.transform(recent["demand_kwh"].values.reshape(-1, 1)).flatten()

    anomaly_events = []

    # Slide a LOOKBACK window over recent data
    for i in range(LOOKBACK, len(values)):
        window = values[i - LOOKBACK:i]
        X = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)  # (1, 48, 1)
        errors, is_anomaly = detect_anomalies(model, X, threshold, DEVICE)

        if is_anomaly[0]:
            error = float(errors[0])
            demand = float(recent.iloc[i]["demand_kwh"])
            ts = str(recent.iloc[i]["datetime"])

            # Severity
            excess = (error - threshold) / threshold
            severity = "high" if excess > 0.5 else ("medium" if excess > 0.2 else "low")

            # Try to use ground truth type if available, else infer
            atype = recent.iloc[i].get("anomaly_type", "unknown")
            if atype == "normal" or atype == "unknown":
                atype = "spike" if demand > recent["demand_kwh"].mean() * 1.5 else "dropout"

            anomaly_events.append(AnomalyEvent(
                timestamp=ts,
                demand_kwh=demand,
                reconstruction_error=round(error, 6),
                anomaly_type=atype,
                severity=severity,
            ))

    return AnomalyResponse(zone=zone, anomalies=anomaly_events, threshold=threshold)


@app.get("/zones")
def get_zones():
    """Returns available zone metadata."""
    zone_labels = {1: "Residential", 2: "Industrial", 3: "Commercial", 4: "Mixed"}
    return [{"id": z, "label": zone_labels[z]} for z in ZONES]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(forecasters.keys())}