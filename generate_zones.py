"""
generate_zones.py
─────────────────
PURPOSE:
  Load UCI household power dataset → resample to hourly →
  replicate into 4 synthetic zones with realistic variation.

WHY THIS APPROACH:
  The UCI dataset is a single household. Real grids have multiple zones
  (industrial, residential, commercial, mixed). We simulate this by:
    - Zone 1 (Residential)  : base signal, evening peak
    - Zone 2 (Industrial)   : morning + afternoon peak, lower night
    - Zone 3 (Commercial)   : flat daytime, sharp drop at night
    - Zone 4 (Mixed)        : blend of all three + most noise

  Each zone gets a different scale factor, phase shift, and noise level.
  This teaches you how domain adaptation works: same underlying signal,
  different real-world characteristics.

OUTPUT:
  data/zones/zone_{1..4}.csv  — hourly demand per zone
  data/zones/combined.csv     — all zones merged
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RAW_DATA_PATH = "household_power_consumption.txt"  # UCI dataset path
OUTPUT_DIR    = Path("data/zones")
RANDOM_SEED   = 42

# Zone characteristics: (scale, phase_shift_hours, noise_std, label)
ZONE_CONFIG = {
    1: dict(scale=1.00, phase=0,  noise=0.05, label="Residential"),
    2: dict(scale=1.35, phase=-2, noise=0.08, label="Industrial"),
    3: dict(scale=0.80, phase=1,  noise=0.04, label="Commercial"),
    4: dict(scale=1.10, phase=0,  noise=0.12, label="Mixed"),
}

# ─── LOAD & CLEAN ─────────────────────────────────────────────────────────────

def load_uci_dataset(path: str) -> pd.DataFrame:
    """
    UCI dataset is semicolon-separated with a Date + Time column.
    Missing values are marked as '?'  — we interpolate them.
    
    WHY interpolate vs drop:
      Dropping creates irregular time series — LSTMs need uniform intervals.
      Linear interpolation is safe here since gaps are short (< 1hr usually).
    """
    print("Loading UCI dataset...")
    df = pd.read_csv(
        path,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        low_memory=False,
        na_values=["?"],
    )
    df = df.sort_values("datetime").set_index("datetime")

    # We only need Global_active_power (kW) as our demand signal
    demand = df["Global_active_power"].astype(float)

    # Interpolate missing values linearly
    n_missing = demand.isna().sum()
    print(f"  → Interpolating {n_missing} missing values ({n_missing/len(demand)*100:.1f}%)")
    demand = demand.interpolate(method="linear").ffill().bfill()

    # Resample from 1-minute to hourly by summing (kWh per hour)
    # WHY sum not mean: power (kW) × time (1/60 hr) = energy (kWh)
    # Summing 60 one-minute readings gives total energy for that hour
    hourly = demand.resample("h").sum()
    print(f"  → Resampled to {len(hourly)} hourly records "
          f"({hourly.index[0]} → {hourly.index[-1]})")
    return hourly


# ─── ZONE REPLICATION ─────────────────────────────────────────────────────────

def add_zone_profile(series: pd.Series, zone_id: int, rng: np.random.Generator) -> pd.Series:
    """
    Transforms the base hourly demand into a zone-specific profile.

    Three transformations:
    1. PHASE SHIFT  — different zones peak at different hours of day
                      (industrial peaks 2h earlier than residential)
    2. SCALE        — zones have different absolute demand magnitudes
    3. NOISE        — realistic measurement + behavioral variation

    WHY additive noise over multiplicative:
      Multiplicative noise creates unrealistically large swings during peaks.
      Additive Gaussian noise better models sensor jitter + small behavioral variance.
    """
    cfg = ZONE_CONFIG[zone_id]

    # Phase shift: roll time series by N hours
    shifted = series.copy()
    if cfg["phase"] != 0:
        shifted = pd.Series(
            np.roll(shifted.values, cfg["phase"]),
            index=shifted.index
        )

    # Scale the zone
    scaled = shifted * cfg["scale"]

    # Add Gaussian noise (proportional to mean to keep SNR consistent)
    noise = rng.normal(0, cfg["noise"] * scaled.mean(), size=len(scaled))
    noisy = (scaled + noise).clip(lower=0)  # demand can't be negative

    noisy.name = f"zone_{zone_id}"
    return noisy


def inject_anomalies(series: pd.Series, rng: np.random.Generator, anomaly_rate: float = 0.01):
    """
    Injects labeled anomalies for evaluation purposes.
    
    WHY we inject anomalies:
      The UCI dataset has natural anomalies but they're unlabeled.
      For a confusion matrix you need ground truth.
      We inject on top of the real data so the model still learns real patterns.
    
    Two anomaly types:
      - SPIKE   : sudden 3–5x demand surge (equipment failure, heat wave)
      - DROPOUT : near-zero demand (outage, sensor failure)
    
    Returns modified series + DataFrame of anomaly timestamps + types.
    """
    n_anomalies = int(len(series) * anomaly_rate)
    anomaly_indices = rng.choice(len(series), size=n_anomalies, replace=False)

    labels = pd.DataFrame(index=series.index)
    labels["is_anomaly"] = 0
    labels["anomaly_type"] = "normal"

    result = series.copy()
    for idx in anomaly_indices:
        atype = rng.choice(["spike", "dropout"])
        if atype == "spike":
            result.iloc[idx] *= rng.uniform(3.0, 5.0)
            labels.iloc[idx, labels.columns.get_loc("anomaly_type")] = "spike"
        else:
            result.iloc[idx] *= rng.uniform(0.0, 0.1)
            labels.iloc[idx, labels.columns.get_loc("anomaly_type")] = "dropout"
        labels.iloc[idx, labels.columns.get_loc("is_anomaly")] = 1

    return result, labels


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    base = load_uci_dataset(RAW_DATA_PATH)

    all_zones = {}
    all_labels = {}

    for zone_id, cfg in ZONE_CONFIG.items():
        print(f"\nGenerating Zone {zone_id} ({cfg['label']})...")
        zone_series = add_zone_profile(base, zone_id, rng)
        zone_series, labels = inject_anomalies(zone_series, rng, anomaly_rate=0.01)

        # Save individual zone
        out = pd.DataFrame({
            "datetime": zone_series.index,
            "demand_kwh": zone_series.values,
            "is_anomaly": labels["is_anomaly"].values,
            "anomaly_type": labels["anomaly_type"].values,
        })
        out.to_csv(OUTPUT_DIR / f"zone_{zone_id}.csv", index=False)
        print(f"  → Saved zone_{zone_id}.csv | {labels['is_anomaly'].sum()} anomalies injected")

        all_zones[f"zone_{zone_id}"] = zone_series
        all_labels[f"zone_{zone_id}_anomaly"] = labels["is_anomaly"]

    # Save combined
    combined = pd.DataFrame(all_zones)
    for k, v in all_labels.items():
        combined[k] = v.values
    combined.to_csv(OUTPUT_DIR / "combined.csv")
    print(f"\n✓ Combined dataset saved → {OUTPUT_DIR}/combined.csv")
    print(f"  Shape: {combined.shape} | Date range: {combined.index[0]} → {combined.index[-1]}")


if __name__ == "__main__":
    main()