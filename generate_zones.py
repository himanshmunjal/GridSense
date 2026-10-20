"""
generate_zones.py
─────────────────
PURPOSE:
  Build 4 grid zones from REAL metered load: the UCI ElectricityLoadDiagrams
  2011–2014 dataset (370 Portuguese clients, 15-minute readings).
    → keep clients metered for the whole 2012–2014 period
    → group clients by the SHAPE of their average weekly load profile
    → each zone = the summed hourly load of one group
    → inject labelled multi-hour anomalies for evaluation

WHY GROUP BY LOAD SHAPE INSTEAD OF RANDOMLY:
  Real distribution zones differ in shape, not just size. From each
  client's normalised 168-hour weekly profile we measure three descriptors
  and fill three zones of ZONE_SIZE clients greedily:
    Zone 3 Weekday-business : lowest weekend/weekday ratio
    Zone 2 Baseload 24/7    : highest night/average ratio (of the rest)
    Zone 1 Evening-peak     : highest evening/daytime ratio (of the rest)
    Zone 4 Mixed            : everyone left
  Plain KMeans on the profiles was tried first: it put the single
  weekday-only outlier client in its own cluster, which is not a zone.
  The dataset does not say which clients are homes or businesses, so the
  zones are named by measured shape rather than by customer type.
  Only the first TRAIN_FRAC of the period is used to define zones, so they
  never see the test period.

WHY DROP CLIENTS ABOVE MAX_CLIENT_KWH:
  Three meters average 11–50 MWh/h — more than whole groups of ordinary
  clients combined; they look like substation-level feeds. Any zone holding
  one would effectively be a single meter.

DATA NOTES (from the UCI dataset description):
  - Values are kW averaged over each 15 minutes → kWh per 15 min = kW / 4.
  - Timestamps are Portuguese local time and label the END of each interval.
  - On the March DST day the 01:00–02:00 hour is all zeros; on the October
    DST day it holds two hours of consumption. Both are repaired below.

DOWNLOAD:
  https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
  → unzip LD2011_2014.txt into data/raw/

OUTPUT:
  data/zones/zone_{1..4}.csv  — hourly demand per zone (kWh), pre-injection
                                demand, anomaly labels
  data/zones/combined.csv     — all zones merged
  data/zones/zone_clients.json — which clients make up each zone
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RAW_DATA_PATH = Path("data/raw/LD2011_2014.txt")
OUTPUT_DIR    = Path("data/zones")
RANDOM_SEED   = 42

START, END    = "2012-01-01 00:00", "2014-12-31 23:00"   # hourly labels kept
TRAIN_FRAC    = 0.70     # must match preprocess.py — clustering uses train only
ZONE_SIZE     = 60       # clients in each shape-defined zone (zone 4 gets the rest)
MAX_CLIENT_KWH = 10_000  # mean kWh/h above which a meter is treated as a feeder
ANOMALY_RATE  = 0.01     # share of hours inside an anomaly event

ZONE_LABELS = {1: "Evening-peak", 2: "Baseload 24/7", 3: "Weekday-business", 4: "Mixed"}

# ─── LOAD & CLEAN ─────────────────────────────────────────────────────────────

def load_hourly_clients(path: Path) -> pd.DataFrame:
    """15-minute kW readings → hourly kWh per client, 2012–2014."""
    print("Loading UCI ElectricityLoadDiagrams...")
    df = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.astype(np.float32) / 4.0       # kW over 15 min → kWh

    # Readings label the END of each 15-min interval: 00:15…01:00 → hour 00:00
    hourly = df.resample("h", closed="right", label="left").sum()
    hourly = hourly.loc[START:END]
    print(f"  → {hourly.shape[1]} clients, {len(hourly)} hours "
          f"({hourly.index[0]} → {hourly.index[-1]})")
    return hourly


def select_active_clients(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Keep clients that are metered throughout (≤1% zero hours, live at both
    ends) and are not feeder-sized (see header).
    """
    zero_share = (hourly <= 0).mean()
    live_start = (hourly.iloc[:24 * 7] > 0).mean() > 0.5
    live_end   = (hourly.iloc[-24 * 7:] > 0).mean() > 0.5
    active = (zero_share <= 0.01) & live_start & live_end
    feeder = hourly.iloc[: int(len(hourly) * TRAIN_FRAC)].mean() > MAX_CLIENT_KWH
    keep = active & ~feeder
    print(f"  → {active.sum()} clients active for the whole period, "
          f"{(active & feeder).sum()} feeder-sized dropped → {keep.sum()} kept")
    return hourly.loc[:, keep]


def fix_dst(zone: pd.Series) -> pd.Series:
    """Repair the DST artefacts described in the header."""
    zone = zone.copy()
    for year in sorted(set(zone.index.year)):
        last_sun = lambda month: max(
            d for d in pd.date_range(f"{year}-{month}-01", periods=31, freq="D")
            if d.month == month and d.dayofweek == 6
        )
        march = last_sun(3) + pd.Timedelta(hours=1)
        october = last_sun(10) + pd.Timedelta(hours=1)
        if march in zone.index:
            zone[march] = np.nan               # missing hour → interpolate
        if october in zone.index:
            zone[october] /= 2                 # two hours summed → one hour
    return zone.interpolate()


# ─── ZONES BY LOAD-SHAPE CLUSTERING ───────────────────────────────────────────

def weekly_profiles(hourly: pd.DataFrame) -> pd.DataFrame:
    """Mean load per (day-of-week, hour), each client normalised by its own mean."""
    idx = hourly.index
    prof = hourly.groupby([idx.dayofweek, idx.hour]).mean()     # (168, clients)
    return (prof / prof.mean()).T                               # (clients, 168)


def profile_stats(profile: np.ndarray) -> dict:
    """Shape descriptors of a normalised 168-h profile (Mon 00h … Sun 23h)."""
    p = profile.reshape(7, 24)
    weekday, weekend = p[:5].mean(), p[5:].mean()
    return {
        "weekend_ratio": weekend / weekday,                          # <1: shuts on weekends
        "night_ratio":   p[:, 0:6].mean() / p.mean(),                # ~1: runs 24/7
        "evening_ratio": p[:, 18:23].mean() / p[:, 9:17].mean(),     # >1: residential
    }


def assign_zones(hourly: pd.DataFrame) -> dict:
    """Train-period weekly profiles → {zone_id: [client names]} (see header)."""
    train = hourly.iloc[: int(len(hourly) * TRAIN_FRAC)]
    profiles = weekly_profiles(train)
    stats = pd.DataFrame([profile_stats(profiles.loc[c].values) for c in profiles.index],
                         index=profiles.index)

    zones, rest = {}, stats
    for zone_id, key, ascending in [(3, "weekend_ratio", True),
                                    (2, "night_ratio", False),
                                    (1, "evening_ratio", False)]:
        members = rest.sort_values(key, ascending=ascending).index[:ZONE_SIZE]
        zones[zone_id] = list(members)
        rest = rest.drop(members)
    zones[4] = list(rest.index)

    for zone_id in sorted(zones):
        load = train[zones[zone_id]]
        s = profile_stats(weekly_profiles(load.sum(axis=1).to_frame()).values[0])
        biggest = load.mean().max() / load.mean().sum()
        print(f"  Zone {zone_id} ({ZONE_LABELS[zone_id]:16s}): {len(zones[zone_id]):3d} clients | "
              f"weekend/weekday {s['weekend_ratio']:.2f} | night {s['night_ratio']:.2f} | "
              f"evening/day {s['evening_ratio']:.2f} | largest client {biggest:.0%}")
    return dict(sorted(zones.items()))


# ─── ANOMALY INJECTION ────────────────────────────────────────────────────────

def inject_anomalies(series: pd.Series, rng: np.random.Generator, anomaly_rate: float):
    """
    Inject labelled multi-hour anomaly events for evaluation.

    WHY we inject anomalies:
      Real meter data has no ground-truth fault labels. Injecting on top of
      real load gives a confusion matrix while the models still learn real
      patterns.

    Two event types, each lasting 1–6 hours:
      - SPIKE   : sustained 1.4–2.5× surge (fault current, unmetered load joining)
      - DROPOUT : partial outage, demand falls to 0–50% (feeder trip, meter fault)

    Magnitudes are deliberately moderate — on smooth aggregate load a 3–5×
    single-hour spike is trivial to catch and would inflate the scores.
    """
    n_target = int(len(series) * anomaly_rate)
    is_anom = np.zeros(len(series), dtype=int)
    types = np.array(["normal"] * len(series), dtype=object)
    values = series.values.copy()

    while is_anom.sum() < n_target:
        length = int(rng.integers(1, 7))
        start = int(rng.integers(0, len(series) - length))
        # keep events separated so each one is a distinct incident
        if is_anom[max(0, start - 24): start + length + 24].any():
            continue
        span = slice(start, start + length)
        if rng.random() < 0.5:
            values[span] *= rng.uniform(1.4, 2.5)
            types[span] = "spike"
        else:
            values[span] *= rng.uniform(0.0, 0.5)
            types[span] = "dropout"
        is_anom[span] = 1

    return pd.Series(values, index=series.index), is_anom, types


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    hourly = select_active_clients(load_hourly_clients(RAW_DATA_PATH))
    zones = assign_zones(hourly)

    combined = {}
    for zone_id, members in zones.items():
        clean = fix_dst(hourly[members].sum(axis=1).astype(float))
        demand, is_anom, types = inject_anomalies(clean, rng, ANOMALY_RATE)

        out = pd.DataFrame({
            "datetime": demand.index,
            "demand_kwh": demand.values,
            # pre-injection load — train.py re-injects fresh anomaly sets
            # into it for a statistically stable anomaly evaluation
            "demand_clean_kwh": clean.values,
            "is_anomaly": is_anom,
            "anomaly_type": types,
        })
        out.to_csv(OUTPUT_DIR / f"zone_{zone_id}.csv", index=False)
        n_events = int((np.diff(np.r_[0, is_anom]) == 1).sum())
        print(f"  → zone_{zone_id}.csv | mean {demand.mean():,.0f} kWh/h | "
              f"{is_anom.sum()} anomalous hours in {n_events} events")

        combined[f"zone_{zone_id}"] = demand
        combined[f"zone_{zone_id}_anomaly"] = is_anom

    pd.DataFrame(combined).to_csv(OUTPUT_DIR / "combined.csv")
    with open(OUTPUT_DIR / "zone_clients.json", "w") as f:
        json.dump({f"zone_{z}": {"label": ZONE_LABELS[z], "clients": m}
                   for z, m in zones.items()}, f, indent=2)
    print(f"\n✓ Zones saved → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
