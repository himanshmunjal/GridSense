# GridSense

Day-ahead (24-hour) electricity demand forecasting and anomaly detection for
four grid zones built from **309 real metered clients** (UCI
ElectricityLoadDiagrams 2011–2014, Portugal), served through a FastAPI backend
and a React dashboard.

- **Forecasters** — an encoder–decoder LSTM and a LightGBM model, both fed
  the past 48 hours plus the known next 24 hours (calendar, public holidays,
  temperature forecast). The served forecast is a validation-weighted blend
  of the two, with calibrated 95% bands from MC Dropout.
- **Baselines** — seasonal naive (same hour yesterday / last week); every
  model is scored against them on the same windows.
- **Anomaly detector** — LSTM autoencoder trained on normal load only; every
  hour is scored by how badly the model can reconstruct it.
- **API** — `/forecast`, `/anomalies`, `/zones`, `/health`.
- **Dashboard** — live forecast chart, anomaly feed and heatmap per zone.

---

## Headline results (test split, last 15% of 2012–2014)

| Zone | MAPE | WAPE | MAE vs best naive | 95% band coverage | Anomaly F1 (20 rounds) | Incidents caught |
|---|---|---|---|---|---|---|
| 1 Evening-peak | 3.39% | 3.17% | −26.5% | 96.4% | 0.727 ± 0.073 | 91.0% |
| 2 Baseload 24/7 | 4.40% | 3.46% | −30.7% | 95.9% | 0.876 ± 0.068 | 93.2% |
| 3 Weekday-business | 3.56% | 3.34% | −30.3% | 96.7% | 0.930 ± 0.054 | 96.4% |
| 4 Mixed | 4.10% | 3.00% | −21.5% | 94.6% | 0.784 ± 0.100 | 82.7% |

*MAE vs best naive* compares the served blend with the better of "same hour
yesterday" and "same hour last week" for that zone (last week for the
weekday-business zone, whose Mondays look nothing like Sundays). Anomaly F1
is the mean ± std over 20 independent random anomaly injections into the
test period.

For reference, the previous version of this project (one household cloned
into four zones) scored 54–63% MAPE, and the same-hour-yesterday baseline
on that data scored 69–84%.

---

## Project structure

```
GridSense/
├── generate_zones.py        # 370 UCI meters → 4 load-shape zones + injected anomaly events
├── preprocess.py            # features (weather, holidays, lags), scaling, windows — shared with API
├── train.py                 # trains, compares and evaluates every model for every zone
├── models/
│   ├── forecaster.py        # encoder–decoder LSTM, MC Dropout inference, band calibration
│   ├── gbm.py               # LightGBM forecaster on the same windows
│   ├── autoencoder.py       # LSTM autoencoder, hourly scoring, threshold calibration
│   └── saved/               # weights, LightGBM models, thresholds.json, model_config.json
├── api/main.py              # FastAPI server
├── dashboard/               # React + Vite + Recharts frontend
├── data/
│   ├── raw/                 # LD2011_2014.txt (downloaded, git-ignored)
│   ├── weather/             # Lisbon hourly weather 2011–2014 (Open-Meteo archive)
│   ├── zones/               # zone_{1..4}.csv, combined.csv, zone_clients.json
│   └── processed/           # fitted scalers per zone
└── results/
    ├── metrics.json             # forecasting metrics: every model, ablations
    └── confusion_matrices.json  # anomaly detection metrics per zone
```

---

## Quick start

### 1. Python environment

Python 3.11 is used during development.

```bash
pip install -r requirements.txt
```

Training uses CUDA, then Apple Silicon (MPS), then CPU.

### 2. Data

The zone CSVs in `data/zones/` and the weather file are committed, so you can
skip this step. To rebuild the zones, download the
[UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
dataset, unzip `LD2011_2014.txt` into `data/raw/`, then:

```bash
python generate_zones.py
```

The weather file was fetched from the
[Open-Meteo historical archive](https://open-meteo.com/en/docs/historical-weather-api)
for Lisbon (38.72 N, −9.14 E), hourly, in UTC.

### 3. Train

```bash
python train.py
```

| File | Contents |
|---|---|
| `models/saved/forecaster_zone_{z}.pt` | encoder–decoder LSTM weights |
| `models/saved/lgbm_zone_{z}.txt` | LightGBM model |
| `models/saved/autoencoder_zone_{z}.pt` | autoencoder weights |
| `models/saved/thresholds.json` | anomaly threshold per zone |
| `models/saved/model_config.json` | architecture, blend weights, calibrated band widths (read by the API) |
| `data/processed/scaler_zone_{z}.pkl` | fitted scaler per zone |
| `results/metrics.json` | forecasting metrics for every model + ablations |
| `results/confusion_matrices.json` | anomaly detection metrics |

Takes roughly an hour on an Apple M1 (8 GB).

### 4. API

```bash
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /forecast?zone=1` | next 24h blended forecast, lower/upper 95% band, last 48h of actuals |
| `GET /anomalies?zone=1&lookback_hours=168` | flagged hours with score, type (spike/dropout) and severity |
| `GET /zones` | zone ids, labels and client counts |
| `GET /health` | loaded models |

The API replays the end of each zone CSV as its "live" stream; the observed
weather of the next 24 hours stands in for a weather forecast.

### 5. Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Expects the API at `http://localhost:8000`.

---

## Data

### Zones (`generate_zones.py`)

The UCI dataset has 15-minute kW readings for 370 Portuguese clients. The
pipeline:

1. Converts to hourly kWh (kW over 15 min ÷ 4, summed per hour; readings
   label the end of their interval).
2. Keeps 2012–2014 and clients metered throughout (≤1% zero hours) → 312.
3. Drops 3 feeder-sized meters (11–50 MWh/h on average) — any zone holding
   one would effectively be a single meter → **309 clients**.
4. Groups clients by the **shape** of their normalised weekly profile,
   measured on the training period only:

| Zone | Name | Clients | Selection rule | Mean load |
|---|---|---|---|---|
| 1 | Evening-peak | 60 | highest evening / daytime ratio | 36.5 MWh/h |
| 2 | Baseload 24/7 | 60 | highest night / average ratio | 9.6 MWh/h |
| 3 | Weekday-business | 60 | lowest weekend / weekday ratio | 15.2 MWh/h |
| 4 | Mixed | 129 | everyone else | 60.5 MWh/h |

   Zones are filled greedily in the order 3 → 2 → 1, so each is 60 clients.
   Plain KMeans was tried first; it isolated a single weekday-only client in
   its own cluster. The dataset does not say which clients are homes or
   businesses, so zones are named by measured shape, not customer type.
5. Repairs the DST artefacts documented by UCI (a zero hour every March, a
   double hour every October).
6. Injects labelled **anomaly events** into 1% of hours, each 1–6 hours long
   and separated by ≥24h:
   - **spike** — sustained ×1.4–2.5 surge
   - **dropout** — partial outage, ×0–0.5

   The pre-injection load is kept in `demand_clean_kwh` for evaluation.

### Weather and holidays

- **Temperature** — Open-Meteo hourly 2 m temperature for Lisbon, converted
  from UTC to Portuguese local time (the load data's clock).
- **Holidays** — Portuguese national public holidays from the `holidays`
  package (which also knows the four holidays suspended in 2013–2015).

---

## Pipeline

### Features (`preprocess.py`)

1. **Clean the training history** — labelled anomalies inside the training
   split are interpolated away so the forecasters learn normal behaviour.
   Validation and test stay raw.
2. **log1p transform** of demand; multiplicative anomalies become additive.
3. **Past features** (15 per hour, encoder input): log demand, lag-24h,
   lag-168h, rolling mean/std 24h, rolling mean 168h, hour / day-of-week /
   day-of-year sin-cos, weekend flag, holiday flag, temperature.
4. **Known-future covariates** (11 per forecast hour, decoder input): lag-24h
   and lag-168h demand of that hour (both ≤ forecast time), hour /
   day-of-week / day-of-year sin-cos, weekend flag, holiday flag,
   **temperature forecast**.
5. **Simulated weather-forecast error** — archived forecasts are not
   available for 2012–2014, so every window's future temperature gets
   Gaussian noise (σ = 1.5 °C, typical of a day-ahead forecast). Using the
   true future temperature would leak information. Noise is seeded, so all
   models see identical inputs.
6. **Split** chronologically 70 / 15 / 15, StandardScaler fit on train only.

`build_features()` is shared with the API so inference always matches training.

### LSTM forecaster (`models/forecaster.py`)

- **Encoder–decoder**: 2-layer encoder LSTM over the past 48h; its final
  state initialises a 2-layer decoder LSTM that walks the 24 target hours
  with their covariates; MC Dropout + per-step MLP head.
- L1 loss in log space (conditional median), AdamW (weight decay 1e-3),
  dropout 0.3, ReduceLROnPlateau, gradient clipping, early stopping.
- Point forecast = mean of 30–50 MC Dropout passes.

### LightGBM forecaster (`models/gbm.py`)

- One model for all 24 horizons: each (window, step) is a row with the
  target hour's future covariates, the step index, and a summary of the
  past window (last readings, 24h mean, rolling stats, lag-48h).
- L1 objective, early stopping on validation.
- Built from the LSTM's exact windows → apples-to-apples comparison.

### Blend and uncertainty

- **Blend** `w · LSTM + (1 − w) · LightGBM`, with `w` chosen on validation
  MAE (grid 0, 0.05, …, 1). `w = 1` or `0` would mean one model alone wins.
- **Calibrated 95% bands** — on validation, measure how many LSTM MC-Dropout
  standard deviations the truth falls from the blended forecast at each
  horizon step; the 95th percentile is the band half-width. Bands are in log
  space, so asymmetric in kWh.

### Anomaly detector (`models/autoencoder.py`)

- LSTM encoder → 8-dim latent → LSTM decoder on scaled log demand only,
  trained on 48h windows with no labelled anomaly.
- **Hourly scoring** — mean squared reconstruction error of each hour across
  the up-to-48 windows covering it.
- **Threshold** — max-F1 on validation, pooled over the fixed injection plus
  20 fresh random injections into the clean validation load (~270 events
  instead of 8–17).
- The API labels a flagged hour as *spike* or *dropout* against the local
  25h median and grades severity by how far the score exceeds the threshold.

---

## Evaluation

### Forecasting (`results/metrics.json`)

All on the test split, every 24h-ahead window, in kWh.

| Metric | Meaning |
|---|---|
| `MAE`, `RMSE` | mean absolute / root mean squared error |
| `MAPE` | mean absolute % error (hours below 10% of mean load ignored) |
| `WAPE` | Σ\|error\| / Σ\|actual\| — share of total energy mis-forecast |
| `models` | the above for `naive_lag24`, `naive_lag168`, `lstm`, `lightgbm`, `ensemble` |
| `skill_vs_naive_MAE` | 1 − MAE_served / MAE_naive-24h |
| `skill_vs_weekly_naive_MAE` | 1 − MAE_served / MAE_naive-168h |
| `interval_coverage_95` | share of actuals inside the 95% band (target 0.95) |
| `ablations` | LightGBM retrained without weather / without holidays |

### Anomaly detection (`results/confusion_matrices.json`)

- **Fixed injection** — confusion matrix and P/R/F1 on the anomalies in the
  zone CSV (what the API shows). Only 6–10 events fall in each test split,
  so these numbers swing a lot: one zone scores F1 = 1.0 by luck.
- **Monte Carlo (`monte_carlo`)** — mean ± std over 20 independent anomaly
  injections into the clean test load. These are the headline numbers.
- **Event level** — `event_recall`: share of incidents with ≥1 flagged hour;
  `event_precision`: share of alarms (runs of flagged hours) that overlap a
  real incident.

---

## Results

### Forecasting — model comparison (test MAPE / MAE in kWh)

| Zone | Naive 24h | Naive 168h | LSTM | LightGBM | **Blend** | LSTM weight |
|---|---|---|---|---|---|---|
| 1 Evening-peak | 4.67% / 1,577 | 5.46% / 1,963 | 3.54% / 1,237 | 3.67% / 1,238 | **3.39% / 1,160** | 0.45 |
| 2 Baseload 24/7 | 5.91% / 490 | 6.13% / 511 | 4.80% / 383 | 4.43% / 342 | **4.40% / 340** | 0.15 |
| 3 Weekday-business | 10.49% / 1,623 | 5.14% / 775 | 4.02% / 609 | 3.68% / 558 | **3.56% / 540** | 0.20 |
| 4 Mixed | 5.09% / 2,316 | 6.58% / 3,240 | 4.44% / 2,010 | 4.22% / 1,864 | **4.10% / 1,819** | 0.50 |

### Forecasting — served blend (test, kWh)

| Zone | MAE | RMSE | MAPE | WAPE | Normal-hours MAPE | Skill vs naive 24h | Skill vs naive 168h | 95% coverage |
|---|---|---|---|---|---|---|---|---|
| 1 Evening-peak | 1,160 | 2,771 | 3.39% | 3.17% | 2.83% | 26.5% | 40.9% | 96.4% |
| 2 Baseload 24/7 | 340 | 941 | 4.40% | 3.46% | 2.62% | 30.7% | 33.5% | 95.9% |
| 3 Weekday-business | 540 | 1,186 | 3.56% | 3.34% | 2.98% | 66.7% | 30.3% | 96.7% |
| 4 Mixed | 1,819 | 5,106 | 4.10% | 3.00% | 2.70% | 21.4% | 43.9% | 94.6% |

### Ablations — LightGBM test MAPE without a feature group

| Zone | Full | No weather | No holidays |
|---|---|---|---|
| 1 Evening-peak | 3.67% | 3.82% (+0.15) | 3.79% (+0.12) |
| 2 Baseload 24/7 | 4.43% | 4.50% (+0.07) | 4.49% (+0.05) |
| 3 Weekday-business | 3.68% | 3.73% (+0.05) | 3.92% (+0.24) |
| 4 Mixed | 4.22% | 4.38% (+0.16) | 4.38% (+0.16) |

### Anomaly detection — 20 random injections into the clean test load (mean ± std)

| Zone | Precision | Recall | F1 | ROC-AUC | Avg. precision | Incidents caught | Alarm precision |
|---|---|---|---|---|---|---|---|
| 1 Evening-peak | 0.623 | 0.892 | 0.727 ± 0.073 | 0.998 | 0.892 | 91.0% | 68.4% |
| 2 Baseload 24/7 | 0.898 | 0.867 | 0.876 ± 0.068 | 0.999 | 0.949 | 93.2% | 90.0% |
| 3 Weekday-business | 0.932 | 0.934 | 0.930 ± 0.054 | 1.000 | 0.977 | 96.4% | 96.9% |
| 4 Mixed | 0.814 | 0.786 | 0.784 ± 0.100 | 0.998 | 0.900 | 82.7% | 81.1% |

### Anomaly detection — fixed injection in the zone CSVs (6–10 events per zone)

| Zone | Precision | Recall | F1 | TP / FN / FP |
|---|---|---|---|---|
| 1 Evening-peak | 0.538 | 0.483 | 0.509 | 14 / 15 / 12 |
| 2 Baseload 24/7 | 0.857 | 0.783 | 0.818 | 36 / 10 / 6 |
| 3 Weekday-business | 1.000 | 1.000 | 1.000 | 22 / 0 / 0 |
| 4 Mixed | 0.935 | 0.879 | 0.906 | 29 / 4 / 2 |

---

## Revision history

### v3 — real data, weather, model comparison

| # | Change | Why |
|---|---|---|
| 1 | One cloned household → 309 real clients in 4 load-shape zones | The old zones were one noisy home ×4 (MAPE stuck ~54%); real aggregate load is what grid operators forecast |
| 2 | Fixed units (kW over 15 min ÷ 4) and DST artefacts | Old `demand_kwh` was 60× too large |
| 3 | Lisbon temperature + Portuguese holidays | Main external drivers of demand |
| 4 | Encoder–decoder LSTM with known-future covariates | The old head could not know tomorrow is a holiday or colder |
| 5 | Simulated 1.5 °C forecast noise on future temperature | Avoid a perfect-weather-forecast leak |
| 6 | LightGBM + seasonal-naive baselines on identical windows; validation-weighted blend | Prove the model choice instead of assuming it |
| 7 | Weather / holiday ablations | Measure what each feature group is worth |
| 8 | Multi-hour, moderate anomaly events (×1.4–2.5, ×0–0.5) | ×3–5 single-hour spikes are trivial on smooth aggregate load |
| 9 | Monte Carlo anomaly evaluation + pooled-threshold calibration | 6–17 events per split made F1 and the threshold noisy |
| 10 | `OMP_NUM_THREADS=1` on macOS | torch and LightGBM bundle separate OpenMP runtimes and segfault together |

### v2 — evaluation and modelling fixes (household data)

| # | Problem | Fix |
|---|---|---|
| 1 | Anomaly detector scored the 48h *input* window but was graded on labels from the *next* 24h | Score and evaluate every hour |
| 2 | Reconstruction error averaged over 48h — a single spiked hour was diluted 48× | Per-hour error, averaged across overlapping windows |
| 3 | Linear scale hid night-time dropouts | log1p transform |
| 4 | "Clean" training windows chosen by a >4σ heuristic | Filter with the real labels |
| 5 | Threshold = mean + 2σ of training error | Max-F1 threshold on validation |
| 6 | Latent dim 32 let the autoencoder copy spikes through | Latent dim 8 |
| 7 | Test metrics from a single random dropout pass | Evaluate the MC mean, as the API serves |
| 8 | 95% bands = mean ± 1.96σ of MC Dropout — far too narrow | Per-step calibration on validation |
| 9 | Forecaster trained on spiky targets; Huber loss in linear space; overfitting | Cleaned history; L1 in log space; dropout 0.3 + AdamW |
| 10 | API duplicated feature code, hardcoded the architecture, leaked the true anomaly label | Shared `build_features()`, `model_config.json`, inferred anomaly type |

---

## Limitations

- **Simulated weather forecast.** Future temperature is the observed value
  plus noise, not an archived forecast; real forecast errors are correlated
  across hours and larger in unusual weather.
- **Single weather station.** One Lisbon point stands in for all of
  Portugal; client locations are not published.
- **Zone definitions.** Zones are groups of real clients, not real feeder
  topology, and are named by load shape because customer types are unknown.
- **Synthetic anomalies.** Events are injected spikes and dropouts with
  known labels; real faults (drift, stuck meters, theft) are not modelled,
  and the threshold uses those labels. Without labels, fall back to a
  percentile of normal scores.
- **The anomaly detector does not know about holidays.** It only sees the
  load signal, so a genuine holiday dip can look like an outage: on the
  API's replay week, zone 1 raises 12 false "dropout" alarms on Christmas
  Day. Scoring the residual against the (holiday-aware) forecast, or feeding
  calendar features to the autoencoder, would fix this.
- **Lisbon is mild.** Temperature helps less here than it would in a
  climate with heavy heating or air-conditioning load.
- **No live data source.** The API replays the end of the zone CSVs.
