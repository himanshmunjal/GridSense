"""
train.py
────────
PURPOSE:
  Orchestrate the full training pipeline for all 4 zones:
    1. Preprocess data per zone (features, weather, holidays, scaling, windows)
    2. Train DemandForecaster (encoder–decoder LSTM)
    3. Train LightGBM on the same windows, plus weather/holiday ablations
    4. Blend LSTM + LightGBM (weight picked on validation), calibrate bands
    5. Compare every model against seasonal-naive baselines on test
    6. Train LSTMAutoencoder on demand-only normal windows, calibrate threshold
    7. Evaluate anomalies: hourly confusion matrix, ROC-AUC, average precision
    8. Save all models + metrics to disk

RUN:
  python train.py

OUTPUT:
  models/saved/forecaster_zone_{z}.pt
  models/saved/lgbm_zone_{z}.txt
  models/saved/autoencoder_zone_{z}.pt
  models/saved/thresholds.json
  models/saved/model_config.json   (hyperparams + calibrated band widths for the API)
  results/metrics.json
  results/confusion_matrices.json
"""

import os
import sys
# On macOS torch and LightGBM each bundle their own libomp. Two OpenMP
# runtimes in one process segfault (in LightGBM fits or CPU LSTM passes)
# unless OpenMP is single-threaded, and this must be set before either
# library is imported. Training runs the networks on MPS, so it costs little.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, average_precision_score,
)

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "models"))

from preprocess import (
    preprocess_zone, unscale_demand, scale_demand, to_model_space, ZONES, INPUT_SIZE, FUTURE_SIZE, FUTURE_COLS,
    LOOKBACK, HORIZON,
)
from models.forecaster import (
    DemandForecaster, train_forecaster, predict_loader, calibrate_intervals,
)
from models.gbm import train_gbm, predict_gbm
from models.autoencoder import (
    LSTMAutoencoder, train_autoencoder, make_normal_loader, calibrate_threshold,
    detect_anomalies, score_series,
)
from generate_zones import inject_anomalies, ANOMALY_RATE

# ─── CONFIG ───────────────────────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SEED = 42

SAVED_DIR   = Path("models/saved")
RESULTS_DIR = Path("results")
SAVED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FORECASTER_EPOCHS  = 50
AUTOENCODER_EPOCHS = 50

# Forecaster hyperparams
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-3  # with DROPOUT=0.3: −1.7% MAE vs 0.2/no decay on zone 1
MC_SAMPLES   = 30    # MC Dropout passes used for evaluation + band calibration

# Feature groups removed in the LightGBM ablations
ABLATIONS = {
    "no_weather": ["fut_temperature", "last_temperature"],
    "no_holiday": ["fut_is_holiday"],
}

# Anomaly evaluation: fresh random anomaly sets injected into the clean
# val/test load. One fixed injection leaves only 6–17 events per split, which
# makes both the max-F1 threshold and the test F1 noisy (a zone with 6 test
# events can score F1 = 1.0 by luck).
INJECTION_ROUNDS = 20

# Autoencoder hyperparams
AE_HIDDEN    = 64
LATENT_DIM   = 8     # tight bottleneck — see models/autoencoder.py
AE_BATCH     = 128

# INPUT_SIZE, LOOKBACK and HORIZON are imported from preprocess.py — single
# source of truth. Never redeclare them here; mismatches caused silent shape
# bugs in the original code.

print(f"Using device : {DEVICE}")
print(f"LOOKBACK     : {LOOKBACK}h  →  HORIZON: {HORIZON}h")
print(f"Input size   : {INPUT_SIZE} features")


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_regression_metrics(y_true_kWh: np.ndarray, y_pred_kWh: np.ndarray) -> dict:
    """
    MAE, RMSE, MAPE and WAPE in kWh.

    WHY threshold-based MAPE mask:
      MAPE = |actual - pred| / |actual|. When actual ≈ 0 (night dropout),
      even a 5 kWh prediction gives MAPE = ∞. We mask values below 10%
      of the mean — a principled cutoff that removes near-zero instants
      without discarding the majority of the data.

    WHY also WAPE (= Σ|error| / Σ|actual|):
      MAPE weights every hour equally, so a 5 kWh miss on a 10 kWh night
      hour counts as much as a 50 kWh miss at the evening peak. WAPE is the
      share of total energy mis-forecast — the number grid operators use.
    """
    err  = y_true_kWh - y_pred_kWh
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    wape = float(np.abs(err).sum() / np.abs(y_true_kWh).sum() * 100)

    threshold = 0.10 * np.mean(np.abs(y_true_kWh))
    mask = np.abs(y_true_kWh) > threshold
    mape = float(np.mean(np.abs(err[mask] / y_true_kWh[mask])) * 100) if mask.any() else float("nan")

    return {
        "MAE":  round(mae,  4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "WAPE": round(wape, 4),
    }


def blend_weight(y_true, lstm, gbm, scaler) -> float:
    """LSTM weight in [0, 1] (step 0.05) minimising validation MAE in kWh."""
    y = unscale_demand(y_true, scaler)
    weights = np.linspace(0, 1, 21)
    maes = [np.abs(y - unscale_demand(w * lstm + (1 - w) * gbm, scaler)).mean() for w in weights]
    return float(weights[int(np.argmin(maes))])


def evaluate_forecasters(lstm_model, booster, loaders, arrays, scaler, device) -> tuple:
    """
    Compare naive baselines, LSTM, LightGBM and their blend on test.

    The served forecast is the validation-weighted blend (weight 1.0 = pure
    LSTM, 0.0 = pure LightGBM). Its 95% bands are calibrated on validation
    in units of the LSTM's MC Dropout std, so they widen where the network
    is unsure.

    Point forecast of the LSTM = MC Dropout mean (exactly what the API
    serves). Plain model(x) would be a single random dropout sample, since
    MCDropout stays active in eval mode.

    Returns:
      metrics : dict — per-model test metrics, served-model metrics
                (all hours + normal hours), skill vs naive, band coverage
      config  : {"lstm_weight", "interval_scale"} for the API
    """
    yv, mv, sv = predict_loader(lstm_model, loaders["val"], MC_SAMPLES, device)
    gv = predict_gbm(booster, *arrays["val"][:2])
    w = blend_weight(yv, mv, gv, scaler)
    interval_scale = calibrate_intervals(yv, w * mv + (1 - w) * gv, sv, coverage=0.95)

    yt, mt, st = predict_loader(lstm_model, loaders["test"], MC_SAMPLES, device)
    gt = predict_gbm(booster, *arrays["test"][:2])
    served = w * mt + (1 - w) * gt
    Xf_test = arrays["test"][1]

    y_true = unscale_demand(yt, scaler)
    lower  = unscale_demand(served - interval_scale * st, scaler)
    upper  = unscale_demand(served + interval_scale * st, scaler)
    # Injected anomalies are unforecastable by definition — also report
    # metrics on normal target hours only
    normal = arrays["test"][3] == 0

    preds = {
        # Seasonal naive: same hour yesterday / last week
        "naive_lag24":  Xf_test[:, :, FUTURE_COLS.index("demand_lag24")],
        "naive_lag168": Xf_test[:, :, FUTURE_COLS.index("demand_lag168")],
        "lstm":         mt,
        "lightgbm":     gt,
        "ensemble":     served,
    }
    models = {name: compute_regression_metrics(y_true, unscale_demand(p, scaler))
              for name, p in preds.items()}
    y_served = unscale_demand(served, scaler)

    metrics = {
        **models["ensemble"],
        "lstm_weight": w,
        "normal_hours": compute_regression_metrics(y_true[normal], y_served[normal]),
        "models": models,
        "naive_lag24": models["naive_lag24"],
        "skill_vs_naive_MAE": round(1 - models["ensemble"]["MAE"] / models["naive_lag24"]["MAE"], 4),
        "skill_vs_weekly_naive_MAE": round(1 - models["ensemble"]["MAE"] / models["naive_lag168"]["MAE"], 4),
        "interval_coverage_95": round(float(((y_true >= lower) & (y_true <= upper)).mean()), 4),
    }
    return metrics, {"lstm_weight": w, "interval_scale": interval_scale.round(4).tolist()}


def run_ablations(arrays, scaler, full_mape) -> dict:
    """Retrain LightGBM without each feature group; report test MAPE change."""
    y_true = unscale_demand(arrays["test"][2].squeeze(-1), scaler)
    out = {}
    for name, drop in ABLATIONS.items():
        booster = train_gbm(arrays["train"], arrays["val"], exclude=drop)
        pred = unscale_demand(predict_gbm(booster, *arrays["test"][:2], exclude=drop), scaler)
        m = compute_regression_metrics(y_true, pred)
        out[name] = {**m, "MAPE_increase": round(m["MAPE"] - full_mape, 4)}
        print(f"  Ablation {name:11s}: MAPE {m['MAPE']:.2f}% "
              f"({m['MAPE'] - full_mape:+.2f} pts vs full LightGBM)")
    return out


def injection_rounds(model, clean_kwh, scaler, seed, device):
    """
    Score INJECTION_ROUNDS copies of a clean split, each with its own random
    anomaly events (same generator as generate_zones.py).
    Returns a list of (scores, labels).
    """
    rounds = []
    for r in range(INJECTION_ROUNDS):
        rng = np.random.default_rng(seed + r)
        demand, labels, _ = inject_anomalies(pd.Series(clean_kwh), rng, ANOMALY_RATE)
        values = scale_demand(to_model_space(demand.values), scaler).astype(np.float32)
        rounds.append((score_series(model, values, device), labels))
    return rounds


def _runs(mask: np.ndarray) -> list:
    """Contiguous True runs of a boolean array as (start, end) pairs."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def event_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """
    Incident-level view: an anomaly event counts as caught if any of its
    hours is flagged; an alarm (run of flagged hours) is true if it overlaps
    an event.
    """
    events, alarms = _runs(labels == 1), _runs(preds == 1)
    caught = sum(preds[a:b].any() for a, b in events)
    true_alarms = sum(labels[a:b].any() for a, b in alarms)
    return {
        "event_recall": caught / max(len(events), 1),
        "event_precision": true_alarms / max(len(alarms), 1),
    }


def hourly_metrics(scores, labels, threshold) -> dict:
    preds = (scores > threshold).astype(int)
    report = classification_report(labels, preds, labels=[0, 1], output_dict=True,
                                   zero_division=0)["1"]
    return {
        "precision": report["precision"], "recall": report["recall"], "f1": report["f1-score"],
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        **event_metrics(labels, preds),
    }


def evaluate_anomaly_detection(model, test_hourly, threshold, scaler, seed, device) -> dict:
    """
    Evaluate the autoencoder hour by hour on test:
      - on the fixed injected anomalies in the zone CSV (what the API shows)
      - mean ± std over INJECTION_ROUNDS fresh injections into the clean
        test load (the headline numbers — far more events, stable estimate)
    """
    scores, is_anomaly = detect_anomalies(model, test_hourly["demand"], threshold, device)
    labels = test_hourly["labels"]
    preds  = is_anomaly.astype(int)

    cm     = confusion_matrix(labels, preds, labels=[0, 1])
    report = classification_report(
        labels, preds,
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )
    roc_auc = float(roc_auc_score(labels, scores))
    avg_prec = float(average_precision_score(labels, scores))

    print(f"\n  Fixed injection — confusion matrix (hourly):")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    anom = report["Anomaly"]
    print(f"  Precision: {anom['precision']:.3f} | "
          f"Recall: {anom['recall']:.3f} | "
          f"F1: {anom['f1-score']:.3f} | "
          f"ROC-AUC: {roc_auc:.3f} | AP: {avg_prec:.3f}")

    rounds = [hourly_metrics(sc, lb, threshold)
              for sc, lb in injection_rounds(model, test_hourly["clean_kwh"], scaler,
                                             seed, device)]
    mc = {k: {"mean": round(float(np.mean([r[k] for r in rounds])), 4),
              "std":  round(float(np.std([r[k] for r in rounds])), 4)}
          for k in rounds[0]}
    print(f"  {INJECTION_ROUNDS} injection rounds — "
          f"P {mc['precision']['mean']:.3f} | R {mc['recall']['mean']:.3f} | "
          f"F1 {mc['f1']['mean']:.3f} ± {mc['f1']['std']:.3f} | "
          f"AP {mc['average_precision']['mean']:.3f} | "
          f"events caught {mc['event_recall']['mean']:.1%} | "
          f"alarm precision {mc['event_precision']['mean']:.1%}")

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(avg_prec, 4),
        "monte_carlo": {"rounds": INJECTION_ROUNDS, **mc},
    }


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_metrics    = {}
    all_cms        = {}
    all_thresholds = {}
    model_config   = {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "forecaster": {
            "input_size": INPUT_SIZE, "future_size": FUTURE_SIZE, "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
        },
        "autoencoder": {"hidden_size": AE_HIDDEN, "latent_dim": LATENT_DIM},
        "interval_scale": {},
        "lstm_weight": {},
    }

    for zone_id in ZONES:
        print(f"\n{'='*60}")
        print(f"  ZONE {zone_id}")
        print(f"{'='*60}")

        # ── LOAD DATA ──────────────────────────────────────────────────────
        train_loader, val_loader, test_loader, scaler, hourly, arrays = \
            preprocess_zone(zone_id, return_arrays=True)
        loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        # ── [1/5] TRAIN LSTM FORECASTER ────────────────────────────────────
        print(f"\n[1/5] Training Forecaster (encoder–decoder LSTM) — Zone {zone_id}...")
        forecaster = DemandForecaster(
            input_size=INPUT_SIZE,    # from preprocess.py — no hardcoding
            future_size=FUTURE_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_steps=HORIZON,
            dropout=DROPOUT,
        )
        forecaster, f_history = train_forecaster(
            forecaster, train_loader, val_loader,
            epochs=FORECASTER_EPOCHS,
            device=DEVICE,
            weight_decay=WEIGHT_DECAY,
        )
        torch.save(forecaster.state_dict(), SAVED_DIR / f"forecaster_zone_{zone_id}.pt")
        print(f"  → Saved forecaster_zone_{zone_id}.pt")

        # ── [2/5] TRAIN LIGHTGBM + ABLATIONS ───────────────────────────────
        print(f"\n[2/5] Training LightGBM — Zone {zone_id}...")
        booster = train_gbm(arrays["train"], arrays["val"])
        booster.save_model(str(SAVED_DIR / f"lgbm_zone_{zone_id}.txt"),
                           num_iteration=booster.best_iteration)
        print(f"  → Saved lgbm_zone_{zone_id}.txt")

        # ── [3/5] COMPARE + BLEND FORECASTERS ──────────────────────────────
        print(f"\n[3/5] Evaluating forecasters — Zone {zone_id}...")
        metrics, served_cfg = evaluate_forecasters(
            forecaster, booster, loaders, arrays, scaler, DEVICE
        )
        model_config["interval_scale"][f"zone_{zone_id}"] = served_cfg["interval_scale"]
        model_config["lstm_weight"][f"zone_{zone_id}"] = served_cfg["lstm_weight"]
        print(f"  {'model':13s} {'MAE':>10s} {'RMSE':>10s} {'MAPE':>8s} {'WAPE':>8s}")
        for name, m in metrics["models"].items():
            print(f"  {name:13s} {m['MAE']:10.1f} {m['RMSE']:10.1f} "
                  f"{m['MAPE']:7.2f}% {m['WAPE']:7.2f}%")
        print(f"  Ensemble LSTM weight: {metrics['lstm_weight']:.2f} | "
              f"skill vs naive-24 (MAE): {metrics['skill_vs_naive_MAE'] * 100:.1f}% | "
              f"95% band coverage: {metrics['interval_coverage_95'] * 100:.1f}%")
        metrics["ablations"] = run_ablations(arrays, scaler, metrics["models"]["lightgbm"]["MAPE"])
        all_metrics[f"zone_{zone_id}"] = metrics

        # ── [4/5] TRAIN AUTOENCODER ────────────────────────────────────────
        # Autoencoder uses the raw (uncleaned) hourly demand only, so the
        # reconstruction error reflects real demand anomalies and isn't
        # diluted by deterministic time-encoding channels.
        print(f"\n[4/5] Training Autoencoder — Zone {zone_id}...")
        normal_train = make_normal_loader(hourly["train"]["demand"], hourly["train"]["labels"],
                                          LOOKBACK, AE_BATCH, shuffle=True)
        normal_val   = make_normal_loader(hourly["val"]["demand"], hourly["val"]["labels"],
                                          LOOKBACK, AE_BATCH, shuffle=False)

        autoencoder = LSTMAutoencoder(
            seq_len=LOOKBACK,     # from preprocess.py — single source of truth
            input_size=1,         # demand only
            hidden_size=AE_HIDDEN,
            latent_dim=LATENT_DIM,
            num_layers=1,
        )
        autoencoder = train_autoencoder(
            autoencoder, normal_train, normal_val,
            epochs=AUTOENCODER_EPOCHS,
            device=DEVICE,
        )

        # Threshold tuned on validation: the fixed injection plus
        # INJECTION_ROUNDS fresh ones, pooled (~20× more events)
        val_scores, _ = detect_anomalies(autoencoder, hourly["val"]["demand"], 0.0, DEVICE)
        val_rounds = [(val_scores, hourly["val"]["labels"])] + injection_rounds(
            autoencoder, hourly["val"]["clean_kwh"], scaler, SEED * 1000 + zone_id * 100, DEVICE)
        threshold, mean_err, std_err = calibrate_threshold(
            np.concatenate([sc for sc, _ in val_rounds]),
            np.concatenate([lb for _, lb in val_rounds]),
        )
        torch.save(autoencoder.state_dict(), SAVED_DIR / f"autoencoder_zone_{zone_id}.pt")
        all_thresholds[f"zone_{zone_id}"] = {
            "threshold": float(threshold),
            "mean_err":  float(mean_err),
            "std_err":   float(std_err),
        }
        print(f"  → Saved autoencoder_zone_{zone_id}.pt")

        # ── [5/5] EVALUATE ANOMALY DETECTION ──────────────────────────────
        print(f"\n[5/5] Evaluating Anomaly Detection — Zone {zone_id}...")
        cm_results = evaluate_anomaly_detection(autoencoder, hourly["test"], threshold, scaler,
                                                SEED * 2000 + zone_id * 100, DEVICE)
        all_cms[f"zone_{zone_id}"] = cm_results

    # ── SAVE ALL RESULTS ───────────────────────────────────────────────────────
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(RESULTS_DIR / "confusion_matrices.json", "w") as f:
        json.dump(all_cms, f, indent=2)

    with open(SAVED_DIR / "thresholds.json", "w") as f:
        json.dump(all_thresholds, f, indent=2)

    with open(SAVED_DIR / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print("\nFinal Metrics Summary:")
    for zone, m in all_metrics.items():
        a = all_cms[zone]["monte_carlo"]
        print(f"  {zone}: MAE={m['MAE']:.1f} kWh | "
              f"RMSE={m['RMSE']:.1f} kWh | "
              f"MAPE={m['MAPE']:.2f}% | "
              f"WAPE={m['WAPE']:.2f}% | "
              f"skill={m['skill_vs_naive_MAE'] * 100:.1f}% | "
              f"anomaly F1={a['f1']['mean']:.3f}±{a['f1']['std']:.3f}")
    print(f"\nAll files saved to {SAVED_DIR} and {RESULTS_DIR}")


if __name__ == "__main__":
    main()