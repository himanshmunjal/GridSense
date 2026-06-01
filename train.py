"""
train.py
────────
PURPOSE:
  Orchestrate the full training pipeline for all 4 zones:
    1. Preprocess data per zone (features, scaling, windows)
    2. Train DemandForecaster (LSTM) on all features
    3. Train LSTMAutoencoder on demand-only normal windows
    4. Calibrate anomaly threshold
    5. Evaluate: MAE, RMSE, MAPE (forecaster) + confusion matrix (autoencoder)
    6. Save all models + metrics to disk

RUN:
  python train.py

OUTPUT:
  models/saved/forecaster_zone_{z}.pt
  models/saved/autoencoder_zone_{z}.pt
  models/saved/thresholds.json
  results/metrics.json
  results/confusion_matrices.json
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import sys

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "models"))

from torch.utils.data import DataLoader, Subset

from preprocess import preprocess_zone, DemandDataset, ZONES, INPUT_SIZE, LOOKBACK
from models.forecaster import DemandForecaster, train_forecaster
from models.autoencoder import LSTMAutoencoder, train_autoencoder, calibrate_threshold, detect_anomalies

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVED_DIR   = Path("models/saved")
RESULTS_DIR = Path("results")
SAVED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FORECASTER_EPOCHS  = 50
AUTOENCODER_EPOCHS = 50

# Forecaster hyperparams
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2

# Autoencoder hyperparams
AE_HIDDEN    = 64
LATENT_DIM   = 32

HORIZON      = 24

# INPUT_SIZE and LOOKBACK are imported from preprocess.py — single source of truth.
# Never redeclare them here; mismatches caused silent shape bugs in the original code.

print(f"Using device : {DEVICE}")
print(f"LOOKBACK     : {LOOKBACK}h  →  HORIZON: {HORIZON}h")
print(f"Input size   : {INPUT_SIZE} features")


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler,
) -> dict:
    """
    Compute MAE, RMSE, and MAPE in original kWh scale.

    WHY inverse_transform:
      The scaler (StandardScaler) was fitted on demand_kwh. y_true and y_pred
      are in StandardScaler space — meaningless to a grid operator.
      Inverse-transforming gives numbers like "±23 kWh error".

    WHY threshold-based MAPE mask:
      MAPE = |actual - pred| / |actual|. When actual ≈ 0 (night dropout),
      even a 5 kWh prediction gives MAPE = ∞. We mask values below 10%
      of the mean — a principled cutoff that removes near-zero instants
      without discarding the majority of the data.

    Args:
      y_true, y_pred : (N, HORIZON) — normalised demand predictions
      scaler         : fitted StandardScaler (single-column, demand only)
    """
    shape = y_true.shape  # (N, HORIZON)

    # The scaler was fit on 6 columns (SCALE_COLS). demand_kwh is column 0.
    # We must NOT pass a 1-column array to a 6-column scaler — shape mismatch.
    # Instead, manually invert using only demand's own mean and std:
    #   original = normalised * std + mean
    demand_mean = scaler.mean_[0]
    demand_std  = scaler.scale_[0]
    y_true_kWh = (y_true * demand_std + demand_mean).reshape(shape)
    y_pred_kWh = (y_pred * demand_std + demand_mean).reshape(shape)

    mae  = float(np.mean(np.abs(y_true_kWh - y_pred_kWh)))
    rmse = float(np.sqrt(np.mean((y_true_kWh - y_pred_kWh) ** 2)))

    # MAPE: exclude near-zero actuals (10% of mean threshold)
    threshold = 0.10 * np.mean(np.abs(y_true_kWh))
    mask = np.abs(y_true_kWh) > threshold
    if mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs(
                (y_true_kWh[mask] - y_pred_kWh[mask]) / y_true_kWh[mask]
            )) * 100
        )

    return {
        "MAE":  round(mae,  4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
    }


def evaluate_forecaster(model, test_loader, scaler, device) -> tuple:
    """
    Run the trained forecaster on the test set and compute metrics.

    DataLoader yields 4-tuples: (X_full, X_demand, y, anomaly)
    Forecaster needs X_full and y.
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for X_full, _, y, _ in test_loader:
            pred = model(X_full.to(device))
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    # Concatenate and squeeze the trailing feature dim
    y_true = np.concatenate(all_true).squeeze(-1)  # (N, HORIZON)
    y_pred = np.concatenate(all_pred).squeeze(-1)

    metrics = compute_regression_metrics(y_true, y_pred, scaler)
    return metrics, y_true, y_pred


def evaluate_anomaly_detection(model, test_loader, threshold, device) -> dict:
    """
    Evaluate the autoencoder against ground truth anomaly labels.

    DataLoader yields 4-tuples: (X_full, X_demand, y, anomaly)
    Autoencoder needs X_demand (demand-only signal, no time features).
    """
    model.eval()
    all_errors, all_pred_labels, all_true_labels = [], [], []

    with torch.no_grad():
        for _, X_demand, _, anomaly in test_loader:
            errors, is_anomaly = detect_anomalies(model, X_demand.to(device), threshold, device)

            # Window is anomalous if ANY timestep in the target range is flagged
            true_window = (anomaly.numpy().sum(axis=1) > 0).astype(int)

            all_errors.extend(errors.tolist())
            all_pred_labels.extend(is_anomaly.astype(int).tolist())
            all_true_labels.extend(true_window.tolist())

    cm     = confusion_matrix(all_true_labels, all_pred_labels)
    report = classification_report(
        all_true_labels, all_pred_labels,
        target_names=["Normal", "Anomaly"],
        output_dict=True,
    )

    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    anom = report["Anomaly"]
    print(f"  Precision: {anom['precision']:.3f} | "
          f"Recall: {anom['recall']:.3f} | "
          f"F1: {anom['f1-score']:.3f}")

    return {"confusion_matrix": cm.tolist(), "classification_report": report}


# ─── FILTER NORMAL WINDOWS ────────────────────────────────────────────────────

def get_normal_only_loader(loader: DataLoader) -> DataLoader:
    """
    Return a DataLoader containing only fully-normal windows.

    A window is considered normal if BOTH:
      (a) the target range (horizon) contains no anomalies  — original check
      (b) the input range (lookback) also contains no anomalies

    WHY check the lookback portion too:
      The autoencoder is trained to reconstruct normal consumption patterns.
      If a window's input contains a spike (even if the target looks normal),
      the autoencoder sees an anomalous sequence during training — it starts
      learning to reconstruct spikes as "normal", which destroys the
      anomaly detection capability.

    DataLoader yields: (X_full, X_demand, y, anomaly)
    `anomaly` covers only the target horizon. We infer lookback cleanliness
    by checking whether the X_demand values are within ±4 std of their
    local mean (a reliable spike proxy since we scaled with StandardScaler).
    """
    dataset = loader.dataset
    normal_indices = []

    for i in range(len(dataset)):
        X_full, X_demand, y, target_anom = dataset[i]

        # Check 1: no ground truth anomalies in the target horizon
        if target_anom.sum() > 0:
            continue

        # Check 2: no obvious spike in the lookback window
        # X_demand is scaled — values > 4σ from 0 (mean after StandardScaler)
        # indicate an injected spike in the lookback portion
        if X_demand.abs().max() > 4.0:
            continue

        normal_indices.append(i)

    pct = len(normal_indices) / len(dataset) * 100
    print(f"  Normal windows: {len(normal_indices)}/{len(dataset)} ({pct:.1f}%)")

    return DataLoader(
        Subset(dataset, normal_indices),
        batch_size=loader.batch_size,
        shuffle=True,
    )


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────

def main():
    all_metrics    = {}
    all_cms        = {}
    all_thresholds = {}

    for zone_id in ZONES:
        print(f"\n{'='*60}")
        print(f"  ZONE {zone_id}")
        print(f"{'='*60}")

        # ── LOAD DATA ──────────────────────────────────────────────────────
        train_loader, val_loader, test_loader, scaler = preprocess_zone(zone_id)

        # ── [1/4] TRAIN FORECASTER ─────────────────────────────────────────
        print(f"\n[1/4] Training Forecaster (LSTM) — Zone {zone_id}...")
        forecaster = DemandForecaster(
            input_size=INPUT_SIZE,    # from preprocess.py — no hardcoding
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_steps=HORIZON,
            dropout=DROPOUT,
        )
        forecaster, f_history = train_forecaster(
            forecaster, train_loader, val_loader,
            epochs=FORECASTER_EPOCHS,
            device=DEVICE,
        )
        torch.save(forecaster.state_dict(), SAVED_DIR / f"forecaster_zone_{zone_id}.pt")
        print(f"  → Saved forecaster_zone_{zone_id}.pt")

        # ── [2/4] EVALUATE FORECASTER ──────────────────────────────────────
        print(f"\n[2/4] Evaluating Forecaster — Zone {zone_id}...")
        metrics, _, _ = evaluate_forecaster(forecaster, test_loader, scaler, DEVICE)
        all_metrics[f"zone_{zone_id}"] = metrics
        print(f"  MAE:  {metrics['MAE']:.2f} kWh")
        print(f"  RMSE: {metrics['RMSE']:.2f} kWh")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")

        # ── [3/4] TRAIN AUTOENCODER ────────────────────────────────────────
        # Autoencoder uses demand-only windows (X_demand) to keep the
        # reconstruction error signal focused on the actual demand anomaly,
        # not diluted by deterministic time-encoding channels.
        print(f"\n[3/4] Training Autoencoder — Zone {zone_id}...")
        normal_train = get_normal_only_loader(train_loader)
        normal_val   = get_normal_only_loader(val_loader)

        autoencoder = LSTMAutoencoder(
            seq_len=LOOKBACK,     # from preprocess.py — single source of truth
            input_size=1,         # demand only (X_demand has 1 channel)
            hidden_size=AE_HIDDEN,
            latent_dim=LATENT_DIM,
            num_layers=1,
        )
        autoencoder = train_autoencoder(
            autoencoder, normal_train, normal_val,
            epochs=AUTOENCODER_EPOCHS,
            device=DEVICE,
        )

        # sigma_multiplier=2.0: ~5% false positive rate under Gaussian errors
        threshold, mean_err, std_err = calibrate_threshold(
            autoencoder, normal_train,
            device=DEVICE,
            sigma_multiplier=2.0,
        )
        torch.save(autoencoder.state_dict(), SAVED_DIR / f"autoencoder_zone_{zone_id}.pt")
        all_thresholds[f"zone_{zone_id}"] = {
            "threshold": float(threshold),
            "mean_err":  float(mean_err),
            "std_err":   float(std_err),
        }
        print(f"  → Saved autoencoder_zone_{zone_id}.pt")

        # ── [4/4] EVALUATE ANOMALY DETECTION ──────────────────────────────
        print(f"\n[4/4] Evaluating Anomaly Detection — Zone {zone_id}...")
        cm_results = evaluate_anomaly_detection(autoencoder, test_loader, threshold, DEVICE)
        all_cms[f"zone_{zone_id}"] = cm_results

    # ── SAVE ALL RESULTS ───────────────────────────────────────────────────────
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(RESULTS_DIR / "confusion_matrices.json", "w") as f:
        json.dump(all_cms, f, indent=2)

    with open(SAVED_DIR / "thresholds.json", "w") as f:
        json.dump(all_thresholds, f, indent=2)

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print("\nFinal Metrics Summary:")
    for zone, m in all_metrics.items():
        print(f"  {zone}: MAE={m['MAE']:.2f} kWh | "
              f"RMSE={m['RMSE']:.2f} kWh | "
              f"MAPE={m['MAPE']:.2f}%")
    print(f"\nAll files saved to {SAVED_DIR} and {RESULTS_DIR}")


if __name__ == "__main__":
    main()