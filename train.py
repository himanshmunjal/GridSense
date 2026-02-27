"""
train.py
────────
PURPOSE:
  Orchestrate the full training pipeline for all 4 zones:
    1. Preprocess data per zone
    2. Train DemandForecaster (LSTM)
    3. Train LSTMAutoencoder on normal windows only
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
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import sys
sys.path.append(str(Path(__file__).resolve().parent))  # Add current directory to Python path
sys.path.append(str(Path(__file__).resolve().parent / "models"))  # Add 'models' directory to Python path
from torch.utils.data import DataLoader, Subset

from preprocess import preprocess_zone, DemandDataset, ZONES
from models.forecaster import DemandForecaster, train_forecaster
from models.autoencoder import LSTMAutoencoder, train_autoencoder, calibrate_threshold, detect_anomalies

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVED_DIR    = Path("models/saved")
RESULTS_DIR  = Path("results")
SAVED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FORECASTER_EPOCHS   = 50
AUTOENCODER_EPOCHS  = 50
HIDDEN_SIZE         = 128
LATENT_DIM          = 32
DROPOUT             = 0.2
LOOKBACK            = 48
HORIZON             = 24

print(f"Using device: {DEVICE}")


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler) -> dict:
    """
    Compute MAE, RMSE, MAPE in original (kWh) scale.
    
    WHY inverse transform:
      All model outputs are in [0,1] normalized scale.
      MAE of 0.05 normalized means nothing to a grid operator.
      They need "your forecast is off by 23 kWh on average."
    
    MAPE note:
      MAPE = mean(|actual - predicted| / |actual|) × 100
      Division by actual causes issues if actual ≈ 0.
      We add a small epsilon or clip to avoid division by zero.
    """
    # Inverse transform: flatten → reshape for scaler → reflatten
    shape = y_true.shape
    y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(shape)
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(shape)

    mae  = np.mean(np.abs(y_true_orig - y_pred_orig))
    rmse = np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))

    # MAPE: avoid division by ~zero
    mask = np.abs(y_true_orig) > 0.1
    mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100

    return {"MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4), "MAPE": round(float(mape), 4)}


def evaluate_forecaster(model, test_loader, scaler, device):
    """Run model on test set, collect predictions, compute metrics."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            X, y = batch[0].to(device), batch[1].to(device)
            pred = model(X)
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0).squeeze(-1)  # (N, horizon)
    y_pred = np.concatenate(all_pred, axis=0).squeeze(-1)

    metrics = compute_regression_metrics(y_true, y_pred, scaler)
    return metrics, y_true, y_pred


def evaluate_anomaly_detection(model, test_loader, threshold, device):
    """
    Evaluate autoencoder anomaly detection against ground truth labels.
    Builds confusion matrix and classification report.
    """
    model.eval()
    all_errors, all_pred_labels, all_true_labels = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            X = batch[0].to(device)
            true_anomaly = batch[2]  # (batch, horizon) — window-level: take any-anomaly

            errors, is_anomaly = detect_anomalies(model, X, threshold, device)

            # Ground truth: is ANY timestep in this window an anomaly?
            true_window_anomaly = (true_anomaly.numpy().sum(axis=1) > 0).astype(int)

            all_errors.extend(errors.tolist())
            all_pred_labels.extend(is_anomaly.astype(int).tolist())
            all_true_labels.extend(true_window_anomaly.tolist())

    cm = confusion_matrix(all_true_labels, all_pred_labels)
    report = classification_report(all_true_labels, all_pred_labels,
                                   target_names=["Normal", "Anomaly"], output_dict=True)

    print(f"\n  Confusion Matrix:\n  TN={cm[0,0]} FP={cm[0,1]}\n  FN={cm[1,0]} TP={cm[1,1]}")
    print(f"  Precision: {report['Anomaly']['precision']:.3f} | "
          f"Recall: {report['Anomaly']['recall']:.3f} | "
          f"F1: {report['Anomaly']['f1-score']:.3f}")

    return {"confusion_matrix": cm.tolist(), "classification_report": report}


# ─── FILTER NORMAL WINDOWS ────────────────────────────────────────────────────

def get_normal_only_loader(loader: DataLoader) -> DataLoader:
    """
    Create a DataLoader containing ONLY normal (non-anomalous) windows.
    
    WHY: Autoencoder must be trained on normal data only.
    We keep windows where NO timestep in the target sequence is anomalous.
    """
    dataset = loader.dataset
    normal_indices = []
    for i in range(len(dataset)):
        _, _, anomaly = dataset[i]
        if anomaly.sum() == 0:  # no anomaly in this window's target range
            normal_indices.append(i)

    normal_subset = Subset(dataset, normal_indices)
    pct = len(normal_indices) / len(dataset) * 100
    print(f"  Normal windows: {len(normal_indices)}/{len(dataset)} ({pct:.1f}%)")

    return DataLoader(normal_subset, batch_size=loader.batch_size, shuffle=True)


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────

def main():
    all_metrics = {}
    all_cms     = {}
    all_thresholds = {}

    for zone_id in ZONES:
        print(f"\n{'='*60}")
        print(f"  ZONE {zone_id}")
        print(f"{'='*60}")

        # ── LOAD DATA ──────────────────────────────────────────────────────
        train_loader, val_loader, test_loader, scaler = preprocess_zone(zone_id)

        # ── TRAIN FORECASTER ───────────────────────────────────────────────
        print(f"\n[1/4] Training Forecaster (LSTM) — Zone {zone_id}...")
        forecaster = DemandForecaster(
            input_size=1, hidden_size=HIDDEN_SIZE,
            num_layers=2, output_steps=HORIZON, dropout=DROPOUT
        )
        forecaster, f_history = train_forecaster(
            forecaster, train_loader, val_loader,
            epochs=FORECASTER_EPOCHS, device=DEVICE
        )
        torch.save(forecaster.state_dict(), SAVED_DIR / f"forecaster_zone_{zone_id}.pt")
        print(f"  → Saved forecaster_zone_{zone_id}.pt")

        # ── EVALUATE FORECASTER ────────────────────────────────────────────
        print(f"\n[2/4] Evaluating Forecaster — Zone {zone_id}...")
        metrics, _, _ = evaluate_forecaster(forecaster, test_loader, scaler, DEVICE)
        all_metrics[f"zone_{zone_id}"] = metrics
        print(f"  MAE: {metrics['MAE']:.2f} kWh | "
              f"RMSE: {metrics['RMSE']:.2f} kWh | "
              f"MAPE: {metrics['MAPE']:.2f}%")

        # ── TRAIN AUTOENCODER (normal windows only) ─────────────────────────
        print(f"\n[3/4] Training Autoencoder — Zone {zone_id}...")
        normal_train_loader = get_normal_only_loader(train_loader)
        normal_val_loader   = get_normal_only_loader(val_loader)

        autoencoder = LSTMAutoencoder(
            seq_len=LOOKBACK, input_size=1,
            hidden_size=64, latent_dim=LATENT_DIM, num_layers=1
        )
        autoencoder = train_autoencoder(
            autoencoder, normal_train_loader, normal_val_loader,
            epochs=AUTOENCODER_EPOCHS, device=DEVICE
        )
        threshold, mean_err, std_err = calibrate_threshold(
            autoencoder, normal_train_loader, device=DEVICE, sigma_multiplier=2.0
        )
        torch.save(autoencoder.state_dict(), SAVED_DIR / f"autoencoder_zone_{zone_id}.pt")
        all_thresholds[f"zone_{zone_id}"] = {
            "threshold": float(threshold),
            "mean_err": float(mean_err),
            "std_err": float(std_err),
        }
        print(f"  → Saved autoencoder_zone_{zone_id}.pt")

        # ── EVALUATE ANOMALY DETECTION ─────────────────────────────────────
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

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print("\nFinal Metrics Summary:")
    for zone, m in all_metrics.items():
        print(f"  {zone}: MAE={m['MAE']} | RMSE={m['RMSE']} | MAPE={m['MAPE']}%")

    print(f"\nAll files saved to {SAVED_DIR} and {RESULTS_DIR}")


if __name__ == "__main__":
    main()