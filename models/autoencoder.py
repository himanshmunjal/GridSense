"""
models/autoencoder.py
─────────────────────
PURPOSE:
  LSTM Autoencoder for unsupervised anomaly detection.
  Trained on NORMAL consumption windows only (log demand signal, 1 channel).
  At inference: score every HOUR by its reconstruction error and flag hours
  above a calibrated threshold.

HOW IT WORKS:
  Encoder: (batch, seq_len, 1) → compressed latent vector (batch, latent_dim)
  Decoder: latent vector → reconstructed sequence (batch, seq_len, 1)

  Normal windows   → low reconstruction error  (model knows these patterns)
  Anomalous windows → high reconstruction error (pattern doesn't match training)

WHY DEMAND-ONLY INPUT (input_size=1):
  The forecaster uses all 10 features. The autoencoder intentionally uses
  only the demand column (X_demand from the DataLoader). Reason: the sin/cos
  time-encoding features are perfectly deterministic — the autoencoder would
  reconstruct them near-perfectly regardless of whether a spike exists,
  diluting the reconstruction error signal. Isolating demand means errors
  are caused only by abnormal consumption patterns, not noise from trivially
  reconstructable channels.

POINT-LEVEL SCORING:
  Anomalies are single hours, so we score hours, not windows. Every hour is
  covered by up to `seq_len` overlapping windows; its score is the mean
  squared reconstruction error at that position across those windows.
  Averaging an error over the whole 48h window would dilute a single spike
  48×, and labelling a window by hours it doesn't contain is meaningless.

THRESHOLD CALIBRATION:
  The threshold is the value that maximises F1 on the VALIDATION split's
  hourly scores. The test split is never used for tuning.

WHY a small latent (8):
  A 32-dim latent for a 48-step window is a loose bottleneck — the model
  can partly copy spikes through. A tighter bottleneck forces it to learn
  only the typical daily shape, so anomalies reconstruct badly.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve


# ─── ENCODER ──────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Compresses a demand sequence into a fixed-size latent vector.

    Input  : (batch, seq_len, 1)    — scaled demand signal
    Output : (batch, latent_dim)    — compressed representation
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]          # (batch, hidden_size) — final layer only
        return self.fc(last_hidden)    # (batch, latent_dim)


# ─── DECODER ──────────────────────────────────────────────────────────────────

class LSTMDecoder(nn.Module):
    """
    Reconstructs the original demand sequence from the latent vector.

    Input  : (batch, latent_dim)    — compressed representation
    Output : (batch, seq_len, 1)    — reconstructed demand sequence

    WHY repeat the latent vector:
      LSTMs require a sequence input. We tile the latent vector `seq_len` times
      so each decoder timestep receives the same context. The LSTM then learns
      to "unroll" the compressed state back into a temporal sequence.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        output_size: int,
        seq_len: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Tile latent across time: (batch, latent_dim) → (batch, seq_len, latent_dim)
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _   = self.lstm(repeated)       # (batch, seq_len, hidden_size)
        return self.fc(out)                  # (batch, seq_len, output_size)


# ─── AUTOENCODER ──────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder: Encoder + Decoder.

    Input and reconstruction are both (batch, seq_len, 1) — demand only.
    Training loss = MSE(reconstruction, input) on normal windows.
    """

    def __init__(
        self,
        seq_len: int     = 48,
        input_size: int  = 1,     # always 1 — demand channel only
        hidden_size: int = 64,
        latent_dim: int  = 8,
        num_layers: int  = 1,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, input_size, seq_len, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x : (batch, seq_len, 1) — X_demand from DataLoader
        Returns:
          recon : (batch, seq_len, 1) — reconstructed demand
        """
        latent = self.encoder(x)
        return self.decoder(latent)


# ─── WINDOWS ──────────────────────────────────────────────────────────────────

def make_windows(values: np.ndarray, seq_len: int) -> np.ndarray:
    """(N,) hourly series → (N - seq_len + 1, seq_len, 1) stride-1 windows."""
    windows = np.lib.stride_tricks.sliding_window_view(values, seq_len)
    return windows[..., None].astype(np.float32)


def make_normal_loader(
    values: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """
    DataLoader over windows that contain NO labelled anomaly anywhere.

    WHY use the labels on the whole window:
      If a training window contains a spike, the autoencoder learns to
      reconstruct spikes as "normal", which destroys detection capability.
    """
    windows = make_windows(values, seq_len)
    dirty   = np.lib.stride_tricks.sliding_window_view(labels, seq_len).sum(axis=1) > 0
    clean   = windows[~dirty]
    print(f"  Normal windows: {len(clean)}/{len(windows)} "
          f"({len(clean) / len(windows) * 100:.1f}%)")
    return DataLoader(TensorDataset(torch.from_numpy(clean)),
                      batch_size=batch_size, shuffle=shuffle)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_autoencoder(
    model: LSTMAutoencoder,
    train_loader,         # normal-only DataLoader from make_normal_loader()
    val_loader,
    epochs: int   = 50,
    lr: float     = 1e-3,
    device: str   = "cpu",
    patience: int = 7,
) -> LSTMAutoencoder:
    """
    Train the autoencoder to reconstruct normal demand windows.

    DataLoader contract: yields 1-tuples (X_demand,) of shape (batch, seq_len, 1).

    Early stopping restores the best validation weights to prevent overfitting
    on the small number of "normal-only" windows.
    """
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss    = float("inf")
    best_weights     = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):

        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for (X_demand,) in train_loader:
            X_demand = X_demand.to(device)

            optimizer.zero_grad()
            recon = model(X_demand)
            loss  = criterion(recon, X_demand)   # reconstruct the INPUT
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (X_demand,) in val_loader:
                X_demand = X_demand.to(device)
                recon    = model(X_demand)
                val_losses.append(criterion(recon, X_demand).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}")

        # ── EARLY STOPPING ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val loss: {best_val_loss:.6f})")
                break

    model.load_state_dict(best_weights)
    return model


# ─── HOURLY SCORING ───────────────────────────────────────────────────────────

@torch.no_grad()
def score_series(
    model: LSTMAutoencoder,
    values: np.ndarray,       # (N,) scaled log demand
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Per-hour anomaly score: mean squared reconstruction error at that hour
    across every window covering it. Returns (N,). Requires N >= seq_len.
    """
    model.eval()
    seq_len = model.decoder.seq_len
    windows = torch.from_numpy(make_windows(values, seq_len))

    errors = []
    for i in range(0, len(windows), batch_size):
        X = windows[i:i + batch_size].to(device)
        errors.append(((model(X) - X) ** 2).squeeze(-1).cpu().numpy())
    errors = np.concatenate(errors)                 # (W, seq_len)

    total = np.zeros(len(values))
    count = np.zeros(len(values))
    for offset in range(seq_len):
        total[offset:offset + len(errors)] += errors[:, offset]
        count[offset:offset + len(errors)] += 1
    return total / count


# ─── THRESHOLD CALIBRATION ────────────────────────────────────────────────────

def calibrate_threshold(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Pick the hourly-score threshold that maximises F1 on the validation split.

    Returns:
      threshold : float — anomaly decision boundary
      mean_err  : float — mean score on normal validation hours
      std_err   : float — std of score on normal validation hours
    """
    precision, recall, cuts = precision_recall_curve(val_labels, val_scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))   # last P/R point has no threshold
    threshold = float(cuts[best])

    normal   = val_scores[val_labels == 0]
    mean_err = float(normal.mean())
    std_err  = float(normal.std())

    print(f"  Anomaly threshold calibrated on validation:")
    print(f"    Mean score (normal hours) : {mean_err:.6f}")
    print(f"    Std  score (normal hours) : {std_err:.6f}")
    print(f"    Threshold (max val F1)    : {threshold:.6f}  "
          f"(val P={precision[best]:.3f} R={recall[best]:.3f} F1={f1[best]:.3f})")

    return threshold, mean_err, std_err


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def detect_anomalies(
    model: LSTMAutoencoder,
    values: np.ndarray,       # (N,) scaled log demand
    threshold: float,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run hourly anomaly detection on a demand series.

    Returns:
      scores     : (N,) float — per-hour reconstruction score
      is_anomaly : (N,) bool  — True where score > threshold
    """
    scores = score_series(model, values, device)
    return scores, scores > threshold
