"""
models/autoencoder.py
─────────────────────
PURPOSE:
  LSTM Autoencoder for unsupervised anomaly detection.
  Trained on NORMAL consumption windows only (demand signal, 1 channel).
  At inference: flag windows with high reconstruction error as anomalies.

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

THRESHOLD CALIBRATION:
  After training, compute reconstruction error on ALL normal training windows.
  Threshold = mean_error + sigma * std_error
  Any test window exceeding this is flagged anomalous.

  WHY 2σ: assumes roughly Gaussian errors → ~5% false positive rate.
  Tune sigma_multiplier up (fewer alerts, riskier) or down (more alerts) 
  depending on grid operator risk tolerance.

DATALOADER CONTRACT:
  All DataLoaders here come from get_normal_only_loader() in train.py, which
  yields 4-tuples: (X_full, X_demand, y, anomaly).
  This file always unpacks index 1 (X_demand) — never index 0 (X_full).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


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
        latent_dim: int  = 32,
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


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_autoencoder(
    model: LSTMAutoencoder,
    train_loader,         # normal-only DataLoader from get_normal_only_loader()
    val_loader,
    epochs: int   = 50,
    lr: float     = 1e-3,
    device: str   = "cpu",
    patience: int = 7,
) -> LSTMAutoencoder:
    """
    Train the autoencoder to reconstruct normal demand windows.

    DataLoader contract: yields 4-tuples (X_full, X_demand, y, anomaly).
    We unpack index 1 (X_demand) — the single-channel demand signal.
    Index 0 (X_full) is the 10-feature forecaster input and must NOT be used
    here: feeding time-encoding channels would dilute reconstruction error.

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
        for batch in train_loader:
            # Unpack index 1: X_demand (batch, seq_len, 1)
            # Index 0 is X_full (10 features) — wrong input for the autoencoder
            _, X_demand, _, _ = batch
            X_demand = X_demand.to(device)

            optimizer.zero_grad()
            recon = model(X_demand)
            loss  = criterion(recon, X_demand)   # reconstruct the INPUT, not y
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                _, X_demand, _, _ = batch
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


# ─── THRESHOLD CALIBRATION ────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_threshold(
    model: LSTMAutoencoder,
    normal_loader,          # normal-only DataLoader (same as training loader)
    device: str = "cpu",
    sigma_multiplier: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Compute reconstruction error distribution on normal training windows.
    Set threshold = mean + sigma_multiplier * std.

    WHY use training data for calibration:
      We want to know "how badly does the model reconstruct its own training
      distribution?" Any test window with error well beyond that is anomalous.
      Using val data here can also work but gives a slightly higher mean
      (val windows weren't seen during training).

    DataLoader contract: yields 4-tuples — unpack X_demand at index 1.

    Returns:
      threshold : float — anomaly decision boundary
      mean_err  : float — mean reconstruction MSE on normal windows
      std_err   : float — std of reconstruction MSE on normal windows
    """
    model.eval()
    errors = []

    for batch in normal_loader:
        _, X_demand, _, _ = batch     # index 1 = demand-only signal
        X_demand = X_demand.to(device)
        recon    = model(X_demand)

        # Per-sample MSE across (seq_len, 1) dims → shape (batch,)
        mse = ((recon - X_demand) ** 2).mean(dim=(1, 2))
        errors.extend(mse.cpu().numpy().tolist())

    errors    = np.array(errors)
    mean_err  = float(errors.mean())
    std_err   = float(errors.std())
    threshold = mean_err + sigma_multiplier * std_err

    print(f"  Anomaly threshold calibrated:")
    print(f"    Mean reconstruction error : {mean_err:.6f}")
    print(f"    Std  reconstruction error : {std_err:.6f}")
    print(f"    Threshold ({sigma_multiplier}σ)           : {threshold:.6f}")

    return threshold, mean_err, std_err


# ─── INFERENCE ────────────────────────────────────────────────────────────────

@torch.no_grad()
def detect_anomalies(
    model: LSTMAutoencoder,
    X_demand: torch.Tensor,    # (batch, seq_len, 1) — demand-only, already on device
    threshold: float,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run anomaly detection on a batch of demand windows.

    Called from evaluate_anomaly_detection() in train.py, which already
    unpacks X_demand from the 4-tuple before passing it here.

    Args:
      X_demand  : (batch, seq_len, 1) — scaled demand signal
      threshold : calibrated anomaly cutoff from calibrate_threshold()

    Returns:
      errors     : (batch,) float — per-window reconstruction MSE
      is_anomaly : (batch,) bool  — True where error > threshold
    """
    model.eval()
    X_demand = X_demand.to(device)
    recon    = model(X_demand)

    errors     = ((recon - X_demand) ** 2).mean(dim=(1, 2)).cpu().numpy()
    is_anomaly = errors > threshold

    return errors, is_anomaly