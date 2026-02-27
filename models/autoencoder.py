"""
models/autoencoder.py
─────────────────────
PURPOSE:
  LSTM Autoencoder for unsupervised anomaly detection.
  Trained on NORMAL consumption windows only.
  At inference: flag windows with high reconstruction error as anomalies.

HOW IT WORKS:
  Encoder: sequence → compressed latent vector
  Decoder: latent vector → reconstructed sequence
  
  Normal windows → low reconstruction error (model knows these patterns)
  Anomalous windows → high reconstruction error (pattern doesn't match training)

WHY THIS APPROACH:
  You don't need labeled anomalies for training. You only need clean "normal"
  data. This is powerful for real-world deployment where anomalies are rare
  and unlabeled (like the UCI dataset).

THRESHOLD CALIBRATION:
  After training, compute reconstruction error on ALL training windows.
  Threshold = mean_error + 2 * std_error
  Any test window exceeding this threshold is flagged as anomalous.
  
  WHY 2σ (as specified in the problem):
    Assumes errors are roughly Gaussian. 2σ means ~5% false positive rate.
    In grid management, you'd tune this: lower σ → more alerts (catch more
    real anomalies but more false alarms), higher → fewer alerts but riskier.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class LSTMEncoder(nn.Module):
    """
    Compresses input sequence to a fixed-size latent vector.
    
    Takes (batch, seq_len, 1) → outputs final hidden state (batch, latent_dim)
    """
    def __init__(self, input_size: int, hidden_size: int, latent_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden) → take last layer
        last_hidden = h_n[-1]              # (batch, hidden)
        latent = self.fc(last_hidden)      # (batch, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    """
    Reconstructs the original sequence from the latent vector.
    
    Takes latent (batch, latent_dim) → outputs (batch, seq_len, 1)
    
    WHY repeat the latent vector:
      LSTM decoder needs a sequence as input. We repeat the latent vector
      `seq_len` times to create input (batch, seq_len, latent_dim).
      Each timestep gets the same context and the LSTM learns to "unroll" it.
    """
    def __init__(self, latent_dim: int, hidden_size: int, output_size: int,
                 seq_len: int, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, latent):
        # Repeat latent across time dimension
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, latent_dim)
        out, _ = self.lstm(repeated)                                # (batch, seq_len, hidden)
        reconstruction = self.fc(out)                               # (batch, seq_len, 1)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder: Encoder + Decoder.
    Input and output are both (batch, seq_len, 1).
    Loss = MSE between input and reconstruction.
    """
    def __init__(
        self,
        seq_len: int = 48,
        input_size: int = 1,
        hidden_size: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, input_size, seq_len, num_layers)

    def forward(self, x):
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return recon


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_autoencoder(
    model: LSTMAutoencoder,
    train_loader,           # IMPORTANT: must be normal-only windows
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 7,
):
    """
    Train on NORMAL windows only.
    
    IMPORTANT: Filter out anomalous windows before passing to this function.
    If anomalies are in training data, the model learns to reconstruct them too,
    destroying the anomaly detection capability.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights  = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        train_losses = []
        for batch in train_loader:
            X = batch[0].to(device)
            optimizer.zero_grad()
            recon = model(X)
            loss = criterion(recon, X)   # reconstruct INPUT, not forecast target
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # VALIDATE
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X = batch[0].to(device)
                recon = model(X)
                val_losses.append(criterion(recon, X).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return model


# ─── THRESHOLD CALIBRATION ────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_threshold(
    model: LSTMAutoencoder,
    train_loader,
    device: str = "cpu",
    sigma_multiplier: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Compute reconstruction error distribution on training data.
    Set threshold = mean + sigma_multiplier * std.

    WHY compute on training data:
      This establishes the baseline for "normal" reconstruction error.
      We're asking: "How poorly does the model reconstruct data it was trained on?"
      Anything much worse than that at test time is suspicious.

    Returns:
      threshold : float — anomaly detection cutoff
      mean_err  : float — mean reconstruction error on normal data
      std_err   : float — std of reconstruction error on normal data
    """
    model.eval()
    errors = []

    for batch in train_loader:
        X = batch[0].to(device)
        recon = model(X)
        # Per-sample MSE: mean over (seq_len, features) dims
        mse = ((recon - X) ** 2).mean(dim=(1, 2))  # (batch,)
        errors.extend(mse.cpu().numpy().tolist())

    errors = np.array(errors)
    mean_err  = errors.mean()
    std_err   = errors.std()
    threshold = mean_err + sigma_multiplier * std_err

    print(f"  Anomaly threshold calibrated:")
    print(f"    Mean reconstruction error : {mean_err:.6f}")
    print(f"    Std reconstruction error  : {std_err:.6f}")
    print(f"    Threshold ({sigma_multiplier}σ)           : {threshold:.6f}")

    return threshold, mean_err, std_err


# ─── INFERENCE ────────────────────────────────────────────────────────────────

@torch.no_grad()
def detect_anomalies(
    model: LSTMAutoencoder,
    X: torch.Tensor,
    threshold: float,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run anomaly detection on input windows.

    Returns:
      errors    : (N,) — reconstruction error per window
      is_anomaly: (N,) bool — True if error > threshold
    """
    model.eval()
    X = X.to(device)
    recon = model(X)
    errors = ((recon - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
    is_anomaly = errors > threshold
    return errors, is_anomaly