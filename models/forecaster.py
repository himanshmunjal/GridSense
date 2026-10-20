"""
models/forecaster.py
────────────────────
PURPOSE:
  Encoder–decoder LSTM multi-step demand forecaster.
  Predicts next 24 hours of demand given the last 48 hours of features plus
  the known-future covariates of the next 24 hours (calendar, holidays,
  same-hour-yesterday/last-week demand, temperature forecast).

ARCHITECTURE:
  X_past   (batch, 48, 15) → 2-layer encoder LSTM ─┐ (h, c)
  X_future (batch, 24, 11) → 2-layer decoder LSTM ←┘
                           → MCDropout → per-step MLP head → (batch, 24, 1)

KEY DESIGN DECISIONS:

  0. WHY encoder–decoder instead of "last hidden state → 24 outputs":
     The old head had to guess the whole next day from one summary vector,
     with no idea that tomorrow is a holiday or 8 °C colder. The decoder
     walks through the 24 target hours with their own covariates, so each
     step's prediction is conditioned on what is known about that hour.

  1. WHY LSTM over GRU:
     LSTM has a separate cell state that acts as long-term memory, useful for
     capturing weekly seasonality on top of daily cycles. GRU is ~5% faster
     with marginal accuracy cost — swap in if training is too slow.

  2. WHY 2 LSTM layers:
     Layer 1 learns short-term patterns (morning ramp, evening peak).
     Layer 2 learns patterns-of-patterns (weekday vs weekend shapes).
     3+ layers yield diminishing returns and harder gradient flow.

  3. WHY MCDropout for confidence intervals:
     Standard LSTM gives a point estimate. Grid operators need uncertainty:
     "demand will be 450 kWh ± 30" vs just "450 kWh".
     MC Dropout: run N forward passes with dropout ACTIVE → get a distribution.
     Mean = forecast, 1.96*std = 95% CI. Zero extra parameters, free uncertainty.

     IMPORTANT: nn.Dropout deactivates during model.eval() by default.
     MCDropout overrides this — it stays active regardless of training mode.
     The LSTM's own inter-layer dropout (set via the dropout param) is handled
     separately: we call model.train() during MC inference to keep it active too.

  4. WHY L1Loss in log space:
     Inputs and targets are log1p(demand), standardised. Minimising L1 there
     makes the model predict the conditional median, which is what MAE and
     MAPE reward on right-skewed household load, and it is robust to any
     spikes that remain in the inputs.

  5. WHY ReduceLROnPlateau scheduler:
     Fixed learning rates stall when loss plateaus. ReduceLROnPlateau halves
     the LR after `patience=3` epochs without val improvement, letting the
     model fine-tune into a sharper minimum without manual tuning.

  6. WHY calibrate the MC Dropout intervals:
     Raw MC Dropout spread (mean ± 1.96·std) is usually far too narrow — it
     only captures model uncertainty, not the noise in the data. We measure
     on the validation split how many stds the true value actually falls
     from the mean, per horizon step, and use the 95th percentile of that
     as the band width. The bands then really cover ~95% of outcomes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


# ─── MC DROPOUT ───────────────────────────────────────────────────────────────

class MCDropout(nn.Dropout):
    """
    Dropout that stays ACTIVE during model.eval().

    Standard nn.Dropout deactivates on eval() — correct for clean inference.
    For MC Dropout uncertainty estimation, we need it always on so repeated
    forward passes produce different stochastic outputs.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Force training=True regardless of module state
        return nn.functional.dropout(x, self.p, training=True, inplace=False)


# ─── MODEL ────────────────────────────────────────────────────────────────────

class DemandForecaster(nn.Module):
    """
    Encoder–decoder LSTM with MC Dropout for 24-hour demand forecasting.

    Args:
      input_size   : past features per timestep (default 15)
      future_size  : known-future covariates per forecast step (default 11)
      hidden_size  : LSTM hidden units per layer (default 128)
      num_layers   : stacked LSTM layers in encoder and decoder (default 2)
      output_steps : forecast horizon in hours (default 24)
      dropout      : dropout probability — applied both inside the LSTMs
                     (inter-layer) and in MCDropout on the decoder output
    """

    def __init__(
        self,
        input_size: int  = 15,
        future_size: int = 11,
        hidden_size: int = 128,
        num_layers: int  = 2,
        output_steps: int = 24,
        dropout: float   = 0.2,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_steps = output_steps
        lstm_dropout = dropout if num_layers > 1 else 0.0

        # batch_first=True so inputs are (batch, seq, features)
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=lstm_dropout)
        # Decoder starts from the encoder's final (h, c) — the summary of
        # the last 48h — and reads one future-covariate row per target hour
        self.decoder = nn.LSTM(future_size, hidden_size, num_layers,
                               batch_first=True, dropout=lstm_dropout)

        # MCDropout on decoder outputs — stays active during eval()
        self.mc_dropout = MCDropout(p=dropout)

        # Per-step head shared across all 24 forecast hours
        mid = hidden_size // 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),                    # GELU slightly outperforms ReLU on time-series
            nn.Linear(mid, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x_past   : (batch, lookback, input_size)
          x_future : (batch, output_steps, future_size)
        Returns:
          forecast : (batch, output_steps, 1)
        """
        _, state = self.encoder(x_past)          # (h, c): (num_layers, batch, hidden)
        dec_out, _ = self.decoder(x_future, state)  # (batch, output_steps, hidden)
        return self.fc(self.mc_dropout(dec_out))    # (batch, output_steps, 1)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_forecaster(
    model: DemandForecaster,
    train_loader,
    val_loader,
    epochs: int   = 50,
    lr: float     = 1e-3,
    device: str   = "cpu",
    patience: int = 7,
    weight_decay: float = 0.0,
) -> Tuple[DemandForecaster, dict]:
    """
    Train the forecaster with early stopping + LR scheduling.

    The DataLoader from preprocess.py yields 4-tuples:
      (X_past, X_future, y, anomaly)

    Early stopping: restore best weights if val loss stagnates for
    `patience` epochs — prevents overfitting on small zone datasets.

    Returns:
      model   : DemandForecaster with best validation weights restored
      history : dict with "train_loss" and "val_loss" lists
    """
    model     = model.to(device)
    # AdamW: decoupled weight decay regularises without fighting Adam's scaling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # L1 in log space → conditional median forecast (see header note 4)
    criterion = nn.L1Loss()

    # Halve LR after 3 epochs without val improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    best_weights  = None
    patience_ctr  = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):

        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for batch in train_loader:
            X_past, X_future, y, _ = batch
            X_past, X_future, y = X_past.to(device), X_future.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X_past, X_future)   # (batch, horizon, 1)
            loss = criterion(pred, y)
            loss.backward()

            # Gradient clipping: BPTT can produce huge gradients in deep LSTMs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X_past, X_future, y, _ = batch
                pred = model(X_past.to(device), X_future.to(device))
                y = y.to(device)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {lr_now:.2e}")

        # ── EARLY STOPPING ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val loss: {best_val_loss:.5f})")
                break

    model.load_state_dict(best_weights)
    return model, history


# ─── MC DROPOUT INFERENCE ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_uncertainty(
    model: DemandForecaster,
    X_past: torch.Tensor,
    X_future: torch.Tensor,
    n_samples: int = 50,
    device: str    = "cpu",
    interval_scale = 1.96,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout inference for calibrated uncertainty intervals.

    Run N stochastic forward passes with dropout ACTIVE.
    The spread across passes gives a free uncertainty estimate.

    WHY model.train() here:
      MCDropout.forward() forces training=True on its own layer,
      but calling model.train() also re-enables the LSTM's inter-layer
      dropout, making the stochasticity consistent across all layers.

    WHY 50 samples:
      30–100 samples stabilises the variance estimate. 50 balances
      accuracy vs latency (typically <100ms on CPU for one zone).

    Args:
      model    : trained DemandForecaster
      X_past   : (batch, lookback, input_size)
      X_future : (batch, horizon, future_size)
      n_samples: number of MC forward passes
      device   : "cpu" or "cuda"
      interval_scale : band half-width in stds — 1.96, or the per-step
                       (horizon,) array from calibrate_intervals()

    Returns:
      mean  : (batch, horizon, 1) — point forecast
      lower : (batch, horizon, 1) — 95% CI lower bound (mean - scale*std)
      upper : (batch, horizon, 1) — 95% CI upper bound (mean + scale*std)
    """
    model.train()   # activate all dropout layers for stochastic inference
    X_past, X_future = X_past.to(device), X_future.to(device)

    # Stack n_samples predictions: (n_samples, batch, horizon, 1)
    samples = torch.stack([model(X_past, X_future) for _ in range(n_samples)], dim=0)

    mean  = samples.mean(dim=0)
    std   = samples.std(dim=0)
    scale = torch.as_tensor(interval_scale, dtype=mean.dtype, device=mean.device)
    if scale.ndim == 1:
        scale = scale.view(1, -1, 1)   # per horizon step
    lower = mean - scale * std
    upper = mean + scale * std

    model.eval()
    return mean.cpu(), lower.cpu(), upper.cpu()


def predict_loader(
    model: DemandForecaster,
    loader,
    n_samples: int = 30,
    device: str    = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MC Dropout forecasts for a whole DataLoader.

    Returns (y_true, mean, std), each (N, horizon) in scaled log space.
    """
    trues, means, stds = [], [], []
    with torch.no_grad():
        for X_past, X_future, y, _ in loader:
            model.train()
            X_past, X_future = X_past.to(device), X_future.to(device)
            samples = torch.stack([model(X_past, X_future) for _ in range(n_samples)])
            trues.append(y.squeeze(-1).numpy())
            means.append(samples.mean(0).squeeze(-1).cpu().numpy())
            stds.append(samples.std(0).squeeze(-1).cpu().numpy())
    model.eval()
    return np.concatenate(trues), np.concatenate(means), np.concatenate(stds)


def calibrate_intervals(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    coverage: float = 0.95,
) -> np.ndarray:
    """
    Per-horizon-step band width (in MC stds) that achieves `coverage` on the
    given (validation) forecasts. Returns (horizon,).
    """
    z = np.abs(y_true - mean) / np.maximum(std, 1e-6)
    return np.quantile(z, coverage, axis=0)