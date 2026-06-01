"""
models/forecaster.py
────────────────────
PURPOSE:
  LSTM-based multi-step demand forecaster.
  Predicts next 24 hours of demand given the last 48 hours + engineered features.

ARCHITECTURE:
  Input (batch, 48, 10) → 2-layer LSTM → MCDropout → FC head → Output (batch, 24, 1)

KEY DESIGN DECISIONS:

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

  4. WHY HuberLoss instead of MSE:
     MSE squares errors, so a single spike anomaly in the training data
     dominates the gradient and pulls weights toward predicting high values.
     HuberLoss is quadratic for small errors and linear for large ones (delta=1.0),
     making training robust to the 1% injected spike anomalies in the dataset.

  5. WHY ReduceLROnPlateau scheduler:
     Fixed learning rates stall when loss plateaus. ReduceLROnPlateau halves
     the LR after `patience=3` epochs without val improvement, letting the
     model fine-tune into a sharper minimum without manual tuning.
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
    2-layer LSTM with MC Dropout for 24-hour demand forecasting.

    Args:
      input_size   : number of input features per timestep (default 10)
      hidden_size  : LSTM hidden units per layer (default 128)
      num_layers   : number of stacked LSTM layers (default 2)
      output_steps : forecast horizon in hours (default 24)
      dropout      : dropout probability — applied both inside LSTM
                     (inter-layer) and in MCDropout after LSTM output
    """

    def __init__(
        self,
        input_size: int  = 10,
        hidden_size: int = 128,
        num_layers: int  = 2,
        output_steps: int = 24,
        dropout: float   = 0.2,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_steps = output_steps

        # LSTM — batch_first=True so input is (batch, seq, features)
        # dropout param activates inter-layer dropout (only when num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,   # ← uses constructor param, not hardcoded
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # ← uses constructor param
        )

        # MCDropout on final LSTM output — stays active during eval()
        self.mc_dropout = MCDropout(p=dropout)

        # Forecast head: hidden_size → output_steps
        # Progressive narrowing gives the network capacity to learn
        # non-linear combinations before the final projection.
        # Using hidden_size (not hardcoded 128) so the class is reusable.
        mid = hidden_size // 2   # 64 when hidden_size=128
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),                    # GELU slightly outperforms ReLU on time-series
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),        # standard dropout (off at eval — intentional here)
            nn.Linear(mid // 2, output_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x : (batch, lookback, input_size)
        Returns:
          forecast : (batch, output_steps, 1)
        """
        # out: (batch, lookback, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, _ = self.lstm(x)

        # Take only the final timestep's hidden state as the sequence summary
        last_hidden = out[:, -1, :]           # (batch, hidden_size)

        # MC Dropout — active even during eval() for uncertainty estimation
        last_hidden = self.mc_dropout(last_hidden)

        # Project to forecast horizon
        forecast = self.fc(last_hidden)       # (batch, output_steps)
        return forecast.unsqueeze(-1)         # (batch, output_steps, 1)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_forecaster(
    model: DemandForecaster,
    train_loader,
    val_loader,
    epochs: int   = 50,
    lr: float     = 1e-3,
    device: str   = "cpu",
    patience: int = 7,
) -> Tuple[DemandForecaster, dict]:
    """
    Train the forecaster with early stopping + LR scheduling.

    The DataLoader from preprocess.py yields 4-tuples:
      (X_full, X_demand, y, anomaly)
    The forecaster uses X_full (all features) and y.

    Early stopping: restore best weights if val loss stagnates for
    `patience` epochs — prevents overfitting on small zone datasets.

    Returns:
      model   : DemandForecaster with best validation weights restored
      history : dict with "train_loss" and "val_loss" lists
    """
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # HuberLoss: robust to spike anomalies in the training targets
    criterion = nn.HuberLoss(delta=1.0)

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
            X_full, _, y, _ = batch          # forecaster only needs X_full and y
            X_full, y = X_full.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X_full)             # (batch, horizon, 1)
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
                X_full, _, y, _ = batch
                X_full, y = X_full.to(device), y.to(device)
                pred = model(X_full)
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
    X: torch.Tensor,
    n_samples: int = 50,
    device: str    = "cpu",
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
      X        : (batch, lookback, input_size) — X_full from DataLoader
      n_samples: number of MC forward passes
      device   : "cpu" or "cuda"

    Returns:
      mean  : (batch, horizon, 1) — point forecast
      lower : (batch, horizon, 1) — 95% CI lower bound (mean - 1.96*std)
      upper : (batch, horizon, 1) — 95% CI upper bound (mean + 1.96*std)
    """
    model.train()   # activate all dropout layers for stochastic inference
    X = X.to(device)

    # Stack n_samples predictions: (n_samples, batch, horizon, 1)
    samples = torch.stack([model(X) for _ in range(n_samples)], dim=0)

    mean  = samples.mean(dim=0)
    std   = samples.std(dim=0)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std

    return mean.cpu(), lower.cpu(), upper.cpu()