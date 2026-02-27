"""
models/forecaster.py
────────────────────
PURPOSE:
  LSTM-based multi-step demand forecaster.
  Predicts next 24 hours given last 48 hours of demand.

ARCHITECTURE:
  Input → LSTM layers → Dropout → FC → Output (24 timesteps)

  Input  : (batch, 48, 1)    — 48 hours of normalized demand
  Output : (batch, 24, 1)    — 24-hour forecast

KEY DESIGN DECISIONS:

  1. WHY LSTM over GRU:
     Both work. LSTM has an extra "cell state" giving it longer memory.
     Useful for capturing weekly demand patterns in noisy industrial data.
     Use GRU if training is too slow — usually <5% accuracy difference.

  2. WHY 2 LSTM layers:
     Layer 1 learns short-term patterns (morning rise, evening peak).
     Layer 2 learns patterns-of-patterns (weekday vs weekend shape).
     More layers = diminishing returns + harder to train.

  3. WHY MC Dropout for confidence intervals:
     Standard LSTM gives a point estimate. Grid operators need to know
     uncertainty: "demand will be 450 kWh ± 30" vs just "450 kWh".
     MC Dropout: run inference N times with dropout ACTIVE → get distribution.
     Mean = forecast, std = uncertainty. Free uncertainty estimation.
     
     IMPORTANT: nn.Dropout is OFF during model.eval() by default.
     We override this by keeping specific dropout layers in train mode.

  4. WHY FC output vs sequence-to-sequence:
     Seq2seq with attention is more accurate but 5x more complex to implement.
     FC output (flatten LSTM hidden state → linear → 24 outputs) is fast,
     explainable, and works well for 24-step horizons.
"""

import torch
import torch.nn as nn
import numpy as np


class MCDropout(nn.Dropout):
    """
    Dropout that stays ACTIVE even during model.eval().
    
    Standard nn.Dropout turns off during eval() — this is what you want
    for a clean prediction. But for MC Dropout uncertainty estimation,
    you NEED it active at inference. This subclass overrides that behavior.
    """
    def forward(self, x):
        # Always apply dropout regardless of self.training flag
        return nn.functional.dropout(x, self.p, training=True, inplace=False)


class DemandForecaster(nn.Module):
    """
    2-layer LSTM with MC Dropout for 24-hour demand forecasting.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_steps: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_steps = output_steps

        # LSTM processes the sequence
        # batch_first=True → input shape (batch, seq, features) — more intuitive
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # inter-layer dropout
        )

        # MCDropout applied to LSTM output before FC
        self.mc_dropout = MCDropout(p=dropout)

        # FC: hidden_size → output_steps (24-step forecast in one shot)
        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        """
        x: (batch, lookback, 1)
        returns: (batch, output_steps, 1)
        """
        # LSTM: out is (batch, seq, hidden), (h_n, c_n)
        # We only need the LAST hidden state → out[:, -1, :]
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]           # (batch, hidden_size)

        dropped = self.mc_dropout(last_hidden) # apply MC Dropout
        forecast = self.fc(dropped)            # (batch, output_steps)

        return forecast.unsqueeze(-1)          # (batch, output_steps, 1)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_forecaster(
    model: DemandForecaster,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 7,
):
    """
    Train the forecaster with early stopping.

    WHY early stopping:
      LSTMs overfit readily on small datasets.
      We monitor val loss — if it doesn't improve for `patience` epochs,
      we restore the best weights. Prevents the model from memorizing training data.

    WHY Adam optimizer:
      Adam adapts learning rate per parameter. Works well for LSTMs where
      gradient magnitudes vary wildly across gates and layers.

    WHY MSELoss over MAELoss:
      MSE penalizes large errors more (squared). For grid management, a 500kW
      miss is much worse than five 100kW misses — MSE reflects this.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    best_weights  = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for batch in train_loader:
            X, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            # Gradient clipping: prevents exploding gradients in LSTMs
            # WHY: BPTT (backprop through time) can accumulate huge gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X, y = batch[0].to(device), batch[1].to(device)
                pred = model(X)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # ── EARLY STOPPING ─────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val loss: {best_val_loss:.5f})")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    return model, history


# ─── MC DROPOUT INFERENCE ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_uncertainty(
    model: DemandForecaster,
    X: torch.Tensor,
    n_samples: int = 50,
    device: str = "cpu",
):
    """
    Monte Carlo Dropout inference.

    Run forward pass N times with dropout ACTIVE.
    Collect N stochastic predictions → compute mean and std.

    WHY 50 samples:
      Empirically, 30–100 samples stabilizes the variance estimate.
      50 is a good balance of accuracy vs compute time.

    Returns:
      mean  : (batch, horizon, 1) — best forecast
      lower : (batch, horizon, 1) — mean - 1.96*std (95% CI lower bound)
      upper : (batch, horizon, 1) — mean + 1.96*std (95% CI upper bound)
    """
    model.train()  # Keep MCDropout active (mc_dropout uses training=True anyway,
                   # but this also activates inter-layer dropout in LSTM)
    X = X.to(device)

    samples = torch.stack([model(X) for _ in range(n_samples)], dim=0)
    # samples: (n_samples, batch, horizon, 1)

    mean  = samples.mean(dim=0)
    std   = samples.std(dim=0)

    lower = mean - 1.96 * std
    upper = mean + 1.96 * std

    return mean.cpu(), lower.cpu(), upper.cpu()