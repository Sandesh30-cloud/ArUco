import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


TARGET_COLS = ["ph", "do", "turbidity", "temperature"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_column(columns: List[str], candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(f"Could not find any of {candidates} in columns: {columns}")


def load_and_prepare(
    csv_path: str,
    time_col: str,
    freq: str,
    max_interp_gap: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in CSV columns")

    # Normalize expected sensor column names with flexible matching.
    ph_col = infer_column(df.columns.tolist(), ["ph", "p_h"])
    do_col = infer_column(df.columns.tolist(), ["do", "dissolved_oxygen"])
    turb_col = infer_column(df.columns.tolist(), ["turbidity", "ntu"])
    temp_col = infer_column(df.columns.tolist(), ["temperature", "temp", "water_temperature"])

    df = df[[time_col, ph_col, do_col, turb_col, temp_col]].copy()
    df.columns = ["timestamp"] + TARGET_COLS
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp")

    # Uniform time step and interpolate short sensor gaps.
    df = df.resample(freq).mean()
    df = df.interpolate(method="time", limit=max_interp_gap, limit_direction="both")
    df = df.dropna()

    if len(df) < 200:
        raise ValueError("Not enough clean points after preprocessing. Need at least ~200 rows.")
    return df


def build_sequences(values: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    max_i = len(values) - lookback - horizon + 1
    for i in range(max_i):
        x = values[i : i + lookback]
        y = values[i + lookback : i + lookback + horizon]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class RNNForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        output_size: int,
        model_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if model_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * output_size),
        )
        self.horizon = horizon
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred.view(-1, self.horizon, self.output_size)


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def split_time_ordered(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> SplitData:
    n = len(x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Invalid split sizes. Adjust train/val ratios.")

    return SplitData(
        x_train=x[:n_train],
        y_train=y[:n_train],
        x_val=x[n_train : n_train + n_val],
        y_val=y[n_train : n_train + n_val],
        x_test=x[n_train + n_val :],
        y_test=y[n_train + n_val :],
    )


def inverse_transform_3d(arr_3d: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, h, f = arr_3d.shape
    flat = arr_3d.reshape(-1, f)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(n, h, f)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for i, col in enumerate(TARGET_COLS):
        true_col = y_true[:, :, i].reshape(-1)
        pred_col = y_pred[:, :, i].reshape(-1)
        rmse = float(np.sqrt(mean_squared_error(true_col, pred_col)))
        mae = float(mean_absolute_error(true_col, pred_col))
        metrics[col] = {"rmse": rmse, "mae": mae}
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RNN forecasting for tilapia water parameters: pH, DO, turbidity, temperature."
    )
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument("--time-col", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--freq", default="10min", help="Resample frequency (e.g., 10min, 30min, 1H).")
    parser.add_argument("--lookback", type=int, default=36, help="Input timesteps.")
    parser.add_argument("--horizon", type=int, default=6, help="Future timesteps to predict.")
    parser.add_argument("--model", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-interp-gap", type=int, default=6)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="tilapia_model")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_and_prepare(args.csv, args.time_col, args.freq, args.max_interp_gap)
    values = df[TARGET_COLS].values.astype(np.float32)

    x_raw, y_raw = build_sequences(values, args.lookback, args.horizon)
    split = split_time_ordered(x_raw, y_raw, args.train_ratio, args.val_ratio)

    scaler = StandardScaler()
    scaler.fit(split.x_train.reshape(-1, split.x_train.shape[-1]))

    def transform_3d(arr: np.ndarray) -> np.ndarray:
        n, t, f = arr.shape
        return scaler.transform(arr.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    x_train = transform_3d(split.x_train)
    y_train = transform_3d(split.y_train)
    x_val = transform_3d(split.x_val)
    y_val = transform_3d(split.y_val)
    x_test = transform_3d(split.x_test)
    y_test = transform_3d(split.y_test)

    train_loader = DataLoader(SeqDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = RNNForecaster(
        input_size=len(TARGET_COLS),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        horizon=args.horizon,
        output_size=len(TARGET_COLS),
        model_type=args.model,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:03d} | train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)

    # Test predictions
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            truths.append(yb.numpy())

    y_pred_scaled = np.concatenate(preds, axis=0)
    y_true_scaled = np.concatenate(truths, axis=0)
    y_pred = inverse_transform_3d(y_pred_scaled, scaler)
    y_true = inverse_transform_3d(y_true_scaled, scaler)

    metrics = evaluate_predictions(y_true, y_pred)
    print("\nTest metrics:")
    for name in TARGET_COLS:
        print(f"{name:12s} RMSE={metrics[name]['rmse']:.4f} | MAE={metrics[name]['mae']:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "tilapia_rnn.pt")
    scaler_path = os.path.join(args.save_dir, "scaler_stats.json")
    metrics_path = os.path.join(args.save_dir, "test_metrics.json")
    config_path = os.path.join(args.save_dir, "config.json")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_order": TARGET_COLS,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            f,
            indent=2,
        )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved scaler stats: {scaler_path}")
    print(f"Saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
