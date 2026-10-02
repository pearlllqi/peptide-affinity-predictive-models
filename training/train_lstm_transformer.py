#!/usr/bin/env python3
"""Train the current mixed-data LSTM and compact Transformer for 200 epochs."""

from pathlib import Path
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/htsk.csv"
DDIM = ROOT / "data/ddim_15_sequences.csv"
OUT = ROOT / "outputs/mixed_lstm_transformer_200epochs"
OUT.mkdir(parents=True, exist_ok=True)
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
SEEDS = (42, 43, 44)
EPOCHS = 200
BATCH_SIZE = 64


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot(sequences):
    x = np.zeros((len(sequences), 20, 20), np.float32)
    for row, sequence in enumerate(sequences):
        for position, aa in enumerate(sequence):
            x[row, position, AA_INDEX[aa]] = 1.0
    return x


def tokens(sequences):
    return np.asarray([[AA_INDEX[aa] for aa in sequence] for sequence in sequences], np.int64)


class LSTMRegressor(nn.Module):
    """Exact LSTM used in the current-logic comparison."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(20, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.20)
        self.dense = nn.Linear(64, 16)
        self.dropout2 = nn.Dropout(0.20)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = self.dropout1(hidden[-1])
        x = torch.tanh(self.dense(x))
        return self.output(self.dropout2(x)).squeeze(-1)


class CompactTransformer(nn.Module):
    """Exact compact Transformer used in the current comparison."""
    def __init__(self):
        super().__init__()
        self.aa = nn.Embedding(20, 64)
        self.position = nn.Parameter(torch.zeros(1, 20, 64))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.15,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(64))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.15), nn.Linear(32, 1))

    def forward(self, x):
        x = self.encoder(self.aa(x) + self.position).mean(dim=1)
        return self.head(x).squeeze(-1)


@torch.no_grad()
def predict(model, x, batch_size=1024):
    model.eval()
    return np.concatenate([
        model(torch.from_numpy(x[start:start + batch_size])).cpu().numpy()
        for start in range(0, len(x), batch_size)
    ])


def metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
        "Pearson": pearsonr(actual, predicted).statistic,
        "Spearman": spearmanr(actual, predicted).statistic,
    }


def train_model(model_name, model, x_train, y_train, x_validation, y_validation, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=0.5)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        train_prediction = predict(model, x_train)
        validation_prediction = predict(model, x_validation)
        train_eval_loss = float(loss_fn(torch.from_numpy(train_prediction), torch.from_numpy(y_train)).item())
        validation_loss = float(loss_fn(torch.from_numpy(validation_prediction), torch.from_numpy(y_validation)).item())
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "Model": model_name, "Seed": seed, "Epoch": epoch,
            "Train_Huber_batch_mean": float(np.mean(batch_losses)),
            "Train_Huber_eval": train_eval_loss,
            "Validation_Huber": validation_loss,
            "Best_so_far": validation_loss <= best_loss,
        })
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{model_name} seed={seed} epoch={epoch} "
                f"train={train_eval_loss:.6f} val={validation_loss:.6f} "
                f"best={best_loss:.6f}@{best_epoch}", flush=True,
            )
    return model, best_state, best_epoch, best_loss, pd.DataFrame(history)


def main():
    torch.set_num_threads(int(os.environ.get("MODEL_NUM_THREADS", "2")))
    raw = pd.read_csv(DATA)
    raw["Sequence"] = raw["Sequence"].astype(str).str.strip().str.upper()
    raw["Delta_G"] = pd.to_numeric(raw["Delta_G"], errors="coerce")
    raw = raw.loc[
        raw["Sequence"].str.fullmatch(f"[{AA}]{{20}}", na=False)
        & raw["Delta_G"].between(-16, -12)
    ].reset_index(drop=True)
    ddim = pd.read_csv(DDIM)
    ddim["Experimental_Delta_G"] = 1.987204258e-3 * 298.15 * np.log(ddim["Experimental_Kd_pM"] * 1e-12)

    metric_rows = []
    histories = []
    ddim_rows = []
    for model_name in ("LSTM", "CompactTransformer"):
        for seed in SEEDS:
            started = time.time()
            set_seed(seed)
            data = raw.sample(frac=1, random_state=seed).reset_index(drop=True)
            train_end = int(0.70 * len(data))
            validation_end = train_end + int(0.15 * len(data))
            y = data["Delta_G"].to_numpy(np.float32)
            mean, std = float(y[:train_end].mean()), float(y[:train_end].std())
            y_scaled = ((y - mean) / (std + 1e-12)).astype(np.float32)
            if model_name == "LSTM":
                x = one_hot(data["Sequence"])
                x_ddim = one_hot(ddim["Model_Input_20aa"])
                model = LSTMRegressor()
            else:
                x = tokens(data["Sequence"])
                x_ddim = tokens(ddim["Model_Input_20aa"])
                model = CompactTransformer()

            model, best_state, best_epoch, best_loss, history = train_model(
                model_name, model, x[:train_end], y_scaled[:train_end],
                x[train_end:validation_end], y_scaled[train_end:validation_end], seed,
            )
            histories.append(history)
            epoch200_state = copy.deepcopy(model.state_dict())
            for checkpoint, state in (("Epoch200", epoch200_state), ("BestValidation", best_state)):
                model.load_state_dict(state)
                test_prediction = predict(model, x[validation_end:]) * std + mean
                metric_rows.append({
                    "Model": model_name, "Seed": seed, "Checkpoint": checkpoint,
                    "Best_validation_epoch": best_epoch, "Best_validation_Huber": best_loss,
                    **metrics(y[validation_end:], test_prediction),
                })
                ddim_prediction = predict(model, x_ddim) * std + mean
                for i, value in enumerate(ddim_prediction):
                    ddim_rows.append({
                        "Model": model_name, "Seed": seed, "Checkpoint": checkpoint,
                        "Clone": ddim.iloc[i]["Name"],
                        "Experimental_Kd_pM": ddim.iloc[i]["Experimental_Kd_pM"],
                        "Experimental_Delta_G": ddim.iloc[i]["Experimental_Delta_G"],
                        "Predicted_Delta_G": value,
                    })
            torch.save({
                "model_name": model_name, "epoch200_state_dict": epoch200_state,
                "best_state_dict": best_state, "best_validation_epoch": best_epoch,
                "best_validation_loss": best_loss, "target_mean": mean, "target_std": std,
            }, OUT / f"{model_name}_seed{seed}_200epochs.pt")
            pd.concat(histories, ignore_index=True).to_csv(OUT / "training_history_partial.csv", index=False)
            pd.DataFrame(metric_rows).to_csv(OUT / "holdout_metrics_partial.csv", index=False)
            pd.DataFrame(ddim_rows).to_csv(OUT / "DDIM15_predictions_partial.csv", index=False)
            print(f"Finished {model_name} seed={seed} in {time.time()-started:.1f}s", flush=True)

    history = pd.concat(histories, ignore_index=True)
    history.to_csv(OUT / "training_history.csv", index=False)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(OUT / "holdout_metrics_per_seed.csv", index=False)
    columns = ["MAE", "MSE", "RMSE", "R2", "Pearson", "Spearman"]
    summary = metrics_frame.groupby(["Model", "Checkpoint"])[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(OUT / "holdout_metrics_3seed_summary.csv", index=False)
    pd.DataFrame(ddim_rows).to_csv(OUT / "DDIM15_predictions_per_seed.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({
        "data": str(DATA), "rows": len(raw), "range": [-16, -12],
        "seeds": SEEDS, "split": [0.70, 0.15, 0.15], "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
        "loss": "Huber(delta=0.5) on training-only z-scored Delta_G",
    }, indent=2))
    print("\nFINAL SUMMARY\n", summary, flush=True)


if __name__ == "__main__":
    main()
