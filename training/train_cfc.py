#!/usr/bin/env python3
"""Train and evaluate the CfC model used for the three-seed manuscript results."""

from pathlib import Path
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from ncps.torch import CfC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(sequences):
    x = np.zeros((len(sequences), 20, 20), np.float32)
    for row, sequence in enumerate(sequences):
        for position, aa in enumerate(sequence):
            x[row, position, AA_INDEX[aa]] = 1.0
    return x


class CurrentCfC(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(20)
        self.d1 = nn.Linear(20, 32)
        self.c1 = CfC(input_size=32, units=32, return_sequences=True, batch_first=True)
        self.dp1 = nn.Dropout(0.20)
        self.d2 = nn.Linear(32, 16)
        self.c2 = CfC(input_size=16, units=16, return_sequences=False, batch_first=True)
        self.dp2 = nn.Dropout(0.20)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.tanh(self.d1(self.norm(x)))
        x, _ = self.c1(x)
        x = self.dp1(x)
        x = torch.tanh(self.d2(x))
        x, _ = self.c2(x)
        return self.out(self.dp2(x)).squeeze(-1)


@torch.no_grad()
def predict(model, x, batch_size=1024):
    model.eval()
    return np.concatenate([
        model(torch.from_numpy(x[start:start + batch_size])).cpu().numpy()
        for start in range(0, len(x), batch_size)
    ])


def metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(actual, predicted)),
        "Pearson": float(pearsonr(actual, predicted).statistic),
        "Spearman": float(spearmanr(actual, predicted).statistic),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="CSV containing Sequence and Delta_G columns")
    parser.add_argument("--out", type=Path, default=Path("outputs/cfc_repeated_holdout"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("MODEL_NUM_THREADS", "2")))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    raw = pd.read_csv(args.data)
    raw["Sequence"] = raw["Sequence"].astype(str).str.strip().str.upper()
    raw["Delta_G"] = pd.to_numeric(raw["Delta_G"], errors="coerce")
    raw = raw.loc[
        raw["Sequence"].str.fullmatch(f"[{AA}]{{20}}", na=False)
        & raw["Delta_G"].between(-16, -12)
    ].reset_index(drop=True)

    all_history = []
    all_metrics = []
    for seed in args.seeds:
        seed_all(seed)
        data = raw.sample(frac=1, random_state=seed).reset_index(drop=True)
        x = encode(data["Sequence"].to_numpy())
        y = data["Delta_G"].to_numpy(np.float32)
        train_end = int(0.70 * len(data))
        validation_end = train_end + int(0.15 * len(data))
        mean, std = float(y[:train_end].mean()), float(y[:train_end].std())
        y_scaled = ((y - mean) / (std + 1e-12)).astype(np.float32)

        model = CurrentCfC()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        loss_fn = nn.HuberLoss(delta=args.huber_delta)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x[:train_end]), torch.from_numpy(y_scaled[:train_end])),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        seed_history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            batch_losses = []
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                batch_losses.append(float(loss.item()))
            validation_prediction = predict(model, x[train_end:validation_end])
            validation_loss = float(loss_fn(
                torch.from_numpy(validation_prediction),
                torch.from_numpy(y_scaled[train_end:validation_end]),
            ).item())
            row = {
                "Epoch": epoch,
                "loss": float(np.mean(batch_losses)),
                "val_loss": validation_loss,
                "Dataset": "primary_-16_to_-12",
                "Model": "CfC",
                "Seed": seed,
            }
            seed_history.append(row)
            all_history.append(row)
            if epoch == 1 or epoch % 10 == 0:
                print(f"CfC seed={seed} epoch={epoch} train={row['loss']:.6f} val={validation_loss:.6f}", flush=True)
        test_prediction = predict(model, x[validation_end:]) * std + mean
        actual = y[validation_end:]
        row = {"Seed": seed, "N_test": len(actual), **metrics(actual, test_prediction)}
        all_metrics.append(row)
        pd.DataFrame(seed_history).to_csv(
            args.out / f"primary_-16_to_-12_cfc_seed{seed}_history.csv", index=False
        )
        pd.DataFrame({
            "Sequence": data["Sequence"].iloc[validation_end:].to_numpy(),
            "Actual_Delta_G": actual,
            "Predicted_Delta_G": test_prediction,
            "Seed": seed,
        }).to_csv(args.out / f"primary_-16_to_-12_cfc_seed{seed}_test_predictions.csv", index=False)
        split_frame = data[["Sequence", "Delta_G"]].copy()
        split_frame["Split"] = (
            ["train"] * train_end
            + ["validation"] * (validation_end - train_end)
            + ["test"] * (len(data) - validation_end)
        )
        split_frame.to_csv(
            args.out / f"primary_-16_to_-12_cfc_seed{seed}_split.csv", index=False
        )
        torch.save({
            "state_dict": model.state_dict(), "target_mean": mean, "target_std": std,
            "seed": seed, "epochs": args.epochs, "test_metrics": row,
        }, args.out / f"CfC_seed{seed}.pt")
    pd.DataFrame(all_history).to_csv(args.out / "mixed_cfc_3seed_training_history.csv", index=False)
    per_seed = pd.DataFrame(all_metrics)
    per_seed.to_csv(args.out / "cfc_per_seed_metrics.csv", index=False)
    columns = ["MAE", "MSE", "RMSE", "R2", "Pearson", "Spearman"]
    summary = per_seed[columns].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["Metric", "Mean", "SD"]
    summary.to_csv(args.out / "cfc_three_seed_summary.csv", index=False)
    config = {
        **vars(args), "data": str(args.data), "out": str(args.out),
        "valid_rows": len(raw), "split": [0.70, 0.15, 0.15],
        "target_scaling": "training-set z-score",
        "initialization": "PyTorch and ncps constructor defaults; no custom reinitialization",
        "early_stopping": False, "reported_checkpoint": f"final epoch {args.epochs}",
    }
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(per_seed.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
