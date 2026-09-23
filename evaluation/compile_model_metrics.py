#!/usr/bin/env python3
"""Compile collaborator CfC, mean, and rerun baseline results."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "current_logic_complete_comparison"
DATA = Path("data/htsk.csv")
SEEDS = [42, 43, 44]

CFC = {
    "primary_-16_to_-12": [
        [42, .273135, .140378, .374671, .436665, .665166, .640708],
        [43, .272294, .134935, .367334, .405971, .645922, .610643],
        [44, .275725, .137877, .371317, .461998, .679965, .635013],
    ],
    "full": [
        [42, .398171, 2.625772, 1.620424, .052782, .238745, .605992],
        [43, .340517, 1.387901, 1.178092, .073515, .277577, .570133],
        [44, .385323, 2.349826, 1.532914, .066737, .271984, .611742],
    ],
}


def metrics(actual, predicted):
    err = predicted - actual
    mse = float(np.mean(err**2))
    return {
        "MAE": float(np.mean(np.abs(err))), "MSE": mse, "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(err**2) / np.sum((actual - actual.mean())**2)),
        "Pearson": np.nan, "Spearman": np.nan,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ridge = pd.read_csv(ROOT / "current_logic_baseline_rerun" / "per_seed_metrics.csv")
    neural = pd.read_csv(ROOT / "current_logic_neural_rerun" / "per_seed_metrics.csv")
    additional = pd.read_csv(ROOT / "current_logic_lstm_gbt" / "per_seed_metrics.csv")
    rows = [ridge, neural, additional]

    cfc_rows = []
    for dataset, values in CFC.items():
        for seed, mae, mse, rmse, r2, pearson, spearman in values:
            cfc_rows.append({"Dataset": dataset, "Model": "CfC", "Seed": seed,
                             "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
                             "Pearson": pearson, "Spearman": spearman,
                             "Selected_alpha": np.nan})
    rows.append(pd.DataFrame(cfc_rows))

    raw = pd.read_csv(DATA)
    mean_rows = []
    for dataset, df in [("primary_-16_to_-12", raw.loc[raw.Delta_G.between(-16, -12)]), ("full", raw)]:
        for seed in SEEDS:
            shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            ntr = int(.70 * len(shuffled)); nv = int(.15 * len(shuffled))
            actual = shuffled.Delta_G.iloc[ntr + nv:].to_numpy()
            predicted = np.full_like(actual, shuffled.Delta_G.iloc[:ntr].mean())
            mean_rows.append({"Dataset": dataset, "Model": "Training mean", "Seed": seed,
                              **metrics(actual, predicted), "Selected_alpha": np.nan})
    rows.append(pd.DataFrame(mean_rows))

    per_seed = pd.concat(rows, ignore_index=True)
    order = {"CfC": 0, "Ridge": 1, "MLP": 2, "CNN": 3, "LSTM": 4,
             "GradientBoostedTrees": 5, "Training mean": 6}
    per_seed["_order"] = per_seed.Model.map(order)
    per_seed = per_seed.sort_values(["Dataset", "_order", "Seed"]).drop(columns="_order")
    per_seed.to_csv(OUT / "all_models_per_seed_metrics.csv", index=False)

    cols = ["MAE", "MSE", "RMSE", "R2", "Pearson", "Spearman"]
    summary = per_seed.groupby(["Dataset", "Model"])[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["_order"] = summary.Model.map(order)
    summary = summary.sort_values(["Dataset", "_order"]).drop(columns="_order")
    summary.to_csv(OUT / "all_models_three_seed_summary.csv", index=False)

    methods = pd.DataFrame([
        ["Shared", "Merged 18,065 unique 20-aa sequences; primary range -16 <= Delta_G <= -12 (17,804 rows) and full range."],
        ["Shared", "Seeds 42/43/44 each reshuffle data and define a new 70/15/15 repeated holdout split; train-only target z-score."],
        ["Shared neural", "70 epochs; batch 64; Huber delta=0.5 on standardized target; lr=1e-4; weight decay=1e-4; global gradient clipping=1.0."],
        ["Ridge", "Flattened 20x20 positional one-hot; alpha selected on validation from 0.1, 1, 10, 100."],
        ["MLP", "Flattened one-hot -> Dense256 ReLU -> Dropout0.2 -> Dense64 ReLU -> Dropout0.2 -> linear output."],
        ["CNN", "Conv1D64(k=3) -> ReLU -> Conv1D64(k=3) -> ReLU -> global max pooling -> Dense32 ReLU -> Dropout0.2 -> output."],
        ["LSTM", "One-hot -> LSTM64 -> Dropout0.2 -> Dense16 tanh -> Dropout0.2 -> output; same 70-epoch Huber/AdamW settings."],
        ["GradientBoostedTrees", "Histogram gradient boosting on flattened one-hot; learning rate, leaves, and iterations selected only on validation."],
        ["CfC", "Per-seed archived metrics; 20x20 one-hot; LayerNorm/Dense32/CfC32/Dropout0.2/Dense16/CfC16/Dropout0.2; Huber setup."],
    ], columns=["Scope_or_model", "Details"])
    methods.to_csv(OUT / "methods_and_fairness_notes.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
