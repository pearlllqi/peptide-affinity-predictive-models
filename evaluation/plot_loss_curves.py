#!/usr/bin/env python3
"""Plot 3-seed mean training/validation curves for Ridge, MLP, and CNN."""

from pathlib import Path
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NEURAL = ROOT / "current_logic_neural_rerun"
RIDGE = ROOT / "current_logic_baseline_rerun" / "ridge_validation_tuning.csv"
OUT = ROOT / "current_logic_baseline_loss_curves"


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def neural_curves(dataset, model):
    paths = sorted(glob.glob(str(NEURAL / f"{dataset}_{model.lower()}_seed*_history.csv")))
    runs = []
    for path in paths:
        rows = read_csv(path)
        runs.append({
            "epoch": np.array([float(r["Epoch"]) for r in rows]),
            "train": np.array([float(r["loss"]) for r in rows]),
            "val": np.array([float(r["val_loss"]) for r in rows]),
        })
    epoch = runs[0]["epoch"]
    train = np.stack([r["train"] for r in runs])
    val = np.stack([r["val"] for r in runs])
    return epoch, train.mean(0), train.std(0, ddof=1), val.mean(0), val.std(0, ddof=1)


def ridge_curve(dataset):
    rows = [r for r in read_csv(RIDGE) if r["Dataset"] == dataset]
    alphas = sorted({float(r["Alpha"]) for r in rows})
    values = []
    for seed in (42, 43, 44):
        sr = {float(r["Alpha"]): float(r["Validation_MSE"]) for r in rows if int(r["Seed"]) == seed}
        values.append([sr[a] for a in alphas])
    values = np.asarray(values)
    return np.asarray(alphas), values.mean(0), values.std(0, ddof=1)


def make_figure(dataset, label):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    alpha, mean, sd = ridge_curve(dataset)
    ax = axes[0]
    ax.plot(alpha, mean, marker="o", color="#2b6cb0", linewidth=2, label="Validation MSE")
    ax.fill_between(alpha, mean - sd, mean + sd, color="#2b6cb0", alpha=0.18, linewidth=0)
    best = int(np.argmin(mean))
    ax.scatter([alpha[best]], [mean[best]], s=70, color="#c53030", zorder=4, label=f"Selected alpha = {alpha[best]:g}")
    ax.set_xscale("log")
    ax.set_xlabel("Ridge alpha")
    ax.set_ylabel("Validation MSE (Delta G units)")
    ax.set_title("Ridge validation tuning")
    ax.legend(frameon=False, fontsize=8)

    for ax, model in zip(axes[1:], ("MLP", "CNN")):
        epoch, tr, tr_sd, va, va_sd = neural_curves(dataset, model)
        ax.plot(epoch, tr, color="#2b6cb0", linewidth=2, label="Training Huber loss")
        ax.fill_between(epoch, tr - tr_sd, tr + tr_sd, color="#2b6cb0", alpha=0.16, linewidth=0)
        ax.plot(epoch, va, color="#dd6b20", linewidth=2, label="Validation Huber loss")
        ax.fill_between(epoch, va - va_sd, va + va_sd, color="#dd6b20", alpha=0.16, linewidth=0)
        best_epoch = int(epoch[np.argmin(va)])
        ax.axvline(best_epoch, color="#718096", linestyle="--", linewidth=1)
        ax.text(best_epoch + 1, ax.get_ylim()[1] - 0.06 * np.ptp(ax.get_ylim()), f"best val epoch {best_epoch}", fontsize=8, color="#4a5568", va="top")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber loss (standardized Delta G)")
        ax.set_title(f"{model} training and validation")
        ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(f"{label}: baseline loss curves (3-seed mean ± SD)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output = OUT / f"{dataset}_ridge_mlp_cnn_loss_curves.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_figure("primary_-16_to_-12", "Primary range (-16 to -12 kcal/mol)")
    make_figure("full", "Full merged dataset")
    print(OUT)


if __name__ == "__main__":
    main()
