#!/usr/bin/env python3
"""Recompute test metrics for the deposited CfC checkpoints."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_cfc import CurrentCfC, encode, metrics  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=root / "models" / "cfc")
    parser.add_argument("--out", type=Path,
                        default=root / "results" / "cfc_recomputed_metrics.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    raw["Sequence"] = raw["Sequence"].astype(str).str.strip().str.upper()
    raw["Delta_G"] = pd.to_numeric(raw["Delta_G"], errors="coerce")
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    raw = raw.loc[
        raw["Sequence"].str.fullmatch(f"[{amino_acids}]{{20}}", na=False)
        & raw["Delta_G"].between(-16, -12)
    ].reset_index(drop=True)

    rows = []
    for seed in args.seeds:
        data = raw.sample(frac=1, random_state=seed).reset_index(drop=True)
        test_start = int(0.70 * len(data)) + int(0.15 * len(data))
        checkpoint = torch.load(
            args.models / f"CfC_seed{seed}.pt", map_location="cpu", weights_only=False
        )
        model = CurrentCfC()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x = encode(data["Sequence"].iloc[test_start:].to_numpy())
        with torch.no_grad():
            prediction = model(torch.from_numpy(x)).numpy()
        prediction = prediction * checkpoint["target_std"] + checkpoint["target_mean"]
        actual = data["Delta_G"].iloc[test_start:].to_numpy(float)
        rows.append({"Seed": seed, "N_test": len(actual), **metrics(actual, prediction)})

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
