#!/usr/bin/env python3
"""Extract the clean Double Mutants training set and manuscript DDIM clone table."""

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "ddim_double_retrained_baselines"
OUT.mkdir(parents=True, exist_ok=True)
AA = "ACDEFGHIKLMNPQRSTVWY"
R = 1.987204258e-3
T = 298.15

double_path = Path("data/Double_Mutants_Kd.xlsx")
paper_table = Path("data/DDIM15_CfC_and_Baseline_Comparison.xlsx")

raw = pd.read_excel(double_path, sheet_name="Sheet1")
sequence = raw["Sequence"].astype("string").str.strip().str.upper()
kd = pd.to_numeric(raw["Kd"], errors="coerce")
valid = sequence.str.fullmatch(f"[{AA}]{{20}}", na=False) & kd.ge(5) & np.isfinite(kd)
training = pd.DataFrame({"Sequence": sequence[valid], "Kd_pM": kd[valid]})
training = training.groupby("Sequence", as_index=False)["Kd_pM"].median()
training["Delta_G"] = R * T * np.log(training["Kd_pM"] * 1e-12)
training.to_csv(OUT / "double_mutants_clean_training.csv", index=False)

clones = pd.read_excel(paper_table, sheet_name="All_Predictions")
clones = clones[[
    "Name", "Full_Sequence", "Experimental_Kd_pM", "Experimental_Delta_G",
    "Double_CfC_Kd_pM", "Double_CfC_Delta_G", "Double_CfC_Rank",
]].copy()
clones["Model_Input_20aa"] = clones["Full_Sequence"].str[1:]
clones.to_csv(OUT / "manuscript_ddim15_reference.csv", index=False)

print(f"Training rows: {len(training):,}")
print(f"DDIM clones: {len(clones)}")
print(f"Exact clone overlaps after removing N-terminal Met: {clones.Model_Input_20aa.isin(training.Sequence).sum()}")
