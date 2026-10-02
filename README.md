# Peptide-affinity predictive models for Bcl-xL

This repository contains the data, training and evaluation code, deposited
model artifacts, and per-seed results supporting the manuscript *Integrating
Diffusion and Liquid AI Models for Predicting Peptide Affinity from mRNA
Display Selections*.

The primary study compares five sequence-based affinity-prediction models on
the same repeated-holdout design:

- convolutional neural network (CNN);
- gradient-boosted decision trees (GBDT);
- compact Transformer;
- long short-term memory network (LSTM);
- closed-form continuous network (CfC).

MLP and ridge-regression baselines and scripts for scoring diffusion-generated
peptide candidates are also included.

## Study design

The primary comparison uses 17,804 valid, unique 20-residue peptide sequences
from the pooled Extension and Doped HTSK dataset, with experimental binding
free energies from -16 to -12 kcal/mol. Seeds 42, 43, and 44 define three
independently shuffled 70% training, 15% validation, and 15% test partitions.
Target standardization is calculated from the training partition only and
reversed before evaluation.

MSE, RMSE, MAE, R², Pearson correlation, and Spearman correlation are computed
separately on each held-out test partition and summarized as the arithmetic
mean ± sample standard deviation across the three trials.

## Repository contents

```text
configs/       shared data, model, training, and evaluation configuration
data/          pooled HTSK data and DDIM candidate tables
training/      training scripts for all compared predictive models
evaluation/    checkpoint verification, metric compilation, and plotting tools
prediction/    scripts for scoring diffusion-generated peptide candidates
checkpoints/   deposited trained CfC checkpoints for seeds 42, 43, and 44
results/       per-seed metrics, mean/SD summaries, and test predictions
docs/          detailed reproducibility instructions
```

## Data files

- `data/pooled_extension_doped_htsk.csv`: pooled HTSK sequence-affinity table
  used for the primary model comparison.
- `data/htsk.csv`: compatibility copy used by existing script defaults.
- `data/ddim_15_sequences.csv`: 15 DDIM/FG candidates and associated
  experimental measurements.
- `data/ddim_htsk5_8_sequences.csv`: DDIM-HTSK5 through DDIM-HTSK8.

The primary table contains `Sequence`, `Kd_M`, and `Delta_G` columns. Model
training uses the 20-residue `Sequence` and the experimental `Delta_G` target.

## Model training scripts

| Model or analysis | Script |
| --- | --- |
| CNN, MLP, and ridge regression | `training/train_cnn_mlp_ridge.py` |
| LSTM and GBDT | `training/train_lstm_gbt.py` |
| Compact Transformer | `training/train_compact_transformer.py` |
| CfC | `training/train_cfc.py` |
| Additional LSTM/Transformer analysis | `training/train_lstm_transformer.py` |
| Extension-only exploratory baselines | `training/extension_*_baseline.py` |

All primary models use seeds 42, 43, and 44 and the repeated 70:15:15 split.
Shared and model-specific settings—including architecture, initialization,
regularization, optimization, and early-stopping behavior—are recorded in
`configs/mixed_dataset_config.yaml` and in the executable scripts.

## Installation

Python 3.11 and pinned package versions are provided in both
`requirements.txt` and `environment.yml`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate peptide-affinity-models
```

## Reproduce the model comparison

Run from the repository root:

```bash
python training/train_cnn_mlp_ridge.py --data data/pooled_extension_doped_htsk.csv
python training/train_lstm_gbt.py --data data/pooled_extension_doped_htsk.csv
COMPACT_TRANSFORMER_DATA=data/pooled_extension_doped_htsk.csv python training/train_compact_transformer.py
python training/train_cfc.py --data data/pooled_extension_doped_htsk.csv --out outputs/cfc_repeated_holdout
```

The CfC training script defaults to all three seeds and saves a checkpoint,
split assignment, training history, test predictions, and metrics for every
trial, plus per-seed and mean/SD summary tables.

See `docs/reproducibility.md` for the complete workflow.

## Deposited checkpoints and results

The reviewer-requested trained CfC models are deposited in `checkpoints/cfc/`:

- `CfC_seed42.pt`
- `CfC_seed43.pt`
- `CfC_seed44.pt`

SHA-256 hashes are supplied in `checkpoints/cfc/SHA256SUMS.txt`. The other
compared models can be retrained using their scripts and the shared
configuration; this release does not claim that their checkpoints are
deposited.

Key result files are:

- `results/per_seed_metrics.csv`;
- `results/aggregate_metrics_mean_sd.csv`;
- `results/predictions/cfc_test_predictions_seed42.csv`;
- `results/predictions/cfc_test_predictions_seed43.csv`;
- `results/predictions/cfc_test_predictions_seed44.csv`.

To recompute the held-out CfC metrics and prediction tables directly from the
deposited checkpoints:

```bash
python evaluation/evaluate_pretrained_cfc.py \
  --data data/pooled_extension_doped_htsk.csv
```

## Citation and archival release

When citing this repository, please cite the associated manuscript and the
versioned archival release. A GitHub release can be linked to Zenodo to provide
a persistent DOI.
