# Reproducibility guide

## Scope

This release supports the three-seed comparison of CNN, GBDT, compact
Transformer, LSTM, and CfC models reported in the manuscript. MLP and ridge
regression are included as additional baselines. Seeds 42, 43, and 44 define
three independently shuffled 70:15:15 train/validation/test partitions.

## Environment

Create the deposited Python 3.11 environment with either:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate peptide-affinity-models
```

## Data

- `data/pooled_extension_doped_htsk.csv` is the pooled HTSK table used for the
  primary comparison. `data/htsk.csv` is retained as a compatibility copy for
  existing script defaults.
- `data/ddim_15_sequences.csv` contains the 15 DDIM/FG sequences and associated
  experimental values used by the DDIM-candidate analysis.
- `data/ddim_htsk5_8_sequences.csv` contains DDIM-HTSK5 through DDIM-HTSK8.

The primary analysis keeps valid 20-residue sequences with Delta_G from -16 to
-12 kcal/mol, yielding 17,804 rows. Target mean and population standard
deviation are calculated from the training partition only.

## Configuration

`configs/mixed_dataset_config.yaml` records the shared split, optimization,
architecture, initialization, regularization, and evaluation settings. The
Python scripts remain the executable source of truth.

## Training

Run from the repository root:

```bash
python training/train_cnn_mlp_ridge.py --data data/pooled_extension_doped_htsk.csv
python training/train_lstm_gbt.py --data data/pooled_extension_doped_htsk.csv
COMPACT_TRANSFORMER_DATA=data/pooled_extension_doped_htsk.csv python training/train_compact_transformer.py
python training/train_cfc.py --data data/pooled_extension_doped_htsk.csv --out outputs/cfc_repeated_holdout
```

`train_cfc.py` defaults to seeds 42, 43, and 44 and writes, for each seed, the
checkpoint, deterministic split assignment, training history, held-out test
predictions, and test metrics. It also writes `cfc_per_seed_metrics.csv` and
`cfc_three_seed_summary.csv`.

## Deposited outputs

- `checkpoints/cfc/` contains the three trained CfC checkpoints requested by
  the reviewer and their SHA-256 hashes.
- `results/per_seed_metrics.csv` contains the individual-seed metrics for the
  model comparison.
- `results/aggregate_metrics_mean_sd.csv` contains mean and sample SD across
  the three trials.
- `results/predictions/cfc_test_predictions_seed*.csv` contains the held-out
  CfC predictions recomputed from the deposited checkpoints.

The other compared models are reproducible from their training scripts and the
shared configuration. Their trained checkpoints are not claimed as deposited
artifacts in this release.

## Verify the deposited CfC checkpoints

```bash
python evaluation/evaluate_pretrained_cfc.py \
  --data data/pooled_extension_doped_htsk.csv
```

This command recomputes per-seed held-out metrics and prediction tables without
retraining the model.
