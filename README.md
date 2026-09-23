# Bcl-xL sequence-affinity modeling code

This archive contains the training and evaluation code, trained CfC checkpoints,
and per-seed results supporting the revised manuscript, *Integrating Diffusion and
Liquid AI Models for Predicting Peptide Affinity from mRNA Display Selections*.

## Primary reproducibility workflow

The primary comparison uses 17,804 unique 20-residue peptide sequences with
binding free energies between -16 and -12 kcal/mol. For each random seed (42,
43, and 44), the rows are reshuffled and divided into 70% training, 15%
validation, and 15% test partitions. Target standardization uses only the
training subset. The reported values are the arithmetic mean and sample standard
deviation across the three repeated holdout trials.

The manuscript CfC architecture is:

`20x20 one-hot -> LayerNorm -> Dense32(tanh) -> CfC32 -> Dropout(0.20) -> Dense16(tanh) -> CfC16 -> Dropout(0.20) -> linear output`

Training uses AdamW, learning rate 1e-4, weight decay 1e-4, batch size 64,
Huber loss with delta 0.5, global gradient clipping at 1.0, and 70 epochs. No
early stopping is used for the reported comparison; the final epoch-70 model is
evaluated. PyTorch and ncps constructor defaults initialize the weights after
setting the Python, NumPy, PyTorch, and data-loader seeds; no custom
reinitialization is applied.

## Included trained model

`models/cfc/` contains the three CfC checkpoints used for the revised analysis:

- `CfC_seed42.pt`
- `CfC_seed43.pt`
- `CfC_seed44.pt`

Each checkpoint contains the model state dictionary and the training-set target
mean and standard deviation required to reverse standardization. SHA-256 hashes
are provided in `models/cfc/SHA256SUMS.txt`.

## Installation

Python 3.11 was used for the deposited run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Input data

The HTSK CSV must contain:

- `Sequence`: exactly 20 standard amino-acid letters after removal of the
  initiating methionine;
- `Delta_G`: experimental binding free energy in kcal/mol.

The HTSK CSV supplied with the manuscript submission should be deposited in the
same archival record as this code. The scripts require an explicit data path and
do not contain author-specific absolute paths.

## Reproduce the CfC analysis

Train all three seeds and write checkpoints, histories, split assignments, test
predictions, per-seed metrics, and the mean/SD summary:

```bash
python training/train_cfc.py --data /path/to/htsk.csv --out outputs/cfc_repeated_holdout
```

Recompute the test metrics from the deposited checkpoints:

```bash
python evaluation/evaluate_pretrained_cfc.py --data /path/to/htsk.csv
```

The manuscript-level metrics used in Table 1 are included in `results/`.

## Other models

The remaining scripts reproduce the CNN, Ridge/MLP, GBDT, LSTM, compact
Transformer, and DDIM-candidate analyses. They use the same seeds and repeated
70:15:15 holdout convention unless a script explicitly identifies itself as an
Extension-only exploratory analysis. The primary manuscript results should not
be regenerated from the Extension-only scripts.

## Archival release

Before public deposition, add the selected software license and replace the DOI
placeholder in the manuscript and response letter with the DOI issued by Zenodo
or an equivalent long-term repository. A GitHub repository can be linked to
Zenodo to create a versioned DOI.
