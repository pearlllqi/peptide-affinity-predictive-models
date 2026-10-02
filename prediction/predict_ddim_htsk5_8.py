#!/usr/bin/env python3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from ncps.torch import CfC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ddim_htsk5_8_final_models"
OUT.mkdir(parents=True, exist_ok=True)
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
SEEDS = (42, 43, 44)

CLONES = (
    pd.read_csv(ROOT / "data" / "ddim_htsk5_8_sequences.csv")
    .rename(columns={"Full_Sequence": "Sequence"})[["Clone", "Sequence"]]
)

def one_hot(sequences):
    x = np.zeros((len(sequences), 20, 20), np.float32)
    for row, sequence in enumerate(sequences):
        # The saved mixed models use the 20-aa input after removing the
        # initiator methionine from the 21-aa manuscript sequences.
        sequence = sequence[1:] if len(sequence) == 21 else sequence
        for position, aa in enumerate(sequence):
            x[row, position, AA_INDEX[aa]] = 1.0
    return x

def tokens(sequences):
    return np.asarray([[AA_INDEX[aa] for aa in (sequence[1:] if len(sequence) == 21 else sequence)] for sequence in sequences], np.int64)

class CurrentCfC(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(20)
        self.d1 = nn.Linear(20, 32)
        self.c1 = CfC(input_size=32, units=32, return_sequences=True, batch_first=True)
        self.dp1 = nn.Dropout(0.2)
        self.d2 = nn.Linear(32, 16)
        self.c2 = CfC(input_size=16, units=16, return_sequences=False, batch_first=True)
        self.dp2 = nn.Dropout(0.2)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.tanh(self.d1(self.norm(x)))
        x, _ = self.c1(x)
        x = self.dp1(x)
        x = torch.tanh(self.d2(x))
        x, _ = self.c2(x)
        return self.out(self.dp2(x)).squeeze(-1)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(20)
        self.conv = nn.Sequential(
            nn.Conv1d(20, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.norm(x).transpose(1, 2)
        return self.head(torch.amax(self.conv(x), dim=2)).squeeze(-1)

class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(20, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.dense = nn.Linear(64, 16)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = self.dropout1(hidden[-1])
        x = torch.tanh(self.dense(x))
        return self.output(self.dropout2(x)).squeeze(-1)

class CompactTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.aa = nn.Embedding(20, 64)
        # Match the saved compact-transformer checkpoints.
        self.pos = nn.Parameter(torch.zeros(1, 20, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.15,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(64))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.15), nn.Linear(32, 1))

    def forward(self, x):
        x = self.encoder(self.aa(x) + self.pos).mean(dim=1)
        return self.head(x).squeeze(-1)

@torch.no_grad()
def predict_torch(model, x):
    model.eval()
    return model(torch.from_numpy(x)).cpu().numpy()

def main():
    torch.set_num_threads(2)
    x_onehot = one_hot(CLONES["Sequence"])
    x_tokens = tokens(CLONES["Sequence"])
    rows = []

    for seed in SEEDS:
        for model_name, model_class in (("CNN", CNN), ("CfC", CurrentCfC)):
            saved = torch.load(
                ROOT / "outputs" / "mixed_train_ddim15" / f"{model_name}_seed{seed}.pt",
                map_location="cpu", weights_only=False,
            )
            model = model_class()
            model.load_state_dict(saved["state_dict"])
            values = predict_torch(model, x_onehot) * saved["target_std"] + saved["target_mean"]
            for clone, sequence, value in zip(CLONES.Clone, CLONES.Sequence, values):
                rows.append([clone, sequence, model_name, seed, float(value), "Epoch70"])

        saved = joblib.load(
            ROOT / "outputs" / "current_logic_lstm_gbt" / f"primary_-16_to_-12_gbt_seed{seed}.joblib"
        )
        values = saved["model"].predict(x_onehot.reshape(len(x_onehot), -1)) * saved["target_std"] + saved["target_mean"]
        for clone, sequence, value in zip(CLONES.Clone, CLONES.Sequence, values):
            rows.append([clone, sequence, "GBT", seed, float(value), "Validation-selected"])

        saved = torch.load(
            ROOT / "outputs" / "mixed_lstm_transformer_200epochs" / f"LSTM_seed{seed}_200epochs.pt",
            map_location="cpu", weights_only=False,
        )
        model = LSTMRegressor()
        model.load_state_dict(saved["best_state_dict"])
        values = predict_torch(model, x_onehot) * saved["target_std"] + saved["target_mean"]
        for clone, sequence, value in zip(CLONES.Clone, CLONES.Sequence, values):
            rows.append([clone, sequence, "LSTM", seed, float(value), f"BestValidation@{saved['best_validation_epoch']}"])

        saved = torch.load(
            ROOT / "outputs" / "mixed_lstm_transformer_200epochs" / f"CompactTransformer_seed{seed}_200epochs.pt",
            map_location="cpu", weights_only=False,
        )
        model = CompactTransformer()
        model.load_state_dict(saved["best_state_dict"])
        values = predict_torch(model, x_tokens) * saved["target_std"] + saved["target_mean"]
        for clone, sequence, value in zip(CLONES.Clone, CLONES.Sequence, values):
            rows.append([clone, sequence, "Compact Transformer", seed, float(value), f"BestValidation@{saved['best_validation_epoch']}"])

    per_seed = pd.DataFrame(rows, columns=[
        "Clone", "Sequence", "Model", "Round", "Predicted_Delta_G_kcal_mol", "Checkpoint"
    ])
    per_seed.to_csv(OUT / "DDIM_HTSK5_8_predictions_per_round.csv", index=False)
    summary = (
        per_seed.groupby(["Clone", "Sequence", "Model"], sort=False)["Predicted_Delta_G_kcal_mol"]
        .agg(Predicted_Delta_G_Mean="mean", Predicted_Delta_G_SD="std", Predicted_Delta_G_Min="min", Predicted_Delta_G_Max="max")
        .reset_index()
    )
    summary["Within_model_rank"] = summary.groupby("Model")["Predicted_Delta_G_Mean"].rank(method="min")
    summary.to_csv(OUT / "DDIM_HTSK5_8_predictions_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
