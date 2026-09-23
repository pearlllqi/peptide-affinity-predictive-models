#!/usr/bin/env python3
"""Extension-only Ridge baseline using the exact CfC preprocessing/split."""

from pathlib import Path
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = Path(os.environ.get("EXTENSION_INPUT", "./Extension_Kd-trunc.xlsx"))
OUT = Path(os.environ.get("EXTENSION_RIDGE_OUT", "Extension_Ridge_results"))
SEED = 42
WT = "IETITIYNYKKAADHFSMSM"
AA = "ACDEFGHIKLMNPQRSTVWY"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
R = 1.987204258e-3  # kcal mol^-1 K^-1
T = 298.15


def prepare():
    df = pd.read_excel(INPUT)
    if "Sequence" not in df or "Kd" not in df:
        raise ValueError("Input must contain Sequence and Kd columns (Kd in pM).")
    seq = df["Sequence"].astype("string").str.strip().str.upper()
    kd = pd.to_numeric(df["Kd"], errors="coerce")
    valid = seq.str.fullmatch(f"[{AA}]{{20}}", na=False) & kd.gt(0) & np.isfinite(kd)
    df = pd.DataFrame({"Sequence": seq[valid], "Kd_pM": kd[valid]})
    df = df.groupby("Sequence", as_index=False)["Kd_pM"].median()
    df["Delta_G"] = R * T * np.log(df["Kd_pM"] * 1e-12)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def encode(sequences):
    aa_i = {a: i for i, a in enumerate(AA)}
    x = np.zeros((len(sequences), 20, 21), dtype=np.float32)
    for n, s in enumerate(sequences):
        for p, a in enumerate(s):
            x[n, p, aa_i[a]] = 1.0
            x[n, p, 20] = float(a != WT[p])
    return x.reshape(len(sequences), -1)


def metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {"R2": r2_score(y, pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y, pred),
            "Pearson": float(np.corrcoef(y, pred)[0, 1]),
            "Spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman"))}


def main():
    np.random.seed(SEED); OUT.mkdir(exist_ok=True)
    df = prepare(); n = len(df); a = int(.70*n); b = a + int(.15*n)
    df["Split"] = np.where(np.arange(n) < a, "train", np.where(np.arange(n) < b, "validation", "test"))
    df.to_csv(OUT / "prepared_data_and_split.csv", index=False)
    x, y = encode(df.Sequence), df.Delta_G.to_numpy()
    xt, xv, xs = x[:a], x[a:b], x[b:]; yt, yv, ys = y[:a], y[a:b], y[b:]
    rows = []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(xt, yt)
        tm=metrics(yt,model.predict(xt)); vm=metrics(yv,model.predict(xv)); sm=metrics(ys,model.predict(xs))
        row = {"alpha":alpha,**{f"Train_{k}":v for k,v in tm.items()},**{f"Validation_{k}":v for k,v in vm.items()},**{f"Test_{k}_diagnostic_only":v for k,v in sm.items()}}; rows.append(row)
    tuning = pd.DataFrame(rows).sort_values("Validation_RMSE")
    tuning.to_csv(OUT / "validation_alpha_tuning.csv", index=False)
    best_alpha = float(tuning.iloc[0].alpha)
    plot_df=pd.DataFrame(rows).sort_values("alpha")
    fig,ax=plt.subplots(figsize=(9,6)); ax.semilogx(plot_df.alpha,plot_df.Train_RMSE,marker="o",linewidth=2,label="Training RMSE"); ax.semilogx(plot_df.alpha,plot_df.Validation_RMSE,marker="o",linewidth=2,label="Validation RMSE"); ax.semilogx(plot_df.alpha,plot_df.Test_RMSE_diagnostic_only,marker="o",linewidth=2,label="Test RMSE (diagnostic only)"); ax.set_xlabel("Ridge alpha (log scale)"); ax.set_ylabel("RMSE (Delta_G)"); ax.grid(alpha=.25); ax.axvline(best_alpha,color="black",linestyle="--",alpha=.65,label=f"Best validation alpha = {best_alpha:g}"); ax.legend(loc="upper center",ncol=2,fontsize=9,framealpha=.9); plt.title("Extension Ridge: train, validation, and test RMSE"); fig.tight_layout(); fig.savefig(OUT/"ridge_train_validation_test_curve.png",dpi=250); plt.close(fig)
    model = Ridge(alpha=best_alpha).fit(xt, yt)
    joblib.dump(model, OUT / "ridge_model.joblib")
    records = []
    all_metrics = []
    for name, sl, xx, yy in [("validation", slice(a,b), xv,yv), ("test", slice(b,n), xs,ys)]:
        pred = model.predict(xx); m = metrics(yy,pred); all_metrics.append({"Split":name, **m})
        part = df.iloc[sl][["Sequence","Kd_pM","Delta_G"]].copy()
        part.insert(0,"Split",name); part.rename(columns={"Delta_G":"Actual_Delta_G"}, inplace=True)
        part["Predicted_Delta_G"] = pred; part["Residual"] = pred-yy; part["Abs_Error"] = np.abs(pred-yy)
        records.append(part)
    pd.concat(records).to_csv(OUT / "holdout_predictions.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(OUT / "metrics.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({"best_alpha":best_alpha,"seed":SEED,"n":n},indent=2))
    print(pd.DataFrame(all_metrics).to_string(index=False)); print(f"Best alpha: {best_alpha}\nSaved to {OUT}")


if __name__ == "__main__": main()
