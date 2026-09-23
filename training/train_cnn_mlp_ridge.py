#!/usr/bin/env python3
"""Rerun Ridge, MLP, and CNN using the manuscript CfC split/target logic.

This implementation uses NumPy only so it remains runnable without TensorFlow or
PyTorch.  It reproduces the data regimes, repeated random holdouts, train-only
target standardization, 70 epochs, batch size 64, Huber loss, AdamW-style weight
decay, and gradient clipping used in the manuscript analysis.
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
SEEDS = [42, 43, 44]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]


def one_hot(sequences):
    x = np.zeros((len(sequences), 20, 20), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            x[i, j, AA_INDEX[aa]] = 1.0
    return x


def prepare(df, seed):
    data = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    x = one_hot(data.Sequence.to_numpy())
    y = data.Delta_G.to_numpy(np.float32)
    n_train = int(0.70 * len(data))
    n_val = int(0.15 * len(data))
    splits = {
        "train": slice(0, n_train),
        "validation": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, len(data)),
    }
    mean = float(y[splits["train"]].mean())
    std = float(y[splits["train"]].std())
    ys = (y - mean) / (std + 1e-12)
    return data, x, y, ys.astype(np.float32), splits, mean, std


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - actual
    mse = float(np.mean(residual**2))
    ss_tot = float(np.sum((actual - actual.mean())**2))
    pearson = float(np.corrcoef(actual, predicted)[0, 1]) if predicted.std() > 0 else np.nan
    if predicted.std() > 0:
        actual_rank = pd.Series(actual).rank(method="average").to_numpy()
        predicted_rank = pd.Series(predicted).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(actual_rank, predicted_rank)[0, 1])
    else:
        spearman = np.nan
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(residual**2) / ss_tot),
        "Pearson": pearson,
        "Spearman": spearman,
    }


def huber_loss_and_grad(pred, target, delta=0.5):
    err = pred - target
    abs_err = np.abs(err)
    loss = np.where(abs_err <= delta, 0.5 * err**2, delta * (abs_err - 0.5 * delta))
    grad = np.where(abs_err <= delta, err, delta * np.sign(err)) / len(target)
    return float(loss.mean()), grad.astype(np.float32)


def global_clip(grads, max_norm=1.0):
    norm = np.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in grads.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        return {k: v * scale for k, v in grads.items()}
    return grads


class AdamW:
    def __init__(self, params, lr=1e-4, weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * (g * g)
            mh = self.m[k] / (1 - 0.9**self.t)
            vh = self.v[k] / (1 - 0.999**self.t)
            if k.startswith("W"):
                p *= 1 - self.lr * self.wd
            p -= self.lr * mh / (np.sqrt(vh) + 1e-7)


class MLP:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.p = {
            "W1": (rng.standard_normal((400, 256)) * np.sqrt(2 / 400)).astype(np.float32),
            "b1": np.zeros(256, np.float32),
            "W2": (rng.standard_normal((256, 64)) * np.sqrt(2 / 256)).astype(np.float32),
            "b2": np.zeros(64, np.float32),
            "W3": (rng.standard_normal((64, 1)) * np.sqrt(1 / 64)).astype(np.float32),
            "b3": np.zeros(1, np.float32),
        }

    def forward(self, x, train=False):
        x = x.reshape(len(x), -1)
        z1 = x @ self.p["W1"] + self.p["b1"]
        h1 = np.maximum(z1, 0)
        m1 = (self.rng.random(h1.shape) >= 0.2).astype(np.float32) / 0.8 if train else np.ones_like(h1)
        d1 = h1 * m1
        z2 = d1 @ self.p["W2"] + self.p["b2"]
        h2 = np.maximum(z2, 0)
        m2 = (self.rng.random(h2.shape) >= 0.2).astype(np.float32) / 0.8 if train else np.ones_like(h2)
        d2 = h2 * m2
        out = (d2 @ self.p["W3"] + self.p["b3"]).reshape(-1)
        return out, (x, z1, m1, d1, z2, m2, d2)

    def backward(self, grad, cache):
        x, z1, m1, d1, z2, m2, d2 = cache
        go = grad[:, None]
        g = {
            "W3": d2.T @ go, "b3": go.sum(0),
        }
        gd2 = go @ self.p["W3"].T
        gz2 = gd2 * m2 * (z2 > 0)
        g["W2"] = d1.T @ gz2; g["b2"] = gz2.sum(0)
        gd1 = gz2 @ self.p["W2"].T
        gz1 = gd1 * m1 * (z1 > 0)
        g["W1"] = x.T @ gz1; g["b1"] = gz1.sum(0)
        return g


def conv_forward(x, w, b):
    xp = np.pad(x, ((0, 0), (1, 1), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, 3, axis=1)
    out = np.einsum("blck,kco->blo", win, w, optimize=True) + b
    return out, win


def conv_backward(grad, win, w):
    gw = np.einsum("blck,blo->kco", win, grad, optimize=True)
    gb = grad.sum(axis=(0, 1))
    gwin = np.einsum("blo,kco->blck", grad, w, optimize=True)
    b, length, channels, kernel = gwin.shape
    gxpad = np.zeros((b, length + 2, channels), np.float32)
    for k in range(kernel):
        gxpad[:, k:k + length, :] += gwin[:, :, :, k]
    return gxpad[:, 1:-1, :], gw, gb


class CNN:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.p = {
            "W1": (rng.standard_normal((3, 20, 64)) * np.sqrt(2 / 60)).astype(np.float32),
            "b1": np.zeros(64, np.float32),
            "W2": (rng.standard_normal((3, 64, 64)) * np.sqrt(2 / 192)).astype(np.float32),
            "b2": np.zeros(64, np.float32),
            "W3": (rng.standard_normal((64, 32)) * np.sqrt(2 / 64)).astype(np.float32),
            "b3": np.zeros(32, np.float32),
            "W4": (rng.standard_normal((32, 1)) * np.sqrt(1 / 32)).astype(np.float32),
            "b4": np.zeros(1, np.float32),
        }

    def forward(self, x, train=False):
        z1, win1 = conv_forward(x, self.p["W1"], self.p["b1"])
        h1 = np.maximum(z1, 0)
        z2, win2 = conv_forward(h1, self.p["W2"], self.p["b2"])
        h2 = np.maximum(z2, 0)
        idx = h2.argmax(axis=1)
        pooled = h2.max(axis=1)
        z3 = pooled @ self.p["W3"] + self.p["b3"]
        h3 = np.maximum(z3, 0)
        mask = (self.rng.random(h3.shape) >= 0.2).astype(np.float32) / 0.8 if train else np.ones_like(h3)
        dropped = h3 * mask
        out = (dropped @ self.p["W4"] + self.p["b4"]).reshape(-1)
        return out, (z1, win1, h1, z2, win2, h2, idx, pooled, z3, mask, dropped)

    def backward(self, grad, cache):
        z1, win1, h1, z2, win2, h2, idx, pooled, z3, mask, dropped = cache
        go = grad[:, None]
        g = {"W4": dropped.T @ go, "b4": go.sum(0)}
        gz3 = (go @ self.p["W4"].T) * mask * (z3 > 0)
        g["W3"] = pooled.T @ gz3; g["b3"] = gz3.sum(0)
        gpool = gz3 @ self.p["W3"].T
        gh2 = np.zeros_like(h2)
        rows = np.arange(len(h2))[:, None]
        chans = np.arange(h2.shape[2])[None, :]
        gh2[rows, idx, chans] = gpool
        gz2 = gh2 * (z2 > 0)
        gh1, g["W2"], g["b2"] = conv_backward(gz2, win2, self.p["W2"])
        gz1 = gh1 * (z1 > 0)
        _, g["W1"], g["b1"] = conv_backward(gz1, win1, self.p["W1"])
        return g


def predict_batches(model, x, batch=512):
    return np.concatenate([model.forward(x[i:i + batch], train=False)[0] for i in range(0, len(x), batch)])


def train_nn(model, x_train, y_train, x_val, y_val, epochs, batch_size, seed):
    opt = AdamW(model.p, lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            pred, cache = model.forward(x_train[ids], train=True)
            loss, grad = huber_loss_and_grad(pred, y_train[ids])
            grads = global_clip(model.backward(grad, cache), 1.0)
            opt.step(grads)
            losses.append(loss)
        val_pred = predict_batches(model, x_val)
        val_loss = huber_loss_and_grad(val_pred, y_val)[0]
        history.append({"Epoch": epoch, "loss": float(np.mean(losses)), "val_loss": val_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}: loss={np.mean(losses):.5f}, val={val_loss:.5f}", flush=True)
    return pd.DataFrame(history)


def ridge_fit(x, y, alpha):
    x = x.reshape(len(x), -1).astype(np.float64)
    mean = x.mean(0)
    xc = x - mean
    yc = y.astype(np.float64) - y.mean()
    # Dual solution is substantially faster because samples < features is false;
    # use primal 400x400 solve.
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ yc)
    intercept = float(y.mean() - mean @ w)
    return w, intercept


def ridge_predict(x, w, intercept):
    return x.reshape(len(x), -1).astype(np.float64) @ w + intercept


def save_predictions(path, data, split_slice, actual, predicted, dataset, model, seed):
    out = pd.DataFrame({
        "Sequence": data.Sequence.iloc[split_slice].to_numpy(),
        "Actual_Delta_G": actual,
        "Predicted_Delta_G": predicted,
    })
    out["Residual"] = out.Predicted_Delta_G - out.Actual_Delta_G
    out["Abs_Error"] = out.Residual.abs()
    out["Dataset"] = dataset; out["Model"] = model; out["Seed"] = seed
    out.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/htsk.csv")
    parser.add_argument("--out", default="outputs/current_logic_baseline_rerun")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", nargs="+", default=["ridge", "mlp", "cnn"])
    parser.add_argument("--datasets", nargs="+", default=["primary", "full"])
    args = parser.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data)
    raw.Sequence = raw.Sequence.astype(str).str.strip().str.upper()
    primary = raw.loc[raw.Delta_G.between(-16, -12)].copy()
    datasets = {"primary_-16_to_-12": primary, "full": raw}
    selected = ["primary_-16_to_-12" if d == "primary" else d for d in args.datasets]
    all_metrics, all_tuning = [], []

    for dataset_name in selected:
        df = datasets[dataset_name]
        for seed in SEEDS:
            print(f"\n=== {dataset_name} | seed {seed} ===", flush=True)
            data, x, y, ys, sp, ymean, ystd = prepare(df, seed)
            tr, va, te = sp["train"], sp["validation"], sp["test"]
            if "ridge" in args.models:
                best = None
                for alpha in RIDGE_ALPHAS:
                    w, b = ridge_fit(x[tr], ys[tr], alpha)
                    pv = ridge_predict(x[va], w, b) * ystd + ymean
                    vm = metrics(y[va], pv)
                    all_tuning.append({"Dataset": dataset_name, "Seed": seed, "Alpha": alpha, **{f"Validation_{k}": v for k, v in vm.items()}})
                    if best is None or vm["MSE"] < best[0]: best = (vm["MSE"], alpha)
                alpha = best[1]
                w, b = ridge_fit(x[tr], ys[tr], alpha)
                pred = ridge_predict(x[te], w, b) * ystd + ymean
                met = metrics(y[te], pred); met.update({"Dataset": dataset_name, "Model": "Ridge", "Seed": seed, "Selected_alpha": alpha})
                all_metrics.append(met)
                np.savez(outdir / f"{dataset_name}_ridge_seed{seed}.npz", weights=w, intercept=b, alpha=alpha, target_mean=ymean, target_std=ystd)
                save_predictions(outdir / f"{dataset_name}_ridge_seed{seed}_test_predictions.csv", data, te, y[te], pred, dataset_name, "Ridge", seed)
                print("  Ridge", met, flush=True)

            for model_name, cls in [("MLP", MLP), ("CNN", CNN)]:
                if model_name.lower() not in args.models: continue
                start = time.time()
                model = cls(seed)
                hist = train_nn(model, x[tr], ys[tr], x[va], ys[va], args.epochs, args.batch_size, seed)
                hist["Dataset"] = dataset_name; hist["Model"] = model_name; hist["Seed"] = seed
                hist.to_csv(outdir / f"{dataset_name}_{model_name.lower()}_seed{seed}_history.csv", index=False)
                pred = predict_batches(model, x[te]) * ystd + ymean
                met = metrics(y[te], pred); met.update({"Dataset": dataset_name, "Model": model_name, "Seed": seed, "Selected_alpha": np.nan})
                all_metrics.append(met)
                np.savez(outdir / f"{dataset_name}_{model_name.lower()}_seed{seed}.npz", **model.p, target_mean=ymean, target_std=ystd)
                save_predictions(outdir / f"{dataset_name}_{model_name.lower()}_seed{seed}_test_predictions.csv", data, te, y[te], pred, dataset_name, model_name, seed)
                print(f"  {model_name} {met} ({time.time()-start:.1f}s)", flush=True)

            pd.DataFrame(all_metrics).to_csv(outdir / "per_seed_metrics_partial.csv", index=False)

    per_seed = pd.DataFrame(all_metrics)
    per_seed.to_csv(outdir / "per_seed_metrics.csv", index=False)
    pd.DataFrame(all_tuning).to_csv(outdir / "ridge_validation_tuning.csv", index=False)
    numeric = ["MAE", "MSE", "RMSE", "R2", "Pearson", "Spearman"]
    summary = per_seed.groupby(["Dataset", "Model"])[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(outdir / "three_seed_summary.csv", index=False)
    with (outdir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    print("\nFINAL SUMMARY\n", summary, flush=True)


if __name__ == "__main__":
    main()
