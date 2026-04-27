from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokamak_tauE_baselines.data import load_dataframe
from tokamak_tauE_baselines.splits import build_split, refit_train_val_split
from tokamak_tauE_baselines.models.mlp import MLPRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/final_baseline.yaml")
    parser.add_argument("--split-type", type=str, choices=["random", "group", "extrap_jet"], default=None)
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--grid-points", type=int, default=200)
    parser.add_argument("--sample-points", type=int, default=1500)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_split_type(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if name.endswith("extrap_jet"):
        return "extrap_jet"
    if name.endswith("group"):
        return "group"
    if name.endswith("random"):
        return "random"
    raise ValueError("Cannot infer split type from run-dir name; pass --split-type.")


def refit_split(train_val_df: pd.DataFrame, split_type: str, group_col: str, seed: int):
    del group_col
    del seed
    raise RuntimeError("refit_split should not be called directly; use refit_train_val_split.")


def prepare_like_training(cfg: Dict, split_type: str):
    df = load_dataframe(
        csv_path=ROOT / cfg["data_path"],
        features=cfg["features"],
        target=cfg["target"],
        metadata_cols=cfg["metadata_cols"],
    )
    split_frames = build_split(
        df=df,
        split_type=split_type,
        group_col=cfg["group_col"],
        target_col=cfg["target"],
        test_size=float(cfg["split"]["test_size"]),
        val_size=float(cfg["split"]["val_size"]),
        seed=int(cfg["seed"]),
    )

    train_val_df = pd.concat([split_frames.train_df, split_frames.val_df], axis=0).reset_index(drop=True)
    refit_train_df, refit_val_df = refit_train_val_split(
        train_val_df=train_val_df,
        split_type=split_type,
        group_col=cfg["group_col"],
        target_col=cfg["target"],
        seed=int(cfg["seed"]),
    )
    test_df = split_frames.test_df.reset_index(drop=True)

    features = cfg["features"]
    target = cfg["target"]

    X_train_log = np.log(refit_train_df[features].astype(float).to_numpy())
    y_train_log = np.log(refit_train_df[target].astype(float).to_numpy()).reshape(-1, 1)

    X_test_log = np.log(test_df[features].astype(float).to_numpy())

    x_scaler = StandardScaler().fit(X_train_log)
    y_scaler = StandardScaler().fit(y_train_log)

    X_test_scaled = x_scaler.transform(X_test_log)

    return {
        "features": features,
        "target": target,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_df": refit_train_df,
        "test_df": test_df,
        "X_test_log": X_test_log,
        "X_test_scaled": X_test_scaled,
    }


def summarize_binned(x: np.ndarray, y: np.ndarray, bins: int) -> pd.DataFrame:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    edges = np.quantile(x_sorted, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (x_sorted >= lo) & (x_sorted < hi) if i < bins - 1 else (x_sorted >= lo) & (x_sorted <= hi)
        if mask.sum() == 0:
            continue
        xb = x_sorted[mask]
        yb = y_sorted[mask]
        rows.append({
            "x_mid_log": float(np.median(xb)),
            "x_mid": float(np.exp(np.median(xb))),
            "n": int(mask.sum()),
            "exp_p10": float(np.quantile(yb, 0.10)),
            "exp_p50": float(np.quantile(yb, 0.50)),
            "exp_p90": float(np.quantile(yb, 0.90)),
        })
    return pd.DataFrame(rows)


def make_reference_grid(train_df: pd.DataFrame, features: List[str], feature: str, grid_points: int):
    log_train = np.log(train_df[features].astype(float))
    ref_log = log_train.median(axis=0).to_numpy(dtype=float)
    j = features.index(feature)
    q_lo = np.quantile(log_train[feature].to_numpy(), 0.02)
    q_hi = np.quantile(log_train[feature].to_numpy(), 0.98)
    x_grid = np.linspace(q_lo, q_hi, grid_points)
    X_ref_log = np.tile(ref_log[None, :], (grid_points, 1))
    X_ref_log[:, j] = x_grid
    return X_ref_log, x_grid


def plot_single_feature(
    feature: str,
    x_grid_log: np.ndarray,
    y_grid_log: np.ndarray,
    exp_grid: np.ndarray,
    binned_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    model_name: str,
):
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot(np.exp(x_grid_log), np.exp(y_grid_log), linewidth=2.2, label=f"{model_name} response")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(feature)
    ax.set_ylabel(r"Predicted $\tau_E$")
    ax.set_title(f"Single-variable response: {feature}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{feature}_response_original.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot(x_grid_log, y_grid_log, linewidth=2.2, label=f"{model_name} response")
    ax.set_xlabel(rf"$\log({feature})$")
    ax.set_ylabel(r"$\log(\tau_E)$")
    ax.set_title(f"Single-variable response in log space: {feature}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{feature}_response_log.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot(np.exp(x_grid_log), exp_grid, linewidth=2.2, label="Reference-state local exponent")
    ax.axhline(0.0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xscale("log")
    ax.set_xlabel(feature)
    ax.set_ylabel(rf"$\partial \log(\tau_E) / \partial \log({feature})$")
    ax.set_title(f"Reference-state local exponent: {feature}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{feature}_local_exponent_reference.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    if len(binned_df) > 0:
        ax.plot(binned_df["x_mid"], binned_df["exp_p50"], marker="o", linewidth=2.0, label="Median")
        ax.fill_between(
            binned_df["x_mid"],
            binned_df["exp_p10"],
            binned_df["exp_p90"],
            alpha=0.25,
            label="10–90 percentile",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xscale("log")
    ax.set_xlabel(feature)
    ax.set_ylabel(rf"$\partial \log(\tau_E) / \partial \log({feature})$")
    ax.set_title(f"Test-set binned local exponent: {feature}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{feature}_local_exponent_binned.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_summary_panel(feature: str, out_dir: Path, dpi: int, model_name: str):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    paths = [
        out_dir / f"{feature}_response_original.png",
        out_dir / f"{feature}_response_log.png",
        out_dir / f"{feature}_local_exponent_reference.png",
        out_dir / f"{feature}_local_exponent_binned.png",
    ]
    titles = ["Response (original scale)", "Response (log scale)", "Local exponent (reference state)", "Local exponent (test-set bins)"]
    for ax, path, title in zip(axs.flat, paths, titles):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{model_name} interpretability summary: {feature}", fontsize=20)
    fig.tight_layout()
    fig.savefig(out_dir / f"{feature}_summary_panel.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_best_params(run_dir: Path) -> Dict:
    with (run_dir / "best_params.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mlp_model(run_dir: Path, best: Dict, input_dim: int, device: str):
    model = MLPRegressor(input_dim=input_dim, hidden_dims=best["hidden_dims"], activation=best["activation"], dropout=float(best["dropout"]))
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def forward_scaled(model, x_scaled: np.ndarray, device: str) -> np.ndarray:
    xt = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(xt).detach().cpu().numpy()
    return np.asarray(out).reshape(-1)


def gradients_wrt_scaled_inputs(model, x_scaled: np.ndarray, device: str) -> np.ndarray:
    xt = torch.tensor(x_scaled, dtype=torch.float32, device=device, requires_grad=True)
    out = model(xt).reshape(-1)
    grads = torch.autograd.grad(outputs=out.sum(), inputs=xt, retain_graph=False, create_graph=False)[0]
    return grads.detach().cpu().numpy()


def local_exponents_from_scaled_grads(grads_scaled: np.ndarray, x_scaler: StandardScaler, y_scaler: StandardScaler):
    return grads_scaled * (float(y_scaler.scale_[0]) / np.asarray(x_scaler.scale_, dtype=float))[None, :]


def predict_log_tau(model, x_scaled: np.ndarray, y_scaler: StandardScaler, device: str) -> np.ndarray:
    y_scaled = forward_scaled(model, x_scaled, device=device).reshape(-1, 1)
    return y_scaler.inverse_transform(y_scaled)[:, 0]


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(ROOT / args.config)
    split_type = args.split_type or infer_split_type(run_dir)
    best = load_best_params(run_dir)
    prep = prepare_like_training(cfg, split_type=split_type)
    features = args.features or prep["features"]
    model = load_mlp_model(run_dir, best, input_dim=len(prep["features"]), device=args.device)
    out_dir = run_dir / "interpretability_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test_scaled = prep["X_test_scaled"]
    X_test_log = prep["X_test_log"]
    test_df = prep["test_df"].copy()
    if args.sample_points and len(test_df) > args.sample_points:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(test_df), size=args.sample_points, replace=False))
        X_test_scaled = X_test_scaled[keep]
        X_test_log = X_test_log[keep]
        test_df = test_df.iloc[keep].reset_index(drop=True)

    grads_scaled = gradients_wrt_scaled_inputs(model, X_test_scaled, device=args.device)
    exponents = local_exponents_from_scaled_grads(grads_scaled, prep["x_scaler"], prep["y_scaler"])

    local_exp_df = test_df[cfg["metadata_cols"]].copy()
    for i, feat in enumerate(prep["features"]):
        local_exp_df[f"log_{feat}"] = X_test_log[:, i]
        local_exp_df[f"{feat}_local_exp"] = exponents[:, i]
    local_exp_df.to_csv(out_dir / "local_exponents_test_samples.csv", index=False, encoding="utf-8-sig")

    rows = []
    for feat in features:
        X_grid_log, x_grid_log = make_reference_grid(prep["train_df"], prep["features"], feat, args.grid_points)
        X_grid_scaled = prep["x_scaler"].transform(X_grid_log)
        y_grid_log = predict_log_tau(model, X_grid_scaled, prep["y_scaler"], device=args.device)
        grid_grads_scaled = gradients_wrt_scaled_inputs(model, X_grid_scaled, device=args.device)
        grid_exponents = local_exponents_from_scaled_grads(grid_grads_scaled, prep["x_scaler"], prep["y_scaler"])
        idx = prep["features"].index(feat)
        exp_grid = grid_exponents[:, idx]
        binned_df = summarize_binned(X_test_log[:, idx], exponents[:, idx], args.bins)
        binned_df.to_csv(out_dir / f"{feat}_binned_local_exponent.csv", index=False, encoding="utf-8-sig")
        plot_single_feature(feat, x_grid_log, y_grid_log, exp_grid, binned_df, out_dir, args.dpi, "MLP")
        make_summary_panel(feat, out_dir, args.dpi, "MLP")
        rows.append({
            "feature": feat,
            "ref_exp_min": float(np.min(exp_grid)),
            "ref_exp_median": float(np.median(exp_grid)),
            "ref_exp_max": float(np.max(exp_grid)),
            "test_exp_p10": float(np.quantile(exponents[:, idx], 0.10)),
            "test_exp_p50": float(np.quantile(exponents[:, idx], 0.50)),
            "test_exp_p90": float(np.quantile(exponents[:, idx], 0.90)),
        })
    pd.DataFrame(rows).to_csv(out_dir / "local_exponent_summary.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "analysis_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir), "split_type": split_type, "features_analyzed": features, "best_params": best}, f, ensure_ascii=False, indent=2)
    print(f"Saved MLP interpretability outputs to: {out_dir}")


if __name__ == "__main__":
    main()
