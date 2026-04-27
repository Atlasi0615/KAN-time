from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import sympy as sp
import torch
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokamak_tauE_baselines.data import load_dataframe
from tokamak_tauE_baselines.metrics import regression_metrics_from_log
from tokamak_tauE_baselines.splits import build_split, refit_train_val_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/final_baseline.yaml")
    parser.add_argument("--split-type", type=str, choices=["random", "group", "extrap_jet"], default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sparsify-steps", type=int, default=40)
    parser.add_argument("--sparsify-lamb", type=float, default=1e-3)
    parser.add_argument("--sparsify-lamb-entropy", type=float, default=2.0)
    parser.add_argument("--attempt-symbolic", action="store_true")
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
    raise ValueError("Cannot infer split type; pass --split-type.")


def prepare_like_training(cfg: Dict, split_type: str):
    df = load_dataframe(csv_path=ROOT / cfg["data_path"], features=cfg["features"], target=cfg["target"], metadata_cols=cfg["metadata_cols"])
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
    train_df, val_df = refit_train_val_split(
        train_val_df=train_val_df,
        split_type=split_type,
        group_col=cfg["group_col"],
        target_col=cfg["target"],
        seed=int(cfg["seed"]),
    )
    X_train_log = np.log(train_df[cfg["features"]].astype(float).to_numpy())
    y_train_log = np.log(train_df[cfg["target"]].astype(float).to_numpy()).reshape(-1, 1)
    X_val_log = np.log(val_df[cfg["features"]].astype(float).to_numpy())
    y_val_log = np.log(val_df[cfg["target"]].astype(float).to_numpy()).reshape(-1, 1)
    holdout_test_df = split_frames.test_df.reset_index(drop=True)
    X_holdout_log = np.log(holdout_test_df[cfg["features"]].astype(float).to_numpy())
    y_holdout_log = np.log(holdout_test_df[cfg["target"]].astype(float).to_numpy()).reshape(-1, 1)
    x_scaler = StandardScaler().fit(X_train_log)
    y_scaler = StandardScaler().fit(y_train_log)
    return {
        "features": cfg["features"],
        "target": cfg["target"],
        "train_input": torch.tensor(x_scaler.transform(X_train_log), dtype=torch.float32),
        "train_label": torch.tensor(y_scaler.transform(y_train_log), dtype=torch.float32),
        "refit_val_input": torch.tensor(x_scaler.transform(X_val_log), dtype=torch.float32),
        "refit_val_label": torch.tensor(y_scaler.transform(y_val_log), dtype=torch.float32),
        "holdout_test_df": holdout_test_df,
        "holdout_test_log": X_holdout_log,
        "holdout_test_true_log": y_holdout_log[:, 0],
        "holdout_test_scaled": x_scaler.transform(X_holdout_log),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def load_best_params(run_dir: Path) -> Dict:
    with (run_dir / "best_params.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def build_kan_from_params(best: Dict, input_dim: int, device: str):
    from kan import KAN
    return KAN(width=[input_dim] + list(best["hidden_dims"]) + [1], grid=int(best["grid"]), k=int(best["k"]), seed=0, device=device)


def load_kan_model(run_dir: Path, best: Dict, input_dim: int, device: str):
    model = build_kan_from_params(best, input_dim=input_dim, device=device)
    model_path = run_dir / "model.pt"
    if hasattr(model, "loadckpt"):
        try:
            model.loadckpt(str(model_path))
            return model
        except Exception:
            pass
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def safe_plot(model, folder: str, mask: bool = False, title: str | None = None, in_vars=None, out_vars=None):
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        if mask:
            return model.plot(folder=folder, mask=True, in_vars=in_vars, out_vars=out_vars, title=title)
        return model.plot(folder=folder, in_vars=in_vars, out_vars=out_vars, title=title)
    except TypeError:
        return model.plot(folder=folder, in_vars=in_vars, out_vars=out_vars, title=title)


def forward_scaled(model, x_scaled: np.ndarray, device: str) -> np.ndarray:
    xt = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(xt)
        if isinstance(out, tuple):
            out = out[0]
        out = out.detach().cpu().numpy()
    out = np.asarray(out)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out[:, 0]
    return out.reshape(-1)


def predict_log_tau(model, x_scaled: np.ndarray, y_scaler: StandardScaler, device: str) -> np.ndarray:
    y_scaled = forward_scaled(model, x_scaled, device=device).reshape(-1, 1)
    return y_scaler.inverse_transform(y_scaled)[:, 0]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def extract_symbolic_expression(formula_obj):
    expr = formula_obj[0] if isinstance(formula_obj, tuple) else formula_obj
    while isinstance(expr, (list, tuple)) and len(expr) == 1:
        expr = expr[0]
    return sp.sympify(expr)


def evaluate_symbolic_expression(expr, symbols, values: np.ndarray) -> np.ndarray:
    fn = sp.lambdify(symbols, expr, modules="numpy")
    cols = [values[:, i] for i in range(values.shape[1])]
    pred = fn(*cols)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 0:
        pred = np.full(values.shape[0], float(pred))
    return pred.reshape(-1)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_yaml(ROOT / args.config)
    split_type = args.split_type or infer_split_type(run_dir)
    prep = prepare_like_training(cfg, split_type)
    best = load_best_params(run_dir)
    model = load_kan_model(run_dir, best, input_dim=len(prep["features"]), device=args.device)
    out_dir = run_dir / "kan_specific"
    out_dir.mkdir(parents=True, exist_ok=True)

    _ = model(prep["train_input"].to(args.device))
    safe_plot(model, folder=str(out_dir / "original_plot"), in_vars=prep["features"], out_vars=[cfg["target"]], title="Original trained KAN")

    summary = {
        "n_edge_before": int(getattr(model, "n_edge", -1)),
        "n_sum_before": str(getattr(model, "n_sum", "NA")),
        "n_mult_before": str(getattr(model, "n_mult", "NA")),
        "best_params": best,
    }

    sparse_model = load_kan_model(run_dir, best, input_dim=len(prep["features"]), device=args.device)
    dataset = {
        "train_input": prep["train_input"].to(args.device),
        "train_label": prep["train_label"].to(args.device),
        "test_input": prep["refit_val_input"].to(args.device),
        "test_label": prep["refit_val_label"].to(args.device),
    }

    sparse_history = None
    try:
        sparse_history = sparse_model.fit(dataset, opt="LBFGS", steps=int(args.sparsify_steps), lamb=float(args.sparsify_lamb), lamb_entropy=float(args.sparsify_lamb_entropy), update_grid=False, log=max(1, int(args.sparsify_steps) + 1))
    except Exception as e:
        summary["sparsify_error"] = str(e)

    _ = sparse_model(prep["train_input"].to(args.device))
    safe_plot(sparse_model, folder=str(out_dir / "sparse_plot"), in_vars=prep["features"], out_vars=[cfg["target"]], title=f"Sparsified KAN (lamb={args.sparsify_lamb})")

    pruned_for_mask = sparse_model
    try:
        maybe_model = pruned_for_mask.prune()
        if maybe_model is not None:
            pruned_for_mask = maybe_model
        _ = pruned_for_mask(prep["train_input"].to(args.device))
        safe_plot(pruned_for_mask, folder=str(out_dir / "pruned_plot"), mask=True, in_vars=prep["features"], out_vars=[cfg["target"]], title="Pruned KAN")
        summary["n_edge_after_prune"] = int(getattr(pruned_for_mask, "n_edge", -1))
        summary["n_sum_after_prune"] = str(getattr(pruned_for_mask, "n_sum", "NA"))
        summary["n_mult_after_prune"] = str(getattr(pruned_for_mask, "n_mult", "NA"))
    except Exception as e:
        summary["prune_error"] = str(e)

    if args.attempt_symbolic:
        try:
            lib = ["x", "x^2", "x^3", "sqrt", "log", "exp", "tanh", "sin", "abs"]
            pruned_for_mask.auto_symbolic(lib=lib)
            var = [sp.Symbol(f"log_{f}") for f in prep["features"]]
            formula = pruned_for_mask.symbolic_formula(var=var)
            expr = extract_symbolic_expression(formula)
            summary["symbolic_formula_repr"] = str(formula)
            (out_dir / "symbolic_formula.txt").write_text(str(formula), encoding="utf-8")

            y_true_log = prep["holdout_test_true_log"]
            y_pred_log_numeric = predict_log_tau(model, prep["holdout_test_scaled"], prep["y_scaler"], device=args.device)
            numeric_metrics = regression_metrics_from_log(y_true_log, y_pred_log_numeric)
            summary["rmse_log_numeric_test"] = float(numeric_metrics["rmse_log"])

            candidates = {}
            for name, candidate_inputs in {
                "raw_log": prep["holdout_test_log"],
                "scaled_input": prep["holdout_test_scaled"],
            }.items():
                try:
                    pred_symbolic = evaluate_symbolic_expression(expr, var, candidate_inputs)
                    candidates[name] = {
                        "rmse_log_symbolic": rmse(y_true_log, pred_symbolic),
                        "rmse_log_symbolic_vs_numeric": rmse(y_pred_log_numeric, pred_symbolic),
                        "predictions": pred_symbolic,
                    }
                except Exception as candidate_error:
                    candidates[name] = {"error": str(candidate_error)}

            summary["symbolic_candidate_metrics"] = {
                name: {
                    key: float(value)
                    for key, value in metrics.items()
                    if key != "predictions"
                }
                for name, metrics in candidates.items()
                if "error" not in metrics
            }
            for name, metrics in candidates.items():
                if "error" in metrics:
                    summary.setdefault("symbolic_candidate_errors", {})[name] = metrics["error"]

            valid_candidates = {name: metrics for name, metrics in candidates.items() if "predictions" in metrics}
            if not valid_candidates:
                raise RuntimeError("Failed to evaluate the symbolic expression in both raw_log and scaled_input spaces.")

            chosen_name, chosen_metrics = min(
                valid_candidates.items(),
                key=lambda item: item[1]["rmse_log_symbolic_vs_numeric"],
            )
            summary["symbolic_input_space"] = chosen_name
            summary["rmse_log_symbolic"] = float(chosen_metrics["rmse_log_symbolic"])
            summary["rmse_log_symbolic_vs_numeric"] = float(chosen_metrics["rmse_log_symbolic_vs_numeric"])

            audit_df = prep["holdout_test_df"][cfg["metadata_cols"]].copy()
            audit_df["y_true_log"] = y_true_log
            audit_df["y_pred_log_numeric"] = y_pred_log_numeric
            audit_df["y_pred_log_symbolic"] = chosen_metrics["predictions"]
            audit_df["abs_diff_symbolic_vs_numeric"] = np.abs(
                audit_df["y_pred_log_symbolic"] - audit_df["y_pred_log_numeric"]
            )
            audit_df.to_csv(out_dir / "symbolic_test_audit.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            summary["symbolic_error"] = str(e)

    if sparse_history is not None:
        (out_dir / "sparsify_history.txt").write_text(str(sparse_history), encoding="utf-8")

    with (out_dir / "kan_specific_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved KAN-specific outputs to: {out_dir}")


if __name__ == "__main__":
    main()
