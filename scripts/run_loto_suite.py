
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokamak_tauE_baselines.config import load_config
from tokamak_tauE_baselines.data import PreparedData, SplitFrames, load_dataframe, prepare_data
from tokamak_tauE_baselines.io_utils import save_dataframe, save_json
from tokamak_tauE_baselines.loto import group_train_val_split, iter_loto_folds
from tokamak_tauE_baselines.metrics import regression_metrics_from_log
from tokamak_tauE_baselines.models.kan_wrapper import predict_kan, train_kan, try_save_kan_state
from tokamak_tauE_baselines.models.mlp import predict_mlp, train_mlp
from tokamak_tauE_baselines.models.ols import fit_ols
from tokamak_tauE_baselines.search import sample_trials
from tokamak_tauE_baselines.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/loto.yaml")
    parser.add_argument("--models", nargs="+", default=["ols", "mlp", "kan"], choices=["ols", "mlp", "kan"])
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Optional cap on the number of held-out tokamaks for debugging.")
    return parser.parse_args()


def make_loto_run_dir(output_root: Path) -> Path:
    from datetime import datetime
    run_dir = output_root / "loto" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_model_dir(base_run_dir: Path, model_name: str) -> Path:
    out = base_run_dir / model_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepare_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict,
) -> PreparedData:
    return prepare_data(
        split_frames=SplitFrames(
            train_df=train_df.reset_index(drop=True),
            val_df=val_df.reset_index(drop=True),
            test_df=test_df.reset_index(drop=True),
        ),
        features=cfg["features"],
        target=cfg["target"],
        metadata_cols=cfg["metadata_cols"],
        log_inputs=bool(cfg["preprocessing"]["log_inputs"]),
        log_target=bool(cfg["preprocessing"]["log_target"]),
        scale_x=bool(cfg["preprocessing"]["scale_x_for_nn"]),
        scale_y=bool(cfg["preprocessing"]["scale_y_for_nn"]),
    )


def fit_eval_ols(train_dev_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict):
    features = cfg["features"]
    target = cfg["target"]
    X_train = np.log(train_dev_df[features].astype(float))
    y_train = np.log(train_dev_df[target].astype(float)).to_numpy()
    X_test = np.log(test_df[features].astype(float))
    y_true_log = np.log(test_df[target].astype(float)).to_numpy()

    ols_result = fit_ols(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        add_constant=bool(cfg["ols"]["add_constant"]),
    )
    y_pred_log = ols_result.predictions
    metrics = regression_metrics_from_log(y_true_log, y_pred_log)

    preds = test_df[cfg["metadata_cols"]].copy()
    preds["y_true_log"] = y_true_log
    preds["y_pred_log"] = y_pred_log
    preds["y_true"] = np.exp(y_true_log)
    preds["y_pred"] = np.exp(y_pred_log)
    return metrics, preds, ols_result


def tune_eval_mlp(train_dev_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict, fold_seed: int):
    train_df, val_df = group_train_val_split(
        train_dev_df=train_dev_df,
        group_col=cfg["group_col"],
        val_size=float(cfg["split"]["val_size"]),
        seed=fold_seed,
    )
    prepared = prepare_split(train_df, val_df, val_df.copy(), cfg)
    trial_space = sample_trials(
        space=cfg["mlp"]["search_space"],
        max_trials=int(cfg["mlp"]["max_trials"]),
        seed=fold_seed,
    )
    train_cfg = cfg["mlp"]["train"]

    best = None
    trial_rows: List[Dict] = []
    for trial_id, params in enumerate(trial_space, start=1):
        output = train_mlp(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
            hidden_dims=params["hidden_dims"],
            activation=params["activation"],
            dropout=float(params["dropout"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            batch_size=int(params["batch_size"]),
            max_epochs=int(train_cfg["max_epochs"]),
            patience=int(train_cfg["patience"]),
            min_delta=float(train_cfg["min_delta"]),
            scheduler_factor=float(train_cfg["scheduler_factor"]),
            scheduler_patience=int(train_cfg["scheduler_patience"]),
            min_lr=float(train_cfg["min_lr"]),
            device=str(train_cfg["device"]),
        )
        val_pred_scaled = predict_mlp(output.model, prepared.X_val, device=str(train_cfg["device"]))
        val_true_log = prepared.inverse_transform_y(prepared.y_val)
        val_pred_log = prepared.inverse_transform_y(val_pred_scaled)
        metrics = regression_metrics_from_log(val_true_log, val_pred_log)
        row = {"trial_id": trial_id, **params, **metrics}
        trial_rows.append(row)
        if best is None or metrics["rmse_log"] < best["rmse_log"]:
            best = row

    refit_train_df, refit_val_df = group_train_val_split(
        train_dev_df=train_dev_df,
        group_col=cfg["group_col"],
        val_size=float(cfg["split"]["val_size"]),
        seed=fold_seed + 1000,
    )
    refit_prepared = prepare_split(refit_train_df, refit_val_df, test_df, cfg)

    final_output = train_mlp(
        X_train=refit_prepared.X_train,
        y_train=refit_prepared.y_train,
        X_val=refit_prepared.X_val,
        y_val=refit_prepared.y_val,
        hidden_dims=best["hidden_dims"],
        activation=best["activation"],
        dropout=float(best["dropout"]),
        lr=float(best["lr"]),
        weight_decay=float(best["weight_decay"]),
        batch_size=int(best["batch_size"]),
        max_epochs=int(train_cfg["max_epochs"]),
        patience=int(train_cfg["patience"]),
        min_delta=float(train_cfg["min_delta"]),
        scheduler_factor=float(train_cfg["scheduler_factor"]),
        scheduler_patience=int(train_cfg["scheduler_patience"]),
        min_lr=float(train_cfg["min_lr"]),
        device=str(train_cfg["device"]),
    )
    test_pred_scaled = predict_mlp(final_output.model, refit_prepared.X_test, device=str(train_cfg["device"]))
    y_true_log = refit_prepared.inverse_transform_y(refit_prepared.y_test)
    y_pred_log = refit_prepared.inverse_transform_y(test_pred_scaled)
    test_metrics = regression_metrics_from_log(y_true_log, y_pred_log)

    preds = refit_prepared.metadata_test.copy()
    preds["y_true_log"] = y_true_log
    preds["y_pred_log"] = y_pred_log
    preds["y_true"] = np.exp(y_true_log)
    preds["y_pred"] = np.exp(y_pred_log)

    return test_metrics, preds, pd.DataFrame(trial_rows), best, final_output


def tune_eval_kan(train_dev_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict, fold_seed: int):
    train_df, val_df = group_train_val_split(
        train_dev_df=train_dev_df,
        group_col=cfg["group_col"],
        val_size=float(cfg["split"]["val_size"]),
        seed=fold_seed,
    )
    prepared = prepare_split(train_df, val_df, val_df.copy(), cfg)
    trial_space = sample_trials(
        space=cfg["kan"]["search_space"],
        max_trials=int(cfg["kan"]["max_trials"]),
        seed=fold_seed,
    )
    fit_cfg = cfg["kan"]["fit"]

    best = None
    trial_rows: List[Dict] = []
    for trial_id, params in enumerate(trial_space, start=1):
        output = train_kan(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_val=prepared.X_val,
            y_val=prepared.y_val,
            hidden_dims=params["hidden_dims"],
            grid=int(params["grid"]),
            k=int(params["k"]),
            adam_steps=int(params["adam_steps"]),
            adam_lr=float(params["adam_lr"]),
            lbfgs_steps=int(params["lbfgs_steps"]),
            lamb=float(params["lamb"]),
            lamb_entropy=float(params["lamb_entropy"]),
            update_grid=bool(fit_cfg["update_grid"]),
            grid_update_num=int(fit_cfg["grid_update_num"]),
            start_grid_update_step=int(fit_cfg["start_grid_update_step"]),
            stop_grid_update_step=int(fit_cfg["stop_grid_update_step"]),
            batch=int(fit_cfg["batch"]),
            seed=fold_seed,
            device=str(fit_cfg["device"]),
        )
        val_pred_scaled = predict_kan(output.model, prepared.X_val, device=str(fit_cfg["device"]))
        val_true_log = prepared.inverse_transform_y(prepared.y_val)
        val_pred_log = prepared.inverse_transform_y(val_pred_scaled)
        metrics = regression_metrics_from_log(val_true_log, val_pred_log)
        row = {"trial_id": trial_id, **params, **metrics}
        trial_rows.append(row)
        if best is None or metrics["rmse_log"] < best["rmse_log"]:
            best = row

    refit_train_df, refit_val_df = group_train_val_split(
        train_dev_df=train_dev_df,
        group_col=cfg["group_col"],
        val_size=float(cfg["split"]["val_size"]),
        seed=fold_seed + 1000,
    )
    refit_prepared = prepare_split(refit_train_df, refit_val_df, test_df, cfg)

    final_output = train_kan(
        X_train=refit_prepared.X_train,
        y_train=refit_prepared.y_train,
        X_val=refit_prepared.X_val,
        y_val=refit_prepared.y_val,
        hidden_dims=best["hidden_dims"],
        grid=int(best["grid"]),
        k=int(best["k"]),
        adam_steps=int(best["adam_steps"]),
        adam_lr=float(best["adam_lr"]),
        lbfgs_steps=int(best["lbfgs_steps"]),
        lamb=float(best["lamb"]),
        lamb_entropy=float(best["lamb_entropy"]),
        update_grid=bool(fit_cfg["update_grid"]),
        grid_update_num=int(fit_cfg["grid_update_num"]),
        start_grid_update_step=int(fit_cfg["start_grid_update_step"]),
        stop_grid_update_step=int(fit_cfg["stop_grid_update_step"]),
        batch=int(fit_cfg["batch"]),
        seed=fold_seed,
        device=str(fit_cfg["device"]),
    )
    test_pred_scaled = predict_kan(final_output.model, refit_prepared.X_test, device=str(fit_cfg["device"]))
    y_true_log = refit_prepared.inverse_transform_y(refit_prepared.y_test)
    y_pred_log = refit_prepared.inverse_transform_y(test_pred_scaled)
    test_metrics = regression_metrics_from_log(y_true_log, y_pred_log)

    preds = refit_prepared.metadata_test.copy()
    preds["y_true_log"] = y_true_log
    preds["y_pred_log"] = y_pred_log
    preds["y_true"] = np.exp(y_true_log)
    preds["y_pred"] = np.exp(y_pred_log)

    return test_metrics, preds, pd.DataFrame(trial_rows), best, final_output


def summarize_model(pred_df: pd.DataFrame, fold_metrics_df: pd.DataFrame) -> Dict:
    overall = regression_metrics_from_log(
        pred_df["y_true_log"].to_numpy(),
        pred_df["y_pred_log"].to_numpy(),
    )
    out = {"overall": overall}
    metric_cols = [c for c in fold_metrics_df.columns if c not in ["held_out_tok", "n_test"]]
    out["fold_mean"] = fold_metrics_df[metric_cols].mean(numeric_only=True).to_dict()
    out["fold_std"] = fold_metrics_df[metric_cols].std(numeric_only=True).to_dict()
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    df = load_dataframe(
        csv_path=ROOT / cfg["data_path"],
        features=cfg["features"],
        target=cfg["target"],
        metadata_cols=cfg["metadata_cols"],
    )

    base_run_dir = make_loto_run_dir(ROOT / cfg["output_root"])
    fold_iter = list(iter_loto_folds(df=df, group_col=cfg["group_col"]))
    if args.max_folds is not None:
        fold_iter = fold_iter[: args.max_folds]

    for model_name in args.models:
        model_dir = make_model_dir(base_run_dir, model_name)
        all_preds = []
        fold_rows = []

        for fold_id, fold in enumerate(fold_iter, start=1):
            fold_seed = int(cfg["seed"]) + fold_id
            fold_name = f"fold_{fold_id:02d}_{fold.held_out_group.replace('/', '_')}"
            fold_dir = model_dir / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)

            if model_name == "ols":
                metrics, preds, ols_result = fit_eval_ols(fold.train_dev_df, fold.test_df, cfg)
                save_dataframe(ols_result.coefficient_frame(), fold_dir / "ols_coefficients.csv")
                best_params = {"model": "ols", "note": "No hyperparameter search"}
            elif model_name == "mlp":
                metrics, preds, trial_df, best_params, final_output = tune_eval_mlp(
                    fold.train_dev_df, fold.test_df, cfg, fold_seed
                )
                save_dataframe(trial_df.sort_values("rmse_log"), fold_dir / "trial_results.csv")
                import torch
                torch.save(final_output.model.state_dict(), fold_dir / "model.pt")
                save_dataframe(pd.DataFrame(final_output.history), fold_dir / "training_history.csv")
            elif model_name == "kan":
                metrics, preds, trial_df, best_params, final_output = tune_eval_kan(
                    fold.train_dev_df, fold.test_df, cfg, fold_seed
                )
                save_dataframe(trial_df.sort_values("rmse_log"), fold_dir / "trial_results.csv")
                save_json(final_output.history, fold_dir / "training_history.json")
                try_save_kan_state(final_output.model, str(fold_dir / "model.pt"))
            else:
                raise ValueError(model_name)

            preds["held_out_tok"] = fold.held_out_group
            save_dataframe(preds, fold_dir / "predictions.csv")
            save_json(metrics, fold_dir / "metrics.json")
            save_json(best_params, fold_dir / "best_params.json")

            all_preds.append(preds)
            fold_rows.append({
                "held_out_tok": fold.held_out_group,
                "n_test": len(preds),
                **metrics,
            })

            print(f"[{model_name.upper()}][{fold_name}] rmse_log={metrics['rmse_log']:.4f} r2_log={metrics['r2_log']:.4f}")

        pred_df = pd.concat(all_preds, axis=0).reset_index(drop=True)
        fold_metrics_df = pd.DataFrame(fold_rows).sort_values("held_out_tok").reset_index(drop=True)

        save_dataframe(pred_df, model_dir / "all_predictions.csv")
        save_dataframe(fold_metrics_df, model_dir / "fold_metrics.csv")
        save_json(summarize_model(pred_df, fold_metrics_df), model_dir / "summary.json")

    print(f"LOTO run finished. Results saved to: {base_run_dir}")


if __name__ == "__main__":
    main()
