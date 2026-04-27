from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokamak_tauE_baselines.config import load_config
from tokamak_tauE_baselines.data import combine_train_val, load_dataframe, prepare_data
from tokamak_tauE_baselines.io_utils import make_run_dir, save_dataframe, save_json
from tokamak_tauE_baselines.metrics import regression_metrics_from_log
from tokamak_tauE_baselines.models.mlp import predict_mlp, train_mlp
from tokamak_tauE_baselines.search import sample_trials
from tokamak_tauE_baselines.seed import set_seed
from tokamak_tauE_baselines.splits import SplitFrames, build_split, refit_train_val_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split-type", type=str, choices=["random", "group", "extrap_jet"], default="random")
    return parser.parse_args()


def refit_split(
    train_val_df: pd.DataFrame,
    split_type: str,
    group_col: str,
    target_col: str,
    seed: int,
) -> SplitFrames:
    train_df, val_df = refit_train_val_split(
        train_val_df=train_val_df,
        split_type=split_type,
        group_col=group_col,
        target_col=target_col,
        seed=seed,
    )
    return SplitFrames(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=train_val_df.iloc[:0].copy().reset_index(drop=True),
    )


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
    split_frames = build_split(
        df=df,
        split_type=args.split_type,
        group_col=cfg["group_col"],
        target_col=cfg["target"],
        test_size=float(cfg["split"]["test_size"]),
        val_size=float(cfg["split"]["val_size"]),
        seed=int(cfg["seed"]),
    )

    prepared = prepare_data(
        split_frames=split_frames,
        features=cfg["features"],
        target=cfg["target"],
        metadata_cols=cfg["metadata_cols"],
        log_inputs=bool(cfg["preprocessing"]["log_inputs"]),
        log_target=bool(cfg["preprocessing"]["log_target"]),
        scale_x=bool(cfg["preprocessing"]["scale_x_for_nn"]),
        scale_y=bool(cfg["preprocessing"]["scale_y_for_nn"]),
    )

    trial_space = sample_trials(
        space=cfg["mlp"]["search_space"],
        max_trials=int(cfg["mlp"]["max_trials"]),
        seed=int(cfg["seed"]),
    )

    run_dir = make_run_dir(ROOT / cfg["output_root"], "mlp", args.split_type)

    trial_rows = []
    best = None
    best_model = None

    train_cfg = cfg["mlp"]["train"]

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
            best_model = output.model

        print(f"[MLP trial {trial_id}/{len(trial_space)}] val_rmse_log={metrics['rmse_log']:.6f} params={params}")

    trial_df = pd.DataFrame(trial_rows).sort_values("rmse_log", ascending=True).reset_index(drop=True)
    save_dataframe(trial_df, run_dir / "trial_results.csv")
    save_json(best, run_dir / "best_params.json")

    # Final refit on train+val with a new internal validation split
    train_val_df = combine_train_val(split_frames)
    refit_frames = refit_split(
        train_val_df=train_val_df,
        split_type=args.split_type,
        group_col=cfg["group_col"],
        target_col=cfg["target"],
        seed=int(cfg["seed"]),
    )
    refit_prepared = prepare_data(
        split_frames=SplitFrames(
            train_df=refit_frames.train_df,
            val_df=refit_frames.val_df,
            test_df=split_frames.test_df,
        ),
        features=cfg["features"],
        target=cfg["target"],
        metadata_cols=cfg["metadata_cols"],
        log_inputs=bool(cfg["preprocessing"]["log_inputs"]),
        log_target=bool(cfg["preprocessing"]["log_target"]),
        scale_x=bool(cfg["preprocessing"]["scale_x_for_nn"]),
        scale_y=bool(cfg["preprocessing"]["scale_y_for_nn"]),
    )

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
    preds["y_true"] = preds["y_true_log"].map(lambda x: __import__("math").exp(float(x)))
    preds["y_pred"] = preds["y_pred_log"].map(lambda x: __import__("math").exp(float(x)))

    save_dataframe(preds, run_dir / "predictions.csv")
    save_json(test_metrics, run_dir / "metrics.json")
    torch.save(final_output.model.state_dict(), run_dir / "model.pt")
    save_dataframe(pd.DataFrame(final_output.history), run_dir / "training_history.csv")

    print(f"[MLP] finished. results saved to: {run_dir}")


if __name__ == "__main__":
    main()
