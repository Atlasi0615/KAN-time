from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokamak_tauE_baselines.config import load_config
from tokamak_tauE_baselines.data import combine_train_val, load_dataframe, prepare_data
from tokamak_tauE_baselines.io_utils import make_run_dir, save_dataframe, save_json
from tokamak_tauE_baselines.metrics import regression_metrics_from_log
from tokamak_tauE_baselines.models.kan_wrapper import predict_kan, train_kan, try_save_kan_state
from tokamak_tauE_baselines.search import sample_trials
from tokamak_tauE_baselines.seed import set_seed
from tokamak_tauE_baselines.splits import SplitFrames, build_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split-type", type=str, choices=["random", "group"], default="random")
    return parser.parse_args()


def refit_split(
    train_val_df: pd.DataFrame,
    split_type: str,
    group_col: str,
    seed: int,
) -> SplitFrames:
    if split_type == "random":
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.10,
            random_state=seed,
            shuffle=True,
        )
    else:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
        groups = train_val_df[group_col].values
        train_idx, val_idx = next(gss.split(train_val_df, groups=groups))
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
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
        space=cfg["kan"]["search_space"],
        max_trials=int(cfg["kan"]["max_trials"]),
        seed=int(cfg["seed"]),
    )

    fit_cfg = cfg["kan"]["fit"]
    run_dir = make_run_dir(ROOT / cfg["output_root"], "kan", args.split_type)

    trial_rows = []
    best = None

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
            seed=int(cfg["seed"]),
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

        print(f"[KAN trial {trial_id}/{len(trial_space)}] val_rmse_log={metrics['rmse_log']:.6f} params={params}")

    trial_df = pd.DataFrame(trial_rows).sort_values("rmse_log", ascending=True).reset_index(drop=True)
    save_dataframe(trial_df, run_dir / "trial_results.csv")
    save_json(best, run_dir / "best_params.json")

    train_val_df = combine_train_val(split_frames)
    refit_frames = refit_split(
        train_val_df=train_val_df,
        split_type=args.split_type,
        group_col=cfg["group_col"],
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
        seed=int(cfg["seed"]),
        device=str(fit_cfg["device"]),
    )

    test_pred_scaled = predict_kan(final_output.model, refit_prepared.X_test, device=str(fit_cfg["device"]))
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
    save_json(final_output.history, run_dir / "training_history.json")
    try_save_kan_state(final_output.model, str(run_dir / "model.pt"))

    print(f"[KAN] finished. results saved to: {run_dir}")


if __name__ == "__main__":
    main()
