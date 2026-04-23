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
from tokamak_tauE_baselines.data import combine_train_val, load_dataframe
from tokamak_tauE_baselines.io_utils import make_run_dir, save_dataframe, save_json
from tokamak_tauE_baselines.metrics import regression_metrics_from_log
from tokamak_tauE_baselines.models.ols import fit_ols
from tokamak_tauE_baselines.seed import set_seed
from tokamak_tauE_baselines.splits import build_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split-type", type=str, choices=["random", "group"], default="random")
    return parser.parse_args()


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

    feature_cols = cfg["features"]
    log_feature_cols = [f"log_{c}" for c in feature_cols]
    target_col = cfg["target"]

    train_val_df = combine_train_val(split_frames).copy()
    test_df = split_frames.test_df.copy()

    for frame in [train_val_df, test_df]:
        for col in feature_cols:
            frame[f"log_{col}"] = frame[col].map(lambda x: __import__("math").log(float(x)))
        frame["log_target"] = frame[target_col].map(lambda x: __import__("math").log(float(x)))

    # OLS only uses train split; validation is kept for bookkeeping consistency.
    ols_result = fit_ols(
        X_train=train_val_df[log_feature_cols],
        y_train=train_val_df["log_target"].to_numpy(),
        X_test=test_df[log_feature_cols],
        add_constant=bool(cfg["ols"]["add_constant"]),
    )

    metrics = regression_metrics_from_log(
        y_true_log=test_df["log_target"].to_numpy(),
        y_pred_log=ols_result.predictions,
    )

    run_dir = make_run_dir(ROOT / cfg["output_root"], "ols", args.split_type)

    preds = test_df[cfg["metadata_cols"]].copy()
    preds["y_true_log"] = test_df["log_target"].to_numpy()
    preds["y_pred_log"] = ols_result.predictions
    preds["y_true"] = preds["y_true_log"].map(lambda x: __import__("math").exp(float(x)))
    preds["y_pred"] = preds["y_pred_log"].map(lambda x: __import__("math").exp(float(x)))

    save_dataframe(preds, run_dir / "predictions.csv")
    save_dataframe(ols_result.coefficient_frame(), run_dir / "ols_coefficients.csv")
    save_json(metrics, run_dir / "metrics.json")
    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(ols_result.model.summary().as_text())

    print(f"[OLS] finished. results saved to: {run_dir}")


if __name__ == "__main__":
    main()
