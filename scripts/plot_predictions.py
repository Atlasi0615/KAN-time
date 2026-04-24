
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path like outputs_final/kan/20260423_175810_random")
    parser.add_argument("--space", type=str, choices=["original", "log"], default="original")
    parser.add_argument("--hue-col", type=str, default=None,
                        help="Optional column in predictions.csv, e.g. TOK")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    pred_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "metrics.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"Cannot find {pred_path}")

    df = pd.read_csv(pred_path)

    if args.space == "original":
        xcol = "y_true"
        ycol = "y_pred"
        xlabel = r"Actual $\tau_E$"
        ylabel = r"Predicted $\tau_E$"
        out_name = "parity_original.png"
        use_log_axis = True
    else:
        xcol = "y_true_log"
        ycol = "y_pred_log"
        xlabel = r"Actual $\log(\tau_E)$"
        ylabel = r"Predicted $\log(\tau_E)$"
        out_name = "parity_log.png"
        use_log_axis = False

    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError(
            f"Required columns not found. Need {xcol} and {ycol}, but got {list(df.columns)}"
        )

    metrics_text = ""
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        lines = []
        if "rmse_log" in metrics:
            lines.append(f"RMSE(log) = {metrics['rmse_log']:.4f}")
        if "mae_log" in metrics:
            lines.append(f"MAE(log) = {metrics['mae_log']:.4f}")
        if "r2_log" in metrics:
            lines.append(f"R²(log) = {metrics['r2_log']:.4f}")
        metrics_text = "\n".join(lines)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(7.2, 6.4))

    scatter_kwargs = dict(data=df, x=xcol, y=ycol, alpha=0.65, s=28, edgecolor=None)
    if args.hue_col is not None and args.hue_col in df.columns:
        ax = sns.scatterplot(**scatter_kwargs, hue=args.hue_col)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    else:
        ax = sns.scatterplot(**scatter_kwargs)

    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()

    if use_log_axis:
        positive = np.concatenate([x[x > 0], y[y > 0]])
        vmin = positive.min()
        vmax = positive.max()
        ax.plot([vmin, vmax], [vmin, vmax], "--", linewidth=1.5, color="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
    else:
        vmin = min(x.min(), y.min())
        vmax = max(x.max(), y.max())
        ax.plot([vmin, vmax], [vmin, vmax], "--", linewidth=1.5, color="black")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Prediction vs Actual ({run_dir.parent.name}, {run_dir.name.split('_')[-1]})")

    if metrics_text:
        ax.text(
            0.03, 0.97,
            metrics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

    plt.tight_layout()
    save_path = run_dir / out_name
    plt.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()
