
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows = []
    for item in manifest["runs"]:
        run_dir = Path(item["run_dir"])
        metrics_path = run_dir / "metrics.json"
        best_path = run_dir / "best_params.json"
        row = {
            "model": item["model"].upper(),
            "split": item["split"],
            "run_dir": str(run_dir),
        }
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            row.update(metrics)
        if best_path.exists():
            with best_path.open("r", encoding="utf-8") as f:
                best = json.load(f)
            row["best_params"] = json.dumps(best, ensure_ascii=False)
        rows.append(row)

    df = pd.DataFrame(rows)
    model_order = pd.CategoricalDtype(categories=["OLS", "MLP", "KAN"], ordered=True)
    split_order = pd.CategoricalDtype(categories=["random", "group"], ordered=True)
    df["model"] = df["model"].astype(model_order)
    df["split"] = df["split"].astype(split_order)
    df = df.sort_values(["split", "model"]).reset_index(drop=True)

    out_csv = manifest_path.parent / "baseline_summary.csv"
    out_md = manifest_path.parent / "baseline_summary.md"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metric_cols = [c for c in ["rmse_log", "mae_log", "r2_log", "median_relative_error", "mean_relative_error"] if c in df.columns]
    md_lines = ["| Model | Split | " + " | ".join(metric_cols) + " |", "|---|---|" + "|".join(["---:" for _ in metric_cols]) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in metric_cols:
            val = r[c]
            vals.append(f"{float(val):.4f}" if pd.notnull(val) else "")
        md_lines.append(f"| {r['model']} | {r['split']} | " + " | ".join(vals) + " |")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(df[["model", "split"] + metric_cols])
    print(f"Saved CSV to {out_csv}")
    print(f"Saved markdown table to {out_md}")


if __name__ == "__main__":
    main()
