
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="例如 outputs_loto/loto/20260424_120000")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    rows = []
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        overall = summary["overall"]
        rows.append({
            "model": model_dir.name.upper(),
            "rmse_log": overall.get("rmse_log"),
            "mae_log": overall.get("mae_log"),
            "r2_log": overall.get("r2_log"),
            "median_relative_error": overall.get("median_relative_error"),
            "mean_relative_error": overall.get("mean_relative_error"),
        })

    df = pd.DataFrame(rows).sort_values("rmse_log").reset_index(drop=True)
    out_csv = run_dir / "loto_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(df)
    print(f"Saved to: {out_csv}")


if __name__ == "__main__":
    main()
