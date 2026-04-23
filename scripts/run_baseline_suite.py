from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split-type", type=str, choices=["random", "group", "both"], default="both")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    split_types = ["random", "group"] if args.split_type == "both" else [args.split_type]

    for split_type in split_types:
        run_command([sys.executable, "scripts/run_ols.py", "--config", args.config, "--split-type", split_type])
        run_command([sys.executable, "scripts/tune_mlp.py", "--config", args.config, "--split-type", split_type])
        run_command([sys.executable, "scripts/tune_kan.py", "--config", args.config, "--split-type", split_type])


if __name__ == "__main__":
    main()
