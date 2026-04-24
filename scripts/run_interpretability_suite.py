from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-run-dir", type=str, required=True)
    parser.add_argument("--kan-run-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/final_baseline.yaml")
    parser.add_argument("--split-type", type=str, choices=["random", "group"], default="random")
    parser.add_argument("--features", nargs="*", default=["IP", "BT", "PL", "NEL", "RGEO"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--attempt-symbolic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    cmds = [
        [py, "scripts/analyze_mlp_interpretability.py", "--run-dir", args.mlp_run_dir, "--config", args.config, "--split-type", args.split_type, "--device", args.device, "--features", *args.features],
        [py, "scripts/analyze_kan_interpretability.py", "--run-dir", args.kan_run_dir, "--config", args.config, "--split-type", args.split_type, "--device", args.device, "--features", *args.features],
        [py, "scripts/analyze_kan_specific.py", "--run-dir", args.kan_run_dir, "--config", args.config, "--split-type", args.split_type, "--device", args.device] + (["--attempt-symbolic"] if args.attempt_symbolic else []),
    ]
    for cmd in cmds:
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print("\nAll interpretability analyses finished.")


if __name__ == "__main__":
    main()
