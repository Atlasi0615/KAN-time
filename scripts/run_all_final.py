
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/final_baseline.yaml")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--split-suite", type=str, choices=["primary", "legacy", "all"], default="primary")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_and_capture(cmd: List[str], log_path: Path, cwd: Path) -> str:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n>>> " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            captured.append(line)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")
    return "".join(captured)


def extract_run_dir(stdout_text: str) -> str:
    markers = ["results saved to:", "finished. results saved to:"]
    for line in reversed(stdout_text.splitlines()):
        lower = line.lower()
        if any(m in lower for m in markers):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Could not parse run directory from command output.")


def main():
    args = parse_args()
    config_path = ROOT / args.config
    session_dir = ROOT / "overnight_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "overnight.log"

    split_suites = {
        "primary": ["extrap_jet", "group"],
        "legacy": ["random", "group"],
        "all": ["random", "extrap_jet", "group"],
    }
    jobs = []
    for split in split_suites[args.split_suite]:
        jobs.extend(
            [
                ("ols", split, [args.python, "scripts/run_ols.py", "--config", args.config, "--split-type", split]),
                ("mlp", split, [args.python, "scripts/tune_mlp.py", "--config", args.config, "--split-type", split]),
                ("kan", split, [args.python, "scripts/tune_kan.py", "--config", args.config, "--split-type", split]),
            ]
        )

    if args.dry_run:
        print("Session dir:", session_dir)
        for _, _, cmd in jobs:
            print(" ".join(cmd))
        return

    manifest = {
        "config": str(config_path),
        "session_dir": str(session_dir),
        "runs": [],
    }

    for model, split, cmd in jobs:
        print("\n" + "=" * 80)
        print(f"Running {model.upper()} | {split}")
        print("=" * 80)
        stdout_text = run_and_capture(cmd, log_path=log_path, cwd=ROOT)
        run_dir = extract_run_dir(stdout_text)
        manifest["runs"].append({"model": model, "split": split, "run_dir": run_dir})

        if not args.skip_plots:
            for space in ["original", "log"]:
                plot_cmd = [args.python, "scripts/plot_predictions.py", "--run-dir", run_dir, "--space", space]
                run_and_capture(plot_cmd, log_path=log_path, cwd=ROOT)

    manifest_path = session_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_cmd = [args.python, "scripts/collect_results.py", "--manifest", str(manifest_path)]
    run_and_capture(summary_cmd, log_path=log_path, cwd=ROOT)

    print("\nAll runs finished.")
    print(f"Session directory: {session_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
