#!/usr/bin/env python3
"""
slice_milestones.py — Post-hoc milestone slicer + comparison runner.

Given full 10M-step eval CSVs, slices them at each milestone step count and
runs compare_metrics.py on the sliced data.

Usage:
  python3 scripts/slice_milestones.py \
    --exp-dir experiments/jsoncpp \
    --milestones 500000,1000000,2000000,10000000 \
    --models m1_0,m1_1,m1_2

  # Query median elapsed_seconds at a milestone (for baseline timing)
  python3 scripts/slice_milestones.py --query-time \
    --exp-dir experiments/jsoncpp --milestone 10000000
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def format_milestone(step: int) -> str:
    """Format step count as human-readable tag: 500000 → '500k', 1000000 → '1m'."""
    if step >= 1_000_000 and step % 1_000_000 == 0:
        return f"{step // 1_000_000}m"
    if step >= 1_000 and step % 1_000 == 0:
        return f"{step // 1_000}k"
    return str(step)


def slice_csv_by_step(csv_path: str, max_step: int) -> pd.DataFrame | None:
    """Load CSV and filter to rows where step <= max_step."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "step" not in df.columns:
        return None
    return df[df["step"] <= max_step].copy()


def slice_csv_by_time(csv_path: str, max_time: float) -> pd.DataFrame | None:
    """Load CSV and filter to rows where elapsed_seconds <= max_time."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "elapsed_seconds" not in df.columns:
        return None
    return df[df["elapsed_seconds"] <= max_time].copy()


def get_elapsed_at_milestone(exp_dir: str, models: list, milestone: int,
                             eval_runs: int = 10) -> float:
    """Find the median elapsed_seconds at the given step milestone across all
    model eval CSVs. Scans both single-file and multi-run layouts."""
    times = []
    for model in models:
        plots_dir = os.path.join(exp_dir, "plots", model)
        # Try single eval CSV
        csv_path = os.path.join(plots_dir, f"rl_metrics_{model}_eval.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "step" in df.columns and "elapsed_seconds" in df.columns:
                at = df[df["step"] <= milestone]
                if not at.empty:
                    times.append(float(at["elapsed_seconds"].iloc[-1]))
        # Also check run_N subdirs
        for i in range(1, eval_runs + 1):
            run_csv = os.path.join(plots_dir, f"run_{i}",
                                   f"rl_metrics_{model}_eval.csv")
            if os.path.exists(run_csv):
                df = pd.read_csv(run_csv)
                if "step" in df.columns and "elapsed_seconds" in df.columns:
                    at = df[df["step"] <= milestone]
                    if not at.empty:
                        times.append(float(at["elapsed_seconds"].iloc[-1]))
    if not times:
        return 0.0
    return float(np.median(times))


def run_comparison(milestone_dir: str, models: list, exp_dir: str):
    """Run compare_metrics.py on sliced CSVs in milestone_dir."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    compare_script = os.path.join(repo_root, "scripts", "compare_metrics.py")
    python = os.path.join(repo_root, ".venv", "bin", "python3")
    if not os.path.isfile(python):
        python = "python3"

    cmd = [python, compare_script,
           "--out", milestone_dir,
           "--phase", "eval",
           "--compare-mode", "steps"]

    for model in models:
        model_dir = os.path.join(milestone_dir, model)
        if os.path.isdir(model_dir):
            flag = f"--{model.replace('_', '-')}"
            cmd += [flag, model_dir]

    baseline_dir = os.path.join(milestone_dir, "baseline")
    if os.path.isdir(baseline_dir):
        cmd += ["--baseline", baseline_dir]

    print(f"  [compare] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser(description="Slice eval CSVs at milestone step counts")
    ap.add_argument("--exp-dir", required=True,
                    help="Experiment directory (e.g. experiments/jsoncpp)")
    ap.add_argument("--milestones", default="500000,1000000,2000000,10000000",
                    help="Comma-separated step counts (default: 500000,1000000,2000000,10000000)")
    ap.add_argument("--models", default="m1_0,m1_1,m1_2",
                    help="Comma-separated model IDs (default: m1_0,m1_1,m1_2)")
    ap.add_argument("--eval-runs", type=int, default=10,
                    help="Max run_N dirs to scan (default: 10)")
    ap.add_argument("--query-time", action="store_true",
                    help="Query mode: print median elapsed_seconds at --milestone")
    ap.add_argument("--milestone", type=int, default=None,
                    help="Single milestone for --query-time mode")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    milestones = [int(s.strip()) for s in args.milestones.split(",")]

    # ── Query mode: just print elapsed_seconds at a milestone ──
    if args.query_time:
        ms = args.milestone or milestones[-1]
        t = get_elapsed_at_milestone(args.exp_dir, models, ms, args.eval_runs)
        print(f"{t:.0f}")
        return

    # ── Slice mode ──
    exp_dir = args.exp_dir
    print(f"\n{'='*60}")
    print(f"  Milestone Slicer")
    print(f"  exp_dir    : {exp_dir}")
    print(f"  models     : {models}")
    print(f"  milestones : {milestones}")
    print(f"{'='*60}\n")

    for milestone in milestones:
        tag = format_milestone(milestone)
        ms_dir = os.path.join(exp_dir, "milestones", tag)
        os.makedirs(ms_dir, exist_ok=True)
        print(f"\n--- Milestone: {tag} ({milestone:,} steps) ---")

        # Slice RL model eval CSVs
        for model in models:
            csv_path = os.path.join(exp_dir, "plots", model,
                                    f"rl_metrics_{model}_eval.csv")
            sliced = slice_csv_by_step(csv_path, milestone)
            if sliced is not None:
                out_model_dir = os.path.join(ms_dir, model)
                os.makedirs(out_model_dir, exist_ok=True)
                out_csv = os.path.join(out_model_dir,
                                       f"rl_metrics_{model}_eval.csv")
                sliced.to_csv(out_csv, index=False)
                print(f"  [slice] {model}: {len(sliced)} rows → {out_csv}")
            else:
                print(f"  [skip]  {model}: no eval CSV found at {csv_path}")

        # Slice baseline (same-steps)
        bl_csv = os.path.join(exp_dir, "plots", "baseline",
                              "rl_metrics_baseline_eval.csv")
        bl_sliced = slice_csv_by_step(bl_csv, milestone)
        if bl_sliced is not None:
            bl_out_dir = os.path.join(ms_dir, "baseline")
            os.makedirs(bl_out_dir, exist_ok=True)
            out_csv = os.path.join(bl_out_dir,
                                   "rl_metrics_baseline_eval.csv")
            bl_sliced.to_csv(out_csv, index=False)
            print(f"  [slice] baseline (steps): {len(bl_sliced)} rows")

        # Slice baseline (same-time): find median elapsed time at this milestone
        t_milestone = get_elapsed_at_milestone(exp_dir, models, milestone,
                                               args.eval_runs)
        if t_milestone > 0:
            bt_csv = os.path.join(exp_dir, "plots", "baseline_time",
                                  "rl_metrics_baseline_time_eval.csv")
            bt_sliced = slice_csv_by_time(bt_csv, t_milestone)
            if bt_sliced is not None:
                bt_out_dir = os.path.join(ms_dir, "baseline_time")
                os.makedirs(bt_out_dir, exist_ok=True)
                out_csv = os.path.join(bt_out_dir,
                                       "rl_metrics_baseline_time_eval.csv")
                bt_sliced.to_csv(out_csv, index=False)
                print(f"  [slice] baseline (time≤{t_milestone:.0f}s): "
                      f"{len(bt_sliced)} rows")

        # Run comparison on the sliced data
        run_comparison(ms_dir, models, exp_dir)

    print(f"\n{'='*60}")
    print(f"  Done. Milestone results in: {exp_dir}/milestones/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
