#!/usr/bin/env python3
"""
compare_metrics.py  —  4-way model comparison

Reads the CSV output dumps produced by run_m0_0.sh / run_m1_0.sh /
run_m1_1.sh / run_m2.sh and generates:

  comparison_report.txt        human-readable summary table
  plot_coverage_train.png      coverage over training steps
  plot_coverage_eval.png       coverage over eval steps
  plot_reward_train.png        smoothed reward over training steps
  plot_action_dist_eval.png    action selection heatmap (eval)
  plot_stability_train.png     edge stability metrics (M1_0, M1_1 only)
  plot_magnitude_m2.png        per-action magnitude snapshot (M2 only)
  comparison_summary.json      machine-readable numbers

Usage:
  python3 compare_metrics.py \\
      --m0-0  results/m0_0 \\
      --m1-0  results/m1_0 \\
      --m1-1  results/m1_1 \\
      --m2    results/m2   \\
      --out   results/comparison

  Any subset of models can be passed; missing ones are skipped gracefully.
  Pass --phase train|eval|both  (default: both)
"""

import argparse, json, math, os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not found — skipping plots, report only")


# ── Constants ──────────────────────────────────────────────────────────────────

MODELS = ["m0_0", "m1_0", "m1_1", "m2"]
MODEL_LABELS = {
    "m0_0": "M0_0 (3-metric)",
    "m1_0": "M1_0 (full-edge dist, 12)",
    "m1_1": "M1_1 (visited-edge dist, 13)",
    "m2":   "M2 (per-mutator mag, 97)",
}
MODEL_COLORS = {
    "m0_0": "#4e79a7",
    "m1_0": "#f28e2b",
    "m1_1": "#59a14f",
    "m2":   "#e15759",
}
SMOOTH_WINDOW = 200    # steps for rolling average on reward / coverage

ACTION_COLUMNS = [
    "DET_FLIP_ONE_BIT","DET_FLIP_TWO_BITS","DET_FLIP_FOUR_BITS",
    "DET_FLIP_ONE_BYTE","DET_FLIP_TWO_BYTES","DET_FLIP_FOUR_BYTES",
    "DET_ARITH_ADD_ONE","DET_ARITH_SUB_ONE",
    "DET_ARITH_ADD_TWO_LE","DET_ARITH_SUB_TWO_LE",
    "DET_ARITH_ADD_TWO_BIG","DET_ARITH_SUB_TWO_BIG",
    "DET_ARITH_ADD_FOUR_LE","DET_ARITH_SUB_FOUR_LE",
    "DET_ARITH_ADD_FOUR_BIG","DET_ARITH_SUB_FOUR_BIG",
    "INTERESTING_BYTE","INTERESTING_TWO_BYTES_LE","INTERESTING_TWO_BYTES_BIG",
    "INTERESTING_FOUR_BYTES_LE","INTERESTING_FOUR_BYTES_BIG",
    "HAVOC_MUT_FLIPBIT","HAVOC_MUT_INTERESTING8",
    "HAVOC_MUT_INTERESTING16","HAVOC_MUT_INTERESTING16BE",
    "HAVOC_MUT_INTERESTING32","HAVOC_MUT_INTERESTING32BE",
    "HAVOC_MUT_ARITH8_","HAVOC_MUT_ARITH8",
    "HAVOC_MUT_ARITH16_","HAVOC_MUT_ARITH16BE_",
    "HAVOC_MUT_ARITH16","HAVOC_MUT_ARITH16BE",
    "HAVOC_MUT_ARITH32_","HAVOC_MUT_ARITH32BE_",
    "HAVOC_MUT_ARITH32","HAVOC_MUT_ARITH32BE",
    "HAVOC_MUT_RAND8","HAVOC_MUT_BYTEADD","HAVOC_MUT_BYTESUB","HAVOC_MUT_FLIP8",
    "DICTIONARY_USER_EXTRAS_OVER","DICTIONARY_USER_EXTRAS_INSERT",
    "DICTIONARY_AUTO_EXTRAS_OVER","DICTIONARY_AUTO_EXTRAS_INSERT",
    "CUSTOM_MUTATOR","HAVOC",
]
N_ACTIONS = len(ACTION_COLUMNS)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        warnings.warn(f"Could not load {path}: {e}")
        return None


def load_fuzzer_stats(path: Path) -> dict:
    """Parse AFL++ fuzzer_stats file into a dict."""
    stats = {}
    if not path.exists():
        return stats
    try:
        for line in path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                stats[k.strip()] = v.strip()
    except Exception:
        pass
    return stats


def load_model_data(model_id: str, results_dir: Path, phase: str) -> dict:
    """Load all available data for one model from its results directory."""
    data = {"id": model_id, "label": MODEL_LABELS[model_id], "dir": results_dir}

    for p in ("train", "eval"):
        if phase not in (p, "both"):
            data[f"df_{p}"] = None
            continue
        csv = results_dir / f"rl_metrics_{model_id}_{p}.csv"
        data[f"df_{p}"] = load_csv(csv)

    data["stats_train"] = load_fuzzer_stats(results_dir / "fuzzer_stats_train.txt")
    data["stats_eval"]  = load_fuzzer_stats(results_dir / "fuzzer_stats_eval.txt")
    return data


# ── Summary statistics ─────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame | None, phase: str) -> dict:
    if df is None or df.empty:
        return {}
    s = {}
    if "coverage" in df.columns:
        s["coverage_final"]  = int(df["coverage"].iloc[-1])
        s["coverage_max"]    = int(df["coverage"].max())
        s["coverage_start"]  = int(df["coverage"].iloc[0])
        s["coverage_gained"] = s["coverage_max"] - s["coverage_start"]
    if "crashes" in df.columns:
        s["crashes_final"] = int(df["crashes"].iloc[-1])
    if "reward" in df.columns:
        s["reward_mean"] = float(df["reward"].mean())
        s["reward_std"]  = float(df["reward"].std())
    if "loss" in df.columns and phase == "train":
        s["loss_final"] = float(df["loss"].iloc[-1])
    if "action" in df.columns:
        counts = df["action"].value_counts()
        top    = counts.idxmax()
        s["top_action"]       = int(top)
        s["top_action_name"]  = ACTION_COLUMNS[int(top)]
        s["top_action_pct"]   = float(counts.max() / len(df) * 100)
        # entropy of action distribution
        probs = counts / counts.sum()
        s["action_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
    if "epsilon" in df.columns and phase == "train":
        s["epsilon_final"] = float(df["epsilon"].iloc[-1])
    s["total_steps"] = len(df)
    return s


# ── Smoothing helper ───────────────────────────────────────────────────────────

def smooth(series: pd.Series, w: int = SMOOTH_WINDOW) -> pd.Series:
    return series.rolling(window=min(w, len(series)), min_periods=1).mean()


# ── Plots ──────────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_coverage(datasets: list[dict], phase: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    any_data = False
    for d in datasets:
        df = d.get(f"df_{phase}")
        if df is None or "coverage" not in df.columns:
            continue
        ax.plot(df["step"], smooth(df["coverage"]),
                label=d["label"], color=MODEL_COLORS[d["id"]], linewidth=1.8)
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("RL Step"); ax.set_ylabel("Coverage (edges hit)")
    ax.set_title(f"Coverage over {phase} steps")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, out_dir / f"plot_coverage_{phase}.png")


def plot_reward(datasets: list[dict], phase: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    any_data = False
    for d in datasets:
        df = d.get(f"df_{phase}")
        if df is None or "reward" not in df.columns:
            continue
        ax.plot(df["step"], smooth(df["reward"]),
                label=d["label"], color=MODEL_COLORS[d["id"]], linewidth=1.8)
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("RL Step"); ax.set_ylabel("Reward (smoothed)")
    ax.set_title(f"Reward over {phase} steps")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    _save(fig, out_dir / f"plot_reward_{phase}.png")


def plot_action_dist(datasets: list[dict], phase: str, out_dir: Path):
    """Heatmap: model × action, showing selection frequency (%)."""
    rows, row_labels = [], []
    for d in datasets:
        df = d.get(f"df_{phase}")
        if df is None or "action" not in df.columns:
            continue
        counts = df["action"].value_counts().reindex(range(N_ACTIONS), fill_value=0)
        rows.append(counts.values / counts.sum() * 100)
        row_labels.append(d["label"])

    if not rows:
        return

    matrix = np.array(rows)   # shape: (n_models, 47)
    fig, ax = plt.subplots(figsize=(18, max(2, len(rows) * 1.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(N_ACTIONS))
    ax.set_xticklabels(ACTION_COLUMNS, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(f"Action selection frequency (%) — {phase}")
    fig.colorbar(im, ax=ax, label="% of steps")
    _save(fig, out_dir / f"plot_action_dist_{phase}.png")


def plot_stability(datasets: list[dict], phase: str, out_dir: Path):
    """Edge stability time series for M1_0 and M1_1 (if columns present)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    plotted = False
    for d in datasets:
        if d["id"] not in ("m1_0", "m1_1"):
            continue
        df = d.get(f"df_{phase}")
        if df is None:
            continue

        if "stability" in df.columns:
            axes[0].plot(df["step"], smooth(df["stability"]),
                         label=d["label"], color=MODEL_COLORS[d["id"]], linewidth=1.8)
            plotted = True

        if d["id"] == "m1_1" and "num_visited" in df.columns:
            axes[1].plot(df["step"], df["num_visited"],
                         label=d["label"], color=MODEL_COLORS[d["id"]], linewidth=1.8)
            plotted = True

    if not plotted:
        plt.close(fig); return

    axes[0].set_title("Mean edge stability ratio"); axes[0].set_xlabel("RL Step")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Visited edges (M1_1 only)"); axes[1].set_xlabel("RL Step")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    _save(fig, out_dir / f"plot_stability_{phase}.png")


def plot_m2_magnitude(datasets: list[dict], phase: str, out_dir: Path):
    """Bar chart of final mean_avg_en / mean_avg_dis for M2."""
    for d in datasets:
        if d["id"] != "m2":
            continue
        df = d.get(f"df_{phase}")
        if df is None or "mean_avg_en" not in df.columns:
            continue

        # Last non-zero row
        last = df[df["mean_avg_en"] > 0]
        if last.empty:
            continue
        last = last.iloc[-1]
        en_val  = float(last["mean_avg_en"])
        dis_val = float(last.get("mean_avg_dis", 0.0))

        # Also try to extract top_en / top_dis action indices if present
        fig, ax = plt.subplots(figsize=(8, 4))
        cats  = ["mean avg_enabled_mag", "mean avg_disabled_mag"]
        vals  = [en_val, dis_val]
        bars  = ax.bar(cats, vals, color=["#4e79a7", "#e15759"], width=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Normalised magnitude (fraction of MAP_SIZE)")
        ax.set_title(f"M2 final per-action magnitude averages — {phase}")
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, out_dir / f"plot_magnitude_m2_{phase}.png")


def plot_coverage_comparison_bar(summaries: dict, phase: str, out_dir: Path):
    """Bar chart comparing final coverage across models."""
    models, vals, colors = [], [], []
    for mid, s in summaries.items():
        ps = s.get(phase, {})
        if "coverage_final" not in ps:
            continue
        models.append(MODEL_LABELS[mid])
        vals.append(ps["coverage_final"])
        colors.append(MODEL_COLORS[mid])

    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Final coverage (edges)")
    ax.set_title(f"Final coverage — {phase}")
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir / f"plot_coverage_bar_{phase}.png")


# ── Text report ────────────────────────────────────────────────────────────────

def write_report(summaries: dict, afl_stats: dict, out_path: Path):
    lines = []
    SEP   = "=" * 70

    lines += [SEP, "  RL FUZZER — 4-MODEL COMPARISON REPORT", SEP, ""]

    for phase in ("train", "eval"):
        lines += [f"{'─'*70}", f"  PHASE: {phase.upper()}", f"{'─'*70}"]

        # Coverage table
        lines += ["", "  Coverage summary:", ""]
        hdr = f"  {'Model':<36} {'Start':>6} {'Final':>6} {'Max':>6} {'Gained':>7} {'Crashes':>8}"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for mid in MODELS:
            ps = summaries.get(mid, {}).get(phase, {})
            if not ps:
                continue
            lines.append(
                f"  {MODEL_LABELS[mid]:<36} "
                f"{ps.get('coverage_start', 0):>6} "
                f"{ps.get('coverage_final', 0):>6} "
                f"{ps.get('coverage_max', 0):>6} "
                f"{ps.get('coverage_gained', 0):>7} "
                f"{ps.get('crashes_final', 0):>8}"
            )

        # Reward table
        lines += ["", "  Reward summary:", ""]
        hdr2 = f"  {'Model':<36} {'Mean R':>8} {'Std R':>8} {'Entropy':>8}"
        lines.append(hdr2)
        lines.append("  " + "-" * (len(hdr2) - 2))
        for mid in MODELS:
            ps = summaries.get(mid, {}).get(phase, {})
            if not ps:
                continue
            lines.append(
                f"  {MODEL_LABELS[mid]:<36} "
                f"{ps.get('reward_mean', float('nan')):>8.3f} "
                f"{ps.get('reward_std', float('nan')):>8.3f} "
                f"{ps.get('action_entropy', float('nan')):>8.3f}"
            )

        # Top action table
        lines += ["", "  Dominant action per model:", ""]
        for mid in MODELS:
            ps = summaries.get(mid, {}).get(phase, {})
            if "top_action_name" not in ps:
                continue
            lines.append(
                f"  {MODEL_LABELS[mid]:<36}  "
                f"#{ps['top_action']:<3} {ps['top_action_name']:<34} "
                f"({ps['top_action_pct']:.1f}%)"
            )

        # AFL++ stats
        lines += ["", "  AFL++ fuzzer_stats:", ""]
        for mid in MODELS:
            fs = afl_stats.get(mid, {}).get(phase, {})
            if not fs:
                continue
            execs = fs.get("execs_done", "?")
            speed = fs.get("execs_per_sec", "?")
            crashes = fs.get("unique_crashes", "?")
            hangs   = fs.get("unique_hangs", "?")
            lines.append(
                f"  {MODEL_LABELS[mid]:<36}  "
                f"execs={execs}  speed={speed}/s  "
                f"unique_crashes={crashes}  hangs={hangs}"
            )
        lines += [""]

    lines += [SEP, "  END OF REPORT", SEP]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  [report] {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global SMOOTH_WINDOW
    ap = argparse.ArgumentParser(
        description="Compare output dumps from all 4 RL fuzzer model runs")
    ap.add_argument("--m0-0",  dest="dir_m0_0",  default=None,
                    help="results dir for M0_0  (default: results/m0_0)")
    ap.add_argument("--m1-0",  dest="dir_m1_0",  default=None,
                    help="results dir for M1_0  (default: results/m1_0)")
    ap.add_argument("--m1-1",  dest="dir_m1_1",  default=None,
                    help="results dir for M1_1  (default: results/m1_1)")
    ap.add_argument("--m2",    dest="dir_m2",    default=None,
                    help="results dir for M2    (default: results/m2)")
    ap.add_argument("--out",   default="results/comparison",
                    help="output directory for plots and report")
    ap.add_argument("--phase", choices=["train", "eval", "both"], default="both",
                    help="which phase CSVs to compare (default: both)")
    ap.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW,
                    help=f"rolling average window for reward/coverage plots (default: {SMOOTH_WINDOW})")
    args = ap.parse_args()

    SMOOTH_WINDOW = args.smooth_window

    # Resolve directories
    dir_map = {
        "m0_0": Path(args.dir_m0_0 or "results/m0_0"),
        "m1_0": Path(args.dir_m1_0 or "results/m1_0"),
        "m1_1": Path(args.dir_m1_1 or "results/m1_1"),
        "m2":   Path(args.dir_m2   or "results/m2"),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    phases = ["train", "eval"] if args.phase == "both" else [args.phase]

    print(f"\n{'='*60}")
    print(f"  RL Fuzzer — Model Comparison")
    print(f"  phase(s) : {args.phase}")
    print(f"  output   : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load all data ─────────────────────────────────────────────────────────
    datasets = []
    summaries = {}
    afl_stats = {}

    for mid in MODELS:
        d = dir_map[mid]
        if not d.exists():
            print(f"  [skip] {mid}: results dir not found: {d}")
            continue
        print(f"  [load] {mid} ← {d}")
        data = load_model_data(mid, d, args.phase)
        datasets.append(data)

        summaries[mid] = {}
        afl_stats[mid] = {}
        for phase in phases:
            summaries[mid][phase] = summarise(data.get(f"df_{phase}"), phase)
            afl_stats[mid][phase] = (
                data["stats_train"] if phase == "train" else data["stats_eval"]
            )

    if not datasets:
        print("[-] No model results found. Run at least one of the run_m*.sh scripts first.")
        sys.exit(1)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if HAS_MPL:
        print("\n  Generating plots...")
        for phase in phases:
            plot_coverage(datasets, phase, out_dir)
            plot_reward(datasets, phase, out_dir)
            plot_action_dist(datasets, phase, out_dir)
            plot_stability(datasets, phase, out_dir)
            plot_m2_magnitude(datasets, phase, out_dir)
            plot_coverage_comparison_bar(summaries, phase, out_dir)
    else:
        print("\n  [warn] matplotlib unavailable — skipping plots")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n  Writing report...")
    write_report(summaries, afl_stats, out_dir / "comparison_report.txt")

    # ── JSON summary ──────────────────────────────────────────────────────────
    json_path = out_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"  [json]   {json_path}")

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    for phase in phases:
        print(f"\n  {phase.upper()} — final coverage:")
        for mid in MODELS:
            ps = summaries.get(mid, {}).get(phase, {})
            if not ps:
                continue
            cov  = ps.get("coverage_final", 0)
            gain = ps.get("coverage_gained", 0)
            cr   = ps.get("crashes_final", 0)
            print(f"    {MODEL_LABELS[mid]:<38} cov={cov:<5}  "
                  f"gained={gain:<5}  crashes={cr}")

    print(f"\n{'='*60}")
    print(f"  Done.  All outputs in: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
