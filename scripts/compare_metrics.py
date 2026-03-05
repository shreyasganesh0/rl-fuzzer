#!/usr/bin/env python3
"""
compare_metrics.py  —  4-way model comparison with multi-run aggregation

Reads CSV output dumps produced by run_m0_0.sh / run_m1_0.sh /
run_m1_1.sh / run_m2.sh and generates:

  comparison_report.txt              human-readable summary table
  plot_coverage_{phase}_steps.png    coverage vs RL step (≈ exec count)
  plot_coverage_{phase}_time.png     coverage vs wall-clock seconds
  plot_reward_{phase}.png            smoothed reward
  plot_action_dist_{phase}.png       action selection heatmap
  plot_stability_{phase}.png         edge stability (M1_0, M1_1 only)
  plot_magnitude_m2_{phase}.png      per-action magnitude snapshot (M2)
  plot_coverage_bar_{phase}.png      coverage_gained bar chart
  plot_throughput_{phase}.png        execs/sec per model
  plot_coverage_per_sec_{phase}.png  edges discovered per second
  comparison_summary.json            machine-readable numbers

Usage:
  python3 compare_metrics.py \\
      --m0-0  results/m0_0 \\
      --m1-0  results/m1_0 \\
      --m1-1  results/m1_1 \\
      --m2    results/m2   \\
      --out   results/comparison

  Any subset of models can be passed; missing ones are skipped gracefully.
  Pass --phase train|eval|both  (default: both)
  Pass --multi-run to aggregate across run_1/, run_2/, ... subdirs
  Pass --compare-mode steps|time to control coverage plot x-axis
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

MODELS = ["m0_0", "m1_0", "m1_1", "m2",
          "m0_0_skip", "m1_0_skip", "m1_1_skip", "m2_skip"]
MODEL_LABELS = {
    "m0_0":         "M0_0 (3-metric)",
    "m1_0":         "M1_0 (full-edge dist, 12)",
    "m1_1":         "M1_1 (visited-edge dist, 13)",
    "m2":           "M2 (per-mutator mag, 97)",
    "m0_0_skip":    "M0_0_SKIP (3-metric, freq=4)",
    "m1_0_skip":    "M1_0_SKIP (full-edge dist, freq=4)",
    "m1_1_skip":    "M1_1_SKIP (visited-edge dist, freq=4)",
    "m2_skip":      "M2_SKIP (per-mutator mag, freq=4)",
    "baseline":     "Baseline (plain AFL++)",
}
MODEL_COLORS = {
    "m0_0":         "#4e79a7",
    "m1_0":         "#f28e2b",
    "m1_1":         "#59a14f",
    "m2":           "#e15759",
    "m0_0_skip":    "#76b7b2",
    "m1_0_skip":    "#edc948",
    "m1_1_skip":    "#b07aa1",
    "m2_skip":      "#ff9da7",
    "baseline":     "#888888",
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

def load_csv(path: Path):
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


def load_multi_run_csvs(model_id: str, results_dir: Path, phase: str, max_runs: int) -> list:
    """Scan results_dir/run_1/, run_2/, ... and load CSVs for the given model/phase."""
    run_dfs = []
    for i in range(1, max_runs + 1):
        run_dir = results_dir / f"run_{i}"
        if not run_dir.exists():
            continue
        csv_path = run_dir / f"rl_metrics_{model_id}_{phase}.csv"
        df = load_csv(csv_path)
        # Fallback: if exact name doesn't match, try any rl_metrics_*_{phase}.csv
        if df is None:
            candidates = sorted(run_dir.glob(f"rl_metrics_*_{phase}.csv"))
            for c in candidates:
                df = load_csv(c)
                if df is not None:
                    break
        if df is not None:
            df = df.copy()
            df["_run_idx"] = i
            run_dfs.append(df)
    return run_dfs


def load_multi_run_stats(results_dir: Path, phase: str, max_runs: int) -> list:
    """Scan results_dir/run_1/, run_2/, ... and load fuzzer_stats for each run."""
    stats_list = []
    for i in range(1, max_runs + 1):
        run_dir = results_dir / f"run_{i}"
        if not run_dir.exists():
            continue
        stats_path = run_dir / f"fuzzer_stats_{phase}.txt"
        s = load_fuzzer_stats(stats_path)
        if s:
            stats_list.append(s)
    return stats_list


def aggregate_multi_run(run_dfs: list):
    """
    Aggregate a list of run DataFrames into one DataFrame.
    For each numeric column: compute nanmean and nanstd across runs.
    Returns DataFrame with original col names (= means) + {col}_std columns
    and a _n_runs column.
    """
    if not run_dfs:
        return None
    if len(run_dfs) == 1:
        df = run_dfs[0].copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col != "_run_idx" and not col.endswith("_std"):
                df[f"{col}_std"] = 0.0
        df["_n_runs"] = 1
        return df

    # Find intersection of step values across all runs
    step_sets = [set(df["step"].values) for df in run_dfs if "step" in df.columns]
    if not step_sets:
        return None
    all_union    = sorted(set().union(*step_sets))
    common_steps = sorted(step_sets[0].intersection(*step_sets[1:]))
    avg_len      = np.mean([len(s) for s in step_sets])
    # Fall back to union when intersection is too small (e.g. baseline CSVs
    # where exec counts differ slightly across runs)
    if not common_steps or len(common_steps) < max(2, avg_len * 0.5):
        common_steps = all_union

    result_rows = []
    for step in common_steps:
        row = {"step": step}
        cols_data: dict = {}
        for df in run_dfs:
            sub = df[df["step"] == step]
            if sub.empty:
                continue
            for col in sub.columns:
                if col in ("step", "_run_idx"):
                    continue
                cols_data.setdefault(col, []).append(sub[col].iloc[0])
        for col, vals in cols_data.items():
            try:
                arr = np.array(vals, dtype=float)
                row[col] = float(np.nanmean(arr))
                row[f"{col}_std"] = float(np.nanstd(arr))
            except (TypeError, ValueError):
                row[col] = vals[0]
        result_rows.append(row)

    if not result_rows:
        return None
    agg = pd.DataFrame(result_rows)
    agg["_n_runs"] = len(run_dfs)
    return agg


def mean_stat(stats_list: list, key: str) -> tuple:
    """Return (mean, std) for a fuzzer_stats key across runs."""
    vals = []
    for s in stats_list:
        v = s.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except ValueError:
                pass
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.mean(vals)), float(np.std(vals)))


def load_model_data(model_id: str, results_dir: Path, phase: str,
                    multi_run: bool = False, max_runs: int = 10) -> dict:
    """Load all available data for one model from its results directory."""
    data = {"id": model_id, "label": MODEL_LABELS.get(model_id, model_id),
            "dir": results_dir}

    for p in ("train", "eval"):
        if phase not in (p, "both"):
            data[f"df_{p}"]       = None
            data[f"df_{p}_runs"]  = []
            data[f"stats_{p}_runs"] = []
            continue

        if multi_run:
            run_dfs    = load_multi_run_csvs(model_id, results_dir, p, max_runs)
            stats_runs = load_multi_run_stats(results_dir, p, max_runs)
            data[f"df_{p}_runs"]    = run_dfs
            data[f"stats_{p}_runs"] = stats_runs
            agg = aggregate_multi_run(run_dfs)
            data[f"df_{p}_agg"] = agg
            data[f"df_{p}"]     = agg  # backward compat
            if not run_dfs:
                # Fall back to a single CSV sitting directly in results_dir
                fallback = load_csv(results_dir / f"rl_metrics_{model_id}_{p}.csv")
                data[f"df_{p}"]     = fallback
                data[f"df_{p}_agg"] = fallback
        else:
            data[f"df_{p}"]       = load_csv(results_dir / f"rl_metrics_{model_id}_{p}.csv")
            data[f"df_{p}_runs"]  = []
            data[f"stats_{p}_runs"] = []

    data["stats_train"] = load_fuzzer_stats(results_dir / "fuzzer_stats_train.txt")
    data["stats_eval"]  = load_fuzzer_stats(results_dir / "fuzzer_stats_eval.txt")
    return data


# ── Summary statistics ─────────────────────────────────────────────────────────

def summarise(df, phase: str, run_dfs: list = None) -> dict:
    if df is None or df.empty:
        return {}
    s = {}
    if "coverage" in df.columns:
        s["coverage_final"]  = int(df["coverage"].iloc[-1])
        s["coverage_max"]    = int(df["coverage"].max())
        s["coverage_start"]  = int(df["coverage"].iloc[0])
        s["coverage_gained"] = s["coverage_max"] - s["coverage_start"]
        if run_dfs:
            per_run = []
            for rdf in run_dfs:
                if "coverage" in rdf.columns and not rdf.empty:
                    per_run.append(int(rdf["coverage"].max()) - int(rdf["coverage"].iloc[0]))
            if per_run:
                s["coverage_gained_mean"] = float(np.mean(per_run))
                s["coverage_gained_std"]  = float(np.std(per_run))
                s["coverage_gained_min"]  = int(min(per_run))
                s["coverage_gained_max"]  = int(max(per_run))
                s["n_runs"] = len(per_run)
    if "crashes" in df.columns:
        s["crashes_final"] = int(df["crashes"].iloc[-1])
    if "reward" in df.columns:
        s["reward_mean"] = float(df["reward"].mean())
        s["reward_std"]  = float(df["reward"].std())
    if "loss" in df.columns and phase == "train":
        s["loss_final"] = float(df["loss"].iloc[-1])
    if "action" in df.columns:
        action_df = df[df["action"] >= 0]
        if not action_df.empty:
            counts = action_df["action"].value_counts()
            top    = counts.idxmax()
            s["top_action"]      = int(top)
            s["top_action_name"] = ACTION_COLUMNS[int(top)] if int(top) < len(ACTION_COLUMNS) else str(top)
            s["top_action_pct"]  = float(counts.max() / len(action_df) * 100)
            probs = counts / counts.sum()
            s["action_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
    if "epsilon" in df.columns and phase == "train":
        s["epsilon_final"] = float(df["epsilon"].iloc[-1])
    # Compute coverage_per_second — prefer per-run data when available
    # (the aggregated DF may have truncated step ranges for baseline data)
    if run_dfs and "coverage" in df.columns:
        per_run_cps = []
        per_run_elapsed = []
        for rdf in run_dfs:
            if "elapsed_seconds" in rdf.columns and "coverage" in rdf.columns and not rdf.empty:
                et = float(rdf["elapsed_seconds"].iloc[-1])
                gained_r = int(rdf["coverage"].max()) - int(rdf["coverage"].iloc[0])
                if et > 0:
                    per_run_cps.append(gained_r / et)
                    per_run_elapsed.append(et)
        if per_run_cps:
            s["coverage_per_second"] = float(np.mean(per_run_cps))
            s["elapsed_seconds"]     = float(np.mean(per_run_elapsed))
    elif "elapsed_seconds" in df.columns:
        elapsed_total = float(df["elapsed_seconds"].iloc[-1])
        if elapsed_total > 0 and "coverage" in df.columns:
            gained = s.get("coverage_gained", 0)
            s["coverage_per_second"] = gained / elapsed_total
            s["elapsed_seconds"]     = elapsed_total
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


def plot_coverage(datasets: list, phase: str, out_dir: Path,
                  compare_mode: str = "steps", multi_run: bool = False):
    """Coverage plot with optional shaded ±1σ bands."""
    fig, ax = plt.subplots(figsize=(10, 5))
    any_data = False
    for d in datasets:
        df = d.get(f"df_{phase}")
        if df is None or "coverage" not in df.columns:
            continue
        is_bl = d["id"] == "baseline"
        color = MODEL_COLORS.get(d["id"], "#aaaaaa")

        if compare_mode == "time":
            if "elapsed_seconds" not in df.columns:
                continue
            x      = df["elapsed_seconds"]
            xlabel = "Wall-clock time (seconds)"
        else:
            x      = df["step"]
            xlabel = "RL Step (≈ AFL++ exec)"

        y = smooth(df["coverage"])
        ax.plot(x, y, label=d["label"], color=color,
                linewidth=1.8,
                linestyle="--" if is_bl else "-",
                alpha=0.75 if is_bl else 1.0)

        if multi_run and "coverage_std" in df.columns:
            std = smooth(df["coverage_std"])
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.18)

        any_data = True

    if not any_data:
        plt.close(fig)
        return

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coverage (edges hit)")
    ax.set_title(f"Coverage over {phase} ({compare_mode}-based)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / f"plot_coverage_{phase}_{compare_mode}.png")


def plot_reward(datasets: list, phase: str, out_dir: Path, multi_run: bool = False):
    fig, ax = plt.subplots(figsize=(10, 5))
    any_data = False
    for d in datasets:
        df = d.get(f"df_{phase}")
        if df is None or "reward" not in df.columns:
            continue
        color = MODEL_COLORS.get(d["id"], "#aaaaaa")
        y = smooth(df["reward"])
        ax.plot(df["step"], y, label=d["label"], color=color, linewidth=1.8)
        if multi_run and "reward_std" in df.columns:
            std = smooth(df["reward_std"])
            ax.fill_between(df["step"], y - std, y + std, color=color, alpha=0.18)
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("RL Step"); ax.set_ylabel("Reward (smoothed)")
    ax.set_title(f"Reward over {phase} steps")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    _save(fig, out_dir / f"plot_reward_{phase}.png")


def plot_action_dist(datasets: list, phase: str, out_dir: Path, multi_run: bool = False):
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

    matrix = np.array(rows)
    fig, ax = plt.subplots(figsize=(18, max(2, len(rows) * 1.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(N_ACTIONS))
    ax.set_xticklabels(ACTION_COLUMNS, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(f"Action selection frequency (%) — {phase}")
    fig.colorbar(im, ax=ax, label="% of steps")
    _save(fig, out_dir / f"plot_action_dist_{phase}.png")


def plot_stability(datasets: list, phase: str, out_dir: Path, multi_run: bool = False):
    """Edge stability time series for M1_0 and M1_1."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    plotted = False
    for d in datasets:
        if d["id"] not in ("m1_0", "m1_1"):
            continue
        df = d.get(f"df_{phase}")
        if df is None:
            continue
        color = MODEL_COLORS.get(d["id"], "#aaaaaa")

        if "stability" in df.columns:
            y = smooth(df["stability"])
            axes[0].plot(df["step"], y, label=d["label"], color=color, linewidth=1.8)
            if multi_run and "stability_std" in df.columns:
                std = smooth(df["stability_std"])
                axes[0].fill_between(df["step"], y - std, y + std, color=color, alpha=0.18)
            plotted = True

        if d["id"] == "m1_1" and "num_visited" in df.columns:
            axes[1].plot(df["step"], df["num_visited"],
                         label=d["label"], color=color, linewidth=1.8)
            plotted = True

    if not plotted:
        plt.close(fig); return

    axes[0].set_title("Mean edge stability ratio"); axes[0].set_xlabel("RL Step")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Visited edges (M1_1 only)"); axes[1].set_xlabel("RL Step")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    _save(fig, out_dir / f"plot_stability_{phase}.png")


def plot_m2_magnitude(datasets: list, phase: str, out_dir: Path, multi_run: bool = False):
    """Bar chart of final mean_avg_en / mean_avg_dis for M2."""
    for d in datasets:
        if d["id"] != "m2":
            continue
        df = d.get(f"df_{phase}")
        if df is None or "mean_avg_en" not in df.columns:
            continue

        last = df[df["mean_avg_en"] > 0]
        if last.empty:
            continue
        last    = last.iloc[-1]
        en_val  = float(last["mean_avg_en"])
        dis_val = float(last.get("mean_avg_dis", 0.0))
        en_err  = float(last.get("mean_avg_en_std",  0.0)) if multi_run else 0.0
        dis_err = float(last.get("mean_avg_dis_std", 0.0)) if multi_run else 0.0

        fig, ax = plt.subplots(figsize=(8, 4))
        cats = ["mean avg_enabled_mag", "mean avg_disabled_mag"]
        vals = [en_val, dis_val]
        errs = [en_err, dis_err]
        bars = ax.bar(cats, vals, color=["#4e79a7", "#e15759"], width=0.4,
                      yerr=errs if multi_run else None,
                      capsize=5 if multi_run else 0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Normalised magnitude (fraction of MAP_SIZE)")
        ax.set_title(f"M2 final per-action magnitude averages — {phase}")
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, out_dir / f"plot_magnitude_m2_{phase}.png")


def plot_coverage_bar(summaries: dict, phase: str, out_dir: Path, multi_run: bool = False):
    """Bar chart of coverage_gained across models, with error bars if multi-run."""
    models, vals, colors, errs = [], [], [], []
    for mid in list(MODELS) + ["baseline"]:
        ps  = summaries.get(mid, {}).get(phase, {})
        key = "coverage_gained_mean" if (multi_run and "coverage_gained_mean" in ps) \
              else "coverage_gained"
        if key not in ps:
            continue
        models.append(MODEL_LABELS.get(mid, mid))
        vals.append(ps[key])
        colors.append(MODEL_COLORS.get(mid, "#aaaaaa"))
        errs.append(ps.get("coverage_gained_std", 0.0) if multi_run else 0.0)

    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x    = range(len(models))
    bars = ax.bar(x, vals, color=colors,
                  yerr=errs if multi_run else None,
                  capsize=5 if multi_run else 0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Coverage gained (edges)")
    ax.set_title(f"Coverage gained — {phase}" + (" (mean ± 1σ)" if multi_run else ""))
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir / f"plot_coverage_bar_{phase}.png")


def _csv_throughput(df) -> float:
    """Derive throughput from CSV data: max(step) / max(elapsed_seconds)."""
    if df is None or df.empty:
        return float("nan")
    if "elapsed_seconds" not in df.columns or "step" not in df.columns:
        return float("nan")
    elapsed = float(df["elapsed_seconds"].iloc[-1])
    steps   = float(df["step"].iloc[-1])
    if elapsed <= 0 or steps <= 0:
        return float("nan")
    return steps / elapsed


def _is_stale_stats(stats: dict) -> bool:
    """Return True if fuzzer_stats looks like calibration-only data."""
    try:
        execs = int(stats.get("execs_done", 0))
        return execs <= 100  # calibration typically shows seed count (~36)
    except (ValueError, TypeError):
        return False


def _get_throughput(d: dict, phase: str, multi_run: bool) -> tuple:
    """Return (throughput, std, source) using fuzzer_stats with CSV fallback.

    source is 'stats' or 'csv' for diagnostics.
    """
    stats_runs = d.get(f"stats_{phase}_runs", [])
    stats_one  = d.get(f"stats_{phase}", {})
    df         = d.get(f"df_{phase}")

    if multi_run and stats_runs:
        mean, std = mean_stat(stats_runs, "execs_per_sec")
        if not math.isnan(mean) and not all(_is_stale_stats(s) for s in stats_runs):
            return (mean, std, "stats")
        # All runs stale — try CSV fallback across individual run DataFrames
        run_dfs = d.get(f"df_{phase}_runs", [])
        if run_dfs:
            csv_vals = [_csv_throughput(rdf) for rdf in run_dfs]
            csv_vals = [v for v in csv_vals if not math.isnan(v)]
            if csv_vals:
                return (float(np.mean(csv_vals)), float(np.std(csv_vals)), "csv")
        # Single aggregated DF fallback
        t = _csv_throughput(df)
        if not math.isnan(t):
            return (t, 0.0, "csv")
        return (float("nan"), float("nan"), "none")

    # Single-run path
    v = stats_one.get("execs_per_sec")
    if v is not None and not _is_stale_stats(stats_one):
        try:
            return (float(v), 0.0, "stats")
        except ValueError:
            pass
    t = _csv_throughput(df)
    if not math.isnan(t):
        return (t, 0.0, "csv")
    return (float("nan"), float("nan"), "none")


def plot_throughput(datasets: list, phase: str, out_dir: Path, multi_run: bool = False):
    """Bar chart of execs_per_sec from fuzzer_stats, with CSV fallback for stale data."""
    models, vals, colors, errs = [], [], [], []
    for d in datasets:
        mid = d["id"]
        mean, std, _src = _get_throughput(d, phase, multi_run)
        if math.isnan(mean):
            continue
        models.append(MODEL_LABELS.get(mid, mid))
        vals.append(mean)
        errs.append(std)
        colors.append(MODEL_COLORS.get(mid, "#aaaaaa"))

    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(models))
    ax.bar(x, vals, color=colors,
           yerr=errs if multi_run else None,
           capsize=5 if multi_run else 0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    vmax = max(vals) if vals else 1
    for i, v in enumerate(vals):
        ax.text(i, v + vmax * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Executions per second")
    ax.set_title(f"Throughput — {phase}" + (" (mean ± 1σ)" if multi_run else ""))
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir / f"plot_throughput_{phase}.png")


def plot_coverage_per_second(summaries: dict, phase: str, out_dir: Path):
    """Bar chart of coverage_gained / elapsed_seconds."""
    models, vals, colors = [], [], []
    for mid in list(MODELS) + ["baseline"]:
        ps = summaries.get(mid, {}).get(phase, {})
        if "coverage_per_second" not in ps:
            continue
        models.append(MODEL_LABELS.get(mid, mid))
        vals.append(ps["coverage_per_second"])
        colors.append(MODEL_COLORS.get(mid, "#aaaaaa"))

    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x    = range(len(models))
    bars = ax.bar(x, vals, color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    vmax = max(vals) if vals else 1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + vmax * 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Edges discovered per second")
    ax.set_title(f"Coverage efficiency — {phase}")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir / f"plot_coverage_per_sec_{phase}.png")


# ── Text report ────────────────────────────────────────────────────────────────

def write_report(summaries: dict, afl_stats: dict, out_path: Path,
                 multi_run: bool = False, compare_mode: str = "steps",
                 datasets: list = None):
    _report_datasets = datasets or []
    lines = []
    SEP   = "=" * 70

    lines += [SEP, "  RL FUZZER — 4-MODEL COMPARISON REPORT", SEP, ""]
    lines += [f"  compare_mode : {compare_mode}", f"  multi_run    : {multi_run}", ""]

    all_mids = list(MODELS) + ["baseline"]

    for phase in ("train", "eval"):
        lines += [f"{'─'*70}", f"  PHASE: {phase.upper()}", f"{'─'*70}"]

        # Coverage table
        lines += ["", "  Coverage summary:", ""]
        if multi_run:
            hdr = f"  {'Model':<36} {'Gained(mean)':>12} {'±std':>7} {'Min':>6} {'Max':>6} {'Runs':>5}"
        else:
            hdr = f"  {'Model':<36} {'Start':>6} {'Final':>6} {'Max':>6} {'Gained':>7} {'Crashes':>8}"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for mid in all_mids:
            ps = summaries.get(mid, {}).get(phase, {})
            if not ps:
                continue
            label = MODEL_LABELS.get(mid, mid)
            if multi_run and "coverage_gained_mean" in ps:
                lines.append(
                    f"  {label:<36} "
                    f"{ps['coverage_gained_mean']:>12.1f} "
                    f"{ps.get('coverage_gained_std', 0.0):>7.1f} "
                    f"{ps.get('coverage_gained_min', 0):>6} "
                    f"{ps.get('coverage_gained_max', 0):>6} "
                    f"{ps.get('n_runs', 1):>5}"
                )
            else:
                lines.append(
                    f"  {label:<36} "
                    f"{ps.get('coverage_start', 0):>6} "
                    f"{ps.get('coverage_final', 0):>6} "
                    f"{ps.get('coverage_max', 0):>6} "
                    f"{ps.get('coverage_gained', 0):>7} "
                    f"{ps.get('crashes_final', 0):>8}"
                )

        # Throughput section
        lines += ["", "  Throughput (execs/sec):", ""]
        for mid in all_mids:
            label = MODEL_LABELS.get(mid, mid)
            # Find the matching dataset to use the CSV fallback logic
            d_match = next((d for d in _report_datasets if d["id"] == mid), None)
            if d_match:
                mean, std, src = _get_throughput(d_match, phase, multi_run)
                if not math.isnan(mean):
                    suffix = f"  [from {src}]" if src == "csv" else ""
                    if multi_run and std > 0:
                        lines.append(f"  {label:<36}  {mean:.1f} ± {std:.1f} execs/s{suffix}")
                    else:
                        lines.append(f"  {label:<36}  {mean:.1f} execs/s{suffix}")
            else:
                # No dataset found — fall back to raw fuzzer_stats
                fs_runs = afl_stats.get(mid, {}).get(f"{phase}_runs", [])
                fs_one  = afl_stats.get(mid, {}).get(phase, {})
                if multi_run and fs_runs:
                    mean, std = mean_stat(fs_runs, "execs_per_sec")
                    if not math.isnan(mean):
                        lines.append(f"  {label:<36}  {mean:.1f} ± {std:.1f} execs/s")
                elif fs_one:
                    speed = fs_one.get("execs_per_sec", "?")
                    lines.append(f"  {label:<36}  {speed} execs/s")

        # Efficiency section
        lines += ["", "  Efficiency (coverage_gained / elapsed_seconds):", ""]
        any_eff = False
        for mid in all_mids:
            ps    = summaries.get(mid, {}).get(phase, {})
            label = MODEL_LABELS.get(mid, mid)
            if "coverage_per_second" in ps:
                lines.append(f"  {label:<36}  {ps['coverage_per_second']:.4f} edges/sec")
                any_eff = True
        if not any_eff:
            lines.append("  (No elapsed_seconds data — run with updated RL servers)")

        # Reward table
        lines += ["", "  Reward summary:", ""]
        hdr2 = f"  {'Model':<36} {'Mean R':>8} {'Std R':>8} {'Entropy':>8}"
        lines.append(hdr2)
        lines.append("  " + "-" * (len(hdr2) - 2))
        for mid in all_mids:
            ps    = summaries.get(mid, {}).get(phase, {})
            label = MODEL_LABELS.get(mid, mid)
            if not ps:
                continue
            lines.append(
                f"  {label:<36} "
                f"{ps.get('reward_mean', float('nan')):>8.3f} "
                f"{ps.get('reward_std',  float('nan')):>8.3f} "
                f"{ps.get('action_entropy', float('nan')):>8.3f}"
            )

        # Top action table
        lines += ["", "  Dominant action per model:", ""]
        for mid in all_mids:
            ps    = summaries.get(mid, {}).get(phase, {})
            label = MODEL_LABELS.get(mid, mid)
            if "top_action_name" not in ps:
                continue
            lines.append(
                f"  {label:<36}  "
                f"#{ps['top_action']:<3} {ps['top_action_name']:<34} "
                f"({ps['top_action_pct']:.1f}%)"
            )

        # AFL++ fuzzer_stats
        lines += ["", "  AFL++ fuzzer_stats:", ""]
        for mid in all_mids:
            fs    = afl_stats.get(mid, {}).get(phase, {})
            label = MODEL_LABELS.get(mid, mid)
            if not fs:
                continue
            execs   = fs.get("execs_done", "?")
            speed   = fs.get("execs_per_sec", "?")
            crashes = fs.get("unique_crashes", "?")
            hangs   = fs.get("unique_hangs", "?")
            lines.append(
                f"  {label:<36}  "
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
    ap.add_argument("--m0-0",      dest="dir_m0_0",    default=None,
                    help="results dir for M0_0  (default: results/m0_0)")
    ap.add_argument("--m1-0",      dest="dir_m1_0",    default=None,
                    help="results dir for M1_0  (default: results/m1_0)")
    ap.add_argument("--m1-1",      dest="dir_m1_1",    default=None,
                    help="results dir for M1_1  (default: results/m1_1)")
    ap.add_argument("--m2",        dest="dir_m2",      default=None,
                    help="results dir for M2    (default: results/m2)")
    ap.add_argument("--m0-0-skip", dest="dir_m0_0_skip", default=None,
                    help="results dir for M0_0_SKIP")
    ap.add_argument("--m1-0-skip", dest="dir_m1_0_skip", default=None,
                    help="results dir for M1_0_SKIP")
    ap.add_argument("--m1-1-skip", dest="dir_m1_1_skip", default=None,
                    help="results dir for M1_1_SKIP")
    ap.add_argument("--m2-skip",   dest="dir_m2_skip",   default=None,
                    help="results dir for M2_SKIP")
    ap.add_argument("--baseline",  dest="dir_baseline", default=None,
                    help="results dir for plain AFL++ baseline (optional)")
    ap.add_argument("--out",       default="results/comparison",
                    help="output directory for plots and report")
    ap.add_argument("--phase",     choices=["train", "eval", "both"], default="both",
                    help="which phase CSVs to compare (default: both)")
    ap.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW,
                    help=f"rolling average window (default: {SMOOTH_WINDOW})")
    ap.add_argument("--multi-run", action="store_true",
                    help="aggregate across run_1/, run_2/, ... subdirs")
    ap.add_argument("--compare-mode", choices=["steps", "time"], default="steps",
                    help="x-axis for coverage plots: steps (default) or time")
    ap.add_argument("--runs",      type=int, default=10,
                    help="max run_N directories to scan (default: 10)")
    args = ap.parse_args()

    SMOOTH_WINDOW = args.smooth_window

    dir_map = {
        "m0_0": Path(args.dir_m0_0 or "results/m0_0"),
        "m1_0": Path(args.dir_m1_0 or "results/m1_0"),
        "m1_1": Path(args.dir_m1_1 or "results/m1_1"),
        "m2":   Path(args.dir_m2   or "results/m2"),
    }
    for skip_id in ("m0_0_skip", "m1_0_skip", "m1_1_skip", "m2_skip"):
        val = getattr(args, f"dir_{skip_id}", None)
        if val:
            dir_map[skip_id] = Path(val)
    if args.dir_baseline:
        dir_map["baseline"] = Path(args.dir_baseline)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    phases = ["train", "eval"] if args.phase == "both" else [args.phase]

    print(f"\n{'='*60}")
    print(f"  RL Fuzzer — Model Comparison")
    print(f"  phase(s)     : {args.phase}")
    print(f"  multi_run    : {args.multi_run}")
    print(f"  compare_mode : {args.compare_mode}")
    print(f"  output       : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load all data ─────────────────────────────────────────────────────────
    datasets  = []
    summaries = {}
    afl_stats = {}

    all_models = list(MODELS) + (["baseline"] if args.dir_baseline else [])
    # Only include models that have a dir_map entry (avoids default paths for _skip)
    all_models = [m for m in all_models if m in dir_map]
    for mid in all_models:
        if mid not in dir_map:
            continue
        d = dir_map[mid]
        if not d.exists():
            print(f"  [skip] {mid}: results dir not found: {d}")
            continue
        print(f"  [load] {mid} ← {d}")
        data = load_model_data(mid, d, args.phase,
                               multi_run=args.multi_run, max_runs=args.runs)
        datasets.append(data)

        summaries[mid] = {}
        afl_stats[mid] = {}
        for phase in phases:
            run_dfs = data.get(f"df_{phase}_runs", []) if args.multi_run else []
            summaries[mid][phase] = summarise(data.get(f"df_{phase}"), phase, run_dfs)
            afl_stats[mid][phase] = (
                data["stats_train"] if phase == "train" else data["stats_eval"]
            )
            afl_stats[mid][f"{phase}_runs"] = data.get(f"stats_{phase}_runs", [])

    if not datasets:
        print("[-] No model results found. Run at least one of the run_m*.sh scripts first.")
        sys.exit(1)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if HAS_MPL:
        print("\n  Generating plots...")
        for phase in phases:
            # Coverage: always generate both step and time variants
            plot_coverage(datasets, phase, out_dir,
                          compare_mode="steps", multi_run=args.multi_run)
            plot_coverage(datasets, phase, out_dir,
                          compare_mode="time",  multi_run=args.multi_run)
            plot_reward(datasets, phase, out_dir, multi_run=args.multi_run)
            plot_action_dist(datasets, phase, out_dir, multi_run=args.multi_run)
            plot_stability(datasets, phase, out_dir, multi_run=args.multi_run)
            plot_m2_magnitude(datasets, phase, out_dir, multi_run=args.multi_run)
            plot_coverage_bar(summaries, phase, out_dir, multi_run=args.multi_run)
            plot_throughput(datasets, phase, out_dir, multi_run=args.multi_run)
            plot_coverage_per_second(summaries, phase, out_dir)
    else:
        print("\n  [warn] matplotlib unavailable — skipping plots")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n  Writing report...")
    write_report(summaries, afl_stats, out_dir / "comparison_report.txt",
                 multi_run=args.multi_run, compare_mode=args.compare_mode,
                 datasets=datasets)

    # ── JSON summary ──────────────────────────────────────────────────────────
    json_path = out_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"  [json]   {json_path}")

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    for phase in phases:
        print(f"\n  {phase.upper()} — coverage gained:")
        for mid in all_models:
            ps    = summaries.get(mid, {}).get(phase, {})
            label = MODEL_LABELS.get(mid, mid)
            if not ps:
                continue
            if args.multi_run and "coverage_gained_mean" in ps:
                mean = ps["coverage_gained_mean"]
                std  = ps.get("coverage_gained_std", 0.0)
                n    = ps.get("n_runs", 1)
                print(f"    {label:<38} gained={mean:.0f} ± {std:.0f}  n={n}")
            else:
                cov  = ps.get("coverage_final", 0)
                gain = ps.get("coverage_gained", 0)
                cr   = ps.get("crashes_final", 0)
                print(f"    {label:<38} cov={cov:<5}  gained={gain:<5}  crashes={cr}")

    print(f"\n{'='*60}")
    print(f"  Done.  All outputs in: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
