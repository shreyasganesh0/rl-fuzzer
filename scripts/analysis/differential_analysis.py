#!/usr/bin/env python3
"""differential_analysis.py — Statistical analysis of Phase 2 differential fuzzing data.

Reads telemetry CSVs from buggy vs fixed libxml2 campaigns and produces:
  1. Coverage trajectory comparison (mean +/- std across seeds)
  2. Differential edge analysis from bitmap snapshots
  3. Mutation effectiveness heatmaps
  4. Coverage landscape feature comparison
  5. Feature importance report (Mann-Whitney U, Vargha-Delaney A12)
  6. Methodology and summary documentation

Usage:
    python3 scripts/analysis/differential_analysis.py \
        --telemetry-dir experiments/differential/telemetry \
        --output-dir experiments/differential/analysis \
        [--bug-pairs xml005,xml017]
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found — statistical tests will be skipped")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not found — plots will be skipped")


# ── Constants ────────────────────────────────────────────────────────────────

VERSIONS = ["xml005_buggy", "xml005_fixed", "xml017_buggy", "xml017_fixed"]
DEFAULT_BUG_PAIRS = ["xml005", "xml017"]
SEEDS = [1, 2, 3]
N_MUTATIONS = 47
MAP_SIZE = 65536

COVERAGE_COLS = [
    "timestamp_us", "total_execs", "total_edges", "new_edges_this_interval",
    "edge_discovery_rate", "crashes_total", "crashes_this_interval",
    "avg_exec_time_us", "corpus_size", "hot_edges", "warm_edges",
    "cool_edges", "cold_edges", "edge_entropy", "edge_hit_mean",
    "edge_hit_std", "edge_hit_max",
]

# Mutation column layout: total_execs, then for NN=00..46:
#   mut_NN_count, mut_NN_new_edges, mut_NN_crashes
MUTATION_NAMES = {
    0: "FLIP_1BIT", 1: "FLIP_2BITS", 2: "FLIP_4BITS",
    3: "FLIP_1BYTE", 4: "FLIP_2BYTES", 5: "FLIP_4BYTES",
    6: "ARITH_ADD1", 7: "ARITH_SUB1",
    8: "ARITH_ADD2LE", 9: "ARITH_SUB2LE",
    10: "ARITH_ADD2BE", 11: "ARITH_SUB2BE",
    12: "ARITH_ADD4LE", 13: "ARITH_SUB4LE",
    14: "ARITH_ADD4BE", 15: "ARITH_SUB4BE",
    16: "INT_BYTE", 17: "INT_2LE", 18: "INT_2BE",
    19: "INT_4LE", 20: "INT_4BE",
    21: "HAVOC_FLIPBIT", 22: "HAVOC_INT8",
    23: "HAVOC_INT16", 24: "HAVOC_INT16BE",
    25: "HAVOC_INT32", 26: "HAVOC_INT32BE",
    27: "HAVOC_ARITH8_", 28: "HAVOC_ARITH8",
    29: "HAVOC_ARITH16_", 30: "HAVOC_ARITH16",
    31: "HAVOC_ARITH16BE", 32: "HAVOC_ARITH16BE_",
    33: "HAVOC_ARITH32_", 34: "HAVOC_ARITH32BE_",
    35: "HAVOC_ARITH32", 36: "HAVOC_ARITH32BE",
    37: "HAVOC_RAND8", 38: "HAVOC_BYTEADD",
    39: "HAVOC_BYTESUB", 40: "HAVOC_FLIP8",
    41: "DICT_USER_OVER", 42: "DICT_USER_INS",
    43: "DICT_AUTO_OVER", 44: "DICT_AUTO_INS",
    45: "CUSTOM_MUTATOR", 46: "HAVOC",
}

# Plot styling
BUGGY_COLOR = "#e15759"
FIXED_COLOR = "#4e79a7"
BUGGY_STYLE = {"color": BUGGY_COLOR, "linewidth": 2.0}
FIXED_STYLE = {"color": FIXED_COLOR, "linewidth": 2.0}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_csv(path):
    """Load a CSV file, returning DataFrame or None on failure."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        warnings.warn(f"Could not load {path}: {e}")
        return None


def load_coverage_data(tel_dir, version, seed):
    """Load coverage dynamics CSV for a given version and seed."""
    fname = f"coverage_dynamics_{version}_seed{seed}.csv"
    return load_csv(os.path.join(tel_dir, fname))


def load_mutation_data(tel_dir, version, seed):
    """Load mutation attribution CSV for a given version and seed."""
    fname = f"mutation_attribution_{version}_seed{seed}.csv"
    return load_csv(os.path.join(tel_dir, fname))


def load_snapshot(tel_dir, version, seed, execs):
    """Load a single bitmap snapshot (65536-byte raw binary)."""
    snap_dir = os.path.join(tel_dir, f"snapshots_{version}_seed{seed}")
    snap_path = os.path.join(snap_dir, f"snapshot_{execs}.bin")
    if not os.path.exists(snap_path):
        return None
    try:
        data = np.fromfile(snap_path, dtype=np.uint8)
        if data.shape[0] != MAP_SIZE:
            warnings.warn(f"Snapshot {snap_path}: expected {MAP_SIZE} bytes, "
                          f"got {data.shape[0]}")
            return None
        return data
    except Exception as e:
        warnings.warn(f"Could not load snapshot {snap_path}: {e}")
        return None


def list_snapshot_execs(tel_dir, version, seed):
    """Return sorted list of exec counts for which snapshots exist."""
    snap_dir = os.path.join(tel_dir, f"snapshots_{version}_seed{seed}")
    if not os.path.isdir(snap_dir):
        return []
    execs = []
    for fname in os.listdir(snap_dir):
        if fname.startswith("snapshot_") and fname.endswith(".bin"):
            try:
                e = int(fname[len("snapshot_"):-len(".bin")])
                execs.append(e)
            except ValueError:
                pass
    return sorted(execs)


def load_all_data(tel_dir, bug_pairs, seeds):
    """Load all telemetry data into a structured dict.

    Returns:
        {pair: {
            "buggy": {seed: {"coverage": df, "mutation": df}},
            "fixed": {seed: {"coverage": df, "mutation": df}},
        }}
    """
    data = {}
    files_loaded = []

    for pair in bug_pairs:
        data[pair] = {"buggy": {}, "fixed": {}}
        for variant in ["buggy", "fixed"]:
            version = f"{pair}_{variant}"
            for seed in seeds:
                cov_df = load_coverage_data(tel_dir, version, seed)
                mut_df = load_mutation_data(tel_dir, version, seed)
                if cov_df is not None or mut_df is not None:
                    data[pair][variant][seed] = {}
                if cov_df is not None:
                    data[pair][variant][seed]["coverage"] = cov_df
                    fname = f"coverage_dynamics_{version}_seed{seed}.csv"
                    files_loaded.append(fname)
                if mut_df is not None:
                    data[pair][variant][seed]["mutation"] = mut_df
                    fname = f"mutation_attribution_{version}_seed{seed}.csv"
                    files_loaded.append(fname)

    return data, files_loaded


# ── Interpolation Helpers ────────────────────────────────────────────────────

def interpolate_to_common_execs(dfs_by_seed, x_col="total_execs",
                                y_col="total_edges", n_points=500):
    """Interpolate coverage curves from multiple seeds to a common x-axis.

    Returns:
        common_x: 1D array of exec counts
        values: 2D array (n_seeds, n_points)  — one row per seed
    """
    if not dfs_by_seed:
        return None, None

    # Determine common range: max of all minimums to min of all maximums
    x_ranges = []
    for seed, seed_data in dfs_by_seed.items():
        df = seed_data.get("coverage")
        if df is None or x_col not in df.columns:
            continue
        x = df[x_col].values
        if len(x) > 0:
            x_ranges.append((x.min(), x.max()))

    if not x_ranges:
        return None, None

    x_lo = max(r[0] for r in x_ranges)
    x_hi = min(r[1] for r in x_ranges)
    if x_lo >= x_hi:
        # Ranges don't overlap; use widest range for partial coverage
        x_lo = min(r[0] for r in x_ranges)
        x_hi = max(r[1] for r in x_ranges)

    common_x = np.linspace(x_lo, x_hi, n_points)
    rows = []

    for seed, seed_data in sorted(dfs_by_seed.items()):
        df = seed_data.get("coverage")
        if df is None or x_col not in df.columns or y_col not in df.columns:
            continue
        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)
        # Deduplicate x values (take last occurrence)
        _, unique_idx = np.unique(x, return_index=True)
        x = x[unique_idx]
        y = y[unique_idx]
        if len(x) < 2:
            continue
        interp_y = np.interp(common_x, x, y)
        rows.append(interp_y)

    if not rows:
        return None, None

    return common_x, np.array(rows)


# ── Analysis 1: Coverage Trajectory Comparison ───────────────────────────────

def compute_divergence_point(common_x, buggy_values, fixed_values):
    """Find the exec count where buggy and fixed curves separate by > 1 std dev.

    Uses the pooled standard deviation at each point. Returns the exec count
    or None if no divergence is found.
    """
    if buggy_values is None or fixed_values is None:
        return None
    if buggy_values.shape[0] < 2 or fixed_values.shape[0] < 2:
        return None

    buggy_mean = buggy_values.mean(axis=0)
    fixed_mean = fixed_values.mean(axis=0)
    buggy_std = buggy_values.std(axis=0, ddof=1) if buggy_values.shape[0] > 1 else np.zeros_like(buggy_mean)
    fixed_std = fixed_values.std(axis=0, ddof=1) if fixed_values.shape[0] > 1 else np.zeros_like(fixed_mean)

    # Pooled standard deviation
    n_b = buggy_values.shape[0]
    n_f = fixed_values.shape[0]
    pooled_std = np.sqrt(((n_b - 1) * buggy_std**2 + (n_f - 1) * fixed_std**2)
                         / max(1, n_b + n_f - 2))

    diff = np.abs(buggy_mean - fixed_mean)
    threshold = pooled_std

    # Avoid division by zero: only consider points where pooled_std > 0
    diverged = np.zeros(len(common_x), dtype=bool)
    nonzero = pooled_std > 0
    diverged[nonzero] = diff[nonzero] > threshold[nonzero]

    # Find first sustained divergence (at least 5 consecutive points)
    run_length = 0
    for i in range(len(diverged)):
        if diverged[i]:
            run_length += 1
            if run_length >= 5:
                return int(common_x[i - 4])
        else:
            run_length = 0

    # Fallback: first single point of divergence
    idx = np.where(diverged)[0]
    if len(idx) > 0:
        return int(common_x[idx[0]])

    return None


def analysis_coverage_trajectory(data, tel_dir, output_dir, bug_pairs, seeds):
    """Produce coverage trajectory plots and divergence analysis."""
    print("[1/6] Coverage trajectory comparison...")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    results = {}

    for pair in bug_pairs:
        buggy_data = data[pair]["buggy"]
        fixed_data = data[pair]["fixed"]

        if not buggy_data and not fixed_data:
            print(f"  {pair}: no data, skipping")
            continue

        common_x_b, buggy_vals = interpolate_to_common_execs(buggy_data)
        common_x_f, fixed_vals = interpolate_to_common_execs(fixed_data)

        # Use the same x-axis for both: re-interpolate to the intersection
        if common_x_b is not None and common_x_f is not None:
            x_lo = max(common_x_b[0], common_x_f[0])
            x_hi = min(common_x_b[-1], common_x_f[-1])
            if x_lo < x_hi:
                n_pts = 500
                common_x = np.linspace(x_lo, x_hi, n_pts)
                buggy_interp = np.array([np.interp(common_x, common_x_b, row)
                                         for row in buggy_vals])
                fixed_interp = np.array([np.interp(common_x, common_x_f, row)
                                         for row in fixed_vals])
            else:
                common_x = common_x_b
                buggy_interp = buggy_vals
                fixed_interp = fixed_vals
        elif common_x_b is not None:
            common_x = common_x_b
            buggy_interp = buggy_vals
            fixed_interp = None
        elif common_x_f is not None:
            common_x = common_x_f
            buggy_interp = None
            fixed_interp = fixed_vals
        else:
            print(f"  {pair}: insufficient coverage data")
            continue

        # Compute divergence
        div_point = compute_divergence_point(common_x, buggy_interp, fixed_interp)

        pair_result = {
            "buggy_seeds": len(buggy_data),
            "fixed_seeds": len(fixed_data),
            "divergence_exec": div_point,
        }

        if buggy_interp is not None:
            pair_result["buggy_final_mean"] = float(buggy_interp[:, -1].mean())
            pair_result["buggy_final_std"] = float(buggy_interp[:, -1].std())
        if fixed_interp is not None:
            pair_result["fixed_final_mean"] = float(fixed_interp[:, -1].mean())
            pair_result["fixed_final_std"] = float(fixed_interp[:, -1].std())

        results[pair] = pair_result
        print(f"  {pair}: buggy_seeds={pair_result['buggy_seeds']}, "
              f"fixed_seeds={pair_result['fixed_seeds']}, "
              f"divergence_exec={div_point}")

        # Plot
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(10, 6))

            if buggy_interp is not None:
                b_mean = buggy_interp.mean(axis=0)
                b_std = buggy_interp.std(axis=0)
                ax.plot(common_x, b_mean, label=f"{pair}_buggy (n={buggy_interp.shape[0]})",
                        **BUGGY_STYLE)
                ax.fill_between(common_x, b_mean - b_std, b_mean + b_std,
                                alpha=0.2, color=BUGGY_COLOR)

            if fixed_interp is not None:
                f_mean = fixed_interp.mean(axis=0)
                f_std = fixed_interp.std(axis=0)
                ax.plot(common_x, f_mean, label=f"{pair}_fixed (n={fixed_interp.shape[0]})",
                        **FIXED_STYLE)
                ax.fill_between(common_x, f_mean - f_std, f_mean + f_std,
                                alpha=0.2, color=FIXED_COLOR)

            if div_point is not None:
                ax.axvline(x=div_point, color="gray", linestyle="--", alpha=0.7,
                           label=f"Divergence @ {div_point:,} execs")

            ax.set_xlabel("Total Executions")
            ax.set_ylabel("Total Edges Discovered")
            ax.set_title(f"Coverage Trajectory: {pair} (buggy vs fixed)")
            ax.legend(loc="lower right")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            plot_path = os.path.join(plots_dir, f"coverage_trajectory_{pair}.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {plot_path}")

    return results


# ── Analysis 2: Differential Edge Analysis ───────────────────────────────────

def analysis_differential_edges(data, tel_dir, output_dir, bug_pairs, seeds):
    """Compare bitmap snapshots to find edges unique to buggy or fixed."""
    print("[2/6] Differential edge analysis...")
    plots_dir = os.path.join(output_dir, "plots")
    diff_dir = os.path.join(output_dir, "differential_edges")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)

    results = {}

    for pair in bug_pairs:
        buggy_ver = f"{pair}_buggy"
        fixed_ver = f"{pair}_fixed"

        # Collect all snapshot exec counts across seeds for each variant
        buggy_execs_by_seed = {}
        fixed_execs_by_seed = {}
        for seed in seeds:
            be = list_snapshot_execs(tel_dir, buggy_ver, seed)
            if be:
                buggy_execs_by_seed[seed] = set(be)
            fe = list_snapshot_execs(tel_dir, fixed_ver, seed)
            if fe:
                fixed_execs_by_seed[seed] = set(fe)

        if not buggy_execs_by_seed or not fixed_execs_by_seed:
            print(f"  {pair}: no snapshots found, skipping")
            continue

        # Find exec counts that appear in at least one seed of both variants
        buggy_all = set()
        for s in buggy_execs_by_seed.values():
            buggy_all |= s
        fixed_all = set()
        for s in fixed_execs_by_seed.values():
            fixed_all |= s
        matched_execs = sorted(buggy_all & fixed_all)

        if not matched_execs:
            print(f"  {pair}: no matching snapshot exec counts, skipping")
            continue

        print(f"  {pair}: {len(matched_execs)} matched snapshot timepoints")

        buggy_only_counts = []
        fixed_only_counts = []
        diff_records = []

        for execs in matched_execs:
            # Load all available seeds, union the edges
            buggy_union = np.zeros(MAP_SIZE, dtype=bool)
            fixed_union = np.zeros(MAP_SIZE, dtype=bool)

            buggy_seed_count = 0
            fixed_seed_count = 0

            for seed in seeds:
                snap = load_snapshot(tel_dir, buggy_ver, seed, execs)
                if snap is not None:
                    buggy_union |= (snap > 0)
                    buggy_seed_count += 1

            for seed in seeds:
                snap = load_snapshot(tel_dir, fixed_ver, seed, execs)
                if snap is not None:
                    fixed_union |= (snap > 0)
                    fixed_seed_count += 1

            if buggy_seed_count == 0 or fixed_seed_count == 0:
                continue

            b_only = int(np.sum(buggy_union & ~fixed_union))
            f_only = int(np.sum(fixed_union & ~buggy_union))
            shared = int(np.sum(buggy_union & fixed_union))

            buggy_only_counts.append(b_only)
            fixed_only_counts.append(f_only)

            record = {
                "total_execs": execs,
                "buggy_total_edges": int(np.sum(buggy_union)),
                "fixed_total_edges": int(np.sum(fixed_union)),
                "buggy_only": b_only,
                "fixed_only": f_only,
                "shared": shared,
                "buggy_seeds_loaded": buggy_seed_count,
                "fixed_seeds_loaded": fixed_seed_count,
            }
            diff_records.append(record)

            # Save per-timepoint JSON
            json_path = os.path.join(diff_dir, f"{pair}_{execs}.json")
            with open(json_path, "w") as f:
                json.dump(record, f, indent=2)

        if not diff_records:
            print(f"  {pair}: no valid differential comparisons")
            continue

        results[pair] = diff_records
        print(f"  {pair}: {len(diff_records)} timepoints analyzed, "
              f"final buggy_only={buggy_only_counts[-1]}, "
              f"fixed_only={fixed_only_counts[-1]}")

        # Plot differential edges over time
        if HAS_MPL and diff_records:
            fig, ax = plt.subplots(figsize=(10, 6))
            execs_arr = [r["total_execs"] for r in diff_records]
            ax.plot(execs_arr, buggy_only_counts, label="Buggy-only edges",
                    color=BUGGY_COLOR, linewidth=2)
            ax.plot(execs_arr, fixed_only_counts, label="Fixed-only edges",
                    color=FIXED_COLOR, linewidth=2)
            ax.fill_between(execs_arr, buggy_only_counts, alpha=0.15,
                            color=BUGGY_COLOR)
            ax.fill_between(execs_arr, fixed_only_counts, alpha=0.15,
                            color=FIXED_COLOR)

            ax.set_xlabel("Total Executions")
            ax.set_ylabel("Differential Edge Count")
            ax.set_title(f"Differential Edges: {pair} (buggy-only vs fixed-only)")
            ax.legend(loc="upper left")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            plot_path = os.path.join(plots_dir, f"differential_edges_{pair}.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {plot_path}")

    return results


# ── Analysis 3: Mutation Effectiveness ───────────────────────────────────────

def compute_mutation_effectiveness(mut_dfs_by_seed, n_windows=20):
    """Compute per-mutation coverage gain rate across time windows.

    Returns:
        windows: list of (execs_start, execs_end) tuples
        effectiveness: array of shape (n_mutations, n_windows)
            Each cell = total new_edges / total count for that mutation in that window
    """
    if not mut_dfs_by_seed:
        return None, None

    # Concatenate all seed data
    all_dfs = []
    for seed, seed_data in mut_dfs_by_seed.items():
        df = seed_data.get("mutation")
        if df is not None:
            df_copy = df.copy()
            df_copy["_seed"] = seed
            all_dfs.append(df_copy)

    if not all_dfs:
        return None, None

    combined = pd.concat(all_dfs, ignore_index=True)
    if "total_execs" not in combined.columns:
        return None, None

    # Determine window boundaries
    x_min = combined["total_execs"].min()
    x_max = combined["total_execs"].max()
    if x_min >= x_max:
        return None, None

    boundaries = np.linspace(x_min, x_max, n_windows + 1)
    windows = [(boundaries[i], boundaries[i + 1]) for i in range(n_windows)]

    effectiveness = np.full((N_MUTATIONS, n_windows), np.nan)

    for w_idx, (w_start, w_end) in enumerate(windows):
        mask = (combined["total_execs"] >= w_start) & (combined["total_execs"] < w_end)
        window_df = combined[mask]
        if window_df.empty:
            continue

        for m in range(N_MUTATIONS):
            count_col = f"mut_{m:02d}_count"
            edges_col = f"mut_{m:02d}_new_edges"
            if count_col in window_df.columns and edges_col in window_df.columns:
                total_count = window_df[count_col].sum()
                total_edges = window_df[edges_col].sum()
                if total_count > 0:
                    effectiveness[m, w_idx] = total_edges / total_count

    return windows, effectiveness


def analysis_mutation_effectiveness(data, tel_dir, output_dir, bug_pairs,
                                    seeds, divergence_results):
    """Compare mutation effectiveness between buggy and fixed versions."""
    print("[3/6] Mutation effectiveness analysis...")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    results = {}

    for pair in bug_pairs:
        buggy_data = data[pair]["buggy"]
        fixed_data = data[pair]["fixed"]

        b_windows, b_eff = compute_mutation_effectiveness(buggy_data)
        f_windows, f_eff = compute_mutation_effectiveness(fixed_data)

        if b_eff is None and f_eff is None:
            print(f"  {pair}: no mutation data, skipping")
            continue

        pair_result = {"mutations": {}}

        # Identify mutations with highest buggy/fixed ratio near divergence
        div_exec = (divergence_results.get(pair, {}).get("divergence_exec")
                    if divergence_results else None)

        if b_eff is not None and f_eff is not None and b_windows and f_windows:
            # Find the window nearest the divergence point (or midpoint)
            target_exec = div_exec if div_exec else (b_windows[0][0] + b_windows[-1][1]) / 2

            # Find nearest window index in buggy windows
            b_mid = [(w[0] + w[1]) / 2 for w in b_windows]
            b_div_idx = int(np.argmin([abs(m - target_exec) for m in b_mid]))
            f_mid = [(w[0] + w[1]) / 2 for w in f_windows]
            f_div_idx = int(np.argmin([abs(m - target_exec) for m in f_mid]))

            for m in range(N_MUTATIONS):
                b_val = b_eff[m, b_div_idx] if not np.isnan(b_eff[m, b_div_idx]) else 0
                f_val = f_eff[m, f_div_idx] if not np.isnan(f_eff[m, f_div_idx]) else 0
                ratio = b_val / f_val if f_val > 0 else (float("inf") if b_val > 0 else 1.0)

                pair_result["mutations"][MUTATION_NAMES.get(m, f"mut_{m:02d}")] = {
                    "buggy_gain_rate": float(b_val),
                    "fixed_gain_rate": float(f_val),
                    "ratio_near_divergence": float(ratio) if np.isfinite(ratio) else None,
                }

            # Sort by ratio
            top_divergent = sorted(
                pair_result["mutations"].items(),
                key=lambda kv: abs((kv[1].get("ratio_near_divergence") or 1.0) - 1.0),
                reverse=True,
            )[:10]
            pair_result["top_divergent_mutations"] = [
                {"name": name, **vals} for name, vals in top_divergent
            ]

            print(f"  {pair}: top divergent mutation = {top_divergent[0][0]}")

        results[pair] = pair_result

        # Plot: side-by-side heatmaps
        if HAS_MPL and (b_eff is not None or f_eff is not None):
            n_show = min(N_MUTATIONS, 47)
            fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

            for ax, eff, windows, title, cmap_color in [
                (axes[0], b_eff, b_windows, f"{pair}_buggy", "Reds"),
                (axes[1], f_eff, f_windows, f"{pair}_fixed", "Blues"),
            ]:
                if eff is None or windows is None:
                    ax.set_title(f"{title}: no data")
                    continue

                # Replace NaN with 0 for display
                display = np.nan_to_num(eff[:n_show, :], nan=0.0)
                # Cap extreme values for better visualization
                vmax = np.percentile(display[display > 0], 95) if np.any(display > 0) else 1.0

                im = ax.imshow(display, aspect="auto", cmap=cmap_color,
                               vmin=0, vmax=max(vmax, 1e-8),
                               interpolation="nearest")

                # Y-axis: mutation names
                ax.set_yticks(range(n_show))
                ax.set_yticklabels([MUTATION_NAMES.get(i, f"mut_{i:02d}")
                                    for i in range(n_show)], fontsize=6)

                # X-axis: time windows
                n_w = len(windows)
                tick_positions = np.linspace(0, n_w - 1, min(6, n_w)).astype(int)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([f"{windows[t][0]/1e3:.0f}K"
                                    for t in tick_positions], fontsize=7)

                ax.set_xlabel("Executions")
                ax.set_title(f"Mutation Effectiveness: {title}")
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                             label="new_edges / count")

            fig.suptitle(f"Mutation Effectiveness Comparison: {pair}", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            plot_path = os.path.join(plots_dir, f"mutation_effectiveness_{pair}.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {plot_path}")

    return results


# ── Analysis 4: Coverage Landscape Features ──────────────────────────────────

def analysis_landscape_features(data, tel_dir, output_dir, bug_pairs, seeds):
    """Compare edge heat ratios, entropy, and coverage velocity."""
    print("[4/6] Coverage landscape features...")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    results = {}

    feature_cols = {
        "hot_ratio": None,     # computed
        "warm_ratio": None,    # computed
        "cool_ratio": None,    # computed
        "cold_ratio": None,    # computed
        "edge_entropy": "edge_entropy",
        "coverage_velocity": None,  # computed (first derivative)
    }

    for pair in bug_pairs:
        pair_result = {}

        for variant in ["buggy", "fixed"]:
            variant_data = data[pair][variant]
            common_x, vals = interpolate_to_common_execs(
                variant_data, y_col="total_edges")
            if common_x is None:
                continue

            # Also interpolate landscape features
            feature_arrays = {}

            for feat_name in ["edge_entropy", "hot_edges", "warm_edges",
                              "cool_edges", "cold_edges"]:
                _, feat_vals = interpolate_to_common_execs(
                    variant_data, y_col=feat_name)
                if feat_vals is not None:
                    feature_arrays[feat_name] = feat_vals

            # Compute ratios
            for heat in ["hot", "warm", "cool", "cold"]:
                heat_key = f"{heat}_edges"
                if heat_key in feature_arrays:
                    total = np.zeros_like(feature_arrays[heat_key])
                    for h in ["hot_edges", "warm_edges", "cool_edges", "cold_edges"]:
                        if h in feature_arrays:
                            total += feature_arrays[h]
                    total = np.where(total > 0, total, 1.0)
                    feature_arrays[f"{heat}_ratio"] = feature_arrays[heat_key] / total

            # Compute coverage velocity (first derivative of total_edges)
            if vals is not None and common_x is not None and len(common_x) > 1:
                dx = np.diff(common_x)
                velocity_rows = []
                for row in vals:
                    dy = np.diff(row)
                    vel = dy / np.where(dx > 0, dx, 1.0)
                    vel = np.concatenate([[vel[0]], vel])  # pad to same length
                    velocity_rows.append(vel)
                feature_arrays["coverage_velocity"] = np.array(velocity_rows)

            pair_result[variant] = {
                "common_x": common_x,
                "features": feature_arrays,
            }

        results[pair] = pair_result

        # Plot: 2x2 grid of landscape features
        if HAS_MPL:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            plot_specs = [
                ("hot_ratio", "Hot Edge Ratio (>128 hits)", axes[0, 0]),
                ("edge_entropy", "Edge Entropy (bits)", axes[0, 1]),
                ("coverage_velocity", "Coverage Velocity (edges/exec)", axes[1, 0]),
                ("warm_ratio", "Warm Edge Ratio (8-128 hits)", axes[1, 1]),
            ]

            for feat_name, ylabel, ax in plot_specs:
                for variant, style, color in [
                    ("buggy", BUGGY_STYLE, BUGGY_COLOR),
                    ("fixed", FIXED_STYLE, FIXED_COLOR),
                ]:
                    vdata = pair_result.get(variant)
                    if vdata is None:
                        continue
                    cx = vdata["common_x"]
                    feat = vdata["features"].get(feat_name)
                    if cx is None or feat is None:
                        continue
                    mean = feat.mean(axis=0)
                    std = feat.std(axis=0)
                    ax.plot(cx, mean, label=f"{pair}_{variant}", **style)
                    ax.fill_between(cx, mean - std, mean + std,
                                    alpha=0.15, color=color)

                ax.set_xlabel("Total Executions")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                    lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

            fig.suptitle(f"Coverage Landscape Features: {pair}", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            plot_path = os.path.join(plots_dir, f"landscape_features_{pair}.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {plot_path}")

    return results


# ── Analysis 5: Feature Importance Report ────────────────────────────────────

def vargha_delaney_a12(group1, group2):
    """Compute Vargha-Delaney A12 effect size.

    A12 > 0.5 means group1 tends to produce larger values.
    A12 = 0.5 means no effect, 0.56 = small, 0.64 = medium, 0.71 = large.
    """
    m = len(group1)
    n = len(group2)
    if m == 0 or n == 0:
        return 0.5

    # Count: how often a value in group1 exceeds one in group2
    r = 0.0
    for a in group1:
        for b in group2:
            if a > b:
                r += 1.0
            elif a == b:
                r += 0.5
    return r / (m * n)


def analysis_feature_importance(data, tel_dir, output_dir, bug_pairs, seeds):
    """Mann-Whitney U + A12 for all features at matched timepoints."""
    print("[5/6] Feature importance report...")

    if not HAS_SCIPY:
        print("  scipy not available, skipping statistical tests")
        return {}

    results = {}

    # Features to test from coverage dynamics CSV
    test_features = [
        "total_edges", "edge_discovery_rate", "crashes_total",
        "avg_exec_time_us", "corpus_size", "hot_edges", "warm_edges",
        "cool_edges", "cold_edges", "edge_entropy",
        "edge_hit_mean", "edge_hit_std", "edge_hit_max",
    ]

    for pair in bug_pairs:
        buggy_data = data[pair]["buggy"]
        fixed_data = data[pair]["fixed"]

        buggy_seeds = [s for s, d in buggy_data.items()
                       if "coverage" in d]
        fixed_seeds = [s for s, d in fixed_data.items()
                       if "coverage" in d]

        if len(buggy_seeds) < 2 or len(fixed_seeds) < 2:
            print(f"  {pair}: need >= 2 seeds per variant for statistical tests "
                  f"(buggy={len(buggy_seeds)}, fixed={len(fixed_seeds)}), skipping")
            continue

        # Find common exec range
        all_max_execs = []
        for s in buggy_seeds:
            df = buggy_data[s]["coverage"]
            all_max_execs.append(df["total_execs"].max())
        for s in fixed_seeds:
            df = fixed_data[s]["coverage"]
            all_max_execs.append(df["total_execs"].max())

        # Test at multiple timepoints (10%, 25%, 50%, 75%, 90% of campaign)
        max_common = min(all_max_execs)
        timepoints = [max_common * frac for frac in [0.10, 0.25, 0.50, 0.75, 0.90]]

        n_tests = len(test_features) * len(timepoints)
        alpha = 0.05
        bonferroni_alpha = alpha / max(n_tests, 1)

        pair_results = {
            "alpha": alpha,
            "bonferroni_alpha": bonferroni_alpha,
            "n_tests": n_tests,
            "timepoints": {},
        }

        for tp in timepoints:
            tp_label = f"{tp:.0f}"
            tp_results = []

            for feat in test_features:
                # Extract feature value at this timepoint from each seed
                buggy_vals = []
                for s in buggy_seeds:
                    df = buggy_data[s]["coverage"]
                    if feat not in df.columns:
                        continue
                    # Find the row closest to this exec count
                    idx = (df["total_execs"] - tp).abs().idxmin()
                    buggy_vals.append(float(df.loc[idx, feat]))

                fixed_vals = []
                for s in fixed_seeds:
                    df = fixed_data[s]["coverage"]
                    if feat not in df.columns:
                        continue
                    idx = (df["total_execs"] - tp).abs().idxmin()
                    fixed_vals.append(float(df.loc[idx, feat]))

                if len(buggy_vals) < 2 or len(fixed_vals) < 2:
                    continue

                # Mann-Whitney U test (two-sided)
                try:
                    stat, pval = mannwhitneyu(buggy_vals, fixed_vals,
                                              alternative="two-sided")
                except ValueError:
                    # All values identical
                    stat, pval = 0.0, 1.0

                a12 = vargha_delaney_a12(buggy_vals, fixed_vals)

                significant = pval < bonferroni_alpha
                effect_label = "negligible"
                a12_dev = abs(a12 - 0.5)
                if a12_dev >= 0.21:
                    effect_label = "large"
                elif a12_dev >= 0.14:
                    effect_label = "medium"
                elif a12_dev >= 0.06:
                    effect_label = "small"

                tp_results.append({
                    "feature": feat,
                    "buggy_mean": float(np.mean(buggy_vals)),
                    "fixed_mean": float(np.mean(fixed_vals)),
                    "U_statistic": float(stat),
                    "p_value": float(pval),
                    "significant_bonferroni": bool(significant),
                    "A12": float(a12),
                    "effect_size": effect_label,
                    "direction": "buggy > fixed" if a12 > 0.5 else "fixed > buggy",
                })

            # Rank by discriminative power (|A12 - 0.5|)
            tp_results.sort(key=lambda r: abs(r["A12"] - 0.5), reverse=True)
            pair_results["timepoints"][tp_label] = tp_results

        results[pair] = pair_results
        n_sig = sum(1 for tp in pair_results["timepoints"].values()
                    for r in tp if r["significant_bonferroni"])
        n_large = sum(1 for tp in pair_results["timepoints"].values()
                      for r in tp if r["effect_size"] in ("large", "medium"))
        print(f"  {pair}: {n_sig} Bonferroni-significant, {n_large} with medium/large effect size "
              f"(Bonferroni alpha={bonferroni_alpha:.6f})")
        if n_sig == 0 and n_large > 0:
            print(f"  NOTE: With only {len(buggy_seeds)} seeds, Bonferroni is very strict. "
                  f"Ranking by effect size (A12) instead.")

    # Save report
    report_path = os.path.join(output_dir, "feature_importance_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {report_path}")

    return results


# ── Analysis 6: Documentation Generation ────────────────────────────────────

def generate_methodology(output_dir, files_loaded, bug_pairs, seeds):
    """Write ANALYSIS_METHODOLOGY.md documenting the analysis pipeline."""
    print("[6/6] Generating documentation...")

    md_path = os.path.join(output_dir, "ANALYSIS_METHODOLOGY.md")
    lines = [
        "# Differential Analysis Methodology",
        "",
        "## Data Sources",
        "",
        "### Files Loaded",
        "",
    ]
    for f in sorted(files_loaded):
        lines.append(f"- `{f}`")
    lines.append("")

    lines.extend([
        "### Bug Pairs Analyzed",
        "",
    ])
    for pair in bug_pairs:
        lines.append(f"- **{pair}**: `{pair}_buggy` vs `{pair}_fixed`")
    lines.append("")

    lines.extend([
        f"### Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "## Statistical Tests",
        "",
        "### Mann-Whitney U Test",
        "",
        "- Non-parametric test for comparing two independent samples",
        "- Null hypothesis: the distributions of buggy and fixed values are equal",
        "- Alternative: two-sided",
        f"- Nominal alpha = 0.05",
        "- Bonferroni correction applied: alpha_corrected = 0.05 / (n_features x n_timepoints)",
        "",
        "### Vargha-Delaney A12 Effect Size",
        "",
        "- Measures the probability that a randomly chosen value from group A",
        "  exceeds a randomly chosen value from group B",
        "- A12 = 0.5: no effect",
        "- |A12 - 0.5| >= 0.06: small effect",
        "- |A12 - 0.5| >= 0.14: medium effect",
        "- |A12 - 0.5| >= 0.21: large effect",
        "",
        "## Divergence Detection Algorithm",
        "",
        "1. Interpolate coverage curves from all seeds to a common execution axis",
        "   (500 equally spaced points across the common range).",
        "2. Compute mean and standard deviation across seeds for each variant.",
        "3. Compute pooled standard deviation at each point:",
        "   `pooled_std = sqrt(((n_b-1)*std_b^2 + (n_f-1)*std_f^2) / (n_b+n_f-2))`",
        "4. A point is considered diverged when `|mean_buggy - mean_fixed| > pooled_std`.",
        "5. The divergence point is the start of the first run of >= 5 consecutive",
        "   diverged points. If no such run exists, the first single diverged point",
        "   is reported.",
        "",
        "## Feature Computation Formulas",
        "",
        "### Coverage Velocity",
        "First difference of total_edges with respect to total_execs:",
        "`velocity[i] = (edges[i] - edges[i-1]) / (execs[i] - execs[i-1])`",
        "",
        "### Edge Heat Ratios",
        "From cumulative bitmap snapshots (65536-byte maps):",
        "- **hot**: `count(map[i] > 128) / count(map[i] > 0)`",
        "- **warm**: `count(8 <= map[i] <= 128) / count(map[i] > 0)`",
        "- **cool**: `count(1 <= map[i] <= 7) / count(map[i] > 0)`",
        "- **cold**: `count(map[i] == 0) / MAP_SIZE`",
        "",
        "Note: ratios from CSV use all 65536 entries as denominator (hot+warm+cool+cold).",
        "",
        "### Edge Entropy",
        "Shannon entropy over 8 power-of-2 hit-count bins (1, 2, 4, 8, 16, 32, 64, 128+):",
        "`entropy = -sum(p_i * log2(p_i))` where `p_i = bin_count_i / nonzero_edges`",
        "",
        "### Mutation Effectiveness (Coverage Gain Rate)",
        "`gain_rate = sum(new_edges) / sum(count)` for each mutation across a time window.",
        "",
        "### Differential Edges",
        "At matched exec counts, load bitmap snapshots from all available seeds.",
        "Union the non-zero bytes across seeds. Then:",
        "- `buggy_only = count(buggy_union & ~fixed_union)`",
        "- `fixed_only = count(fixed_union & ~buggy_union)`",
        "- `shared = count(buggy_union & fixed_union)`",
        "",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {md_path}")
    return md_path


def generate_summary(output_dir, coverage_results, differential_results,
                     mutation_results, importance_results, bug_pairs):
    """Write summary.md with human-readable findings."""
    lines = [
        "# Differential Fuzzing Analysis Summary",
        "",
    ]

    for pair in bug_pairs:
        lines.extend([
            f"## Bug Pair: {pair}",
            "",
        ])

        # Coverage trajectory
        cov = coverage_results.get(pair, {})
        if cov:
            lines.append("### Coverage Trajectory")
            lines.append("")
            b_mean = cov.get("buggy_final_mean")
            f_mean = cov.get("fixed_final_mean")
            if b_mean is not None:
                lines.append(f"- Buggy final coverage: {b_mean:.1f} "
                             f"(+/- {cov.get('buggy_final_std', 0):.1f}) edges "
                             f"across {cov['buggy_seeds']} seeds")
            if f_mean is not None:
                lines.append(f"- Fixed final coverage: {f_mean:.1f} "
                             f"(+/- {cov.get('fixed_final_std', 0):.1f}) edges "
                             f"across {cov['fixed_seeds']} seeds")
            div = cov.get("divergence_exec")
            if div is not None:
                lines.append(f"- **Divergence point**: {div:,} executions")
            else:
                lines.append("- No statistically significant divergence detected")
            if b_mean is not None and f_mean is not None:
                diff = b_mean - f_mean
                direction = "buggy > fixed" if diff > 0 else "fixed > buggy"
                lines.append(f"- Final coverage difference: {abs(diff):.1f} edges ({direction})")
            lines.append("")

        # Differential edges
        diff_data = differential_results.get(pair, [])
        if diff_data:
            last = diff_data[-1]
            lines.append("### Differential Edges (at final timepoint)")
            lines.append("")
            lines.append(f"- Buggy-only edges: {last['buggy_only']}")
            lines.append(f"- Fixed-only edges: {last['fixed_only']}")
            lines.append(f"- Shared edges: {last['shared']}")
            lines.append(f"- Buggy total: {last['buggy_total_edges']}, "
                         f"Fixed total: {last['fixed_total_edges']}")
            lines.append("")

        # Mutation effectiveness
        mut = mutation_results.get(pair, {})
        top_muts = mut.get("top_divergent_mutations", [])
        if top_muts:
            lines.append("### Most Differentially Effective Mutations")
            lines.append("")
            lines.append("Mutations with the largest buggy/fixed effectiveness "
                         "ratio near the divergence point:")
            lines.append("")
            for i, m in enumerate(top_muts[:5], 1):
                ratio = m.get("ratio_near_divergence")
                ratio_str = f"{ratio:.3f}" if ratio is not None else "inf"
                lines.append(f"{i}. **{m['name']}**: buggy gain rate = "
                             f"{m['buggy_gain_rate']:.6f}, "
                             f"fixed gain rate = {m['fixed_gain_rate']:.6f}, "
                             f"ratio = {ratio_str}")
            lines.append("")

        # Feature importance
        fi = importance_results.get(pair, {})
        timepoints = fi.get("timepoints", {})
        if timepoints:
            lines.append("### Feature Importance (Mann-Whitney U)")
            lines.append("")
            lines.append(f"- Bonferroni-corrected alpha: {fi.get('bonferroni_alpha', 'N/A')}")
            lines.append("")

            # Show the most discriminative features across all timepoints
            all_features = defaultdict(list)
            for tp_label, tp_results in timepoints.items():
                for r in tp_results:
                    all_features[r["feature"]].append(abs(r["A12"] - 0.5))

            ranked = sorted(all_features.items(),
                            key=lambda kv: np.mean(kv[1]), reverse=True)
            lines.append("Top discriminative features (by mean |A12 - 0.5| "
                         "across timepoints):")
            lines.append("")
            for feat, deviations in ranked[:5]:
                mean_dev = np.mean(deviations)
                lines.append(f"- **{feat}**: mean |A12 - 0.5| = {mean_dev:.3f}")
            lines.append("")

    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {summary_path}")
    return summary_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Differential analysis of buggy vs fixed fuzzing telemetry")
    parser.add_argument("--telemetry-dir",
                        default="experiments/differential/telemetry",
                        help="Directory containing telemetry CSVs and snapshots")
    parser.add_argument("--output-dir",
                        default="experiments/differential/analysis",
                        help="Directory for analysis output")
    parser.add_argument("--bug-pairs", type=str, default=None,
                        help="Comma-separated bug pair prefixes "
                             "(default: xml005,xml017)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed numbers (default: 1,2,3)")
    args = parser.parse_args()

    tel_dir = args.telemetry_dir
    output_dir = args.output_dir

    bug_pairs = (args.bug_pairs.split(",") if args.bug_pairs
                 else DEFAULT_BUG_PAIRS)
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else SEEDS)

    print("=" * 60)
    print("  Differential Fuzzing Analysis")
    print("=" * 60)
    print(f"  Telemetry dir: {os.path.abspath(tel_dir)}")
    print(f"  Output dir:    {os.path.abspath(output_dir)}")
    print(f"  Bug pairs:     {', '.join(bug_pairs)}")
    print(f"  Seeds:         {', '.join(str(s) for s in seeds)}")
    print()

    if not os.path.isdir(tel_dir):
        print(f"[-] Telemetry directory does not exist: {tel_dir}")
        print("    Run the telemetry campaign first:")
        print("    bash scripts/run_telemetry_campaign.sh")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading telemetry data...")
    data, files_loaded = load_all_data(tel_dir, bug_pairs, seeds)

    total_loaded = len(files_loaded)
    if total_loaded == 0:
        print("[-] No telemetry data found. Expected files like:")
        print("    coverage_dynamics_xml005_buggy_seed1.csv")
        print("    mutation_attribution_xml005_buggy_seed1.csv")
        sys.exit(1)

    print(f"  Loaded {total_loaded} files")
    for pair in bug_pairs:
        n_buggy = len(data[pair]["buggy"])
        n_fixed = len(data[pair]["fixed"])
        print(f"  {pair}: buggy={n_buggy} seeds, fixed={n_fixed} seeds")
    print()

    # ── Run analyses ─────────────────────────────────────────────────────
    coverage_results = analysis_coverage_trajectory(
        data, tel_dir, output_dir, bug_pairs, seeds)
    print()

    differential_results = analysis_differential_edges(
        data, tel_dir, output_dir, bug_pairs, seeds)
    print()

    mutation_results = analysis_mutation_effectiveness(
        data, tel_dir, output_dir, bug_pairs, seeds, coverage_results)
    print()

    landscape_results = analysis_landscape_features(
        data, tel_dir, output_dir, bug_pairs, seeds)
    print()

    importance_results = analysis_feature_importance(
        data, tel_dir, output_dir, bug_pairs, seeds)
    print()

    # ── Documentation ────────────────────────────────────────────────────
    generate_methodology(output_dir, files_loaded, bug_pairs, seeds)
    generate_summary(output_dir, coverage_results, differential_results,
                     mutation_results, importance_results, bug_pairs)

    print()
    print("=" * 60)
    print("  Analysis complete")
    print("=" * 60)
    print(f"  Output: {os.path.abspath(output_dir)}")
    print()

    # List output files
    for dirpath, _dirnames, filenames in os.walk(output_dir):
        for fname in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, fname), output_dir)
            print(f"    {rel}")


if __name__ == "__main__":
    main()
