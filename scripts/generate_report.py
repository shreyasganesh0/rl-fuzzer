#!/usr/bin/env python3
"""generate_report.py — Detailed statistical comparison of all experiment data.

Reads eval CSVs and fuzzer_stats from experiments/ and produces a comprehensive
report covering coverage, throughput, timing, action distributions, and
model-vs-baseline comparisons.

Usage:
    python3 scripts/generate_report.py [--exp-root experiments] [--out report.txt]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Configuration ────────────────────────────────────────────────────────────

BENCHMARKS = ["jsoncpp", "freetype2", "libxml2", "re2", "harfbuzz", "libpng"]
MODELS = ["m1_0", "m1_1", "m1_2"]
BASELINES = ["baseline", "baseline_time"]
ALL_AGENTS = MODELS + BASELINES
MILESTONES = ["500k", "1m", "2m", "10m"]
MILESTONE_STEPS = {"500k": 500_000, "1m": 1_000_000, "2m": 2_000_000, "10m": 10_000_000}

MODEL_LABELS = {
    "m1_0": "M1_0 (full-edge dist, 12-dim)",
    "m1_1": "M1_1 (visited-edge dist, 13-dim)",
    "m1_2": "M1_2 (visited + input buf, 64-dim)",
    "baseline": "Baseline (same steps)",
    "baseline_time": "Baseline (same time)",
}

W = 100  # report width


# ── Helpers ──────────────────────────────────────────────────────────────────

def hr(char="─"):
    return char * W


def heading(text, char="═"):
    pad = max(0, W - len(text) - 4)
    return f"{char * 2} {text} {char * pad}"


def load_eval_csv(path):
    """Load an eval CSV, return DataFrame or None."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None


def load_fuzzer_stats(path):
    """Parse AFL++ fuzzer_stats file into a dict."""
    if not os.path.exists(path):
        return None
    stats = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    v = float(v) if "." in v else int(v)
                except ValueError:
                    pass
                stats[k] = v
    return stats


def fmt_num(n, width=10):
    """Format number with commas."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—".rjust(width)
    if isinstance(n, float):
        return f"{n:,.2f}".rjust(width)
    return f"{n:,}".rjust(width)


def fmt_pct(n, width=8):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—".rjust(width)
    return f"{n:.1f}%".rjust(width)


def safe_div(a, b):
    if b == 0 or b is None:
        return None
    return a / b


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data(exp_root):
    """Load all experiment data into a structured dict."""
    data = {}
    for bench in BENCHMARKS:
        bench_dir = Path(exp_root) / bench
        if not bench_dir.exists():
            continue
        data[bench] = {"plots": {}, "milestones": {}, "fuzzer_stats": {}}

        # Full eval CSVs
        for agent in ALL_AGENTS:
            csv_path = bench_dir / "plots" / agent / f"rl_metrics_{agent}_eval.csv"
            df = load_eval_csv(csv_path)
            if df is not None:
                data[bench]["plots"][agent] = df

            stats_path = bench_dir / "plots" / agent / "fuzzer_stats_eval.txt"
            stats = load_fuzzer_stats(stats_path)
            if stats is not None:
                data[bench]["fuzzer_stats"][agent] = stats

        # Milestone CSVs
        for ms in MILESTONES:
            data[bench]["milestones"][ms] = {}
            for agent in ALL_AGENTS:
                csv_path = bench_dir / "milestones" / ms / agent / f"rl_metrics_{agent}_eval.csv"
                df = load_eval_csv(csv_path)
                if df is not None:
                    data[bench]["milestones"][ms][agent] = df

    return data


# ── Report Sections ──────────────────────────────────────────────────────────

def section_overview(data, lines):
    lines.append(heading("EXPERIMENT OVERVIEW"))
    lines.append("")
    lines.append(f"  Benchmarks: {', '.join(b for b in BENCHMARKS if b in data)}")
    lines.append(f"  Models:     {', '.join(MODELS)}")
    lines.append(f"  Baselines:  same-steps, same-time")
    lines.append(f"  Milestones: {', '.join(MILESTONES)}")
    lines.append("")

    # Per-benchmark data availability
    lines.append("  Data availability:")
    lines.append("")
    header = f"  {'Benchmark':<14}"
    for agent in ALL_AGENTS:
        label = agent.replace("baseline_time", "bl_time").replace("baseline", "bl_steps")
        header += f" {label:>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for bench in BENCHMARKS:
        if bench not in data:
            continue
        row = f"  {bench:<14}"
        for agent in ALL_AGENTS:
            if agent in data[bench]["plots"]:
                df = data[bench]["plots"][agent]
                max_step = int(df["step"].max())
                if max_step >= 9_000_000:
                    row += f" {'10M':>10}"
                elif max_step >= 1_800_000:
                    row += f" {'~{:.1f}M'.format(max_step/1e6):>10}"
                else:
                    row += f" {f'{max_step/1e3:.0f}K':>10}"
            else:
                row += f" {'—':>10}"
        lines.append(row)
    lines.append("")


def section_per_benchmark_detail(data, lines):
    lines.append(heading("PER-BENCHMARK DETAILED STATISTICS"))
    lines.append("")

    for bench in BENCHMARKS:
        if bench not in data:
            continue
        lines.append(hr("━"))
        lines.append(f"  BENCHMARK: {bench.upper()}")
        lines.append(hr("━"))
        lines.append("")

        # ── Coverage summary table ───────────────────────────────────
        lines.append("  Coverage (edges discovered):")
        lines.append("")
        header = f"    {'Agent':<18} {'Final':>10} {'@500K':>10} {'@1M':>10} {'@2M':>10} {'@10M':>10} {'Time(s)':>10} {'Exec/s':>10}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))

        for agent in ALL_AGENTS:
            df_full = data[bench]["plots"].get(agent)
            stats = data[bench]["fuzzer_stats"].get(agent)

            final_cov = None
            elapsed = None
            if df_full is not None:
                final_cov = int(df_full["coverage"].iloc[-1])
                if "elapsed_seconds" in df_full.columns:
                    elapsed = float(df_full["elapsed_seconds"].iloc[-1])

            execs_per_sec = None
            if stats and "execs_per_sec" in stats:
                execs_per_sec = stats["execs_per_sec"]

            ms_covs = {}
            for ms in MILESTONES:
                ms_df = data[bench]["milestones"].get(ms, {}).get(agent)
                if ms_df is not None:
                    ms_covs[ms] = int(ms_df["coverage"].iloc[-1])

            label = agent.replace("baseline_time", "bl_time").replace("baseline", "bl_steps")
            row = f"    {label:<18}"
            row += fmt_num(final_cov)
            for ms in MILESTONES:
                row += fmt_num(ms_covs.get(ms))
            row += fmt_num(elapsed)
            row += fmt_num(execs_per_sec)
            lines.append(row)

        lines.append("")

        # ── Coverage over time analysis ──────────────────────────────
        lines.append("  Coverage dynamics:")
        lines.append("")
        for agent in MODELS:
            df = data[bench]["plots"].get(agent)
            if df is None:
                continue
            label = MODEL_LABELS.get(agent, agent)
            cov = df["coverage"]
            steps = df["step"]

            # Find when coverage plateaus (last 10% of steps, coverage change < 1%)
            n = len(df)
            if n > 100:
                tail_start = int(n * 0.9)
                tail_cov = cov.iloc[tail_start:]
                plateau_pct = safe_div(tail_cov.max() - tail_cov.min(), max(1, cov.max())) or 0
                plateau_pct *= 100

                # Find step where 90% of final coverage was reached
                final_cov = cov.iloc[-1]
                if final_cov > 0:
                    target_90 = final_cov * 0.9
                    idx_90 = cov[cov >= target_90].index[0] if (cov >= target_90).any() else n - 1
                    step_90 = int(steps.iloc[idx_90])
                    time_90 = float(df["elapsed_seconds"].iloc[idx_90]) if "elapsed_seconds" in df.columns else None
                else:
                    step_90 = 0
                    time_90 = 0

                lines.append(f"    {agent}:")
                lines.append(f"      Final coverage: {int(final_cov):,} edges")
                lines.append(f"      90% coverage reached at step {step_90:,}" +
                             (f" ({time_90:.0f}s)" if time_90 else ""))
                lines.append(f"      Tail variability (last 10%): {plateau_pct:.2f}%")
                lines.append("")

        # ── Coverage AUC ─────────────────────────────────────────────
        lines.append("  Coverage AUC (edge·seconds — higher = faster coverage ramp):")
        lines.append("")
        aucs = {}
        for agent in ALL_AGENTS:
            df = data[bench]["plots"].get(agent)
            if df is not None and "elapsed_seconds" in df.columns and len(df) > 1:
                auc = float(np.trapezoid(df["coverage"], df["elapsed_seconds"]))
                aucs[agent] = auc
        if aucs:
            ranked = sorted(aucs.items(), key=lambda x: -x[1])
            for rank, (agent, auc) in enumerate(ranked, 1):
                label = agent.replace("baseline_time", "bl_time").replace("baseline", "bl_steps")
                marker = " ← best" if rank == 1 else ""
                lines.append(f"    #{rank}  {label:<18} {auc:>16,.0f}{marker}")
            lines.append("")

        # ── Throughput comparison ────────────────────────────────────
        lines.append("  Throughput (from fuzzer_stats):")
        lines.append("")
        for agent in ALL_AGENTS:
            stats = data[bench]["fuzzer_stats"].get(agent)
            if stats is None:
                continue
            label = agent.replace("baseline_time", "bl_time").replace("baseline", "bl_steps")
            eps = stats.get("execs_per_sec", "—")
            total = stats.get("execs_done", "—")
            runtime = stats.get("run_time", "—")
            corpus = stats.get("corpus_count", "—")
            favored = stats.get("corpus_favored", "—")
            crashes = stats.get("saved_crashes", 0)
            lines.append(f"    {label:<18} exec/s={fmt_num(eps, 8)}  "
                         f"total_execs={fmt_num(total, 12)}  "
                         f"runtime={fmt_num(runtime, 7)}s  "
                         f"corpus={fmt_num(corpus, 6)}  "
                         f"favored={fmt_num(favored, 5)}  "
                         f"crashes={crashes}")
        lines.append("")

        # ── RL overhead (time per step comparison) ───────────────────
        rl_times = {}
        bl_time = None
        for agent in ALL_AGENTS:
            df = data[bench]["plots"].get(agent)
            if df is not None and "elapsed_seconds" in df.columns and len(df) > 1:
                total_time = float(df["elapsed_seconds"].iloc[-1])
                total_steps = int(df["step"].iloc[-1])
                if total_steps > 0:
                    us_per_step = (total_time / total_steps) * 1e6
                    if agent == "baseline":
                        bl_time = us_per_step
                    elif agent in MODELS:
                        rl_times[agent] = us_per_step

        if rl_times and bl_time:
            lines.append("  RL overhead (µs/step vs baseline):")
            lines.append("")
            lines.append(f"    {'baseline':<18} {bl_time:>8.1f} µs/step")
            for agent, t in sorted(rl_times.items()):
                overhead = safe_div(t - bl_time, bl_time)
                overhead_str = f" (+{overhead*100:.1f}%)" if overhead is not None else ""
                lines.append(f"    {agent:<18} {t:>8.1f} µs/step{overhead_str}")
            lines.append("")

        # ── Action distribution (top 5 most-used actions) ────────────
        lines.append("  Action distribution (top 5 most-selected mutations):")
        lines.append("")

        ACTION_NAMES = {
            0: "FLIP_1BIT", 1: "FLIP_2BITS", 2: "FLIP_4BITS", 3: "FLIP_1BYTE",
            4: "FLIP_2BYTES", 5: "FLIP_4BYTES", 6: "ARITH_ADD1", 7: "ARITH_SUB1",
            8: "ARITH_ADD2LE", 9: "ARITH_SUB2LE", 10: "ARITH_ADD2BE", 11: "ARITH_SUB2BE",
            12: "ARITH_ADD4LE", 13: "ARITH_SUB4LE", 14: "ARITH_ADD4BE", 15: "ARITH_SUB4BE",
            16: "INT_BYTE", 17: "INT_2LE", 18: "INT_2BE", 19: "INT_4LE", 20: "INT_4BE",
            21: "HAVOC_FLIPBIT", 22: "HAVOC_INT8", 23: "HAVOC_INT16", 24: "HAVOC_INT16BE",
            25: "HAVOC_INT32", 26: "HAVOC_INT32BE", 27: "HAVOC_ARITH8_", 28: "HAVOC_ARITH8",
            29: "HAVOC_ARITH16_", 30: "HAVOC_ARITH16", 31: "HAVOC_ARITH16BE",
            32: "HAVOC_ARITH16BE_", 33: "HAVOC_ARITH32_", 34: "HAVOC_ARITH32BE_",
            35: "HAVOC_ARITH32", 36: "HAVOC_ARITH32BE", 37: "HAVOC_RAND8",
            38: "HAVOC_BYTEADD", 39: "HAVOC_BYTESUB", 40: "HAVOC_FLIP8",
            41: "DICT_USER_OVER", 42: "DICT_USER_INS", 43: "DICT_AUTO_OVER",
            44: "DICT_AUTO_INS", 45: "CUSTOM_MUT", 46: "HAVOC",
        }

        for agent in MODELS:
            df = data[bench]["plots"].get(agent)
            if df is None or "action" not in df.columns:
                continue
            counts = df["action"].value_counts().head(5)
            total = len(df)
            lines.append(f"    {agent}:")
            for act_id, count in counts.items():
                name = ACTION_NAMES.get(int(act_id), f"ACT_{act_id}")
                pct = count / total * 100
                lines.append(f"      {name:<22} {count:>8,}x  ({pct:>5.1f}%)")
            lines.append("")

        # ── Statistical test: RL vs baseline ─────────────────────────
        if HAS_SCIPY:
            bl_df = data[bench]["plots"].get("baseline")
            if bl_df is not None and "coverage" in bl_df.columns:
                lines.append("  Mann-Whitney U test (RL coverage vs baseline, last 20% of steps):")
                lines.append("")
                bl_tail = bl_df["coverage"].iloc[int(len(bl_df) * 0.8):]
                for agent in MODELS:
                    df = data[bench]["plots"].get(agent)
                    if df is None or "coverage" not in df.columns:
                        continue
                    rl_tail = df["coverage"].iloc[int(len(df) * 0.8):]
                    try:
                        stat, pval = mannwhitneyu(rl_tail, bl_tail, alternative="greater")
                        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                        rl_med = float(rl_tail.median())
                        bl_med = float(bl_tail.median())
                        lines.append(f"    {agent} vs baseline:  U={stat:,.0f}  p={pval:.4e}  "
                                     f"[{sig}]  median: {rl_med:,.0f} vs {bl_med:,.0f}")
                    except Exception as e:
                        lines.append(f"    {agent} vs baseline:  error — {e}")
                lines.append("")

        lines.append("")


def section_cross_benchmark(data, lines):
    lines.append(heading("CROSS-BENCHMARK COMPARISON"))
    lines.append("")

    for ms in MILESTONES:
        lines.append(hr("─"))
        lines.append(f"  Milestone: {ms} ({MILESTONE_STEPS[ms]:,} steps)")
        lines.append(hr("─"))
        lines.append("")

        # Coverage table
        lines.append("  Final coverage per agent:")
        lines.append("")
        header = f"    {'Benchmark':<14}"
        for agent in ALL_AGENTS:
            label = agent.replace("baseline_time", "bl_time").replace("baseline", "bl_steps")
            header += f" {label:>10}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))

        agent_covs = defaultdict(list)
        for bench in BENCHMARKS:
            if bench not in data:
                continue
            row = f"    {bench:<14}"
            for agent in ALL_AGENTS:
                ms_df = data[bench]["milestones"].get(ms, {}).get(agent)
                if ms_df is not None:
                    cov = int(ms_df["coverage"].iloc[-1])
                    row += fmt_num(cov)
                    agent_covs[agent].append(cov)
                else:
                    row += fmt_num(None)
            lines.append(row)

        lines.append("")

        # Average coverage
        lines.append("  Average coverage across benchmarks:")
        lines.append("")
        ranked = []
        for agent in ALL_AGENTS:
            if agent_covs[agent]:
                avg = np.mean(agent_covs[agent])
                ranked.append((agent, avg, len(agent_covs[agent])))
        ranked.sort(key=lambda x: -x[1])
        for rank, (agent, avg, n) in enumerate(ranked, 1):
            label = agent.replace("baseline_time", "bl_time").replace("baseline", "bl_steps")
            lines.append(f"    #{rank}  {label:<18} avg={avg:>10,.1f} edges  (across {n} benchmarks)")
        lines.append("")

        # Coverage gain over baseline
        lines.append("  Coverage gain vs same-steps baseline:")
        lines.append("")
        header = f"    {'Benchmark':<14}"
        for m in MODELS:
            header += f" {m:>10}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))

        model_gains = defaultdict(list)
        for bench in BENCHMARKS:
            if bench not in data:
                continue
            bl_df = data[bench]["milestones"].get(ms, {}).get("baseline")
            bl_cov = int(bl_df["coverage"].iloc[-1]) if bl_df is not None else None

            row = f"    {bench:<14}"
            for m in MODELS:
                m_df = data[bench]["milestones"].get(ms, {}).get(m)
                if m_df is not None and bl_cov is not None:
                    m_cov = int(m_df["coverage"].iloc[-1])
                    gain = m_cov - bl_cov
                    row += f" {gain:>+10,}"
                    model_gains[m].append(gain)
                else:
                    row += fmt_num(None)
            lines.append(row)
        lines.append("")

        # Win/loss record
        lines.append("  Win/loss vs baseline (benchmarks where RL > baseline):")
        lines.append("")
        for m in MODELS:
            gains = model_gains[m]
            if gains:
                wins = sum(1 for g in gains if g > 0)
                losses = sum(1 for g in gains if g < 0)
                ties = sum(1 for g in gains if g == 0)
                lines.append(f"    {m:<18} wins={wins}  losses={losses}  ties={ties}  "
                             f"(out of {len(gains)} benchmarks)")
        lines.append("")


def section_timing_summary(data, lines):
    lines.append(heading("TIMING SUMMARY"))
    lines.append("")
    lines.append("  Wall-clock time for full 10M-step eval:")
    lines.append("")
    header = f"    {'Benchmark':<14}"
    for agent in MODELS + ["baseline"]:
        label = agent.replace("baseline", "bl_steps")
        header += f" {label:>12}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for bench in BENCHMARKS:
        if bench not in data:
            continue
        row = f"    {bench:<14}"
        for agent in MODELS + ["baseline"]:
            df = data[bench]["plots"].get(agent)
            if df is not None and "elapsed_seconds" in df.columns:
                t = float(df["elapsed_seconds"].iloc[-1])
                hours = t / 3600
                row += f" {hours:>10.1f}h"
            else:
                row += f" {'—':>12}"
        lines.append(row)
    lines.append("")

    # Same-time baseline coverage comparison
    lines.append("  Same-time baseline: coverage achieved in same wall-clock as RL eval:")
    lines.append("")
    header = f"    {'Benchmark':<14} {'BL_time cov':>12} {'Best RL cov':>12} {'Best RL':>10} {'RL wins?':>10}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for bench in BENCHMARKS:
        if bench not in data:
            continue
        bt_df = data[bench]["plots"].get("baseline_time")
        bt_cov = int(bt_df["coverage"].iloc[-1]) if bt_df is not None else None

        best_rl = None
        best_rl_name = None
        for m in MODELS:
            df = data[bench]["plots"].get(m)
            if df is not None:
                c = int(df["coverage"].iloc[-1])
                if best_rl is None or c > best_rl:
                    best_rl = c
                    best_rl_name = m

        row = f"    {bench:<14}"
        row += fmt_num(bt_cov, 12)
        row += fmt_num(best_rl, 12)
        row += f" {best_rl_name or '—':>10}"
        if bt_cov is not None and best_rl is not None:
            row += f" {'YES' if best_rl > bt_cov else 'NO':>10}"
        else:
            row += f" {'—':>10}"
        lines.append(row)
    lines.append("")


def section_model_comparison(data, lines):
    lines.append(heading("MODEL HEAD-TO-HEAD"))
    lines.append("")
    lines.append("  Pairwise wins across all benchmarks at 10M steps (by coverage):")
    lines.append("")

    # Build coverage matrix
    cov_at_10m = {}
    for bench in BENCHMARKS:
        if bench not in data:
            continue
        cov_at_10m[bench] = {}
        for agent in ALL_AGENTS:
            ms_df = data[bench]["milestones"].get("10m", {}).get(agent)
            if ms_df is not None:
                cov_at_10m[bench][agent] = int(ms_df["coverage"].iloc[-1])

    agents_to_compare = MODELS + ["baseline"]
    header = f"    {'vs':<14}"
    for a in agents_to_compare:
        label = a.replace("baseline", "bl_steps")
        header += f" {label:>10}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for a1 in agents_to_compare:
        label1 = a1.replace("baseline", "bl_steps")
        row = f"    {label1:<14}"
        for a2 in agents_to_compare:
            if a1 == a2:
                row += f" {'—':>10}"
                continue
            wins = 0
            total = 0
            for bench in cov_at_10m:
                if a1 in cov_at_10m[bench] and a2 in cov_at_10m[bench]:
                    total += 1
                    if cov_at_10m[bench][a1] > cov_at_10m[bench][a2]:
                        wins += 1
            row += f" {f'{wins}/{total}':>10}"
        lines.append(row)
    lines.append("")

    # Same at 500k
    lines.append("  Pairwise wins at 500K steps (early advantage):")
    lines.append("")
    cov_at_500k = {}
    for bench in BENCHMARKS:
        if bench not in data:
            continue
        cov_at_500k[bench] = {}
        for agent in ALL_AGENTS:
            ms_df = data[bench]["milestones"].get("500k", {}).get(agent)
            if ms_df is not None:
                cov_at_500k[bench][agent] = int(ms_df["coverage"].iloc[-1])

    header = f"    {'vs':<14}"
    for a in agents_to_compare:
        label = a.replace("baseline", "bl_steps")
        header += f" {label:>10}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))

    for a1 in agents_to_compare:
        label1 = a1.replace("baseline", "bl_steps")
        row = f"    {label1:<14}"
        for a2 in agents_to_compare:
            if a1 == a2:
                row += f" {'—':>10}"
                continue
            wins = 0
            total = 0
            for bench in cov_at_500k:
                if a1 in cov_at_500k[bench] and a2 in cov_at_500k[bench]:
                    total += 1
                    if cov_at_500k[bench][a1] > cov_at_500k[bench][a2]:
                        wins += 1
            row += f" {f'{wins}/{total}':>10}"
        lines.append(row)
    lines.append("")


def section_key_findings(data, lines):
    lines.append(heading("KEY FINDINGS"))
    lines.append("")

    findings = []

    # 1. Early vs late advantage
    early_rl_wins = 0
    early_total = 0
    late_rl_wins = 0
    late_total = 0
    for bench in BENCHMARKS:
        if bench not in data:
            continue
        for ms_tag, counter_wins, counter_total in [
            ("500k", None, None), ("10m", None, None)
        ]:
            bl_df = data[bench]["milestones"].get(ms_tag, {}).get("baseline")
            if bl_df is None:
                continue
            bl_cov = int(bl_df["coverage"].iloc[-1])
            for m in MODELS:
                m_df = data[bench]["milestones"].get(ms_tag, {}).get(m)
                if m_df is None:
                    continue
                m_cov = int(m_df["coverage"].iloc[-1])
                if ms_tag == "500k":
                    early_total += 1
                    if m_cov > bl_cov:
                        early_rl_wins += 1
                else:
                    late_total += 1
                    if m_cov > bl_cov:
                        late_rl_wins += 1

    if early_total > 0:
        findings.append(f"  1. EARLY ADVANTAGE: At 500K steps, RL models beat baseline in "
                        f"{early_rl_wins}/{early_total} benchmark×model combinations "
                        f"({early_rl_wins/early_total*100:.0f}%).")
    if late_total > 0:
        findings.append(f"  2. LATE CONVERGENCE: At 10M steps, RL models beat baseline in "
                        f"{late_rl_wins}/{late_total} combinations "
                        f"({late_rl_wins/late_total*100:.0f}%). "
                        f"Baseline catches up with more steps.")

    # 2. Best model per benchmark
    findings.append("")
    findings.append("  3. BEST MODEL PER BENCHMARK (at 10M steps):")
    for bench in BENCHMARKS:
        if bench not in data:
            continue
        best_agent = None
        best_cov = -1
        for agent in ALL_AGENTS:
            ms_df = data[bench]["milestones"].get("10m", {}).get(agent)
            if ms_df is not None:
                cov = int(ms_df["coverage"].iloc[-1])
                if cov > best_cov:
                    best_cov = cov
                    best_agent = agent
        if best_agent:
            label = best_agent.replace("baseline_time", "bl_time").replace("baseline", "bl_steps")
            findings.append(f"     {bench:<14} → {label} ({best_cov:,} edges)")

    # 3. RL overhead
    findings.append("")
    findings.append("  4. RL OVERHEAD:")
    overheads = []
    for bench in BENCHMARKS:
        if bench not in data:
            continue
        bl_df = data[bench]["plots"].get("baseline")
        if bl_df is None or "elapsed_seconds" not in bl_df.columns:
            continue
        bl_rate = float(bl_df["step"].iloc[-1]) / float(bl_df["elapsed_seconds"].iloc[-1])
        for m in MODELS:
            df = data[bench]["plots"].get(m)
            if df is None or "elapsed_seconds" not in df.columns:
                continue
            rl_rate = float(df["step"].iloc[-1]) / float(df["elapsed_seconds"].iloc[-1])
            slowdown = safe_div(bl_rate - rl_rate, bl_rate)
            if slowdown is not None:
                overheads.append((bench, m, slowdown * 100))

    if overheads:
        avg_overhead = np.mean([o[2] for o in overheads])
        max_oh = max(overheads, key=lambda x: x[2])
        min_oh = min(overheads, key=lambda x: x[2])
        findings.append(f"     Average throughput reduction: {avg_overhead:.1f}%")
        findings.append(f"     Worst: {max_oh[1]} on {max_oh[0]} ({max_oh[2]:.1f}%)")
        findings.append(f"     Best:  {min_oh[1]} on {min_oh[0]} ({min_oh[2]:.1f}%)")

    for f in findings:
        lines.append(f)
    lines.append("")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate detailed experiment report")
    ap.add_argument("--exp-root", default="experiments", help="Experiment root dir")
    ap.add_argument("--out", default=None, help="Output file (default: stdout + experiments/detailed_report.txt)")
    args = ap.parse_args()

    data = load_all_data(args.exp_root)
    if not data:
        print("No experiment data found.", file=sys.stderr)
        sys.exit(1)

    lines = []
    lines.append("=" * W)
    lines.append("  RL-FUZZER: DETAILED EXPERIMENT REPORT")
    lines.append(f"  Generated from: {os.path.abspath(args.exp_root)}")
    lines.append("=" * W)
    lines.append("")

    section_overview(data, lines)
    section_per_benchmark_detail(data, lines)
    section_cross_benchmark(data, lines)
    section_timing_summary(data, lines)
    section_model_comparison(data, lines)
    section_key_findings(data, lines)

    lines.append("=" * W)
    lines.append("  END OF REPORT")
    lines.append("=" * W)

    report = "\n".join(lines)

    out_path = args.out or os.path.join(args.exp_root, "detailed_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\n[+] Report saved to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
