#!/usr/bin/env python3
"""
summarize_benchmarks.py — Cross-benchmark summary for multi-benchmark experiments.

Reads comparison_summary.json from each benchmark's milestone dirs and produces:
  - A summary table (text) ranking models across benchmarks
  - FuzzBench-style average score ranking
  - Pairwise A12 effect sizes aggregated across benchmarks
  - JSON output for programmatic access

Usage:
  python3 scripts/summarize_benchmarks.py \
    --exp-root experiments/ \
    --benchmarks jsoncpp,freetype2,libxml2,re2,harfbuzz,libpng \
    --models m1_0,m1_1,m1_2 \
    --milestones 500000,1000000,2000000,10000000
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


MODEL_LABELS = {
    "m0_0":      "M0_0",
    "m1_0":      "M1_0",
    "m1_1":      "M1_1",
    "m1_2":      "M1_2",
    "m2":        "M2",
    "baseline":  "Baseline",
}


def format_milestone(step: int) -> str:
    if step >= 1_000_000 and step % 1_000_000 == 0:
        return f"{step // 1_000_000}m"
    if step >= 1_000 and step % 1_000 == 0:
        return f"{step // 1_000}k"
    return str(step)


def load_milestone_summary(exp_root, benchmark, milestone_tag):
    """Load comparison_summary.json for a benchmark at a milestone."""
    path = os.path.join(exp_root, benchmark, "milestones", milestone_tag,
                        "comparison_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_fuzzbench_score(coverage_values: dict) -> dict:
    """Compute FuzzBench-style average score.

    Each model gets score = (its median coverage / max median coverage) * 100.
    """
    if not coverage_values:
        return {}
    max_cov = max(coverage_values.values())
    if max_cov == 0:
        return {m: 0.0 for m in coverage_values}
    return {m: (v / max_cov) * 100.0 for m, v in coverage_values.items()}


def vargha_delaney_a12(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.5
    more = np.sum(x[:, None] > y[None, :])
    equal = np.sum(x[:, None] == y[None, :])
    return float((more + 0.5 * equal) / (nx * ny))


def main():
    ap = argparse.ArgumentParser(description="Cross-benchmark summary")
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--benchmarks", default="jsoncpp,freetype2,libxml2,re2,harfbuzz,libpng")
    ap.add_argument("--models", default="m1_0,m1_1,m1_2")
    ap.add_argument("--milestones", default="500000,1000000,2000000,10000000")
    args = ap.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    milestones = [int(s.strip()) for s in args.milestones.split(",")]
    all_ids = models + ["baseline"]

    summary_dir = os.path.join(args.exp_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    lines = []
    SEP = "=" * 80
    json_out = {}

    lines += [SEP, "  CROSS-BENCHMARK SUMMARY", SEP, ""]

    for milestone in milestones:
        tag = format_milestone(milestone)
        lines += [f"{'─'*80}", f"  MILESTONE: {tag} ({milestone:,} steps)", f"{'─'*80}"]
        json_out[tag] = {}

        # Collect coverage_gained per model per benchmark
        per_benchmark_gains = {}  # {benchmark: {model: coverage_gained}}
        per_benchmark_auc = {}

        for benchmark in benchmarks:
            data = load_milestone_summary(args.exp_root, benchmark, tag)
            if data is None:
                lines.append(f"  [skip] {benchmark}: no data at milestone {tag}")
                continue

            summaries = data.get("summaries", data)  # handle both old/new format
            per_benchmark_gains[benchmark] = {}
            per_benchmark_auc[benchmark] = {}

            for mid in all_ids:
                ps = summaries.get(mid, {}).get("eval", {})
                if not ps:
                    continue
                gained = ps.get("coverage_gained_mean", ps.get("coverage_gained", 0))
                per_benchmark_gains[benchmark][mid] = gained
                if "coverage_auc" in ps:
                    per_benchmark_auc[benchmark][mid] = ps["coverage_auc"]

        if not per_benchmark_gains:
            lines.append("  No benchmark data found for this milestone.")
            continue

        # ── Per-benchmark coverage table ──
        lines += ["", "  Coverage gained per benchmark:", ""]
        hdr = f"  {'Benchmark':<16}"
        for mid in all_ids:
            label = MODEL_LABELS.get(mid, mid)
            hdr += f" {label:>12}"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))

        for benchmark in benchmarks:
            gains = per_benchmark_gains.get(benchmark)
            if not gains:
                continue
            row = f"  {benchmark:<16}"
            for mid in all_ids:
                val = gains.get(mid)
                if val is not None:
                    row += f" {val:>12.0f}"
                else:
                    row += f" {'—':>12}"
            lines.append(row)

        # ── FuzzBench-style average score ──
        lines += ["", "  FuzzBench-style average score (higher = better):", ""]
        model_scores = {mid: [] for mid in all_ids}

        for benchmark, gains in per_benchmark_gains.items():
            scores = compute_fuzzbench_score(gains)
            for mid, score in scores.items():
                model_scores[mid].append(score)

        avg_scores = {}
        for mid in all_ids:
            if model_scores[mid]:
                avg_scores[mid] = float(np.mean(model_scores[mid]))

        # Sort by score descending
        ranked = sorted(avg_scores.items(), key=lambda x: -x[1])
        for rank, (mid, score) in enumerate(ranked, 1):
            label = MODEL_LABELS.get(mid, mid)
            n = len(model_scores[mid])
            lines.append(f"  #{rank}  {label:<20}  {score:>6.1f}%  (across {n} benchmarks)")

        json_out[tag]["avg_scores"] = avg_scores
        json_out[tag]["per_benchmark"] = per_benchmark_gains

        # ── Average rank ──
        lines += ["", "  Average rank (lower = better):", ""]
        model_ranks = {mid: [] for mid in all_ids}
        for benchmark, gains in per_benchmark_gains.items():
            sorted_models = sorted(gains.items(), key=lambda x: -x[1])
            for rank, (mid, _) in enumerate(sorted_models, 1):
                model_ranks[mid].append(rank)

        avg_ranks = {}
        for mid in all_ids:
            if model_ranks[mid]:
                avg_ranks[mid] = float(np.mean(model_ranks[mid]))

        ranked_by_rank = sorted(avg_ranks.items(), key=lambda x: x[1])
        for mid, avg_rank in ranked_by_rank:
            label = MODEL_LABELS.get(mid, mid)
            lines.append(f"  {label:<20}  avg_rank={avg_rank:.2f}")

        json_out[tag]["avg_ranks"] = avg_ranks

        # ── Coverage AUC table ──
        if per_benchmark_auc:
            lines += ["", "  Coverage AUC per benchmark (edge·seconds):", ""]
            hdr_auc = f"  {'Benchmark':<16}"
            for mid in all_ids:
                hdr_auc += f" {MODEL_LABELS.get(mid, mid):>14}"
            lines.append(hdr_auc)
            lines.append("  " + "-" * (len(hdr_auc) - 2))
            for benchmark in benchmarks:
                aucs = per_benchmark_auc.get(benchmark)
                if not aucs:
                    continue
                row = f"  {benchmark:<16}"
                for mid in all_ids:
                    val = aucs.get(mid)
                    if val is not None:
                        row += f" {val:>14,.0f}"
                    else:
                        row += f" {'—':>14}"
                lines.append(row)

        lines += [""]

    lines += [SEP, "  END OF CROSS-BENCHMARK SUMMARY", SEP]

    # Write outputs
    report_path = os.path.join(summary_dir, "cross_benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [report] {report_path}")

    json_path = os.path.join(summary_dir, "cross_benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_out, indent=2, fp=f, default=str)
    print(f"  [json]   {json_path}")


if __name__ == "__main__":
    main()
