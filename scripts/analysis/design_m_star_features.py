#!/usr/bin/env python3
"""
design_m_star_features.py — Generate M* feature specification from differential analysis.

Reads the feature importance report produced by differential_analysis.py and
outputs a fully self-contained feature specification for the M* model, including
SHM layout, normalization details, and C implementation hints.

Usage:
    python3 scripts/analysis/design_m_star_features.py \
        --report experiments/differential/analysis/feature_importance_report.json \
        --output experiments/differential/analysis/m_star_feature_spec.json \
        [--top-k 15] \
        [--alpha 0.05]
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone


# ── Constants ───────────────────────────────────────────────────────────────

ACTION_DIM = 47  # must match common.py / all mutators

# Mapping from source_csv + source_column to C implementation hints.
# Keys are (source_csv, source_column) tuples; if only source_column matches,
# the source_csv-agnostic fallback is used.
_C_IMPL_HINTS = {
    # ── Coverage dynamics (from telemetry coverage_dynamics CSV) ─────────
    ("coverage_dynamics", "total_edges"):
        "count_coverage(afl) — count bytes != 0xFF in afl->virgin_bits",
    ("coverage_dynamics", "new_edges_this_interval"):
        "current_coverage - prev_coverage (delta between consecutive calls)",
    ("coverage_dynamics", "edge_discovery_rate"):
        "(current_coverage - prev_coverage) / (float)interval_size",
    ("coverage_dynamics", "crashes_total"):
        "afl->saved_crashes (or afl->unique_crashes)",
    ("coverage_dynamics", "crashes_this_interval"):
        "afl->saved_crashes - prev_crashes",
    ("coverage_dynamics", "avg_exec_time_us"):
        "elapsed_us / execs_this_interval (clock_gettime CLOCK_MONOTONIC delta)",
    ("coverage_dynamics", "corpus_size"):
        "afl->queued_items",
    ("coverage_dynamics", "hot_edges"):
        "count edges in cumulative_map with hit_count > 128",
    ("coverage_dynamics", "warm_edges"):
        "count edges in cumulative_map with hit_count in [8, 128]",
    ("coverage_dynamics", "cool_edges"):
        "count edges in cumulative_map with hit_count in [1, 7]",
    ("coverage_dynamics", "cold_edges"):
        "MAP_SIZE - total_edges (edges never hit)",
    ("coverage_dynamics", "edge_entropy"):
        "Shannon entropy over nonzero hit-count bins: "
        "-sum(p_i * log2(p_i)) where p_i = count_in_bin / total_nonzero",
    ("coverage_dynamics", "edge_hit_mean"):
        "sum of nonzero hit counts / number of nonzero edges",
    ("coverage_dynamics", "edge_hit_std"):
        "sqrt(sum((hit_i - mean)^2) / n_nonzero) over cumulative_map",
    ("coverage_dynamics", "edge_hit_max"):
        "max value in cumulative_map[0..MAP_SIZE-1]",
    ("coverage_dynamics", "total_execs"):
        "afl->fsrv.total_execs",
    ("coverage_dynamics", "timestamp_us"):
        "clock_gettime(CLOCK_MONOTONIC) converted to microseconds since start",

    # ── Mutation attribution (from telemetry mutation_attribution CSV) ───
    ("mutation_attribution", "mut_effectiveness_best"):
        "max over all 47 actions of (new_edges_for_action / count_for_action)",
    ("mutation_attribution", "mut_effectiveness_mean"):
        "mean over all 47 actions of (new_edges_for_action / count_for_action)",
    ("mutation_attribution", "mut_concentration"):
        "fraction of new edges attributed to the single most-effective mutation",
    ("mutation_attribution", "mut_diversity"):
        "Shannon entropy of mutation usage counts: -sum(p_i * log2(p_i))",

    # ── Edge distribution derived features ──────────────────────────────
    ("coverage_dynamics", "hot_edge_ratio"):
        "(float)hot_edges / max(total_edges, 1)",
    ("coverage_dynamics", "warm_edge_ratio"):
        "(float)warm_edges / max(total_edges, 1)",
    ("coverage_dynamics", "cool_edge_ratio"):
        "(float)cool_edges / max(total_edges, 1)",
    ("coverage_dynamics", "coverage_velocity"):
        "(current_coverage - coverage_N_steps_ago) / N (first derivative)",
    ("coverage_dynamics", "coverage_acceleration"):
        "(velocity_now - velocity_N_steps_ago) / N (second derivative)",

    # ── Per-mutation columns (mut_NN_count, mut_NN_new_edges, etc.) ─────
    # Handled by the _get_per_mutation_hint() fallback below.
}

# Normalization recommendations by feature source/type.
_NORMALIZATION_HINTS = {
    "total_edges":              "divide by MAP_SIZE (65536.0)",
    "new_edges_this_interval":  "divide by MAX_NEW_EDGES (100.0)",
    "edge_discovery_rate":      "clip to [0, 1] (already a rate)",
    "crashes_total":            "log1p(x) / log1p(MAX_CRASHES)",
    "crashes_this_interval":    "log1p(x) / log1p(100)",
    "avg_exec_time_us":         "log1p(x) / log1p(1e6)",
    "corpus_size":              "log1p(x) / log1p(100000)",
    "hot_edges":                "divide by MAP_SIZE (65536.0)",
    "warm_edges":               "divide by MAP_SIZE (65536.0)",
    "cool_edges":               "divide by MAP_SIZE (65536.0)",
    "cold_edges":               "divide by MAP_SIZE (65536.0)",
    "edge_entropy":             "divide by 8.0 (max entropy for 256 bins)",
    "edge_hit_mean":            "divide by 255.0",
    "edge_hit_std":             "divide by 128.0",
    "edge_hit_max":             "divide by 255.0",
    "total_execs":              "log1p(x) / log1p(train_steps)",
    "timestamp_us":             "not used as a feature directly",
    "hot_edge_ratio":           "already in [0, 1]",
    "warm_edge_ratio":          "already in [0, 1]",
    "cool_edge_ratio":          "already in [0, 1]",
    "coverage_velocity":        "divide by MAX_NEW_EDGES (100.0)",
    "coverage_acceleration":    "clip to [-1, 1] after dividing by MAX_NEW_EDGES",
    "mut_effectiveness_best":   "clip to [0, 1]",
    "mut_effectiveness_mean":   "clip to [0, 1]",
    "mut_concentration":        "already in [0, 1]",
    "mut_diversity":            "divide by log2(47) (max entropy over 47 actions)",
}

# SHM data type mapping by column name pattern.
_SHM_FORMATS = {
    "total_execs":   "uint64",
    "timestamp_us":  "uint64",
    "crashes_total": "uint32",
    "corpus_size":   "uint32",
    "hot_edges":     "uint32",
    "warm_edges":    "uint32",
    "cool_edges":    "uint32",
    "cold_edges":    "uint32",
    "edge_hit_max":  "uint32",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_per_mutation_hint(col: str) -> str:
    """Generate a C implementation hint for per-mutation columns like
    mut_03_count, mut_12_new_edges, etc."""
    parts = col.split("_")
    # Expected patterns: mut_NN_count, mut_NN_new_edges, mut_NN_crashes
    if len(parts) >= 3 and parts[0] == "mut":
        action_idx = parts[1]
        metric = "_".join(parts[2:])
        if metric == "count":
            return (f"increment mut_count[{action_idx}] each time action "
                    f"{action_idx} is selected in afl_custom_fuzz()")
        elif metric == "new_edges":
            return (f"after execution, if coverage increased, add delta to "
                    f"mut_new_edges[{action_idx}] for the previously selected action")
        elif metric == "crashes":
            return (f"if afl->saved_crashes increased after execution with action "
                    f"{action_idx}, increment mut_crashes[{action_idx}]")
    return f"track per-interval accumulator for column '{col}' in the C mutator struct"


def _get_c_implementation(source_csv: str, source_column: str) -> str:
    """Look up or generate a C implementation hint for a given feature."""
    key = (source_csv, source_column)
    if key in _C_IMPL_HINTS:
        return _C_IMPL_HINTS[key]
    # Try source_column alone across all source_csvs
    for (csv, col), hint in _C_IMPL_HINTS.items():
        if col == source_column:
            return hint
    # Per-mutation column pattern
    if source_column.startswith("mut_"):
        return _get_per_mutation_hint(source_column)
    # Generic fallback
    return (f"read from AFL++ state or compute in afl_custom_fuzz() — "
            f"source: {source_csv}.{source_column}")


def _get_normalization(source_column: str) -> str:
    """Look up normalization recommendation for a feature."""
    if source_column in _NORMALIZATION_HINTS:
        return _NORMALIZATION_HINTS[source_column]
    # Per-mutation columns
    if source_column.startswith("mut_") and source_column.endswith("_count"):
        return "divide by log_interval (executions per telemetry interval)"
    if source_column.startswith("mut_") and "new_edges" in source_column:
        return "divide by MAX_NEW_EDGES (100.0)"
    if source_column.startswith("mut_") and "crashes" in source_column:
        return "log1p(x) / log1p(10)"
    # Derived / ratio features
    if "ratio" in source_column or "rate" in source_column:
        return "already in [0, 1] or clip to [0, 1]"
    if "entropy" in source_column:
        return "divide by 8.0"
    return "min-max scale to [0, 1] based on observed range during training"


def _get_shm_format(source_column: str) -> str:
    """Determine SHM wire format for a feature."""
    if source_column in _SHM_FORMATS:
        return _SHM_FORMATS[source_column]
    # Integer counts
    if source_column.startswith("mut_") and source_column.endswith("_count"):
        return "uint32"
    if source_column.startswith("mut_") and "crashes" in source_column:
        return "uint32"
    # Everything else is float32 (normalized in C or in Python)
    return "float32"


def _shm_type_size(fmt: str) -> int:
    """Return byte size for a SHM format string."""
    return {"float32": 4, "uint32": 4, "uint64": 8}[fmt]


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n, minimum 128."""
    if n <= 128:
        return 128
    return 1 << (n - 1).bit_length()


def _dedup_feature_name(name: str) -> str:
    """Produce a canonical key for deduplication.

    Features that are the same metric at different timepoints (e.g.
    'edge_entropy@50000' and 'edge_entropy@100000') share the same base
    metric.  We strip any @<timepoint> or _at_<timepoint> suffix and
    also strip the bug-pair tag if present.
    """
    # Remove @timepoint suffix
    base = name.split("@")[0]
    # Remove _at_NNNNN suffix
    parts = base.rsplit("_at_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
    return base


def _generate_rationale(feat: dict) -> str:
    """Generate a human-readable rationale for why a feature was selected."""
    es = feat.get("effect_size_a12", 0.5)
    p = feat.get("p_value", 1.0)
    bug_pair = feat.get("bug_pair", "unknown")

    direction = "higher in buggy" if es > 0.5 else "lower in buggy"
    strength = "large" if abs(es - 0.5) > 0.28 else "medium"
    return (f"{strength} effect ({direction}, A12={es:.2f}) "
            f"distinguishing buggy from fixed versions of {bug_pair} "
            f"(p={p:.1e})")


# ── Core logic ──────────────────────────────────────────────────────────────

def load_report(report_path: str) -> dict:
    """Load and validate the feature importance report JSON."""
    if not os.path.exists(report_path):
        print(f"[ERROR] Report not found: {report_path}", file=sys.stderr)
        sys.exit(1)
    with open(report_path) as f:
        report = json.load(f)
    if "features" not in report:
        print(f"[ERROR] Report missing 'features' key: {report_path}",
              file=sys.stderr)
        sys.exit(1)
    return report


def filter_and_dedup(features: list, alpha: float, top_k: int) -> tuple:
    """Filter by p-value, deduplicate by base metric, return (selected, excluded).

    Returns:
        (selected_features, excluded_features) where each excluded entry
        has a reason string.
    """
    # Phase 1: split into significant / not significant
    significant = []
    excluded = []
    for feat in features:
        pv = feat.get("p_value", 1.0)
        if pv >= alpha:
            excluded.append({
                "name": feat["name"],
                "reason": f"p_value ({pv:.4f}) >= alpha ({alpha})",
                "p_value": pv,
            })
        else:
            significant.append(feat)

    # Phase 2: deduplicate — keep the one with smallest p-value per base metric
    best_by_key = {}
    for feat in significant:
        key = _dedup_feature_name(feat["name"])
        if key not in best_by_key or feat["p_value"] < best_by_key[key]["p_value"]:
            if key in best_by_key:
                old = best_by_key[key]
                excluded.append({
                    "name": old["name"],
                    "reason": (f"deduplicated — same metric as '{feat['name']}' "
                               f"which has smaller p_value "
                               f"({feat['p_value']:.4e} < {old['p_value']:.4e})"),
                    "p_value": old["p_value"],
                })
            best_by_key[key] = feat
        else:
            excluded.append({
                "name": feat["name"],
                "reason": (f"deduplicated — same metric as "
                           f"'{best_by_key[key]['name']}' which has smaller "
                           f"p_value ({best_by_key[key]['p_value']:.4e})"),
                "p_value": feat["p_value"],
            })

    # Phase 3: sort by effect size (distance from 0.5), then p-value as tiebreak
    deduped = list(best_by_key.values())
    deduped.sort(key=lambda f: (-abs(f.get("effect_size_a12", 0.5) - 0.5),
                                 f.get("p_value", 1.0)))

    # Phase 4: take top-K
    selected = deduped[:top_k]
    for feat in deduped[top_k:]:
        excluded.append({
            "name": feat["name"],
            "reason": f"ranked below top-{top_k} after deduplication",
            "p_value": feat.get("p_value", 1.0),
        })

    return selected, excluded


def build_shm_layout(feature_specs: list) -> dict:
    """Compute the SHM byte layout.

    Layout:
        offset 0:  sequence_counter  (uint32, 4 bytes)
        offset 4:  feature values    (consecutive float32/uint32/uint64)
        ...
        offset N:  action_seq        (uint32, 4 bytes)
        offset N+4: action           (int32,  4 bytes)
        total:     rounded up to next power of 2 >= needed
    """
    feature_start = 4  # after sequence counter
    offset = feature_start
    for spec in feature_specs:
        spec["shm_offset_bytes"] = offset
        offset += _shm_type_size(spec["shm_format"])

    action_seq_offset = offset
    action_offset = action_seq_offset + 4
    total_needed = action_offset + 4  # action is 4 bytes
    total_bytes = _next_power_of_two(total_needed)

    return {
        "total_bytes": total_bytes,
        "sequence_counter_offset": 0,
        "sequence_counter_size": 4,
        "feature_start_offset": feature_start,
        "feature_format": (f"{len(feature_specs)} consecutive values "
                           f"(float32/uint32/uint64 per feature spec)"),
        "action_seq_offset": action_seq_offset,
        "action_seq_size": 4,
        "action_offset": action_offset,
        "action_size": 4,
    }


def build_spec(selected: list, excluded: list, report: dict,
               alpha: float) -> dict:
    """Build the complete m_star_feature_spec.json structure."""

    state_dim = len(selected)

    # Build per-feature specs
    feature_specs = []
    for rank, feat in enumerate(selected):
        source_csv = feat.get("source_csv", "unknown")
        source_col = feat.get("source_column", feat["name"])
        spec = {
            "index": rank,
            "name": feat["name"],
            "description": feat.get("description", ""),
            "source": f"{source_csv}.{source_col}",
            "c_implementation": _get_c_implementation(source_csv, source_col),
            "normalization": _get_normalization(source_col),
            "shm_format": _get_shm_format(source_col),
            # shm_offset_bytes is filled in by build_shm_layout
            "shm_offset_bytes": 0,
            "importance_rank": rank + 1,
            "p_value": feat.get("p_value"),
            "effect_size": feat.get("effect_size_a12"),
            "rationale": _generate_rationale(feat),
        }
        feature_specs.append(spec)

    shm_layout = build_shm_layout(feature_specs)

    # Architecture recommendation
    if state_dim <= 20:
        hidden = [128, 128, 64]
        rationale = (f"state_dim={state_dim} <= 20: using compact "
                     f"[128, 128, 64] architecture (same as M0_0, M1_1)")
    else:
        hidden = [256, 256, 128]
        rationale = (f"state_dim={state_dim} > 20: using wider "
                     f"[256, 256, 128] architecture (same as M1_2, M2)")

    spec = {
        "model_name": "m_star",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "design_m_star_features.py",
        "source_report": os.path.basename(
            report.get("_source_path", "feature_importance_report.json")),
        "selection_criteria": {
            "alpha": alpha,
            "top_k": len(selected),
            "deduplication": "keep smallest p-value per base metric name",
            "ranking": "by |effect_size_a12 - 0.5| descending, p_value ascending",
        },
        "state_dim": state_dim,
        "action_dim": ACTION_DIM,
        "recommended_architecture": {
            "hidden_layers": hidden,
            "activation": "relu",
            "rationale": rationale,
        },
        "recommended_algorithm": "contextual_bandit_thompson",
        "features": feature_specs,
        "shm_layout": shm_layout,
        "excluded_features": excluded,
    }

    return spec


# ── Output ──────────────────────────────────────────────────────────────────

def print_summary(spec: dict):
    """Print a human-readable summary to stdout."""
    W = 72
    print()
    print("=" * W)
    print("  M* Feature Specification Summary")
    print("=" * W)
    print()
    print(f"  State dimension:    {spec['state_dim']}")
    print(f"  Action dimension:   {spec['action_dim']}")
    arch = spec["recommended_architecture"]
    print(f"  Hidden layers:      {arch['hidden_layers']}")
    print(f"  Activation:         {arch['activation']}")
    print(f"  Algorithm:          {spec['recommended_algorithm']}")
    print(f"  SHM total bytes:    {spec['shm_layout']['total_bytes']}")
    print()
    print("-" * W)
    print("  Selected features (ranked by discriminative power):")
    print("-" * W)
    print()
    print(f"  {'#':>3}  {'Name':<32} {'Effect':>7} {'p-value':>10} {'Format':<8}")
    print(f"  {'':>3}  {'':<32} {'(A12)':>7} {'':>10} {'':>8}")
    print(f"  {'-'*3}  {'-'*32} {'-'*7} {'-'*10} {'-'*8}")

    for feat in spec["features"]:
        es = feat["effect_size"]
        pv = feat["p_value"]
        es_str = f"{es:.3f}" if es is not None else "   -"
        pv_str = f"{pv:.2e}" if pv is not None else "    -"
        print(f"  {feat['importance_rank']:>3}  {feat['name']:<32} "
              f"{es_str:>7} {pv_str:>10} {feat['shm_format']:<8}")

    print()
    print("-" * W)
    print(f"  Excluded: {len(spec['excluded_features'])} features")
    print("-" * W)
    print()

    # Group exclusion reasons
    reasons = {}
    for ex in spec["excluded_features"]:
        key = ex["reason"].split(" — ")[0] if " — " in ex["reason"] else ex["reason"]
        # Simplify to first clause
        if key.startswith("p_value"):
            key = "p_value >= alpha"
        elif key.startswith("deduplicated"):
            key = "deduplicated (same base metric)"
        elif key.startswith("ranked below"):
            key = "below top-K cutoff"
        reasons[key] = reasons.get(key, 0) + 1

    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {count:>3} features: {reason}")

    print()

    # SHM layout summary
    layout = spec["shm_layout"]
    print("-" * W)
    print("  SHM Layout:")
    print("-" * W)
    print()
    print(f"    [  0 ..   3]  sequence_counter  (uint32)")
    feat_end = layout["action_seq_offset"] - 1
    print(f"    [  4 .. {feat_end:>3}]  "
          f"{spec['state_dim']} feature values  ({layout['feature_format']})")
    aseq = layout["action_seq_offset"]
    aoff = layout["action_offset"]
    print(f"    [{aseq:>3} .. {aseq+3:>3}]  action_seq        (uint32)")
    print(f"    [{aoff:>3} .. {aoff+3:>3}]  action            (int32)")
    print(f"    Total: {layout['total_bytes']} bytes")
    print()
    print("=" * W)
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate M* feature specification from differential "
                    "analysis feature importance report.")
    ap.add_argument(
        "--report", required=True,
        help="Path to feature_importance_report.json from differential_analysis.py")
    ap.add_argument(
        "--output", required=True,
        help="Output path for m_star_feature_spec.json")
    ap.add_argument(
        "--top-k", type=int, default=15,
        help="Maximum number of features to select (default: 15)")
    ap.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold for p-value filtering (default: 0.05)")
    args = ap.parse_args()

    # Load report
    report = load_report(args.report)
    report["_source_path"] = args.report
    features = report["features"]
    print(f"[+] Loaded {len(features)} features from {args.report}")

    # Filter and deduplicate
    selected, excluded = filter_and_dedup(features, args.alpha, args.top_k)
    if not selected:
        print("[ERROR] No features passed the significance filter "
              f"(alpha={args.alpha}). Try increasing --alpha or check "
              "the report.", file=sys.stderr)
        sys.exit(1)

    print(f"[+] After filtering (alpha={args.alpha}) and deduplication: "
          f"{len(selected)} features selected, {len(excluded)} excluded")

    # Build spec
    spec = build_spec(selected, excluded, report, args.alpha)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[+] Wrote feature spec to {args.output}")

    # Print summary
    print_summary(spec)


if __name__ == "__main__":
    main()
