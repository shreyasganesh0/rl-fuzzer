# Experiment 1: jsoncpp 500K Steps — Full Model Comparison

## Overview

| Property | Value |
|----------|-------|
| **Date** | March 5, 2026 (09:55 – 13:39, ~3.7 hours) |
| **Target** | jsoncpp (`jsoncpp_jsoncpp_fuzzer`, FuzzBench commit `8190e06`) |
| **Train steps** | 500K (plateau early-stopping triggered at ~350K-362K) |
| **Eval steps** | 500K |
| **Eval runs** | 5 per model (multi-run aggregation, mean ± std) |
| **Models** | M0_0, M1_0, M1_1, M2 (standard) + skip variants (train freq=4) |
| **Baselines** | Same-steps (500K execs), Same-time (203s = median RL eval time) |
| **Script** | `scripts/build_and_compare.sh` → `scripts/run_experiment.sh` |
| **Raw data** | Root-level `plots/`, `comparison_results/` |

This was the first full experiment, run before the multi-benchmark framework
(`run_full_experiment.sh`) and before M1_2 was implemented.

---

## Models Tested

| Model | State Dim | Key Features | Hidden Layers | Train Freq |
|-------|-----------|-------------|---------------|-----------|
| **M0_0** | 3 | coverage, new_edges, crashes | [128, 128, 64] | Every step |
| **M1_0** | 12 | + edge distribution stats (en/dis mean, std, max, nonzero) | [128, 128, 64] | Every step |
| **M1_1** | 13 | + visited-edge count | [128, 128, 64] | Every step |
| **M2** | 97 | + per-mutator trace-bit magnitudes (47 en + 47 dis) | [256, 256, 128] | Every step |
| **M0_0_skip** | 3 | Same as M0_0 | [128, 128, 64] | Every 4 steps |
| **M1_0_skip** | 12 | Same as M1_0 | [128, 128, 64] | Every 4 steps |
| **M1_1_skip** | 13 | Same as M1_1 | [128, 128, 64] | Every 4 steps |
| **M2_skip** | 97 | Same as M2 | [256, 256, 128] | Every 4 steps |

---

## Training Results

All models used plateau early-stopping (10K-step window with <1 edge delta + epsilon < 0.06).
Training stopped well before the 500K step limit.

| Model | Steps at Plateau | Elapsed (s) | Coverage Gained | Throughput (steps/s) |
|-------|-----------------|-------------|-----------------|---------------------|
| M0_0 | 354,900 | 623 | 651 | 576 |
| M1_0 | 362,500 | 696 | 654 | 528 |
| M1_1 | 350,000 | 657 | 650 | 540 |
| M2 | 354,900 | 711 | 642 | 506 |
| M0_0_skip | — | — | 680 | 1,615 |
| M1_0_skip | — | — | 675 | 1,433 |
| M1_1_skip | — | — | 689 | 1,469 |
| M2_skip | — | — | 678 | 1,392 |

### Training Observations

- All standard models converge to ~650 edges with similar coverage
- **Skip variants are 2.5-3x faster** (1,400-1,615 steps/s vs 506-576 steps/s) and achieve
  slightly higher coverage (675-689 vs 642-654)
- Training action entropy is 3.5-3.8 (diverse exploration during epsilon-greedy)
- All models favor `DICTIONARY_USER_EXTRAS_INSERT` (#42) as dominant training action (5-15%)

---

## Eval Results — Same-Steps Comparison (5 runs, mean ± std)

Each model evaluated 5 times at 500K steps with frozen policy (epsilon = 0.05).

| Model | Gained (mean ± std) | Min | Max | Throughput (execs/s) |
|-------|---------------------|-----|-----|---------------------|
| M0_0 | 387.8 ± 27.7 | 335 | 416 | 2,827 ± 23 |
| **M1_0** | 561.8 ± 48.7 | 505 | 649 | 2,467 ± 45 |
| M1_1 | 566.4 ± 11.6 | 556 | 582 | 2,497 ± 14 |
| M2 | 603.4 ± 1.5 | 601 | 605 | 2,504 ± 23 |
| M0_0_skip | 243.6 ± 0.8 | 243 | 245 | 2,850 ± 16 |
| **M1_0_skip** | **626.4 ± 15.4** | 604 | 644 | 2,374 ± 21 |
| M1_1_skip | 578.2 ± 3.8 | 573 | 583 | 2,403 ± 19 |
| M2_skip | 568.6 ± 25.5 | 536 | 599 | 2,549 ± 25 |
| **Baseline** | **5,605.8 ± 81.0** | 5,472 | 5,708 | **59,306 ± 501** |

### Eval Action Degeneration

| Model | Dominant Action | % |
|-------|----------------|---|
| M0_0 | #36 HAVOC_MUT_ARITH32BE | 100.0% |
| M1_0 | #15 DET_ARITH_SUB_FOUR_BIG | 53.7% |
| M1_1 | #41 DICTIONARY_USER_EXTRAS_OVER | 100.0% |
| M2 | #42 DICTIONARY_USER_EXTRAS_INSERT | 100.0% |
| M0_0_skip | #2 DET_FLIP_FOUR_BITS | 100.0% |
| M1_0_skip | #45 CUSTOM_MUTATOR | 41.8% |
| M1_1_skip | #11 DET_ARITH_SUB_TWO_BIG | 86.7% |
| M2_skip | #45 CUSTOM_MUTATOR | 100.0% |

Only M1_0 (53.7%) and M1_0_skip (41.8%) retained any action diversity at eval.
All others collapsed to single-action policies.

---

## Eval Results — Same-Time Comparison (5 runs, mean ± std)

Baseline given 203 seconds (median wall-clock time of RL eval runs).

| Model | Gained (mean ± std) | Efficiency (edges/s) |
|-------|---------------------|---------------------|
| M0_0 | 387.8 ± 27.7 | 2.17 |
| M1_0 | 561.8 ± 48.7 | 2.71 |
| M1_1 | 566.4 ± 11.6 | 2.78 |
| M2 | 603.4 ± 1.5 | 2.97 |
| M0_0_skip | 243.6 ± 0.8 | 1.38 |
| M1_0_skip | 626.4 ± 15.4 | 2.88 |
| M1_1_skip | 578.2 ± 3.8 | 2.75 |
| M2_skip | 568.6 ± 25.5 | 2.84 |
| **Baseline (same-time)** | **5,790.4 ± 5.2** | **28.52** |

Baseline achieves 28.5 edges/s vs best RL at 2.97 edges/s — a **9.6x efficiency gap**.

---

## Key Findings

### 1. Baseline dominates at 500K steps

Baseline discovers 5,606 ± 81 edges vs best RL model (M1_0_skip) at 626 ± 15 edges.
That's a **9x coverage gap** driven entirely by throughput: baseline runs at 59,306 execs/s
vs RL at 2,374-2,850 execs/s.

### 2. Skip variants improve training throughput but not eval coverage

| Comparison | Standard | Skip (freq=4) | Improvement |
|-----------|----------|---------------|------------|
| Train throughput | 506-576 steps/s | 1,392-1,615 steps/s | **2.5-3x faster** |
| Train coverage | 642-654 | 675-689 | Slightly better |
| Eval coverage | 388-603 | 244-626 | Mixed (M1_0_skip best, M0_0_skip worst) |

Skip variants train faster and achieve comparable coverage, but the eval
results are mixed — M1_0_skip is the best overall RL model (626 edges) but
M0_0_skip is the worst (244 edges).

### 3. Policy degeneration is the core problem

6/8 models converge to 100% single-action policies at eval. Only M1_0 and
M1_0_skip retain action diversity (53.7% and 41.8% dominant action). This
suggests the edge distribution features in M1_0 provide a slightly richer
learning signal that resists full collapse.

### 4. M2 (per-mutator magnitudes) adds complexity without benefit

M2 has the largest state space (97 dims) but achieves only 603 edges (vs M1_0_skip
at 626 with 12 dims). The per-mutator magnitude features don't help — and M2's
training throughput is the lowest (506 steps/s) due to the larger state.

### 5. Variance is low across runs

Most models show std < 5% of mean, indicating reproducible results. The baseline
is especially stable (5,606 ± 81, CV = 1.4%). M1_0 has the highest variance
(562 ± 49, CV = 8.7%), likely because it's the only model with non-degenerate
action selection.

---

## Comparison with Experiment 2 (Multi-Benchmark, 10M Steps)

| Aspect | Experiment 1 | Experiment 2 |
|--------|-------------|-------------|
| Target | jsoncpp only | 6 benchmarks |
| Steps | 500K | 10M |
| Models | M0_0, M1_0, M1_1, M2 + skip | M1_0, M1_1, M1_2 |
| Multi-run | 5 eval runs | 1 eval run |
| Best RL (jsoncpp) | M1_0_skip: 626 ± 15 | M1_2: 317 |
| Baseline (jsoncpp) | 5,606 ± 81 | 7,444 |
| RL throughput | 2,374-2,850 steps/s | 2,644-2,822 steps/s |
| Baseline throughput | 59,306 execs/s | 44,654 execs/s |

The lower RL coverage at 10M steps in Experiment 2 (317 vs 626 at 500K) reflects
a different model (M1_2 vs M1_0_skip) and different training conditions (10M train
steps with full degeneration vs 350K plateau-stopped training).

---

## Raw Data Locations

| Data | Path |
|------|------|
| Single-run report | `comparison_results/comparison_report.txt` |
| Multi-run same-steps report | `comparison_results/same_steps/comparison_report.txt` |
| Multi-run same-time report | `comparison_results/same_time/comparison_report.txt` |
| Machine-readable summary | `comparison_results/comparison_summary.json` |
| Experiment log | `comparison_results/experiment.log` |
| Training CSVs | `plots/<model>/rl_metrics_<model>_train.csv` |
| Eval CSVs (run 1) | `plots/<model>/rl_metrics_<model>_eval.csv` |
| Eval CSVs (runs 2-5) | `plots/<model>/run_{2..5}/rl_metrics_<model>_eval.csv` |
| Baseline CSV | `plots/baseline/rl_metrics_baseline_eval.csv` |
| Plots | `comparison_results/plot_*.png` |
