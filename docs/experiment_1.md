# Experiment 1 — jsoncpp 500K Steps, Full Model Comparison

Single target (jsoncpp), 5–8 RL models trained once and evaluated N times
against the same checkpoint, optionally compared against a vanilla AFL++
baseline in two flavours (same-steps and same-time).

---

## How to Run

### Default invocation (full experiment, ~55 min)

```bash
bash scripts/experiment1.sh
```

Builds & trains all default models on `bin/target`, runs 5 eval rounds, and
produces a same-steps comparison report at `comparison_results/same_steps/`.

### Quick smoke test (~3 min)

```bash
bash scripts/build_benchmark.sh jsoncpp           # one-time target build
bash scripts/experiment1.sh \
    --models m0_0 \
    --train-steps 2000 \
    --eval-steps 1000 \
    --eval-runs 2 \
    --no-plateau \
    --out comparison_results/smoke1
```

Validates: build → train → checkpoint → eval → CSV → compare. Confirm
`comparison_results/smoke1/same_steps/comparison_report.txt` exists. Add
`--run-baseline` to also smoke the baseline + same-time path.

### Flags

| Flag | Default | Effect |
|---|---|---|
| `--skip-train` | off | Reuse `bin/rl_<model>.pt` checkpoints; skips the ~10-min/model training phase. Verifies all checkpoints exist before starting. |
| `--eval-runs N` | `5` | Total eval rounds. The first eval is harvested from the end of training (`run_1`); rounds 2..N are pure `--eval-only`. |
| `--train-steps N` | `500000` | Per-model training step cap. |
| `--eval-steps N` | `500000` | Per-eval-round step cap. |
| `--models CSV` | `m0_0,m1_0,m1_1,m1_2,m2` | Subset of models. Add `_skip` variants (e.g. `m1_0,m1_0_skip`). |
| `--run-baseline` | off | Adds vanilla AFL++ runs in two flavours: same-steps (`baseline/`) and same-time (`baseline_time/`) at the median RL eval wall-clock. |
| `--no-plateau` | off | Disable plateau early-stopping in `rl_server.py`. |
| `--compare-mode` | `steps` | Affects CSV output dir naming; the script always runs both same-steps and (if baseline enabled) same-time comparisons regardless. |
| `--out DIR` | `comparison_results` | Where the final reports land. |

### Phases

1. **Phase 0** — clear `outputs_eval/` and `--out` dir. If not `--skip-train`, also clear `outputs/`, `plots/`, and old `bin/mutator_*.so`.
2. **Phase 1** — training. Calls `build_and_compare.sh` without `--eval-only` so it does train + first eval; that first eval becomes `run_1`.
3. **Phase 2** — N-1 (or N if `--skip-train`) more eval rounds, each preceded by `rm -rf outputs_eval/` to defeat `AFL_AUTORESUME` queue contamination.
4. **Phase 2b** (if `--run-baseline`) — time-based baseline at median RL eval wall-clock.
5. **Phase 3** — verification: counts CSV files per model.
6. **Phase 4a/4b** — `scripts/visuals/compare_metrics.py` produces `comparison_report.txt` plus PNG plots.

### `_skip` model variants

The `_skip` variants of M0_0/M1_0/M1_1/M1_2/M2 are identical in state space
and architecture to the originals but train every 4th step instead of every
step (passed via `--train-freq 4` to `rl_server.py`). The DQN backprop is
~1050 µs/step while action selection is ~50 µs/step; reducing training
frequency drops amortised per-step cost from ~1290 µs to ~355 µs (≈3.6× RL
server throughput) at the cost of slower learning. Useful for measuring the
coverage impact of training-frequency tradeoffs against the originals and
the plain AFL++ baseline.

### Outputs

| Path | Contents |
|---|---|
| `plots/<model>/run_N/rl_metrics_<model>_eval.csv` | Per-eval-run metrics CSV |
| `plots/<model>/rl_metrics_<model>_train.csv` | Training metrics CSV |
| `comparison_results/same_steps/comparison_report.txt` | Same-steps text report |
| `comparison_results/same_time/comparison_report.txt` | Same-time text report (if baseline) |
| `comparison_results/{same_steps,same_time}/plot_*.png` | Coverage, reward, action-distribution, throughput plots |
| `comparison_results/experiment.log` | Master orchestrator log |

---

## Results

### Overview

| Property | Value |
|----------|-------|
| **Date** | March 5, 2026 (09:55 – 13:39, ~3.7 hours) |
| **Target** | jsoncpp (`jsoncpp_jsoncpp_fuzzer`, FuzzBench commit `8190e06`) |
| **Train steps** | 500K (plateau early-stopping triggered at ~350K-362K) |
| **Eval steps** | 500K |
| **Eval runs** | 5 per model (multi-run aggregation, mean ± std) |
| **Models** | M0_0, M1_0, M1_1, M2 (standard) + skip variants (train freq=4) |
| **Baselines** | Same-steps (500K execs), Same-time (203s = median RL eval time) |
| **Script** | `scripts/experiment1.sh` (orchestrator) → `scripts/build_and_compare.sh` → `scripts/run_model.sh` |
| **Raw data** | Root-level `plots/`, `comparison_results/` |

This was the first full experiment, run before the multi-benchmark framework
(now `experiment2.sh`) and before M1_2 was implemented.

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
| M0_0 | #36 HAVOC_MUT_ARITH32BE | [100.0%](#3-policy-degeneration-is-the-core-problem) |
| M1_0 | #15 DET_ARITH_SUB_FOUR_BIG | [53.7%](#3-policy-degeneration-is-the-core-problem) |
| M1_1 | #41 DICTIONARY_USER_EXTRAS_OVER | [100.0%](#3-policy-degeneration-is-the-core-problem) |
| M2 | #42 DICTIONARY_USER_EXTRAS_INSERT | [100.0%](#3-policy-degeneration-is-the-core-problem) |
| M0_0_skip | #2 DET_FLIP_FOUR_BITS | [100.0%](#3-policy-degeneration-is-the-core-problem) |
| M1_0_skip | #45 CUSTOM_MUTATOR | [41.8%](#3-policy-degeneration-is-the-core-problem) |
| M1_1_skip | #11 DET_ARITH_SUB_TWO_BIG | [86.7%](#3-policy-degeneration-is-the-core-problem) |
| M2_skip | #45 CUSTOM_MUTATOR | [100.0%](#3-policy-degeneration-is-the-core-problem) |

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

## Comparison with Experiment 2

See [experiment_2.md](experiment_2.md) for the full multi-benchmark 10M-step experiment report.

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
