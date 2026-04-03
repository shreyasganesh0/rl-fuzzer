# Experiment 2: Multi-Benchmark 10M Steps — Full Model Comparison

## Overview

| Property | Value |
|----------|-------|
| **Date** | March 2026 (~6 days total, 4 crash recoveries) |
| **Targets** | 6 FuzzBench benchmarks: jsoncpp, freetype2, libxml2, re2, harfbuzz, libpng |
| **Train steps** | 10M (no plateau early-stopping) |
| **Eval steps** | 10M |
| **Eval runs** | 1 per model (no multi-run aggregation) |
| **Milestones** | 500K, 1M, 2M, 10M |
| **Models** | M1_0, M1_1, M1_2 |
| **Baselines** | Same-steps (10M execs), Same-time (median RL eval wall-clock) |
| **Script** | `scripts/run_full_experiment.sh` |
| **Raw data** | `experiments/<benchmark>/plots/`, `experiments/<benchmark>/milestones/` |

This was the second experiment, run after M1_2 was implemented and the
multi-benchmark framework (`run_full_experiment.sh`, `build_benchmark.sh`,
per-benchmark recipes) was built. Unlike Experiment 1 (jsoncpp-only, 500K
steps, 8 models, 5 eval runs), this experiment tests 3 models across 6
targets at full 10M-step scale with milestone snapshots.

---

## Models Tested

| Model | State Dim | Key Features | Hidden Layers | SHM Size |
|-------|-----------|-------------|---------------|----------|
| **M1_0** | 12 | Coverage, crashes, edge distribution stats (en/dis mean, std, max, nonzero), stability | [128, 128, 64] | 256 B |
| **M1_1** | 13 | M1_0 features but normalized over visited edges + visited-edge count | [128, 128, 64] | 256 B |
| **M1_2** | 64 | M1_1 + input buffer features: buf_len, entropy, printable_ratio, histogram[16], first_32_bytes | [256, 256, 128] | 512 B |

### Design Rationale

- **M1_0**: Can the agent learn from global coverage statistics alone?
- **M1_1**: Does knowing exploration breadth (visited edges) help?
- **M1_2**: Does observing the actual input being mutated enable smarter mutation selection?

### Shared Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Gamma | 0.99 |
| Learning rate | 1e-4 (Adam) |
| Replay buffer | 100,000 transitions |
| Target network sync | Every 1,000 steps |
| Epsilon schedule | 1.0 -> 0.05 linear over 6M steps (60% of training) |
| Entropy coefficient | 0.01 |
| Gradient clipping | 10.0 |
| Training frequency | Every step |

---

## Benchmarks

| Benchmark | Domain | Harness Source | Seeds | Dictionary |
|-----------|--------|---------------|-------|------------|
| **jsoncpp** | JSON parser | Source tree (`fuzz.cpp`) | Recipe-generated (4 JSON files) | Source (`fuzz.dict`) |
| **freetype2** | Font renderer | Source tree (`ftfuzzer.cc`) | Fallback (`"FUZZ"`) | Fallback minimal |
| **libxml2** | XML parser | FuzzBench (`target.cc`) | Fallback (`"FUZZ"`) | Source (`fuzz/xml.dict`) |
| **re2** | Regex engine | FuzzBench (`target.cc`) | Fallback (`"FUZZ"`) | Fallback minimal |
| **harfbuzz** | Text shaping | Source tree (`hb-shape-fuzzer.cc`) | Fallback (`"FUZZ"`) | Fallback minimal |
| **libpng** | Image decoder | Source tree (oss-fuzz) | FuzzBench (`seed.png`) | Source (`png.dict`) |

All harnesses use `LLVMFuzzerTestOneInput` (libFuzzer API), compatible with
AFL++ via `libAFLDriver.a`. Targets compiled with `afl-clang-fast++` instrumentation.

---

## Experiment Protocol

```
For each benchmark (jsoncpp, freetype2, libxml2, re2, harfbuzz, libpng):
  Phase 0: Build target           (build_benchmark.sh)
  Phase 1: Train all 3 models     (run_model.sh --train-steps 10M --milestones 500k,1m,2m,10m)
  Phase 2: Eval all 3 models      (run_model.sh --eval-steps 10M, frozen policy, epsilon=0.05)
  Phase 3: Baseline same-steps    (plain AFL++ with -E 10M)
  Phase 4: Baseline same-time     (plain AFL++ with -V <median_RL_seconds>)
  Phase 5: Slice milestones       (slice_milestones.py → 500K/1M/2M/10M snapshots)
  Phase 6: Compare at each milestone (compare_metrics.py)

After all benchmarks:
  Phase 7: Cross-benchmark summary (summarize_benchmarks.py)
```

**Crash recovery**: The orchestrator checks for existing checkpoints and eval
CSVs before each step. Interrupted runs automatically skip completed work on
restart. The experiment survived 4 laptop crashes over 6 days.

---

## Results — Coverage Gained at 500K Steps

| Benchmark | M1_0 | M1_1 | M1_2 | Baseline |
|-----------|------|------|------|----------|
| jsoncpp | 230 | 89 | 313 | **979** |
| freetype2 | 450 | 342 | 0 | **1,467** |
| libxml2 | 971 | **1,213** | 156 | 556 |
| re2 | 1,775 | 1,068 | 1,078 | **6,359** |
| harfbuzz | 1,453 | 380 | **1,866** | 1,494 |
| libpng | 0 | 0 | 0 | 0 |

**Baseline leads on 3/5 benchmarks** (jsoncpp, freetype2, re2). RL models
outperform baseline only on **libxml2** (M1_1: 1,213 vs 556) and **harfbuzz**
(M1_2: 1,866 vs 1,494).

Note: Baseline "step" = AFL++ execs_done, which jumps by millions between
1-second polls. Coverage at 500K execs is linearly interpolated from surrounding
data points.

---

## Results — Coverage Gained at 10M Steps

| Benchmark | M1_0 | M1_1 | M1_2 | Baseline (same-steps) | Baseline (same-time) |
|-----------|------|------|------|-----------------------|---------------------|
| jsoncpp | 234 | 93 | 317 | **7,444** | **7,634** |
| freetype2 | 864 | 399 | 57 | **7,608** | **8,493** |
| libxml2 | 1,065 | 1,560 | 250 | **3,512** | **5,505** |
| re2 | 2,505 | 1,335 | 1,301 | **22,931** | **23,343** |
| harfbuzz | 2,111 | 1,175 | 2,780 | **7,038** | **7,241** |
| libpng | 4 | 4 | 4 | **72** | **72** |

**Baseline outperforms all RL models on every benchmark at every milestone.**

The same-time baseline (running AFL++ for the same wall-clock duration as RL)
performs even better, executing 10-30x more total test cases.

---

## Coverage AUC (area under coverage-vs-time curve, edge-seconds)

Higher = faster and/or higher coverage over the full eval duration.

| Benchmark | M1_0 | M1_1 | M1_2 | Baseline |
|-----------|------|------|------|----------|
| jsoncpp | 844K | 329K | **1,196K** | 1,157K |
| freetype2 | **3,718K** | 1,480K | 205K | 2,308K |
| libxml2 | 5,181K | **6,911K** | 1,251K | 669K |
| re2 | **14,066K** | 6,206K | 6,079K | 9,787K |
| harfbuzz | 11,289K | 11,027K | **15,886K** | 3,195K |
| libpng | **19K** | 14K | 14K | 10K |

RL models often have higher AUC than the same-steps baseline despite lower
final coverage. This is because RL runs 10-30x longer in wall-clock time
(3,500-9,400s vs 137-493s), accumulating more area simply by running longer,
not by finding coverage faster.

---

## Throughput Comparison

| Benchmark | Baseline (execs/s) | M1_0 (steps/s) | M1_1 (steps/s) | M1_2 (steps/s) | RL Overhead |
|-----------|-------------------|-----------------|-----------------|-----------------|-------------|
| jsoncpp | 44,654 | 2,768 | — | — | **94%** |
| freetype2 | 24,345 | 2,031 | — | — | **92%** |
| libxml2 | 39,203 | 2,046 | — | — | **95%** |
| re2 | 19,069 | 1,736 | — | — | **91%** |
| harfbuzz | 19,750 | 1,858 | — | — | **91%** |
| libpng | 63,850 | 2,123 | — | — | **97%** |

**Average overhead: 93% throughput reduction.** RL processes ~2,000 steps/s
regardless of target, while baseline varies 19K-64K execs/s. The bottleneck
is SHM communication + Python DQN inference on every mutation step.

### Where the Time Goes (per mutation step)

| Component | Time |
|-----------|------|
| Read coverage map from SHM (64KB) | ~50 us |
| Compute state features (edge stats) | ~100 us |
| Write state to SHM, read action back | ~10 us |
| Wait for RL server (DQN forward pass) | ~200 us |
| Apply selected mutation | ~1 us |
| **Total with RL** | **~360 us/step** |
| **Total without RL** | **~20-50 us/step** |

Per-step overhead is 7-18x higher with RL.

---

## Same-Time Baseline Comparison

When given equal wall-clock time, does RL have an advantage?

| Benchmark | Baseline (same-time) | Best RL | Best Model | RL Wins? |
|-----------|---------------------|---------|------------|----------|
| jsoncpp | 5,983 | 313 | M1_2 | **NO** |
| freetype2 | 7,707 | 793 | M1_0 | **NO** |
| libxml2 | 4,948 | 1,465 | M1_1 | **NO** |
| re2 | 15,768 | 2,230 | M1_0 | **NO** |
| harfbuzz | 4,089 | 1,985 | M1_2 | **NO** |
| libpng | 0 | 0 | — | TIE |

**RL loses on all benchmarks when controlling for wall-clock time.** The
same-time baseline executes 10-30x more test cases because it doesn't have
the SHM + inference overhead.

---

## Head-to-Head Model Comparison

### Pairwise Wins at 10M Steps (coverage gained)

|  | M1_0 | M1_1 | M1_2 | Baseline |
|--|------|------|------|----------|
| **M1_0** | -- | 4/6 | 3/6 | 0/6 |
| **M1_1** | 1/6 | -- | 2/6 | 0/6 |
| **M1_2** | 2/6 | 3/6 | -- | 0/6 |
| **Baseline** | **5/6** | **5/6** | **5/6** | -- |

### Pairwise Wins at 500K Steps

|  | M1_0 | M1_1 | M1_2 | Baseline |
|--|------|------|------|----------|
| **M1_0** | -- | 4/6 | 3/6 | 1/6 |
| **M1_1** | 1/6 | -- | 2/6 | 1/6 |
| **M1_2** | 2/6 | 3/6 | -- | 1/6 |
| **Baseline** | **4/6** | **4/6** | **4/6** | -- |

M1_0 is the most consistent RL model. Baseline dominates at all horizons.

---

## Per-Benchmark Analysis

### jsoncpp (JSON parser)

- **Best RL**: M1_2 at 317 edges (vs baseline 7,444) -- 23x gap
- Coverage plateaus instantly for RL (by step 124K)
- Baseline discovers 23x more edges
- All agents converge to single-action policies: M1_0 -> action #1, M1_1 -> action #18, M1_2 -> action #35

### freetype2 (Font renderer)

- **Best RL**: M1_0 at 864 edges (vs baseline 7,608) -- 9x gap
- M1_0 shows late coverage jump (step 6.1M -> 864 edges)
- M1_2 discovers only 57 edges -- input buffer features don't help here
- Baseline reaches 5,747 edges by 2M steps, eventually 7,608

### libxml2 (XML parser)

- **Best RL**: M1_1 at 1,560 edges (vs baseline 3,512) -- 2.3x gap
- M1_1 (visited-edge tracking) outperforms M1_0 (1,065) -- edge visit info helps
- M1_2 achieves only 250 edges -- input buffer features don't help here
- `xml.dict` provides good dictionary, benefits both RL and baseline
- **Closest RL-to-baseline gap** of any benchmark

### re2 (Regex engine)

- **Best RL**: M1_0 at 2,505 edges (vs baseline 22,931) -- 9x gap
- Largest absolute gap -- baseline finds 9x more edges
- re2 has deep state space that rewards extensive corpus exploration
- RL throughput: ~1,736 steps/s vs baseline 19,069 steps/s

### harfbuzz (Text shaping engine)

- **Best RL**: M1_2 at 2,780 edges (vs baseline 7,038) -- 2.5x gap
- M1_2's input-buffer awareness provides measurable advantage over M1_0 (2,111) and M1_1 (1,175)
- Coverage AUC: M1_2 leads at 15.9M edge-seconds vs M1_0 at 11.3M
- **Best showcase for M1_2's input-aware features**

### libpng (Image decoder)

- **All models**: 4 edges (effectively zero useful coverage)
- Only 1 FuzzBench seed (`seed.png`), AFL++ exhausts the queue immediately
- State shows `finished` -- no new inputs discovered
- Baseline achieves 72 edges -- low but nonzero
- **Conclusion**: Insufficient seed corpus makes this benchmark uninformative

---

## Key Findings

### 1. Baseline dominates at all horizons

At both 500K and 10M steps, plain AFL++ outperforms RL models on most
benchmarks (4/6 at 500K, 5/6 at 10M). The throughput gap (10-30x) is
insurmountable -- more mutations means more chances to discover new coverage.

### 2. RL wins only on select benchmarks at early horizons

M1_1 outperforms baseline on libxml2 at 500K steps (1,213 vs 556), and M1_2
on harfbuzz (1,866 vs 1,494). These are the **only** cases where RL beats
baseline. Both advantages disappear by 10M steps as baseline catches up.

### 3. Throughput is the bottleneck (93% overhead)

RL processes ~2,000 steps/s regardless of target, while baseline runs at
19K-64K execs/s. The SHM communication + Python DQN inference per mutation
step creates a 10-30x throughput gap that dominates all other factors.

### 4. Sparse rewards lead to degenerate policies

All agents converge to single-action strategies (100% of steps use one
mutation). Coverage stagnates within the first 0.1-1% of the run on most
benchmarks. The reward signal (new edges only) is too sparse for the DQN
to learn meaningful mutation preferences.

### 5. Input-aware features (M1_2) show promise on harfbuzz

M1_2 achieves the best RL coverage (2,780 edges) and highest AUC (15.9M)
on harfbuzz, suggesting input characteristics can inform mutation selection
for complex structured-input targets. However, M1_2 performs poorly on
freetype2 (57 edges) and libxml2 (250 edges), indicating the extra state
complexity can hurt when the features aren't informative.

### 6. Model ranking varies by benchmark

No single RL model is best on all targets:
- **M1_0** wins on freetype2 and re2 (simpler state, less overfitting)
- **M1_1** wins on libxml2 (visited-edge tracking helps)
- **M1_2** wins on jsoncpp and harfbuzz (input features help for structured inputs)

### 7. 10M training steps cause worse degeneration than 500K

Comparing with Experiment 1's plateau-stopped training (~350K steps):
- Experiment 1 best RL (jsoncpp): M1_0_skip at 626 edges (500K eval)
- Experiment 2 best RL (jsoncpp): M1_2 at 317 edges (10M eval)

Longer training without plateau-stopping leads to more severe policy collapse.

---

## Comparison with Experiment 1

| Aspect | Experiment 1 | Experiment 2 |
|--------|-------------|-------------|
| **Date** | Mar 5, 2026 (3.7h) | Mar 2026 (~6 days) |
| **Script** | `run_experiment.sh` | `run_full_experiment.sh` |
| **Targets** | jsoncpp only | 6 FuzzBench benchmarks |
| **Models** | M0_0, M1_0, M1_1, M2 + skip (8) | M1_0, M1_1, M1_2 (3) |
| **Train steps** | 500K (plateau stopped ~350K) | 10M (no plateau) |
| **Eval steps** | 500K | 10M |
| **Eval runs** | 5 per model | 1 per model |
| **Best RL (jsoncpp)** | M1_0_skip: 626 +/- 15 | M1_2: 317 |
| **Baseline (jsoncpp)** | 5,606 +/- 81 | 7,444 |
| **RL throughput** | 2,374-2,850 steps/s | ~2,000 steps/s |
| **Baseline throughput** | 59,306 execs/s | 19K-64K execs/s |
| **Coverage gap** | ~9x | ~23x (jsoncpp) |
| **Policy degeneration** | 6/8 models single-action | All models single-action |

The lower RL coverage at 10M steps (317 vs 626 at 500K) reflects longer
training without early-stopping causing more severe policy collapse, plus
a different model set (M1_2 vs M1_0_skip).

---

## Infrastructure Built for This Experiment

| Component | File | Purpose |
|-----------|------|---------|
| Benchmark build framework | `scripts/build_benchmark.sh` | Universal target builder |
| 6 benchmark recipes | `benchmarks/*/build_recipe.sh` | Per-target build + link steps |
| Experiment orchestrator | `scripts/run_full_experiment.sh` | Multi-benchmark loop with crash recovery |
| Milestone checkpoints | `scripts/rl_server.py --milestones` | Save .pt copies at step counts |
| Per-experiment directories | `scripts/run_model.sh --exp-dir` | Prevent multi-benchmark clobbering |
| Milestone slicer | `scripts/slice_milestones.py` | Post-hoc CSV truncation at milestones |
| Cross-benchmark summary | `scripts/summarize_benchmarks.py` | FuzzBench-style scores and rankings |
| M1_2 model | `scripts/models/m1_2.py` + `src/mutator_m1_2.c` | 64-dim input-aware DQN |
| Local dependency builds | `packages/local/` | liblzma + libarchive (no sudo) |
| Report generator | `scripts/generate_report.py` | Detailed statistical reports |

### Compute Scale

- 6 benchmarks x 3 models x 10M steps = **180M RL training steps**
- Plus 10M eval steps per model, 2 baselines per benchmark
- Total: ~6 days on a single laptop (with 4 crash recoveries)

---

## Output Structure

```
experiments/
├── jsoncpp/
│   ├── bin/                    # checkpoints: rl_m1_0.pt, rl_m1_1.pt, rl_m1_2.pt
│   ├── outputs/                # AFL++ training output
│   ├── outputs_eval/           # AFL++ eval output
│   ├── plots/
│   │   ├── m1_0/               # rl_metrics_m1_0_{train,eval}.csv
│   │   ├── m1_1/
│   │   ├── m1_2/
│   │   ├── baseline/           # rl_metrics_baseline_eval.csv
│   │   └── baseline_time/      # rl_metrics_baseline_time_eval.csv
│   └── milestones/
│       ├── 500k/               # sliced CSVs + comparison plots + reports
│       ├── 1m/
│       ├── 2m/
│       └── 10m/
├── freetype2/                  # same structure
├── libxml2/
├── re2/
├── harfbuzz/
├── libpng/
├── summary/                    # cross-benchmark tables and rankings
│   ├── cross_benchmark_report.txt
│   └── cross_benchmark_summary.json
└── full_experiment.log         # master orchestrator log
```
