# Experiment 2 — Multi-Benchmark 10M Steps, Milestone-Snapshotted

Multi-benchmark, multi-model, milestone-snapshotted comparison. Per-benchmark
loop with a per-model nested loop. Eval runs once per model and is sliced
post-hoc at each milestone via `scripts/visuals/slice_milestones.py`.
Designed to be resumable — partial runs skip work that's already complete.

---

## How to Run

### Default invocation (full experiment, hours-to-days)

```bash
bash scripts/experiment2.sh
```

Iterates `jsoncpp,freetype2,libxml2,re2,harfbuzz,libpng` × `m1_0,m1_1,m1_2`.
For each benchmark: builds it, trains every model once with milestone
checkpoints at 500K/1M/2M/10M, runs same-steps and same-time baselines, then
slices and summarizes.

### Quick smoke test (~5 min)

```bash
bash scripts/experiment2.sh \
    --benchmarks jsoncpp \
    --models m1_0 \
    --milestones 1000,2000 \
    --train-steps 2000 \
    --eval-steps 1000 \
    --eval-runs 1 \
    --no-plateau \
    --exp-root experiments/smoke2
```

Validates: per-benchmark build → train → milestone checkpoints
(`bin/rl_m1_0.pt.1k`, `.2k`) → eval → baseline (steps + time) →
`slice_milestones.py` → `summarize_benchmarks.py`. Check
`experiments/smoke2/jsoncpp/milestones/` and `experiments/smoke2/summary/`.
Add `--skip-build` and reuse an existing `bin/target` to skip the lengthy
benchmark build.

### Flags

| Flag | Default | Effect |
|---|---|---|
| `--benchmarks CSV` | all six | Restrict to a subset. Each name must have `benchmarks/<name>/build_recipe.sh`. |
| `--models CSV` | `m1_0,m1_1,m1_2` | Models to train per benchmark. |
| `--milestones CSV` | `500000,1000000,2000000,10000000` | Step counts at which to snapshot the training checkpoint (`run_model.sh` calls `cp` per milestone). Max milestone also drives the `--train-steps`/`--eval-steps` defaults. |
| `--eval-runs N` | `1` | Eval repetitions per model. |
| `--skip-train` | off | Reuse `<exp_dir>/bin/rl_<model>.pt`; runs eval-only. Recovery: skips per-model eval if its CSV already covers ≥95% of `--eval-steps`. |
| `--skip-build` | off | Reuse existing `bin/target`. Asserts it exists. |
| `--no-plateau` | off | Forwarded to `run_model.sh`. |
| `--exp-root DIR` | `experiments/` | Per-benchmark trees go under `<exp-root>/<benchmark>/`. |
| `--train-steps N` | max milestone | Override default. |
| `--eval-steps N` | max milestone | Override default. |

### Phases (per benchmark)

1. **Phase 0** — build the benchmark via `scripts/build_benchmark.sh <name>` unless `--skip-build`.
2. **Phase 1** — train each model in sequence; save milestone checkpoints. With `--skip-train`, runs eval-only and resumes via `csv_has_enough_steps`.
3. **Phase 2** — same-steps baseline via local `run_baseline()` helper using `afl-fuzz -E`.
4. **Phase 3** — same-time baseline. RL median wall-clock computed by `slice_milestones.py --query-time`.
5. **Phase 4** — `slice_milestones.py` cuts each eval CSV at every milestone and produces per-milestone reports under `<exp-dir>/milestones/`.
6. **Cross-benchmark summary** — `summarize_benchmarks.py` aggregates across benchmarks into `<exp-root>/summary/`.

---

## Building Benchmark Targets (`scripts/build_benchmark.sh`)

`build_benchmark.sh` is a generic driver. It does not know how to build any
specific target — it sources a recipe at `benchmarks/<name>/build_recipe.sh`
that supplies the project-specific bits (clone URL, build steps, link steps).
The driver handles cloning at a pinned commit, wiring AFL++'s compiler
wrappers, copying seeds and dictionary, and smoke-testing the binary.

### Available recipes

| Benchmark  | FuzzBench Name                  | Build System | Dependencies       |
|------------|--------------------------------|-------------|-------------------|
| jsoncpp    | jsoncpp_jsoncpp_fuzzer         | cmake       | —                 |
| freetype2  | freetype2_ftfuzzer             | autotools   | zlib              |
| libxml2    | libxml2_xml                    | autotools   | zlib, liblzma     |
| re2        | re2_fuzzer                     | make        | —                 |
| harfbuzz   | harfbuzz_hb-shape-fuzzer       | meson       | meson, ninja      |
| libpng     | libpng_libpng_read_fuzzer      | autotools   | zlib              |

### Prerequisites

| Path | Default | Purpose |
|---|---|---|
| `$AFL_ROOT/afl-clang-fast++` | `~/packages/AFLplusplus` | AFL-instrumented C++ compiler. Build AFL++ from source if missing. |
| `$AFL_ROOT/libAFLDriver.a` | same | libFuzzer-compatible driver providing `main()`. Built via `cd $AFL_ROOT && make libAFLDriver.a`. |
| `$FUZZBENCH` | `~/fuzzbench` | Source of pinned commit (`benchmark.yaml`) and harness (`benchmarks/<name>/target.cc`). Clone from `github.com/google/fuzzbench`. |
| `$TARGETS_DIR/<project>/src/` | `~/targets/<project>/src/` (created lazily) | Where the project source is cloned. Override with `export TARGETS_DIR=...`. The driver only re-clones if `.git` is missing. |
| `packages/local/` (optional) | repo-local | Headers/libs added to `CFLAGS`/`LDFLAGS`/`PKG_CONFIG_PATH` if present. Use for dependencies you can't install with sudo. |
| `.venv/bin/meson` (optional) | repo-local | Auto-added to `PATH` if present. Required for meson benchmarks (freetype2, harfbuzz). |

### Recipe contract

A recipe is a sourced shell file. The driver requires four things:

| Element | Meaning |
|---|---|
| `FUZZBENCH_NAME=...` | Lookup key under `$FUZZBENCH/benchmarks/`. Drives commit pinning, harness path, and (if present) seed corpus. |
| `GIT_URL=...` | Where to clone the project. Cloned to `$TARGETS_DIR/<basename>/src` (default `~/targets/<basename>/src`). |
| `BUILD_STEPS()` | Function producing a static library at a known location. The driver doesn't link inside this — it just expects something usable to exist when the function returns. |
| `LINK_STEPS()` | Function that links the static lib + harness + `$AFLDRIVER` into `$RL_FUZZER/bin/target`. |

Optional: `DICT_PATH` (set inside `BUILD_STEPS()` to point at a dict the
project ships) and `SEEDS_DIR` (path to a seeds directory the recipe wants
to install).

Variables passed to recipes: `CC`, `CXX`, `CFLAGS`, `CXXFLAGS`, `LDFLAGS`,
`AFLDRIVER`, `SRC_DIR`, `RL_FUZZER`, `FUZZBENCH`.

### Dictionary wiring (priority order)

1. **Recipe-set `DICT_PATH`** (e.g., jsoncpp's `fuzz.dict` from inside the source tree).
2. **A `.dict` file under `$FUZZBENCH/benchmarks/<FUZZBENCH_NAME>/`** or anywhere in the cloned source. The first match is copied.
3. **Synthetic 1-line fallback** `kw1="FUZZ"` so the dict argument never points at a missing file.

Destination is always `dictionaries/target.dict`. `run_model.sh` picks it up
if present and adds `-x dictionaries/target.dict` to `afl-fuzz`.

### Seeds wiring (same priority)

1. **Recipe-set `SEEDS_DIR`** copied into `inputs/` if `inputs/` is empty.
2. **`$FUZZBENCH/benchmarks/<FUZZBENCH_NAME>/seeds/`** if it exists.
3. **Synthetic `inputs/seed_default`** containing the literal bytes `FUZZ`.

`inputs/` is left alone if it already has files. Switching benchmarks
requires `rm -rf inputs/*` first.

### One-time setup

```bash
git clone https://github.com/AFLplusplus/AFLplusplus ~/packages/AFLplusplus
cd ~/packages/AFLplusplus && make -j$(nproc) && make libAFLDriver.a

git clone --depth=1 https://github.com/google/fuzzbench.git ~/fuzzbench

cd ~/projects/rl-fuzzer
bash scripts/build_benchmark.sh jsoncpp        # or libxml2, freetype2, etc.
```

After a successful run you have `bin/target`, `dictionaries/target.dict`,
`inputs/`, and empty `outputs/`, `outputs_eval/`, `plots/` ready for
`run_model.sh`.

---

## Results

### Overview

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
| **Script** | `scripts/experiment2.sh` |
| **Raw data** | `experiments/<benchmark>/plots/`, `experiments/<benchmark>/milestones/` |

This was the second experiment, run after M1_2 was implemented and the
multi-benchmark framework (`experiment2.sh`, `build_benchmark.sh`,
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
| [jsoncpp](#jsoncpp-json-parser) | 230 | 89 | 313 | **979** |
| [freetype2](#freetype2-font-renderer) | 450 | 342 | 0 | **1,467** |
| [libxml2](#libxml2-xml-parser) | 971 | **1,213** | 156 | 556 |
| [re2](#re2-regex-engine) | 1,775 | 1,068 | 1,078 | **6,359** |
| [harfbuzz](#harfbuzz-text-shaping-engine) | 1,453 | 380 | **1,866** | 1,494 |
| [libpng](#libpng-image-decoder) | 0 | 0 | 0 | 0 |

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
| [jsoncpp](#jsoncpp-json-parser) | 234 | 93 | 317 | **7,444** | **7,634** |
| [freetype2](#freetype2-font-renderer) | 864 | 399 | 57 | **7,608** | **8,493** |
| [libxml2](#libxml2-xml-parser) | 1,065 | 1,560 | 250 | **3,512** | **5,505** |
| [re2](#re2-regex-engine) | 2,505 | 1,335 | 1,301 | **22,931** | **23,343** |
| [harfbuzz](#harfbuzz-text-shaping-engine) | 2,111 | 1,175 | 2,780 | **7,038** | **7,241** |
| [libpng](#libpng-image-decoder) | 4 | 4 | 4 | **72** | **72** |

**Baseline outperforms all RL models on every benchmark at every milestone.**

The same-time baseline (running AFL++ for the same wall-clock duration as RL)
performs even better, executing 10-30x more total test cases.

---

## Coverage AUC (area under coverage-vs-time curve, edge-seconds)

Higher = faster and/or higher coverage over the full eval duration.

| Benchmark | M1_0 | M1_1 | M1_2 | Baseline |
|-----------|------|------|------|----------|
| [jsoncpp](#jsoncpp-json-parser) | 844K | 329K | **1,196K** | 1,157K |
| [freetype2](#freetype2-font-renderer) | **3,718K** | 1,480K | 205K | 2,308K |
| [libxml2](#libxml2-xml-parser) | 5,181K | **6,911K** | 1,251K | 669K |
| [re2](#re2-regex-engine) | **14,066K** | 6,206K | 6,079K | 9,787K |
| [harfbuzz](#harfbuzz-text-shaping-engine) | 11,289K | 11,027K | **15,886K** | 3,195K |
| [libpng](#libpng-image-decoder) | **19K** | 14K | 14K | 10K |

RL models often have higher AUC than the same-steps baseline despite lower
final coverage. This is because RL runs 10-30x longer in wall-clock time
(3,500-9,400s vs 137-493s), accumulating more area simply by running longer,
not by finding coverage faster.

---

## Throughput Comparison

| Benchmark | Baseline (execs/s) | M1_0 (steps/s) | M1_1 (steps/s) | M1_2 (steps/s) | RL Overhead |
|-----------|-------------------|-----------------|-----------------|-----------------|-------------|
| jsoncpp | 44,654 | 2,768 | — | — | [**94%**](#where-the-time-goes-per-mutation-step) |
| freetype2 | 24,345 | 2,031 | — | — | [**92%**](#where-the-time-goes-per-mutation-step) |
| libxml2 | 39,203 | 2,046 | — | — | [**95%**](#where-the-time-goes-per-mutation-step) |
| re2 | 19,069 | 1,736 | — | — | [**91%**](#where-the-time-goes-per-mutation-step) |
| harfbuzz | 19,750 | 1,858 | — | — | [**91%**](#where-the-time-goes-per-mutation-step) |
| libpng | 63,850 | 2,123 | — | — | [**97%**](#where-the-time-goes-per-mutation-step) |

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
| [jsoncpp](#jsoncpp-json-parser) | 5,983 | 313 | M1_2 | **NO** |
| [freetype2](#freetype2-font-renderer) | 7,707 | 793 | M1_0 | **NO** |
| [libxml2](#libxml2-xml-parser) | 4,948 | 1,465 | M1_1 | **NO** |
| [re2](#re2-regex-engine) | 15,768 | 2,230 | M1_0 | **NO** |
| [harfbuzz](#harfbuzz-text-shaping-engine) | 4,089 | 1,985 | M1_2 | **NO** |
| [libpng](#libpng-image-decoder) | 0 | 0 | — | TIE |

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
| **Script** | `experiment1.sh` | `experiment2.sh` |
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
| Experiment orchestrator | `scripts/experiment2.sh` | Multi-benchmark loop with crash recovery |
| Milestone checkpoints | `scripts/rl_server.py --milestones` | Save .pt copies at step counts |
| Per-experiment directories | `scripts/run_model.sh --exp-dir` | Prevent multi-benchmark clobbering |
| Milestone slicer | `scripts/visuals/slice_milestones.py` | Post-hoc CSV truncation at milestones |
| Cross-benchmark summary | `scripts/visuals/summarize_benchmarks.py` | FuzzBench-style scores and rankings |
| M1_2 model | `scripts/models/m1_2.py` + `src/mutator_m1_2.c` | 64-dim input-aware DQN |
| Local dependency builds | `packages/local/` | liblzma + libarchive (no sudo) |

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
