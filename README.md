# RL-Guided AFL++ Fuzzer

This is a research fuzzing framework that integrates **AFL++** with **Deep Q-Network (DQN)** agents to learn and optimise mutation strategies at runtime. Instead of applying mutations randomly, It treats mutation selection as a sequential decision problem and learns which of 47 AFL++ mutation primitives is most likely to discover new code coverage for a given execution context.

The project benchmarks **4 DQN model variants** (differing in state representation complexity) against a **plain AFL++ baseline**, producing comparison plots and statistical reports.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Models](#models)
3. [Architecture](#architecture)
4. [Action Space](#action-space)
5. [Reward Function](#reward-function)
6. [IPC: Shared Memory Protocol](#ipc-shared-memory-protocol)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Output Structure](#output-structure)
11. [Metrics and Analysis](#metrics-and-analysis)

---

## System Overview

Each model variant consists of two cooperating processes that communicate through a memory-mapped file:

| Component | Language | Role |
|---|---|---|
| **Custom Mutator** (`src/mutator_m*.c`) | C | AFL++ plugin; collects execution state, writes to SHM, reads action, executes the chosen mutation |
| **RL Server** (`scripts/rl_server.py`) | Python / PyTorch | Hosts the DQN; reads state from SHM, computes actions, runs the training loop, writes CSV metrics |

```
┌──────────────────────────────────────────────────────┐
│  AFL++ Process                                        │
│                                                       │
│  afl_custom_fuzz() — mutator_m*.c                     │
│    1. Collect coverage, edges, crashes from trace_bits │
│    2. Write state to SHM  (RELEASE store)        ─────┤──→  /tmp/rl_shm_<model_id>
│    3. Poll SHM for action (ACQUIRE load)         ←────┤──   (mmap file)
│    4. Apply mutation primitive                        │
└──────────────────────────────────────────────────────┘
                         ↑↓
┌──────────────────────────────────────────────────────┐
│  RL Server — rl_server.py --model-id <id>             │
│    1. Poll SHM for new state  (state_seq sentinel)    │
│    2. Build state vector (model-specific)              │
│    3. DQN forward pass → choose action                 │
│    4. Write action to SHM (action_seq sentinel)        │
│    5. Store (s, a, r, s') → replay buffer → backprop   │
└──────────────────────────────────────────────────────┘
```

---

## Models

All models share the same 47-action discrete action space and the same RL
training loop, differing only in **state representation** and **SHM layout**:

| Model | State Dims | SHM Size | Hidden Layers | State Description |
|---|---|---|---|---|
| **M0_0** | 3 | 128 B | [128, 128, 64] | Basic: `[coverage_n, new_edges_n, crashes_n]` |
| **M1_0** | 12 | 256 B | [128, 128, 64] | Edge stability distribution over all 65536 edges |
| **M1_1** | 13 | 256 B | [128, 128, 64] | Edge stability over visited edges only + visit count |
| **M1_2** | 64 | 512 B | [256, 256, 128] | M1_1 + input-buffer features (length, entropy, byte histogram) |
| **M2** | 97 | 1024 B | [256, 256, 128] | Per-mutator trace-bit magnitudes (47 enabled + 47 disabled averages) |
| **M3_0** | 13 | 128 B | [128, 128, 64] | Differential-derived features: heat distribution, entropy, timing, velocity |

Each model's configuration is defined in a self-contained module under
`scripts/models/m*.py`, exporting SHM layout constants, `build_state()`,
`shm_read()`, and CSV column definitions. M3_0 additionally supports a
contextual-bandit variant (`--algorithm bandit`) alongside the default DQN.

`_skip` variants of M0_0/M1_0/M1_1/M1_2/M2 train every 4th step instead of
every step (≈3.6× RL server throughput at the cost of slower learning).

---

## Action Space

The agent selects from 47 discrete mutation primitives that map directly to AFL++'s internal mutator IDs:

- **Deterministic stages** (16): bit flips (1/2/4 bits, 1/2/4 bytes), arithmetic add/sub (8/16/32-bit, LE/BE)
- **Interesting values** (5): boundary constants (8/16/32-bit, LE/BE)
- **Havoc mutations** (18): random bit flips, arithmetic, byte operations
- **Dictionary operations** (4): user/auto extras overwrite/insert
- **Meta** (2): custom mutator, full havoc
- **Total: 47 actions** (enforced by `assert ACTION_SIZE == 47`)

---

## Reward Function

```
reward = (coverage_now - coverage_prev) + (log1p(crashes_now) - log1p(crashes_prev)) * 1000
```

Coverage deltas are measured in raw edge counts (one new edge = +1.0 reward). Crash discovery provides a large bonus through log-scaled crash count deltas. No step cost penalty is applied (`STEP_COST = 0.0`).

---

## IPC: Shared Memory Protocol

Each model uses its own SHM file at `/tmp/rl_shm_<model_id>`. The C mutator writes execution state (coverage, edges, crashes, and model-specific features) and the Python server reads state, computes an action, and writes it back. Synchronisation uses monotonically incrementing sequence numbers with GCC atomic builtins (`__ATOMIC_RELEASE` / `__ATOMIC_ACQUIRE`).

---

## Installation

### External dependencies and where they live

The build framework expects three things outside this repo. Defaults can be
overridden with environment variables.

| Dependency | Default location | Override | Notes |
|---|---|---|---|
| **AFL++** (built from source) | `~/packages/AFLplusplus` | `$AFL_ROOT` | Must be built including `libAFLDriver.a`. |
| **FuzzBench** clone | `~/fuzzbench` | `$FUZZBENCH` | Source of pinned commits and harnesses for each benchmark recipe. |
| **Benchmark sources** | `~/targets/<project>/src` | `$TARGETS_DIR` | `build_benchmark.sh` clones each project under `$TARGETS_DIR/<project>/src` at the FuzzBench-pinned commit on first build. |

System packages (Debian/Ubuntu names): `build-essential g++ libstdc++-dev git clang-18 zlib1g-dev liblzma-dev autoconf automake libtool pkg-config`.

Build tools that recipes need (any of these — install via apt, or pip into `.venv`; the build framework auto-adds `.venv/bin` to `PATH`):

| Tool | Used by recipes | apt package | venv install |
|---|---|---|---|
| `cmake` | jsoncpp | `cmake` | `pip install cmake` |
| `meson` + `ninja` | freetype2, harfbuzz | `meson ninja-build` | `pip install meson ninja` |
| `make` | re2, libxml2, libpng | `build-essential` (already required) | — |

If `libstdc++-dev` is not installed but `g++` is (only `libstdc++.so.6` exists on the search path, no unversioned `libstdc++.so`), `build_benchmark.sh` auto-detects this and points clang at the gcc install dir via `--gcc-install-dir=` so the link step succeeds anyway.

### One-time setup

```bash
# 1. AFL++ — build from source, including the libFuzzer driver
git clone https://github.com/AFLplusplus/AFLplusplus ~/packages/AFLplusplus
cd ~/packages/AFLplusplus
make -j"$(nproc)"
make libAFLDriver.a
LLVM_CONFIG=llvm-config-18 make -f GNUmakefile.llvm    # LLVM mode (afl-clang-fast)

# 2. FuzzBench — needed for benchmark commit pinning, harnesses, dictionaries, seeds
git clone --depth=1 https://github.com/google/fuzzbench.git ~/fuzzbench

# 3. This project — Python venv with PyTorch + plotting + (optional) build tools
cd ~/projects/rl-fuzzer
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scipy matplotlib
pip install cmake meson ninja          # only if not installed system-wide

# 4. Optional — point the build framework at non-default locations
export AFL_ROOT=~/packages/AFLplusplus
export FUZZBENCH=~/fuzzbench
```

### Build a target

Build any FuzzBench benchmark target (`jsoncpp`, `freetype2`, `libxml2`,
`re2`, `harfbuzz`, `libpng`):

```bash
bash scripts/build_benchmark.sh jsoncpp        # or any other recipe name
```

The driver clones the project source into `~/targets/<project>/src` at the
FuzzBench-pinned commit, builds with AFL++ instrumentation, and writes:
`bin/target` (instrumented binary), `dictionaries/target.dict`, and seeds
into `inputs/`. See [`docs/experiment_2.md#building-benchmark-targets-scriptsbuild_benchmarksh`](docs/experiment_2.md#building-benchmark-targets-scriptsbuild_benchmarksh)
for the recipe contract and per-benchmark quirks.

For **Experiment 3**, the four libxml2 differential targets must be
pre-built into `experiments/differential/targets/{xml005_buggy,xml005_fixed,xml017_buggy,xml017_fixed}/target`.
See [`docs/experiment_3.md#prerequisites`](docs/experiment_3.md#prerequisites)
for the build commands.

### Optional kernel tuning (recommended for stable benchmarks)

```bash
echo core | sudo tee /proc/sys/kernel/core_pattern
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Usage

Three end-to-end experiment scripts live under `scripts/`. Each command
below uses sensible defaults and is enough to run the experiment as-is —
no other reading required. For tuning, every flag is documented in the
linked per-experiment doc.

### Experiment 1 — single-target, multi-model with statistical aggregation

```bash
bash scripts/build_benchmark.sh jsoncpp     # one-time: builds bin/target

# Reproduce the published Experiment 1 results (8 models + same-steps & same-time baselines, ~3.7h):
bash scripts/experiment1.sh \
    --models m0_0,m1_0,m1_1,m2,m0_0_skip,m1_0_skip,m1_1_skip,m2_skip \
    --run-baseline

# Or a quicker variant (5 standard models, no baseline, ~55 min):
bash scripts/experiment1.sh
```

Trains each RL model on `bin/target`, runs 5 eval rounds per model with
the frozen policy, and writes the same-steps comparison report and plots
to `comparison_results/same_steps/` (plus `same_time/` if `--run-baseline`).
The published doc was produced by the first command; the bare default uses
the modern model set (includes M1_2 added later, omits skip variants and
baselines) and is fine for re-runs without strict reproduction.

→ Full doc: [`docs/experiment_1.md`](docs/experiment_1.md)
&nbsp;·&nbsp; [Every flag](docs/experiment_1.md#flags)
&nbsp;·&nbsp; [Smoke test](docs/experiment_1.md#quick-smoke-test-3-min)
&nbsp;·&nbsp; [Results](docs/experiment_1.md#results)

### Experiment 2 — multi-benchmark with milestone snapshots

```bash
bash scripts/experiment2.sh                 # full run, hours-to-days
```

Iterates the six FuzzBench benchmarks
(`jsoncpp,freetype2,libxml2,re2,harfbuzz,libpng`) × three models
(`m1_0,m1_1,m1_2`). Builds each target via `build_benchmark.sh`, trains
with milestone checkpoints at 500K/1M/2M/10M steps, runs same-steps and
same-time baselines, slices each eval CSV at every milestone, then emits a
cross-benchmark summary. Resumable — partial runs skip work that's already
complete.

→ Full doc: [`docs/experiment_2.md`](docs/experiment_2.md)
&nbsp;·&nbsp; [Every flag](docs/experiment_2.md#flags)
&nbsp;·&nbsp; [Benchmark build framework](docs/experiment_2.md#building-benchmark-targets-scriptsbuild_benchmarksh)
&nbsp;·&nbsp; [Smoke test](docs/experiment_2.md#quick-smoke-test-5-min)
&nbsp;·&nbsp; [Results](docs/experiment_2.md#results)

### Experiment 3 — differential-informed RL fuzzing (M3_0)

```bash
bash experiments/differential/build/build_libxml2_targets.sh   # one-time: builds 4 ASAN libxml2 targets, ~15 min
bash scripts/experiment3.sh                                    # full run, ~12 hours
```

The first command clones libxml2 at four git tags (v2.9.3, v2.9.4 ×2, v2.9.5),
builds each with AFL++ instrumentation + ASAN, and writes the four target
binaries into `experiments/differential/targets/{xml005_buggy,xml005_fixed,xml017_buggy,xml017_fixed}/target`.
Pass `--clean` to force a fresh re-clone.

Trains three variants (M3_0 DQN, M3_0 contextual bandit, M1_0) on
`xml005_buggy`, then evaluates each plus a vanilla AFL++ baseline on
`xml005_buggy` (in-distribution) and `xml017_buggy` (transfer), 5 runs per
target × variant.

→ Full doc: [`docs/experiment_3.md`](docs/experiment_3.md)
&nbsp;·&nbsp; [Every flag](docs/experiment_3.md#flags)
&nbsp;·&nbsp; [Prerequisites](docs/experiment_3.md#prerequisites)
&nbsp;·&nbsp; [Smoke test](docs/experiment_3.md#quick-smoke-test-4-min)
&nbsp;·&nbsp; [Full technical report](docs/experiment_3.md#full-technical-report--setup-methodology-results-and-analysis)
&nbsp;·&nbsp; [Known bugs](docs/experiment_3.md#appendix-a--static-analysis-known-bugs-in-m3_0-implementation)

### Quick smoke tests (~3–5 min each)

If you just want to confirm the toolchain works end-to-end before
committing to a full run. Each is self-contained for a fresh clone — no
pre-existing checkpoints required, just the per-experiment build step
described above.

```bash
# After: bash scripts/build_benchmark.sh jsoncpp
bash scripts/experiment1.sh --train-steps 2000 --eval-steps 3000 --eval-runs 1 --models m0_0 --no-plateau --out comparison_results/smoke1

# Self-contained (uses --skip-build to reuse the jsoncpp bin/target above):
bash scripts/experiment2.sh --skip-build --benchmarks jsoncpp --models m1_0 --milestones 2000 --train-steps 2000 --eval-steps 2000 --eval-runs 1 --no-plateau --exp-root experiments/smoke2

# After: bash experiments/differential/build/build_libxml2_targets.sh
bash scripts/experiment3.sh --train-steps 1500 --eval-steps 1500 --eval-runs 1
```

### Lower-level entry points

If you only need to drive a single model (no comparison/aggregation):

```bash
bash scripts/run_model.sh --model-id m0_0 --train-steps 50000 --eval-steps 20000
bash scripts/run_model.sh --model-id m2   --eval-only        --eval-steps 20000
```

Or invoke the RL server directly:

```bash
python3 scripts/rl_server.py --model-id m0_0 --mode train --train-steps 50000
python3 scripts/rl_server.py --model-id m2   --mode eval  --eval-steps 20000 --model bin/rl_m2.pt
```

---

## Project Structure

```
rl-fuzzer/
├── src/
│   ├── mutator_m0_0.c          # Basic 3-feature state (128 B SHM)
│   ├── mutator_m1_0.c          # Edge-stability state (256 B SHM)
│   ├── mutator_m1_1.c          # Visited-edge stability + visit count (256 B SHM)
│   ├── mutator_m1_2.c          # M1_1 + input-buffer features (512 B SHM)
│   ├── mutator_m2.c            # Per-action magnitudes (1024 B SHM)
│   ├── mutator_m3_0.c          # 13-feature differential-derived state (Experiment 3)
│   └── mutator_telemetry.c     # Random-mutation telemetry collection (used to derive M3_0)
├── scripts/
│   ├── experiment1.sh          # Entry point: single-target multi-model with stats
│   ├── experiment2.sh          # Entry point: multi-benchmark with milestones
│   ├── experiment3.sh          # Entry point: M3_0 differential fuzzing
│   ├── build_and_compare.sh    # Orchestrator used by experiment1
│   ├── build_benchmark.sh      # Generic FuzzBench target builder
│   ├── run_model.sh            # Unified train+eval shell runner (--model-id)
│   ├── rl_server.py            # Unified RL server entry point (--model-id)
│   ├── models/                 # Per-model state/SHM modules + DQN/Bandit common code
│   └── visuals/                # Plotting + comparison + cross-benchmark summary
│       ├── compare_metrics.py
│       ├── slice_milestones.py
│       └── summarize_benchmarks.py
├── benchmarks/                 # Per-benchmark build_recipe.sh files
├── docs/
│   ├── experiment_1.md         # Run instructions + results for experiment1.sh
│   ├── experiment_2.md         # Run instructions + benchmark build framework + results
│   └── experiment_3.md         # Run instructions + technical report + known bugs
├── inputs/                     # Seed corpus
├── dictionaries/               # AFL++ dictionaries
└── bin/                        # Build outputs (mutator .so, target, .pt checkpoints)
```

### Adding a New Model

1. Create `scripts/models/m_new.py` implementing the module interface:
   - `STATE_SIZE`, `SHM_SIZE`, `SHM_PATH`, `MODEL_PATH_DEFAULT`, `LABEL`, `HIDDEN_LAYERS`
   - `STATE_SEQ_OFF`, `ACTION_OFF`, `ACTION_SEQ_OFF`
   - `CSV_EXTRA_HEADER`
   - `shm_read(shm, shm_size) -> dict`
   - `build_state(d, train_steps) -> np.ndarray`
   - `zero_state_data() -> dict`
   - `csv_extra_fields(d, args) -> str`
   - `log_extra(d, args) -> str`
   - `exit_summary(d, step, cov, cr, epsilon, tag) -> None`
2. Add the ID to `MODEL_IDS` in `scripts/models/__init__.py`
3. Create a corresponding `src/mutator_m_new.c` with matching SHM layout
4. Run: `bash scripts/run_model.sh --model-id m_new`

---

## Output Structure

```
plots/<model_id>/
  rl_metrics_<model_id>_train.csv    # Training metrics (every 100 steps)
  rl_metrics_<model_id>_eval.csv     # Eval metrics
  fuzzer_stats_train.txt             # AFL++ fuzzer_stats snapshot
  fuzzer_stats_eval.txt

comparison_results/
  comparison_report.txt              # Mean +/- std table (if --multi-run)
  comparison_summary.json
  plot_coverage_eval_steps.png
  plot_coverage_eval_time.png
  plot_coverage_bar_eval.png
  plot_throughput_eval.png
  plot_coverage_per_sec_eval.png
```

---

## Metrics and Analysis

Training and eval metrics are written to CSV every 100 steps. The base columns shared by all models:

| Column | Description |
|---|---|
| `step` | Global step counter |
| `reward` | Reward received |
| `coverage_term` | Coverage delta component of reward |
| `crash_term` | Crash delta component of reward |
| `loss` | DQN TD loss |
| `epsilon` | Current exploration rate |
| `coverage` | AFL++ edge coverage count |
| `crashes` | Total crashes |
| `action` | Action chosen by the agent |
| `elapsed_seconds` | Wall-clock time since start |

Model-specific extra columns:

| Model | Extra Columns |
|---|---|
| M0_0 | (none) |
| M1_0 | `en_mean_n`, `dis_mean_n`, `stability` |
| M1_1 | `num_visited`, `stability` |
| M2 | `mean_avg_en`, `mean_avg_dis`, `top_en_action`, `top_dis_action`, `nonzero_mag_frac` |

Generate comparison plots and reports:

```bash
python3 scripts/visuals/compare_metrics.py \
    --m0-0 plots/m0_0 --m1-0 plots/m1_0 --m1-1 plots/m1_1 --m2 plots/m2 \
    --out comparison_results/ --phase eval
```
