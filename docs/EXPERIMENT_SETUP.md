# Multi-Benchmark Experiment Setup

This document summarizes all changes made to take the RL-fuzzer project from
single-benchmark (jsoncpp-only) to a 6-benchmark, 3-model, milestone-based
experiment framework. It covers what was built, why, and how the pieces fit
together.

---

## 1. Goal

Run 3 DQN models (M1_0, M1_1, M1_2) + 2 baselines (same-steps, same-time)
across 6 FuzzBench benchmarks, with eval snapshots at 500K, 1M, 2M, and 10M
steps. Produce per-benchmark milestone comparisons and a cross-benchmark
summary.

## 2. New Model: M1_2

**Files:** `scripts/models/m1_2.py`, `scripts/models/m1_2_skip.py`, `src/mutator_m1_2.c`

M1_2 extends M1_1's 13-dimensional state with 51 input-buffer features:
- `buf_len_norm`, `entropy_norm`, `printable_ratio` (3 scalar features)
- `histogram[16]` — byte-value histogram binned into 16 buckets
- `first_32_bytes[32]` — normalized first 32 bytes of the input

Total state dimension: 64. Network: [256, 256, 128] hidden layers.
SHM region: 512 bytes at `/tmp/rl_shm_m1_2` (action at offset 256).

Registered in `scripts/models/__init__.py` and `scripts/compare_metrics.py`.

## 3. Generic Benchmark Build Framework

**Files:** `scripts/build_benchmark.sh`, `benchmarks/*/build_recipe.sh`

Previously, only `scripts/build_jsoncpp.sh` existed (a monolithic 239-line
script). This was replaced with a generic framework:

- `scripts/build_benchmark.sh` — universal entry point that handles:
  - AFL++ prerequisite checks
  - LLVM detection
  - Source cloning at FuzzBench-pinned commits
  - Delegating to per-benchmark recipes
  - Dictionary and seed corpus installation
  - Smoke testing
  - Directory structure setup

- `benchmarks/<name>/build_recipe.sh` — each recipe defines:
  - `FUZZBENCH_NAME` — FuzzBench benchmark directory name
  - `GIT_URL` — upstream source repository
  - `BUILD_STEPS()` — configure + compile with AFL++ instrumentation
  - `LINK_STEPS()` — link harness + static lib + AFL driver into `bin/target`

### Available Benchmarks

| Benchmark | FUZZBENCH_NAME | Harness Source | Dependencies |
|-----------|---------------|----------------|--------------|
| jsoncpp | jsoncpp_jsoncpp_fuzzer | Source tree (`fuzz.cpp`) | none |
| freetype2 | freetype2_ftfuzzer | Source tree (`ftfuzzer.cc`) | libarchive (local) |
| libxml2 | libxml2_xml | FuzzBench (`target.cc`) | liblzma (local) |
| re2 | re2_fuzzer | FuzzBench (`target.cc`) | none |
| harfbuzz | harfbuzz_hb-shape-fuzzer | Source tree (`hb-shape-fuzzer.cc`) | meson (from .venv) |
| libpng | libpng_libpng_read_fuzzer | Source tree (`libpng_read_fuzzer.cc`) | none |

### Seeds and Dictionaries

| Benchmark | Seeds | Dictionary |
|-----------|-------|------------|
| jsoncpp | Recipe-generated JSON files | Source tree (`fuzz.dict`) |
| freetype2 | Fallback (`"FUZZ"`) | Fallback minimal |
| libxml2 | Fallback (`"FUZZ"`) | Source tree (`fuzz/xml.dict`) |
| re2 | Fallback (`"FUZZ"`) | Fallback minimal |
| harfbuzz | Fallback (`"FUZZ"`) | Fallback minimal |
| libpng | FuzzBench (`seed.png`) | Source tree (`png.dict`) |

## 4. Local Dependency Builds

**Directory:** `packages/local/` (gitignored)

Two system packages (`libarchive-dev`, `liblzma-dev`) were unavailable and we
lacked sudo. They were built from source:

```
packages/xz/        → packages/local/lib/liblzma.a
packages/libarchive/ → packages/local/lib/libarchive.a
```

`build_benchmark.sh` wires these in automatically:
- Sets `PKG_CONFIG_PATH` to `packages/local/lib/pkgconfig`
- Appends `-I` and `-L` flags to `CFLAGS`, `CXXFLAGS`, `LDFLAGS`
- Adds `.venv/bin` to `PATH` so meson is discoverable

## 5. Experiment Orchestration

### run_full_experiment.sh

**File:** `scripts/run_full_experiment.sh`

Top-level orchestrator that loops over benchmarks and models:

```
For each benchmark:
  Phase 0: Build target (build_benchmark.sh)
  Phase 1: Train all models (run_model.sh with --milestones)
  Phase 2: Baseline same-steps (AFL++ with -E <steps>)
  Phase 3: Baseline same-time (AFL++ with -V <seconds>)
  Phase 4: Slice milestones + compare (slice_milestones.py)

After all benchmarks:
  Cross-benchmark summary (summarize_benchmarks.py)
```

Key flags:
- `--benchmarks LIST` — comma-separated benchmark names
- `--models LIST` — comma-separated model IDs (default: m1_0,m1_1,m1_2)
- `--milestones LIST` — step counts for snapshots (default: 500k,1m,2m,10m)
- `--no-plateau` — disable early-stopping
- `--skip-train` — reuse existing checkpoints
- `--skip-build` — reuse existing bin/target

**Crash recovery:** The script checks for existing checkpoints and eval CSVs
before each step. If a run is interrupted and restarted, it skips completed
work automatically.

### run_model.sh --exp-dir

`run_model.sh` gained `--exp-dir DIR` to redirect all outputs (checkpoints,
AFL outputs, plots) into an experiment-specific directory tree, preventing
multi-benchmark runs from clobbering each other.

### rl_server.py --milestones

`rl_server.py` gained `--milestones LIST` to save checkpoint copies at
specified training step counts (e.g. `rl_m1_0.pt.500k`, `rl_m1_0.pt.1m`).
CSV writes are now streamed with `flush()` for crash resilience.

### slice_milestones.py

**File:** `scripts/slice_milestones.py`

Post-hoc CSV slicer. Takes a full 10M-step eval CSV and produces milestone
snapshots by truncating at each milestone step count. Also supports
`--query-time` to find the median RL eval wall-clock time (used to set the
same-time baseline budget).

### summarize_benchmarks.py

**File:** `scripts/summarize_benchmarks.py`

Cross-benchmark aggregation. Reads milestone results from all benchmarks and
produces summary tables and plots comparing model performance across targets.

## 6. Output Structure

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
│       ├── 500k/               # sliced CSVs for each model + baselines
│       ├── 1m/
│       ├── 2m/
│       └── 10m/
├── freetype2/
│   └── (same structure)
├── ...
└── summary/                    # cross-benchmark tables and plots
```

## 7. Comparison Enhancements

**File:** `scripts/compare_metrics.py`

- Added M1_2 model definition and color scheme
- Added **coverage AUC** metric (area under coverage-over-time curve)
- Added optional **Mann-Whitney U** statistical tests (via scipy)

## 8. Performance Optimizations

- **Mutator edge counting**: All 4 mutators (m0_0, m1_0, m1_1, m2) use
  8-byte chunk scanning to skip fully-saturated regions in the coverage map
- **DQNAgent inference**: Pre-allocated tensor buffer avoids per-step
  allocation in `select_action()`
- **Mutator compilation**: Added `-march=native -ffast-math` flags

## 9. Commit History

```
f0a68f6 fix: correct libpng harness path to oss-fuzz/libpng_read_fuzzer.cc
baf9394 feat: add local dependency builds and wire into benchmark framework
e931799 feat: add M1_2 model with input-buffer features (64-dim state)
e12e3e8 perf: optimize edge counting in mutators with 8-byte chunk scan
ef4344d feat: add milestone checkpoints and per-experiment directory support
992ff31 feat: add coverage AUC, Mann-Whitney U tests, and M1_2 to comparison
e725d13 feat: add multi-benchmark experiment orchestration
754110e docs: add benchmark build guide and experiment status script
```

## 10. Running the Experiment

```bash
# Build all dependencies (one-time, if packages/local/ doesn't exist)
cd packages && git clone --depth=1 https://github.com/tukaani-project/xz.git
cd xz && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../local -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc) && make install && cd ../../..

cd packages && git clone --depth=1 https://github.com/libarchive/libarchive.git
cd libarchive && mkdir cmake-build && cd cmake-build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../local -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DENABLE_TEST=OFF -DENABLE_TAR=OFF -DENABLE_CPIO=OFF -DENABLE_CAT=OFF
make -j$(nproc) && make install && cd ../../..

# Run full experiment (all 6 benchmarks)
bash scripts/run_full_experiment.sh --no-plateau

# Or specific benchmarks
bash scripts/run_full_experiment.sh --benchmarks jsoncpp,re2 --no-plateau

# Resume after crash (skips completed work automatically)
bash scripts/run_full_experiment.sh --benchmarks <remaining> --no-plateau

# Check status
bash scripts/status.sh
```
