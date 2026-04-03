# Differential Fuzzing Experiment — Claude Code Execution Plan

## Revision History

This document was reviewed and 10 ambiguities/contradictions were fixed:

| # | Issue | Resolution |
|---|-------|------------|
| 1 | ASAN: plan said `-fsanitize=address` but FUZZBENCH_SETUP.md says never use it manually | Changed to `AFL_USE_ASAN=1` (AFL++ handles ASAN consistently) |
| 2 | Prerequisites assumed `~/fuzzbench/` and `~/packages/AFLplusplus/` exist | Added Phase 0 prerequisite verification |
| 3 | FuzzBench path `libxml2-v2.9.2` doesn't match existing recipe `libxml2_xml` | Changed all references to `libxml2_xml` |
| 4 | CVE-2016-1762 fix commit was unspecified | Pinned to `v2.9.4` tag, fix commit `a7a94612aa3b16779e2c74e1fa353b5d9786c602` |
| 5 | Telemetry mutator can't observe AFL++ internal stages with MUTATOR_ONLY=0 | Changed to MUTATOR_ONLY=1 with random mutation selection (full attribution) |
| 6 | Separate `rl_server_m_star.py` breaks unified server architecture | Integrated: add `ContextualBanditAgent` to `common.py`, `--algorithm` to `rl_server.py` |
| 7 | `packages/local/` for liblzma doesn't exist | Check system `liblzma-dev` first, build from source only if needed |
| 8 | Bitmap snapshot dumps `afl->shm.map` which resets between execs | Changed to `cumulative_map` maintained in mutator; `total_edges` uses `virgin_bits` |
| 9 | Harness compatibility across v2.9.3-v2.9.5 unverified | Try FuzzBench `target.cc`, fall back to `fuzz/xml.c`; STOP on failure |
| 10 | Comparison model `M1_0_skip` not in Experiment 2 | Changed to `M1_0` (most consistent RL model from Experiment 2) |

## Context

You are modifying an existing RL-guided AFL++ fuzzing project located at `~/projects/rl-fuzzer/`. The project has this structure:

```
~/projects/rl-fuzzer/
  src/                    # C mutator source files (mutator_m0_0.c, etc.)
  scripts/                # Python RL servers, run scripts, comparison tools
  bin/                    # Compiled binaries (.so mutators, .pt models, target binary)
  inputs/                 # Seed corpus
  dictionaries/           # Fuzzing dictionaries (target.dict)
  outputs/                # AFL++ training output per model
  outputs_eval/           # AFL++ eval output per model
  plots/                  # Metrics CSVs per model
  comparison_results/     # Final comparison reports
```

Supporting infrastructure:
- AFL++ is installed at `~/packages/AFLplusplus` (built from source, LLVM mode with afl-clang-fast)
- FuzzBench repo is at `~/fuzzbench/`
- The project currently fuzzes jsoncpp using custom AFL++ mutator plugins (.so) that communicate with a Python DQN RL server via mmap'd shared memory at `/tmp/rl_shm_m*`
- Python environment uses PyTorch (CPU), NumPy, Matplotlib

## Goal

Build an instrumented differential fuzzing experiment that:
1. Fuzzes a BUGGY vs FIXED version of libxml2 using vanilla AFL++ (no RL) while collecting rich coverage telemetry
2. Analyzes the differential telemetry to identify features that correlate with bug-proximity
3. Designs and trains a new model M* using those features within the existing RL setup
4. Evaluates whether M* finds bugs faster than baseline AFL++ and existing M1_0 model

## Phase 0: Prerequisites & Environment Verification

Before any build work, verify all dependencies exist:

1. **AFL++**: `~/packages/AFLplusplus/` must exist with LLVM mode built. If missing:
   ```bash
   git clone https://github.com/AFLplusplus/AFLplusplus.git ~/packages/AFLplusplus
   cd ~/packages/AFLplusplus && make -j$(nproc)
   LLVM_CONFIG=llvm-config-16 make -f GNUmakefile.llvm
   ```
2. **FuzzBench**: `~/fuzzbench/` must exist. If missing:
   ```bash
   git clone --depth=1 https://github.com/google/fuzzbench.git ~/fuzzbench
   ```
3. **System libraries**: `pkg-config --libs liblzma` and `pkg-config --libs zlib` must succeed.
4. **Python venv**: `.venv` with torch, numpy, matplotlib, pandas, scipy.
5. **Kernel tuning**: `core_pattern` set to `core`, CPU governor set to `performance`.

**STOP if** any prerequisite cannot be satisfied. Report exactly what failed.

After verification, confirm the FuzzBench libxml2 benchmark exists:
- Check `~/fuzzbench/benchmarks/libxml2_xml/` (this is the correct FuzzBench name, matching `benchmarks/libxml2/build_recipe.sh`)
- Read `benchmark.yaml` for the pinned commit
- Locate harness file, seeds, and dictionary
- Report all exact paths found

## Phase 1: Build Infrastructure for Buggy/Fixed libxml2 Pairs

### Step 1.1: Create the experiment directory structure

```
~/projects/rl-fuzzer/experiments/differential/
  build/                  # Build scripts
  targets/                # Compiled buggy/fixed binaries
    xml005_buggy/         # CVE-2017-5130 (integer overflow in xmlmemory.c)
    xml005_fixed/
    xml017_buggy/         # CVE-2016-1762 (heap overread in parser.c)
    xml017_fixed/
  seeds/                  # libxml2 seed corpus
  dictionaries/           # libxml2 fuzzing dictionary
  telemetry/              # Raw telemetry output
  analysis/               # Analysis scripts and results
```

### Step 1.2: Write `build_libxml2_targets.sh`

**CRITICAL STANDARDIZATION RULE**: Everything except the libxml2 source commit MUST come from FuzzBench's `libxml2_xml` benchmark (located at `~/fuzzbench/benchmarks/libxml2_xml/`). This ensures our differential experiment is comparable to published FuzzBench results. Follow the same pattern as the existing `benchmarks/libxml2/build_recipe.sh` in this repo.

This script must:

1. **Extract FuzzBench's build recipe, harness, seeds, and dictionary FIRST:**
   ```bash
   FUZZBENCH_DIR="$HOME/fuzzbench"
   BENCHMARK_DIR="$FUZZBENCH_DIR/benchmarks/libxml2_xml"
   ```
   - Read `$BENCHMARK_DIR/benchmark.yaml` to understand FuzzBench's build configuration.
   - Check `$BENCHMARK_DIR/` for `Dockerfile`, `build.sh`, and harness files (`target.cc`).
   - If the benchmark uses the OSS-Fuzz integration pattern (no local Dockerfile/build.sh), pull the build.sh from OSS-Fuzz's libxml2 project at `https://github.com/google/oss-fuzz/tree/master/projects/libxml2`.
   - **Extract the harness source file** — try `$BENCHMARK_DIR/target.cc` first, then fall back to `$SRC_DIR/fuzz/xml.c`. Copy to `experiments/differential/build/harness.c`. This SAME harness file is used for ALL 4 builds.
   - **Copy seeds** from FuzzBench's seed corpus for libxml2 to `experiments/differential/seeds/`. These SAME seeds are used for ALL runs.
   - **Copy dictionary** from FuzzBench's benchmark directory to `experiments/differential/dictionaries/libxml2.dict`. This SAME dictionary is used for ALL runs.

2. **Clone libxml2 and checkout the 4 versions:**
   Clone from `https://gitlab.gnome.org/GNOME/libxml2.git`, then for each bug pair:
   - **XML005 (CVE-2017-5130)**: Use tag `v2.9.4` as buggy, `v2.9.5` as fixed. The fix commit is `897dffbae322b46b83f99a607d527058a72c51ed` (adds overflow check in xmlMemoryStrdup).
   - **XML017 (CVE-2016-1762)**: Heap buffer overread in xmlNextChar in parserInternals.c. Use `v2.9.3` as buggy, `v2.9.4` as fixed. The fix commit is `a7a94612aa3b16779e2c74e1fa353b5d9786c602` (committed 2016-02-09, included in v2.9.4 release). Note: `v2.9.4` serves double duty — it is xml005_buggy (has CVE-2017-5130) AND xml017_fixed (has CVE-2016-1762 fix). A known PoC exists from Bugzilla #759671.

3. **For each of the 4 versions, build with the SAME flags as FuzzBench:**
   Use the configure flags from the existing `benchmarks/libxml2/build_recipe.sh`, with AFL++ instrumentation and ASAN via `AFL_USE_ASAN=1`:
   ```bash
   export CC="$HOME/packages/AFLplusplus/afl-clang-fast"
   export CXX="$HOME/packages/AFLplusplus/afl-clang-fast++"
   export AFL_USE_ASAN=1   # AFL++ injects -fsanitize=address consistently
   export CFLAGS="-g -O2"
   export CXXFLAGS="-g -O2"
   ./autogen.sh \
       --prefix="$BUILD_DIR/install" \
       --disable-shared --without-python --without-debug \
       --without-ftp --without-http \
       CC="$CC" CXX="$CXX" CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS"
   make -j$(nproc)
   make install
   ```
   **Why `AFL_USE_ASAN=1` instead of `-fsanitize=address`**: The project's FUZZBENCH_SETUP.md documents that manually adding `-fsanitize=address` to CFLAGS causes ASAN module constructors in library objects without the ASAN runtime linked, resulting in SIGSEGV before the fork server starts. `AFL_USE_ASAN=1` tells AFL++ to handle ASAN consistently across both compilation and linking. This applies to ALL 4 builds for comparable throughput.

   Then compile the FuzzBench harness against each version's `libxml2.a`:
   ```bash
   AFL_USE_ASAN=1 $CXX $CXXFLAGS harness.c \
       -I $BUILD_DIR/install/include/libxml2 \
       $BUILD_DIR/install/lib/libxml2.a \
       $HOME/packages/AFLplusplus/libAFLDriver.a \
       -lz -llzma -o target
   ```

4. **Output compiled binaries** to `targets/xml005_buggy/target`, `targets/xml005_fixed/target`, `targets/xml017_buggy/target`, `targets/xml017_fixed/target`.

5. **Verify builds** by running: `echo '<a/>' | ./target` — should not crash on any version.

**Why this matters**: If we invent our own harness/seeds/dict, our baseline AFL++ numbers won't be comparable to published FuzzBench results, and reviewers will question whether observed differences are due to the RL model or due to different test infrastructure.

### Step 1.3: Verify bug reachability

Write `verify_bugs.sh` that:
1. If OSS-Fuzz PoC inputs exist for these CVEs, download them and run through the buggy target (should crash with ASAN) and fixed target (should not crash).
2. If no PoC available, do a quick 5-minute vanilla AFL++ run on the buggy target and check if crashes appear.
3. Log results to `experiments/differential/build/verification.log`.

## Phase 2: Instrumented Telemetry Collection

### Step 2.1: Write `mutator_telemetry.c` — a custom AFL++ mutator that logs coverage telemetry

This is a NEW mutator file in `src/`. It does NOT do RL action selection — it selects mutations **randomly** (uniform over the same 47 actions as the RL mutators) and logs every choice and its coverage effect. It runs with `AFL_CUSTOM_MUTATOR_ONLY=1` so it controls ALL mutations, enabling complete per-mutation attribution.

It must collect and log these metrics at configurable intervals (default: every 1,000 executions):

**Coverage dynamics (logged as a CSV row per interval):**
- `timestamp_us`: microsecond timestamp
- `total_execs`: cumulative executions so far
- `total_edges`: count of discovered edges via `count_coverage(afl)` using `afl->virgin_bits` (matching existing mutators — counts bytes != 0xFF)
- `new_edges_this_interval`: edges discovered since last snapshot
- `edge_discovery_rate`: new_edges_this_interval / interval_size
- `crashes_total`: cumulative unique crashes
- `crashes_this_interval`: crashes since last snapshot
- `avg_exec_time_us`: average execution time in this interval (from `afl->fsrv.total_execs` and timing)
- `corpus_size`: number of entries in AFL++'s queue
- `favored_count`: number of favored queue entries (if accessible)

**Edge heat distribution (logged every interval):**
- `hot_edges`: count of edges with hit_count > 128
- `warm_edges`: count of edges with hit_count 8-128
- `cool_edges`: count of edges with hit_count 1-7
- `cold_edges`: count of edges with hit_count 0 (or: 65536 - total_edges)
- `edge_entropy`: Shannon entropy of the hit-count distribution over nonzero edges
- `edge_hit_mean`: mean hit count over nonzero edges
- `edge_hit_std`: standard deviation of hit counts over nonzero edges
- `edge_hit_max`: maximum hit count

**Per-mutation attribution (logged every interval):**
- For each of the 47 mutation types, track:
  - `mut_N_count`: times mutation N was used this interval
  - `mut_N_new_edges`: new edges attributed to mutation N this interval
  - `mut_N_crashes`: crashes attributed to mutation N this interval
- Implementation: in `afl_custom_fuzz()`, select a random action (0-46 uniform), apply it using the same `apply_mutation()` dispatch as existing RL mutators (copied from `mutator_m0_0.c`). Record the action chosen. After the next call to `afl_custom_fuzz()`, check if coverage changed (compare `count_coverage()` before/after) and attribute new edges to the previous action.
- NOTE: The telemetry mutator runs with `AFL_CUSTOM_MUTATOR_ONLY=1`, meaning it controls ALL mutations. This enables complete per-mutation attribution. The mutator uses the same 47-action dispatch table as the RL mutators, making telemetry data directly comparable to RL training data.

**Snapshot of cumulative edge bitmap (logged less frequently — every 10,000 executions):**
- The mutator maintains a `uint8_t cumulative_map[65536]` updated each execution: `cumulative_map[i] = max(cumulative_map[i], afl->shm.map[i])`. This is necessary because `afl->shm.map` is reset between executions and only reflects the last run.
- Dump the full 65,536-byte `cumulative_map` as a raw binary file: `snapshot_<total_execs>.bin`.
- This enables offline analysis of which specific edges were hit at each point in time and their maximum hit counts.

**Output format:**
- Primary CSV: `telemetry/coverage_dynamics_<version>.csv` with one row per interval
- Mutation CSV: `telemetry/mutation_attribution_<version>.csv` with one row per interval (47 * 3 = 141 columns for mutation data)
- Snapshot dir: `telemetry/snapshots_<version>/` with bitmap dumps

**Implementation approach:**
The telemetry mutator should be compiled as a `.so` and loaded via `AFL_CUSTOM_MUTATOR_LIBRARY`. It hooks into `afl_custom_fuzz()` to observe mutations and `afl_custom_init()` / periodic callbacks to snapshot state. Use `afl_custom_queue_new_entry()` to detect new coverage-gaining inputs.

Key AFL++ API functions available in the custom mutator:
- `afl->shm.map` — the 65,536-byte shared trace bitmap
- `afl->queued_items` — number of items in the queue
- `afl->fsrv.total_execs` — total executions
- `afl->stage_cur` — current mutation stage index
- `afl->unique_crashes` — unique crash count

Reference your existing `src/mutator_m0_0.c` for the struct layout and API conventions. The telemetry mutator uses the same `#include "afl-fuzz.h"` interface but does NOT need the SHM IPC to a Python server — it just writes to log files.

### Step 2.2: Write `run_telemetry_campaign.sh`

This script runs the instrumented fuzzing campaign. For each target (xml005_buggy, xml005_fixed, xml017_buggy, xml017_fixed):

1. Compile `mutator_telemetry.c` into `bin/mutator_telemetry.so`
2. Run AFL++ for a fixed time budget (suggest: 1 hour per target, configurable via `--duration` flag):
   ```bash
   # SEEDS_DIR and DICT_FILE MUST point to the FuzzBench-sourced copies:
   SEEDS_DIR="$REPO_ROOT/experiments/differential/seeds"        # copied from FuzzBench in Phase 1
   DICT_FILE="$REPO_ROOT/experiments/differential/dictionaries/libxml2.dict"  # copied from FuzzBench in Phase 1

   AFL_CUSTOM_MUTATOR_LIBRARY="$REPO_ROOT/bin/mutator_telemetry.so" \
   AFL_CUSTOM_MUTATOR_ONLY=1 \
   AFL_USE_ASAN=1 \
   "$AFL_DIR/afl-fuzz" \
       -i "$SEEDS_DIR" \
       -o "$TELEMETRY_DIR/afl_out_${version}" \
       -x "$DICT_FILE" \
       -t 1000 \
       -s "$RANDOM_SEED" \
       -- "$TARGET_BIN"
   ```
   Note: `AFL_CUSTOM_MUTATOR_ONLY=1` means the telemetry mutator controls ALL mutations (random uniform selection over 47 actions), enabling full per-mutation attribution. The same seeds and dictionary are used for ALL 4 target versions — the ONLY variable is the target binary.
3. After each run completes, copy the telemetry CSVs to `telemetry/`.
4. Log wall-clock time, final coverage, and crash count to `telemetry/campaign_summary.json`.

**Important:** Run each version 3 times with different random seeds (`-s 1`, `-s 2`, `-s 3`) for statistical significance. This means 4 versions × 3 seeds = 12 runs total.

### Step 2.3: Write `run_baseline_vanilla.sh`

A simpler script that runs vanilla AFL++ (NO custom mutator) on the same 4 targets for the same duration, also 3 seeds each. This gives you a ground truth for "what does coverage look like without any mutator overhead." Collect standard AFL++ stats from `fuzzer_stats` and `plot_data` files.

## Phase 3: Differential Analysis

### Step 3.1: Write `analysis/differential_analysis.py`

This is the core analysis script. It reads the telemetry CSVs from Phase 2 and produces:

**3.1.1: Coverage trajectory comparison**
- Plot coverage-over-time curves for buggy vs fixed versions (mean ± std across 3 seeds)
- Identify the time window where coverage trajectories diverge between buggy and fixed versions
- Calculate the "divergence point" — the execution count where buggy and fixed coverage curves separate by > 1 std dev

**3.1.2: Differential edge analysis**
- Load the bitmap snapshots from both versions at matched execution counts
- Compute the set difference: edges present in buggy but not in fixed, and vice versa
- These "differential edges" are the bug neighborhood
- Count differential edges over time — when do they first appear?

**3.1.3: Mutation effectiveness analysis**
- For each mutation type, compute the "coverage gain rate" (new edges per use) on buggy vs fixed
- Identify mutations that are disproportionately effective on the buggy version near the divergence point
- Compute per-mutation coverage-gain time series and look for divergence between buggy/fixed

**3.1.4: Coverage landscape feature analysis**
- Compare edge heat distributions (hot/warm/cool/cold ratios) between buggy and fixed at matched timepoints
- Compare edge entropy trajectories
- Look at coverage velocity (first derivative of coverage curve) — does the buggy version show acceleration or deceleration near bug-triggering regions?

**3.1.5: Feature importance report**
- Rank all collected features by their discriminative power between buggy and fixed versions
- Use simple methods: Mann-Whitney U test for each feature at key timepoints, or mutual information between feature values and "buggy vs fixed" label
- Output a ranked list of features with p-values and effect sizes
- Save to `analysis/feature_importance_report.json`

**Output:**
- `analysis/plots/` — all generated plots (PNG)
- `analysis/feature_importance_report.json` — ranked features
- `analysis/differential_edges/` — edge set differences per timepoint
- `analysis/summary.md` — human-readable summary of findings

### Step 3.2: Write `analysis/design_m_star_features.py`

Based on the feature importance report from 3.1.5, this script:
1. Reads the ranked feature list
2. Selects the top-K features (configurable, default K=15) that show statistically significant divergence between buggy and fixed
3. Outputs a feature specification file `analysis/m_star_feature_spec.json` that defines:
   - Feature name, source (which telemetry field), normalization method
   - State vector dimension for M*
   - Recommended network architecture (based on dimension — use [128,128,64] for dim ≤ 20, [256,256,128] for dim > 20)

## Phase 4: Model M* Implementation

### Step 4.1: Write `src/mutator_m_star.c`

A new C mutator that:
1. Collects the features identified in Phase 3 from AFL++'s runtime state
2. Writes them to SHM at `/tmp/rl_shm_m_star` (same IPC protocol as existing mutators)
3. Reads back an action (0-46) from the RL server
4. Applies the selected mutation

This should follow the exact same pattern as `src/mutator_m1_0.c` but with the feature vector defined by `m_star_feature_spec.json`. The feature computation logic goes in the C mutator's `afl_custom_fuzz()` function.

### Step 4.2: Write `scripts/models/m_star.py`

A standard model module following the `m1_2.py` pattern. Exports all required constants and functions (`STATE_SIZE`, `SHM_SIZE`, `SHM_PATH`, `shm_read()`, `build_state()`, etc.) based on the feature spec from `analysis/m_star_feature_spec.json`.

Register `"m_star"` in `scripts/models/__init__.py`.

### Step 4.3: Modify `scripts/models/common.py` — add `ContextualBanditAgent`

Add a `ContextualBanditAgent` class alongside the existing `DQNAgent`:
- Two-head network: per-action mean + log-variance
- Thompson sampling: sample from N(mean, exp(log_var)) per action, pick argmax of samples
- Loss: negative log-likelihood of observed reward under predicted distribution
- No replay buffer, no target network, no discount factor
- Same interface as `DQNAgent` (`select_action`, `train_step`, `save`, `load`)

### Step 4.4: Modify `scripts/rl_server.py` — add `--algorithm` flag

Add `--algorithm {dqn,bandit}` argument (default: `dqn`):
- `dqn`: Uses existing `DQNAgent` (no change to existing behavior)
- `bandit`: Uses new `ContextualBanditAgent`
- Model-id dispatch via `importlib.import_module` remains unchanged

**Why integrate instead of separate server**: The existing architecture uses a single `rl_server.py` with model-id dispatch. Creating a separate `rl_server_m_star.py` would duplicate the training loop, CSV logging, SHM protocol, and milestone handling. Adding `--algorithm` is a minimal change that preserves the established pattern.

### Step 4.5: M* uses existing `scripts/run_model.sh`

No new run script needed. M* is invoked via:
```bash
bash scripts/run_model.sh --model-id m_star --algorithm bandit --train-steps 500000
```

### Step 4.4: Write `scripts/run_m_star_experiment.sh`

The full evaluation script:
1. Train M* on the BUGGY version of xml005 for 500K steps
2. Evaluate M* on:
   - xml005_buggy (same bug, in-distribution) — does it find the bug faster?
   - xml017_buggy (different bug, transfer) — does the learned policy generalize?
3. Compare against:
   - Vanilla AFL++ baseline (same duration)
   - M1_0 (most consistent RL model from Experiment 2, 12-dim state)
4. Metrics: time-to-first-crash, coverage-over-time, mutation diversity (action entropy)
5. 5 eval runs per configuration for statistical significance

Output: `experiments/differential/results/m_star_evaluation_report.md`

## Summary of Files to Create

### Build infrastructure (Phase 1):
- `experiments/differential/build/build_libxml2_targets.sh`
- `experiments/differential/build/verify_bugs.sh`

### Telemetry collection (Phase 2):
- `src/mutator_telemetry.c`
- `scripts/run_telemetry_campaign.sh`
- `scripts/run_baseline_vanilla.sh`

### Analysis (Phase 3):
- `scripts/analysis/differential_analysis.py`
- `scripts/analysis/design_m_star_features.py`

### Model M* (Phase 4):
- `src/mutator_m_star.c`
- `scripts/models/m_star.py`
- `scripts/run_m_star_experiment.sh`
- Modified: `scripts/models/common.py` (add `ContextualBanditAgent`)
- Modified: `scripts/rl_server.py` (add `--algorithm` flag)
- Modified: `scripts/models/__init__.py` (add `m_star` to `MODEL_IDS`)

## Execution Order

Run phases sequentially. After each phase, the user will verify outputs and provide feedback before proceeding:

1. Phase 1 → user verifies builds work and bugs are reachable
2. Phase 2 → user runs campaigns (12+ hours of compute), provides telemetry CSVs
3. Phase 3 → user reviews analysis, confirms feature selection makes sense
4. Phase 4 → user runs M* training and evaluation

## Critical Implementation Notes

1. **Do NOT modify any existing files** in src/ or scripts/ unless explicitly told to. All new files go alongside existing ones.
2. **Follow existing code conventions**: look at `src/mutator_m0_0.c` for C style, `scripts/rl_server.py` for Python style (unified server, not per-model), `scripts/run_model.sh` for shell script style (unified runner with `--model-id` dispatch).
3. **The SHM IPC protocol**: 128+ byte mmap'd file at `/tmp/rl_shm_*`. C side writes state, sets sequence counter. Python side reads state, writes action, increments counter. Use atomic `__sync_*` builtins for the counter. Reference `mutator_m0_0.c` for exact byte layout.
4. **AFL++ custom mutator API**: The `.so` must export `afl_custom_init`, `afl_custom_fuzz`, `afl_custom_deinit`, and optionally `afl_custom_queue_new_entry`. The `afl` struct pointer gives access to `shm.map`, `queued_items`, `fsrv.total_execs`, etc.
5. **libxml2 build**: Use `afl-clang-fast` (not gcc). The `--disable-shared` flag is critical — AFL++ needs a static library to link the harness correctly. `--without-python` avoids Python binding build issues.
6. **ASAN compatibility**: Use `AFL_USE_ASAN=1` environment variable (do NOT manually add `-fsanitize=address` to CFLAGS — see FUZZBENCH_SETUP.md for why). AFL++ will inject ASAN consistently across compile and link steps. This is essential for buggy versions to crash on bug-triggering inputs. Apply to ALL 4 builds for comparable throughput.
7. **The telemetry mutator must not slow down fuzzing significantly.** Write CSVs in append mode, flush every interval (not every execution). Use stack-allocated buffers for formatting. The bitmap snapshots are the expensive operation — do them 10x less frequently than the CSV rows.
8. **All paths should be relative to `REPO_ROOT`** which is auto-detected as `$(cd "$(dirname "$0")/.." && pwd)` in shell scripts, matching the existing convention.
9. **FuzzBench standardization is mandatory.** The harness, seeds, dictionary, and build flags MUST come from FuzzBench's `~/fuzzbench/benchmarks/libxml2_xml/` benchmark (note: `libxml2_xml`, NOT `libxml2-v2.9.2`). Do NOT write a custom harness, generate synthetic seeds, or invent configure flags. The ONLY thing that varies across the 4 builds is the libxml2 source commit. Follow the same extraction pattern used in the existing `benchmarks/libxml2/build_recipe.sh` (which uses `FUZZBENCH_NAME="libxml2_xml"`). If the FuzzBench benchmark uses OSS-Fuzz integration (i.e., no local `build.sh`), pull the build recipe from `https://github.com/google/oss-fuzz/tree/master/projects/libxml2`.
10. **Verify harness API compatibility.** The FuzzBench harness was written for libxml2 v2.9.2, but we're building against v2.9.3 through v2.9.5. Before proceeding past Phase 1, verify that the harness compiles cleanly against all 4 versions. The `xmlReadMemory()` API is stable across these versions, so this should work, but check for any renamed headers or struct changes.

## Mandatory Transparency Rules — NEVER Silently Substitute

Claude Code MUST follow these rules in every phase. Violating any of these is a blocking error.

1. **If a FuzzBench file is missing or a path doesn't resolve, STOP and tell the user.** Do NOT write a replacement harness, generate synthetic seeds, invent a dictionary, or guess at build flags. Print exactly which path you tried, what you expected to find, and what you actually found. Then wait for user input.

2. **If the harness doesn't compile against one of the 4 libxml2 versions, STOP and tell the user.** Print the exact compiler error. Do NOT patch the harness, change compiler flags, or comment out code to make it work. The user needs to decide whether the fix is acceptable or whether a different version pair should be used.

3. **If a Magma bug patch doesn't apply cleanly or the expected commit hash doesn't exist, STOP and tell the user.** Do NOT pick a different nearby commit or guess at the right tag. Print what you tried and what went wrong.

4. **If any build step requires a dependency that isn't already installed, tell the user** what needs to be installed and why, rather than silently running `apt install` or `pip install`.

5. **At the end of each phase, print a summary of every decision that was made,** including:
   - Which exact FuzzBench paths were used for harness, seeds, dictionary, and build recipe
   - Which exact git tags/commits were checked out for each of the 4 builds
   - Any warnings encountered during compilation
   - File sizes and locations of all outputs
   - Whether the user needs to take any manual action before the next phase

6. **If any metric, feature, or telemetry field is derived or computed (not directly read from AFL++ state), document the formula.** For example, if `edge_entropy` is Shannon entropy computed over hit-count bins, write out the formula in the metadata, not just the column name.

7. **Never rename, reformat, or silently merge the user's existing files.** All new files go in `experiments/differential/` or alongside existing files in `src/` and `scripts/`. Existing run scripts, mutators, and RL servers must remain untouched.

## Self-Documenting Telemetry — Output Requirements

All telemetry output must be readable and interpretable by a future Claude instance (or any researcher) that has NEVER seen this conversation. Every output directory must include metadata files that make the data fully self-contained.

### Required metadata files:

**`experiments/differential/telemetry/EXPERIMENT_MANIFEST.json`**

Generated at the start of Phase 2, updated after each run. Contains:
```json
{
  "experiment_name": "differential_fuzzing_libxml2",
  "created": "ISO-8601 timestamp",
  "repo_root": "/home/shreyasganesh/projects/rl-fuzzer",
  "afl_version": "output of afl-fuzz --version",
  "targets": {
    "xml005_buggy": {
      "library": "libxml2",
      "git_tag": "v2.9.4",
      "git_commit_hash": "actual full SHA",
      "cve": "CVE-2017-5130",
      "bug_type": "integer overflow in xmlMemoryStrdup (xmlmemory.c)",
      "bug_description": "Size parameter overflows, causing undersized malloc and subsequent heap buffer overflow on strcpy",
      "patch_summary": "Adds overflow check: if (size > SIZE_MAX - RESERVE_SIZE) return NULL",
      "is_buggy": true,
      "binary_path": "experiments/differential/targets/xml005_buggy/target",
      "binary_sha256": "sha256 of the compiled binary"
    },
    "xml005_fixed": { "...same fields...", "is_buggy": false, "git_tag": "v2.9.5" },
    "xml017_buggy": { "...same fields..." },
    "xml017_fixed": { "...same fields..." }
  },
  "fuzzbench_source": {
    "benchmark_dir": "~/fuzzbench/benchmarks/libxml2_xml",
    "harness_file": "exact filename and path that was used",
    "harness_sha256": "sha256 of the harness source file",
    "seeds_dir": "path to seeds, with count and total size",
    "dictionary_file": "path to dictionary, with entry count",
    "configure_flags": "exact ./configure invocation used for all builds",
    "compiler": "afl-clang-fast version string",
    "cflags": "exact CFLAGS used",
    "cxxflags": "exact CXXFLAGS used"
  },
  "runs": [
    {
      "run_id": "xml005_buggy_seed1",
      "target": "xml005_buggy",
      "random_seed": 1,
      "duration_seconds": 3600,
      "start_time": "ISO-8601",
      "end_time": "ISO-8601",
      "final_total_edges": 12345,
      "final_unique_crashes": 3,
      "final_total_execs": 500000,
      "telemetry_csv": "telemetry/coverage_dynamics_xml005_buggy_seed1.csv",
      "mutation_csv": "telemetry/mutation_attribution_xml005_buggy_seed1.csv",
      "snapshot_dir": "telemetry/snapshots_xml005_buggy_seed1/"
    }
  ]
}
```

**`experiments/differential/telemetry/COLUMN_DICTIONARY.md`**

A human-and-machine-readable file that defines EVERY column in every CSV. Generated alongside the CSVs. Must include:

```markdown
# Telemetry Column Dictionary

## coverage_dynamics_*.csv

| Column | Type | Unit | Description | Source | Normalization |
|--------|------|------|-------------|--------|---------------|
| timestamp_us | int64 | microseconds | Wall-clock time since campaign start | clock_gettime(CLOCK_MONOTONIC) | None (raw) |
| total_execs | int64 | count | Cumulative executions since campaign start | afl->fsrv.total_execs | None (raw) |
| total_edges | int32 | count | Number of discovered edges via count_coverage() using afl->virgin_bits (bytes != 0xFF) | count_coverage(afl) matching existing mutators | None (raw) |
| new_edges_this_interval | int32 | count | total_edges[now] - total_edges[prev_interval] | Computed | None (raw) |
| edge_discovery_rate | float64 | edges/exec | new_edges_this_interval / interval_size | Computed | None (raw) |
| ... (every column documented) |

## mutation_attribution_*.csv

| Column | Type | Description |
|--------|------|-------------|
| mut_00_count | int32 | Times mutation type 0 (bitflip 1/1) was used this interval |
| mut_00_new_edges | int32 | New edges discovered when mutation type 0 was active |
| mut_00_crashes | int32 | Crashes when mutation type 0 was active |
| mut_01_count | int32 | Times mutation type 1 (bitflip 2/1) was used this interval |
| ... |

## Mutation Type ID Mapping

This table maps the 47 mutation type IDs (0-46) used in column names to AFL++ mutation primitives:

| ID | AFL++ Stage Name | Category | Description |
|----|-----------------|----------|-------------|
| 0 | bitflip 1/1 | deterministic | Flip single bit at each position |
| 1 | bitflip 2/1 | deterministic | Flip two adjacent bits at each position |
| 2 | bitflip 4/1 | deterministic | Flip four adjacent bits at each position |
| 3 | bitflip 8/8 | deterministic | Flip single byte at each position |
| 4 | bitflip 16/8 | deterministic | Flip two adjacent bytes |
| 5 | bitflip 32/8 | deterministic | Flip four adjacent bytes |
| 6 | arith 8/8 | deterministic | Add/subtract small integers to each byte |
| 7 | arith 16/8 | deterministic | Add/subtract small integers to each 16-bit word |
| 8 | arith 32/8 | deterministic | Add/subtract small integers to each 32-bit dword |
| ... (all 47 mapped) |

Source: These IDs correspond to the mutation stage indices in AFL++'s
`afl-fuzz-one.c`. The mapping is derived from the order in which AFL++
applies mutations in its deterministic and havoc stages.
```

**`experiments/differential/telemetry/snapshots_*/README.md`**

Each snapshot directory must contain a README explaining the binary format:
```markdown
# Bitmap Snapshot Format

Each file is named `snapshot_{total_execs}.bin`.

Format: Raw 65536-byte dump of the mutator's cumulative_map at the recorded execution count.
- cumulative_map[i] = max over all executions so far of afl->shm.map[i]
- Byte i represents the maximum hit count for edge i (0-255, AFL++ bucketed)
- Nonzero bytes indicate edges that were exercised at some point
- Value 0 means the edge was never hit across any execution up to this snapshot

To load in Python:
    import numpy as np
    bitmap = np.fromfile("snapshot_100000.bin", dtype=np.uint8)
    active_edges = np.nonzero(bitmap)[0]
    hit_counts = bitmap[active_edges]
```

### Required analysis output metadata:

**`experiments/differential/analysis/ANALYSIS_METHODOLOGY.md`**

Generated by `differential_analysis.py` alongside its results. Documents:
- Which telemetry files were loaded and how they were aligned (by execution count? by wall-clock time?)
- How statistical tests were configured (Mann-Whitney U: alternative hypothesis, significance level, correction method)
- How feature importance was computed (mutual information: number of bins, normalization method)
- What the "divergence point" means operationally and how it was detected
- All derived features: name, formula, input columns, normalization, and rationale for inclusion
- Which features were excluded and why

**`experiments/differential/analysis/m_star_feature_spec.json`**

The feature specification for M*, generated by `design_m_star_features.py`. Must be fully self-contained — a future Claude instance reading ONLY this file should be able to implement the C mutator and Python RL server:

```json
{
  "model_name": "m_star",
  "state_dim": 15,
  "action_dim": 47,
  "recommended_architecture": {
    "hidden_layers": [128, 128, 64],
    "activation": "relu",
    "rationale": "dim <= 20, matching M1_0 architecture which showed best results"
  },
  "recommended_algorithm": "contextual_bandit_thompson",
  "features": [
    {
      "index": 0,
      "name": "edge_discovery_rate",
      "description": "New edges discovered per execution in the last 1000-exec interval",
      "source": "computed from total_edges delta / interval_size",
      "c_implementation": "count nonzero bytes in shm.map, subtract previous count, divide by 1000",
      "normalization": "divide by 0.01 (empirical max from telemetry data)",
      "shm_offset_bytes": 8,
      "shm_format": "float32",
      "importance_rank": 1,
      "p_value": 0.00023,
      "effect_size": 0.82,
      "rationale": "Strongest divergence between buggy and fixed versions near bug-adjacent regions"
    }
  ],
  "shm_layout": {
    "total_bytes": 128,
    "sequence_counter_offset": 0,
    "sequence_counter_size": 8,
    "action_offset": 120,
    "action_size": 4,
    "feature_start_offset": 8,
    "feature_format": "15 consecutive float32 values"
  }
}
```

### Why this matters:

Without these metadata files, the telemetry is just columns of numbers. A future Claude instance (or you, or your professor) reading `mut_23_new_edges = 7` has no way to know that mutation 23 is "havoc: random byte XOR" or that the value 7 is high or low for this experiment. The EXPERIMENT_MANIFEST.json ties every CSV back to exactly which binary was fuzzed, which commit it came from, and whether it's the buggy or fixed version. The COLUMN_DICTIONARY.md makes every column interpretable without needing access to the C source code. The m_star_feature_spec.json makes it possible to implement M* from the spec alone without needing the analysis conversation.
