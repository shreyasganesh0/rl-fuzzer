# Experiment 3: Differential-Informed RL-Guided Fuzzing (M3_0)

## Full Technical Report — Setup, Methodology, Results, and Analysis

**Date**: April 4, 2026
**Status**: Complete
**Repository**: `rl-fuzzer/` (commit `3af5c72` and descendants)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Motivation](#2-background-and-motivation)
3. [Research Question](#3-research-question)
4. [System Architecture](#4-system-architecture)
5. [Phase 0 — Prerequisites and Infrastructure](#5-phase-0--prerequisites-and-infrastructure)
6. [Phase 1 — Target Construction](#6-phase-1--target-construction)
7. [Phase 2 — Telemetry Collection](#7-phase-2--telemetry-collection)
8. [Phase 3 — Differential Analysis and Feature Derivation](#8-phase-3--differential-analysis-and-feature-derivation)
9. [Phase 4 — M3_0 Model Implementation](#9-phase-4--m3_0-model-implementation)
10. [Phase 5 — Training and Evaluation](#10-phase-5--training-and-evaluation)
11. [Results](#11-results)
12. [Analysis and Discussion](#12-analysis-and-discussion)
13. [Reproducing This Experiment](#13-reproducing-this-experiment)
14. [File Inventory](#14-file-inventory)

---

## 1. Executive Summary

This experiment investigates whether reinforcement learning can learn to guide fuzz testing more effectively by using state features derived from differential analysis of buggy vs. fixed software. We build two versions of libxml2 (one containing a known CVE, one with the fix applied), fuzz both with identical seeds and infrastructure, and statistically compare their execution profiles to identify which observable features best distinguish vulnerability-adjacent code exploration. These features become the state representation for an RL agent (M3_0) that selects mutation strategies during fuzzing.

**Key result**: M3_0 (DQN variant) achieves 10.3% more edge coverage than the prior best RL model (M1_0) on the in-distribution target and 9.3% on a transfer target, validating that differential-informed features capture meaningful structural properties of code exploration. However, vanilla AFL++ still outperforms all RL variants by ~7%, primarily due to policy collapse — the RL agent converges to a narrow mutation distribution during evaluation.

---

## 2. Background and Motivation

### 2.1 The Problem: State Representation for RL-Guided Fuzzing

AFL++ is a coverage-guided fuzzer that selects mutations randomly from a fixed distribution. The hypothesis of RL-guided fuzzing is that an agent can learn to select mutations adaptively based on observed execution feedback, improving coverage discovery rate.

The central challenge is **state representation**: what should the RL agent observe? Prior models in this project used increasingly sophisticated features:

| Model | Dimensions | Features | Limitation |
|-------|-----------|----------|------------|
| M0_0 | 3 | coverage, new_edges, crashes | Too sparse — agent can't distinguish qualitatively different coverage states |
| M1_0 | 12 | Edge stability distribution (enabled/disabled split, mean/std/max/density) | Features designed by intuition, not empirically validated |
| M1_1 | 13 | Visited-edge tracking (per-edge hit counts, stability) | High overhead, features not proven to correlate with bug-finding |
| M2 | 3+context | M0_0 + contextual bandit | Algorithm change, same sparse state |

All prior models share a fundamental weakness: their features were designed by human intuition about what *should* matter, without empirical evidence that these features actually distinguish productive from unproductive fuzzing states.

### 2.2 The Differential Insight

Instead of guessing which features matter, we can **measure** them. Given two nearly-identical programs — one with a known vulnerability, one without — we can fuzz both and observe which execution metrics diverge. Features that systematically differ between buggy and fixed runs are, by definition, the features that correlate with vulnerability-adjacent code exploration.

This is the core idea behind M3_0: derive the state representation empirically from differential fuzzing data, rather than designing it by hand.

### 2.3 Why libxml2?

libxml2 is chosen for several reasons:
- Well-studied target with multiple documented CVEs in the Magma benchmark
- FuzzBench provides a standardized harness (`libxml2_xml/target.cc`), seeds, and dictionary
- Bugs span different vulnerability classes (integer overflow, heap overread), testing feature generalization
- Version-controlled source allows building exact buggy/fixed pairs from git tags

---

## 3. Research Question

**Primary**: Can RL state features derived from differential analysis of buggy vs. fixed software outperform hand-designed features for mutation selection?

**Secondary questions**:
1. Do differential features transfer to unseen targets (different CVE, same codebase)?
2. Does DQN or contextual bandit learn better policies from these features?
3. What is the current gap between RL-guided and vanilla AFL++, and what causes it?

---

## 4. System Architecture

### 4.1 Component Overview

The system has three runtime components:

```
┌─────────────────┐     mmap'd SHM (128 bytes)     ┌──────────────────┐
│   AFL++ Fuzzer   │◄──────────────────────────────►│  RL Server (Py)  │
│                  │   state_seq, 13 features,       │                  │
│  + Custom        │   action_seq, action             │  DQN / Bandit    │
│    Mutator (.so) │                                 │  Agent           │
└────────┬─────────┘                                 └──────────────────┘
         │
         ▼
   Instrumented Target Binary
   (AFL++ forkserver mode)
```

**AFL++ Fuzzer**: Industry-standard coverage-guided fuzzer. Manages the test case queue, forkserver, and coverage bitmap. We use `AFL_CUSTOM_MUTATOR_ONLY=1` to route all mutation decisions through our plugin.

**Custom Mutator Plugin** (`mutator_m3_0.so`): A shared library loaded by AFL++ via the custom mutator API. On each fuzzing iteration:
1. Reads the coverage bitmap from `afl->fsrv.trace_bits`
2. Computes 13 features (edge heat, entropy, timing, velocity)
3. Writes features to shared memory
4. Waits for the RL server to write an action
5. Applies the selected mutation to the test case

**RL Server** (`rl_server.py`): A Python process that:
1. Reads the 13-feature state vector from shared memory
2. Selects an action (one of 47 mutations) using the learned policy
3. Writes the action back to shared memory
4. Trains the neural network on accumulated experience

### 4.2 Shared Memory IPC Protocol

Communication uses a 128-byte memory-mapped file (`/tmp/rl_shm_m3_0`) with atomic sequencing:

```
Offset  Size   Type     Field              Direction
──────  ─────  ───────  ─────────────────  ─────────
0       4      uint32   state_seq          C → Py    (incremented atomically by C)
4       4      uint32   total_edges        C → Py
8       4      uint32   cold_edges         C → Py
12      4      uint32   hot_edges          C → Py
16      4      uint32   warm_edges         C → Py
20      4      uint32   cool_edges         C → Py
24      4      float32  edge_entropy       C → Py    (pre-normalized: / 3.0)
28      4      float32  edge_hit_mean      C → Py    (pre-normalized: / 255.0)
32      4      float32  edge_hit_std       C → Py    (pre-normalized: / 255.0)
36      4      uint32   corpus_size        C → Py
40      4      uint32   crashes            C → Py
44      4      uint32   new_edges          C → Py
48      4      float32  avg_exec_time      C → Py    (pre-normalized: log1p/log1p(100000))
52      4      float32  coverage_velocity  C → Py    (pre-normalized: / 0.1, clipped)
56-63   8      —        (padding)
64      4      uint32   action_seq         Py → C    (incremented atomically by Py)
68      4      int32    action             Py → C    (0–46)
72-127  56     —        (padding)
```

**Synchronization protocol**:
1. C writes all feature fields, then increments `state_seq` with `__atomic_store_n(..., __ATOMIC_RELEASE)`
2. Python busy-polls `state_seq` with `__ATOMIC_ACQUIRE` semantics (via struct unpacking)
3. Python writes `action`, then increments `action_seq`
4. C busy-polls `action_seq` to detect the new action

This lock-free design avoids mutex overhead. The 100μs poll sleep in Python prevents CPU waste.

### 4.3 Mutation Action Space

Both the telemetry mutator and RL mutators share an identical 47-action mutation table:

| Actions | Category | Description |
|---------|----------|-------------|
| 0–5 | Deterministic flips | FLIP_1BIT through FLIP_4BYTES |
| 6–15 | Arithmetic | ADD/SUB on 1/2/4-byte values (LE + BE) |
| 16–20 | Interesting values | Boundary values (0, 1, MAX, -128, etc.) in 1/2/4-byte slots |
| 21–40 | Havoc (single-op) | Random byte set, clone, delete, insert, overwrite, crossover |
| 41–44 | Dictionary | User/auto dictionary token overwrite/insert |
| 45 | Custom multi-op | Multiple focused mutations in sequence |
| 46 | Full havoc | Stacked random mutations (AFL++ style) |

This action space is identical across all RL models (M0_0 through M3_0), ensuring that differences in performance are attributable to the state representation, not the action space.

---

## 5. Phase 0 — Prerequisites and Infrastructure

### 5.1 Software Dependencies

| Component | Version / Source | Purpose |
|-----------|-----------------|---------|
| AFL++ | Latest from `~/packages/AFLplusplus` | Fuzzer framework |
| FuzzBench | `google/fuzzbench` repo | Standardized harness, seeds, dictionary for libxml2 |
| libxml2 | Git tags v2.9.3 through v2.9.5 | Target library (buggy and fixed versions) |
| Python 3 | System + venv | RL server, analysis scripts |
| PyTorch | Latest stable | Neural network training |
| clang-18 | System compiler | Compiling instrumented targets and mutator plugins |
| ASAN | Via `AFL_USE_ASAN=1` | Memory error detection in targets |

### 5.2 Kernel Configuration

```bash
echo core > /proc/sys/kernel/core_pattern   # Required for crash detection
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor  # Stable benchmarks
```

### 5.3 Asset Provenance

All assets are cryptographically verified to ensure reproducibility:

| Asset | Source | Verification |
|-------|--------|-------------|
| Harness | `FuzzBench/benchmarks/libxml2_xml/target.cc` | SHA256: `bd91b6d1...` (byte-identical, no modifications) |
| Seeds | 38 XML files from `FuzzBench/benchmarks/libxml2_xml/seeds/` | 188KB total corpus |
| Dictionary | `libxml2/fuzz/xml.dict` | SHA256: `1a6c8d15...` |

**Why use the unmodified FuzzBench harness?** To eliminate harness variation as a confound. The harness is compiled with `-include cstdint -include cstddef` flags to handle missing standard type includes, rather than modifying the source. This maintains byte-identical provenance.

---

## 6. Phase 1 — Target Construction

### 6.1 CVE Selection

Two CVEs were selected from the Magma benchmark's libxml2 entries, chosen to span different vulnerability classes:

| ID | CVE | Class | Location | Mechanism |
|----|-----|-------|----------|-----------|
| xml005 | CVE-2017-5130 | Integer overflow | `xmlmemory.c:xmlMemStrdupLoc()` | `strlen(str) + 1` overflows for crafted large strings; `malloc(RESERVE_SIZE + size)` allocates undersized buffer |
| xml017 | CVE-2016-1762 | Heap buffer overread | `parserInternals.c:xmlNextChar()` | UTF-8 multi-byte sequence parsing jumps into handler without validating remaining buffer length |

**Why these two CVEs?**
- Different vulnerability classes test whether differential features capture general structural properties vs. bug-specific patterns
- Different libxml2 subsystems (memory management vs. parser internals) test cross-module generalization
- Both have clean buggy/fixed version pairs available via git tags

### 6.2 Building the Four Targets

Each CVE produces two binaries: one from the vulnerable version, one from the patched version.

| Target | libxml2 Tag | CVE Status | Purpose |
|--------|-------------|------------|---------|
| `xml005_buggy` | v2.9.4 | Contains CVE-2017-5130 | Differential "positive" |
| `xml005_fixed` | v2.9.5 | CVE-2017-5130 patched | Differential "negative" |
| `xml017_buggy` | v2.9.3 | Contains CVE-2016-1762 | Differential "positive" |
| `xml017_fixed` | v2.9.4 | CVE-2016-1762 patched | Differential "negative" |

**Build process** (for each target):

```bash
# 1. Clone libxml2 at the specific tag
git clone --branch v2.9.4 --depth 1 https://gitlab.gnome.org/GNOME/libxml2.git

# 2. Configure with AFL++ instrumentation + ASAN
CC=afl-clang-fast CXX=afl-clang-fast++ \
AFL_USE_ASAN=1 \
./autogen.sh --disable-shared --without-python --without-zlib --without-lzma

# 3. Build static library
make -j$(nproc)

# 4. Compile harness against the static library
afl-clang-fast++ -fsanitize=address \
    -include cstdint -include cstddef \
    -I./include \
    target.cc ./libxml2/.libs/libxml2.a \
    -lz -llzma -o target
```

**Key build decisions:**
- **Static linking**: Ensures the fuzzer exercises the specific libxml2 version, not a system-installed one
- **ASAN**: Catches memory errors that might not cause immediate crashes (heap overreads, use-after-free)
- **`AFL_USE_ASAN=1`**: Uses AFL++'s ASAN integration rather than manual `-fsanitize` flags, ensuring correct interaction with the forkserver
- **`-include cstdint -include cstddef`**: Portability fix applied via compiler flags, not source modification, preserving harness provenance

### 6.3 Static Bug Verification

Before proceeding, we verified that the buggy versions actually contain the vulnerability and the fixed versions actually contain the patch, via source code inspection:

**CVE-2017-5130 (xml005)**:
- **Buggy** (`xml005_buggy/src/xmlmemory.c:496-502`):
  ```c
  size = strlen(str) + 1;                    // No overflow check
  p = (MEMHDR *) malloc(RESERVE_SIZE+size);  // Undersized if overflow
  ```
- **Fixed** (`xml005_fixed/src/xmlmemory.c:516-521`):
  ```c
  if (size > (MAX_SIZE_T - RESERVE_SIZE))    // Overflow guard added
      goto error;
  p = (MEMHDR *) malloc(RESERVE_SIZE+size);
  ```

**CVE-2016-1762 (xml017)**:
- **Buggy** (`xml017_buggy/src/parserInternals.c:419`): `xmlNextChar` jumps directly into UTF-8 multi-byte parsing with no buffer bounds validation
- **Fixed** (`xml017_fixed/src/parserInternals.c:429-434`): Adds `VALID_CTXT(ctxt)` check plus restructured early-return before entering UTF-8 handler

---

## 7. Phase 2 — Telemetry Collection

### 7.1 Telemetry Mutator Design

A special-purpose mutator (`src/mutator_telemetry.c`) was built for data collection. Unlike the RL mutators, it:
- Selects mutations **uniformly at random** (no RL feedback)
- Logs comprehensive per-step metrics to CSV files
- Saves full 65,536-byte coverage bitmaps at regular intervals
- Does **not** communicate with any external process (no SHM IPC)

This design isolates the data collection from any learning effects — we observe what AFL++ discovers naturally, without RL interference.

**Logged metrics per step** (17 columns per CSV row):
1. `timestamp_us` — wall-clock microseconds
2. `total_execs` — cumulative execution count
3. `total_edges` — unique edges discovered
4. `new_edges_this_interval` — edges found since last log
5. `edge_discovery_rate` — new edges per execution
6. `crashes_total` — cumulative unique crashes
7. `crashes_this_interval` — crashes since last log
8. `avg_exec_time_us` — mean execution time in microseconds
9. `corpus_size` — number of interesting inputs in queue (`afl->queued_items`)
10. `hot_edges` — edges hit > 128 times (in cumulative map)
11. `warm_edges` — edges hit 8–128 times
12. `cool_edges` — edges hit 1–7 times
13. `cold_edges` — edges never hit (= MAP_SIZE - total_edges)
14. `edge_entropy` — Shannon entropy of hit-count distribution over 8 power-of-2 bins
15. `edge_hit_mean` — mean hit count across all non-zero edges
16. `edge_hit_std` — standard deviation of hit counts
17. `edge_hit_max` — maximum hit count on any single edge

**Cumulative bitmap**: The telemetry mutator maintains a `cumulative_map[65536]` array. After each execution, it performs a max-merge: `cumulative_map[i] = max(cumulative_map[i], trace_bits[i])`. This captures the maximum observed hit count per edge across all executions, unlike the per-execution `trace_bits` which resets each run.

**Edge heat classification** bins each edge by its cumulative hit count:
- **Hot** (> 128 hits): Heavily exercised code — loops, frequently reached branches
- **Warm** (8–128 hits): Moderately exercised — conditional branches taken by multiple inputs
- **Cool** (1–7 hits): Lightly exercised — rare code paths, edge-case handling
- **Cold** (0 hits): Unreached code — the exploration frontier

**Shannon entropy** is computed over 8 power-of-2 bins: `[1,2), [2,4), [4,8), [8,16), [16,32), [32,64), [64,128), [128,256)`. Each non-zero edge is classified into its bin, and entropy is `-Σ(p_i · log₂(p_i))`. High entropy means coverage is evenly distributed across intensity levels; low entropy means most edges cluster in a few bins.

### 7.2 Campaign Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Targets | 4 (xml005_buggy/fixed, xml017_buggy/fixed) | 2 CVEs × 2 versions |
| Seeds per target | 3 (random selection from 38-file corpus) | Statistical replication |
| Telemetry runs | 12 (4 targets × 3 seeds) | Full factorial design |
| Baseline runs | 12 (identical config, no custom mutator) | Control group |
| Total runs | 24 | |
| Duration | ~9.7 hours per run | Until coverage saturation |
| Total executions | 33–45 million per run | Varies by target complexity |
| Log interval | Every 1,000 executions | Coverage dynamics CSV |
| Snapshot interval | Every 10,000 executions | Full bitmap dumps |
| Total data generated | ~3.4 GB | ~40,000 snapshots + CSVs |

**Why 3 seeds?** This is the minimum for computing variance estimates. With n=3, Mann-Whitney U has a minimum achievable p-value of 0.05, which means nothing can pass Bonferroni correction (α/65 ≈ 0.00077). This is a known limitation, addressed by using effect size (A12) rather than p-values for feature ranking.

### 7.3 Saturation Criterion

Campaigns ran until coverage saturation: the point where the edge discovery rate drops below 3% of the initial rate. All 4 targets reached saturation:

| Target | Final Edges (mean ± std) | Total Execs |
|--------|-------------------------|-------------|
| xml005_buggy | 5,371 ± 25.7 | ~35M |
| xml005_fixed | 5,165 ± 9.4 | ~35M |
| xml017_buggy | 5,784 ± 31.6 | ~45M |
| xml017_fixed | 5,488 ± 41.2 | ~45M |

---

## 8. Phase 3 — Differential Analysis and Feature Derivation

### 8.1 Analysis Pipeline

The analysis script (`scripts/analysis/differential_analysis.py`) performs 6 stages:

1. **Data loading**: Parse all 12 coverage_dynamics CSVs, 12 mutation_attribution CSVs, and ~40,000 bitmap snapshots into a hierarchical structure: `{cve_pair → {buggy/fixed → {seed → {coverage/mutation: DataFrame}}}}`

2. **Curve interpolation**: Align coverage curves from different seeds to 500 common execution-count points using linear interpolation. This handles the fact that different seeds reach different execution counts at different wall-clock times.

3. **Coverage trajectory comparison**: Compute mean ± std coverage curves for buggy and fixed variants, identify the divergence point where `|mean_buggy - mean_fixed| > pooled_std` for ≥5 consecutive points.

4. **Differential edge analysis**: Compare bitmap snapshots at matched execution counts to identify edges unique to buggy, unique to fixed, or shared.

5. **Mutation effectiveness comparison**: For each of the 47 mutations, compute `effectiveness = Σ(new_edges) / Σ(count)` separately for buggy and fixed, then rank by `ratio = eff_buggy / eff_fixed`.

6. **Feature importance ranking**: At 5 timepoints per CVE pair, compute Mann-Whitney U test and Vargha-Delaney A12 effect size for each candidate feature.

### 8.2 Statistical Methods

**Mann-Whitney U test**: Non-parametric test comparing two independent samples. With n=3 per group, the minimum p-value is 0.05. After Bonferroni correction (α = 0.05 / 65 timepoint-feature combinations ≈ 0.00077), no feature achieves statistical significance. This is expected and not a flaw — it reflects the small sample size inherent in expensive fuzzing campaigns.

**Vargha-Delaney A12 effect size**: `A12 = P(X_buggy > X_fixed) + 0.5 · P(X_buggy = X_fixed)`. Ranges from 0 to 1, with 0.5 indicating no effect. We report `|A12 - 0.5|` as the "A12 deviation":
- ≥ 0.21: Large effect
- ≥ 0.14: Medium effect
- ≥ 0.06: Small effect
- < 0.06: Negligible

**Why A12 over p-values?** A12 measures the probability that a randomly chosen buggy observation exceeds a randomly chosen fixed observation. Unlike p-values, it does not depend on sample size — it answers "how often does this feature differ?" rather than "are we confident it differs?". For feature selection (which features to include in the state vector), the magnitude of difference matters more than statistical certainty about its existence.

**Divergence detection**: After interpolating coverage curves to 500 common points, pooled standard deviation is computed as:
```
pooled_std = sqrt(((n_b - 1) · std_b² + (n_f - 1) · std_f²) / (n_b + n_f - 2))
```
A divergence point is the first of ≥5 consecutive points where `|mean_buggy - mean_fixed| > pooled_std`.

### 8.3 Key Findings from Differential Analysis

#### Coverage Divergence

| CVE Pair | Divergence Point | Buggy Final | Fixed Final | Differential |
|----------|-----------------|-------------|-------------|-------------|
| xml005 (CVE-2017-5130) | 3,486 execs (very early) | 5,371 ± 26 | 5,165 ± 9 | +206 edges |
| xml017 (CVE-2016-1762) | 238,453 execs (late) | 5,784 ± 32 | 5,488 ± 41 | +296 edges |

**Interpretation**: Buggy versions consistently discover more edges because vulnerability-adjacent code creates additional reachable paths (error handlers, fallthrough cases, memory corruption leading to unusual control flow). xml005 diverges almost immediately because the integer overflow is on a hot path; xml017 diverges late because the UTF-8 parsing bug requires specific multi-byte sequences to trigger.

#### Differential Edges

At matched execution counts, bitmap comparison reveals:
- xml005: 3,342 buggy-only edges, 3,078 fixed-only edges, 1,975 shared
- xml017: Similar pattern with late-emerging differentials

#### Mutation Effectiveness

The mutations most favoring buggy-version coverage differ by CVE:

**xml005 (integer overflow)**: Arithmetic mutations dominate
- ARITH_SUB4LE (ratio 2.217) — 4-byte subtraction, likely triggers size underflow
- HAVOC_ARITH16BE (2.061) — 2-byte big-endian arithmetic
- INT_2BE (1.956) — 2-byte interesting value insertion

**xml017 (heap overread)**: Dictionary/structural mutations dominate
- HAVOC_INT32 (1.780) — 4-byte interesting value
- HAVOC_INT16BE (1.756) — 2-byte value insertion
- FLIP_2BITS (1.562) — subtle bit-level changes to encoding bytes

**Interpretation**: The bug class determines which mutations are productive. Integer overflow bugs are triggered by arithmetic boundary values; parser bugs are triggered by structural content changes. A good RL agent should learn to favor different mutation strategies based on the observed coverage state.

#### Crash Differential

| CVE Pair | Buggy Crashes | Fixed Crashes | Signal Quality |
|----------|--------------|---------------|----------------|
| xml005 | 278 ± var | 0 | Clean differential (crash = bug) |
| xml017 | 97 ± var | 87 ± var | Noisy (ASAN catches non-CVE issues too) |

xml005 has a clean crash signal: every crash is the CVE. xml017's crash counts overlap because ASAN detects other memory issues in both versions, making crash count alone unreliable as a bug discriminator.

### 8.4 Feature Selection and Ranking

Candidate features were ranked by mean A12 deviation across all 5 timepoints and both CVE pairs:

| Rank | Feature | Mean A12 Dev | Category | Rationale |
|------|---------|-------------|----------|-----------|
| 1 | `total_edges` | 0.389 | Coverage | Strongest discriminator — buggy versions consistently reach more edges |
| 2 | `cold_edges` | 0.267 | Frontier | Unreached code (MAP_SIZE - total_edges). Fixed versions have more frontier remaining |
| 3 | `corpus_size` | 0.244 | Productivity | Number of "interesting" inputs discovered. More edges → more queue entries |
| 4 | `hot_edges` | 0.244 | Heat | Concentration of heavily-exercised code changes with bug presence |
| 5 | `cool_edges` | 0.233 | Heat | Lightly-touched edges — the discovery frontier within reached code |
| 6 | `avg_exec_time` | 0.222 | Timing | Bug-adjacent code creates execution time anomalies (extra error handling paths) |
| 7 | `edge_hit_mean` | 0.211 | Depth | Average execution depth across all reached edges |
| 8 | `warm_edges` | 0.200 | Heat | Transition zone between hot and cool — moderately exercised branches |
| 9 | `edge_hit_std` | 0.200 | Distribution | Variance of hit counts — how "peaked" vs "flat" the execution profile is |
| 10 | `edge_entropy` | 0.189 | Distribution | Shannon entropy — compact summary of coverage distribution shape |
| 11 | `crashes` | 0.133 | Reward | Direct bug signal, but noisy (xml017) and sparse |
| 12 | `new_edges` | 0.0 | Reward | No discriminative power (identical per-step rates), but essential as immediate RL feedback |
| 13 | `coverage_velocity` | 0.0 | Temporal | No discriminative power, but provides exploration-vs-exploitation signal over time |

**Excluded features** (A12 = 0.5 everywhere, no effect):
- `edge_hit_max`: Dominated by a single hot loop edge, same across buggy/fixed
- `edge_discovery_rate`: Identical per-step rate between variants (only cumulative totals differ)

### 8.5 Why Include Zero-Discriminative Features?

`new_edges` and `coverage_velocity` have A12 = 0.0 (no difference between buggy and fixed), yet are included in the 13-feature state vector. This is intentional:

- **`new_edges`**: The primary reward signal for RL. Without it, the agent cannot observe the immediate consequence of its mutation choice. It's not meant to distinguish buggy/fixed — it's meant to guide learning within a single run.

- **`coverage_velocity`**: Provides temporal context — is coverage accelerating or decelerating? This helps the agent distinguish "early exploration" (velocity high, try diverse mutations) from "late saturation" (velocity low, try more aggressive mutations). Again, not a bug discriminator but a necessary RL signal.

### 8.6 Generalization Argument

The 13 features capture **structural properties** of code exploration, not target-specific coverage counts:

1. **Edge heat distribution** (hot/warm/cool/cold ratios): Any vulnerability creates reachable code paths that shift the heat distribution — more error handlers, more fallthrough cases, more reachable code.

2. **Shannon entropy**: A compact summary of how evenly execution effort is distributed. Bug-adjacent code typically creates "hot spots" that reduce entropy.

3. **Execution time**: Vulnerability-adjacent code often involves additional processing (error handling, memory operations) that creates measurable timing anomalies.

4. **Coverage velocity**: The rate of discovery naturally changes as the fuzzer explores different regions of the code graph.

These properties should transfer to any target where bugs create structural differences in the reachable code graph. The xml017 transfer evaluation tests this hypothesis.

### 8.7 Redundancy Analysis

**Concern**: `cold_edges = MAP_SIZE - total_edges`, so these are linearly dependent.

**Resolution**: In the state vector, `total_edges` and `cold_edges` are both normalized by `MAP_SIZE` (65536), so they are indeed redundant as raw values. However, the neural network can still use both without harm — at worst, one weight goes to zero. The 13-feature state with a [128, 128, 64] network (128·13 + 128·128 + 128·64 + 64·47 = 22,256 parameters) is well within capacity and not at risk of overfitting.

---

## 9. Phase 4 — M3_0 Model Implementation

### 9.1 C Mutator (`src/mutator_m3_0.c`)

The mutator is a ~480-line C file implementing the AFL++ custom mutator API. Key design decisions:

**Data structures in `my_mutator_t`**:
```c
typedef struct my_mutator {
    afl_state_t *afl;            // AFL++ state pointer
    uint8_t     *mutated_buf;    // Scratch buffer for mutations
    int          shm_fd;         // SHM file descriptor
    void        *shm;            // mmap'd SHM pointer
    uint32_t     prev_coverage;  // Coverage at previous step (for new_edges)
    uint32_t     state_seq;      // Monotonic state sequence counter
    uint32_t     last_action_seq;// Last observed action from RL server
    uint32_t     step_count;     // Total steps executed
    uint8_t      cumulative_map[MAP_SIZE];  // Max-merged coverage (65536 bytes, inline)
    uint32_t     edge_ring[VELOCITY_WINDOW]; // Coverage values ring buffer (1000 entries)
    uint32_t     ring_idx;       // Current ring buffer write position
    int          ring_full;      // Whether ring buffer has wrapped
    struct timespec last_exec_time; // Timestamp of last execution
    float        avg_exec_time_us;  // EMA of per-execution time
} my_mutator_t;
```

**Feature computation in `shm_push_state()`**: A single O(MAP_SIZE) loop computes all coverage-derived features:

```c
// 1. Max-merge trace_bits into cumulative_map
for (uint32_t i = 0; i < MAP_SIZE; i++) {
    if (trace[i] > cumulative_map[i]) cumulative_map[i] = trace[i];
}

// 2. Single pass: classify, accumulate sum/sum_sq, bin for entropy
for (uint32_t i = 0; i < MAP_SIZE; i++) {
    uint8_t v = cumulative_map[i];
    if (v == 0) { cold++; continue; }
    total++;
    sum += v; sum_sq += (uint64_t)v * v;
    if (v > 128) hot++;
    else if (v >= 8) warm++;
    else cool++;
    // Bin into 8 power-of-2 buckets for entropy
    int bin = 0; uint8_t tmp = v;
    while (tmp > 1 && bin < 7) { tmp >>= 1; bin++; }
    bins[bin]++;
}

// 3. Compute entropy
float entropy = 0.0f;
for (int b = 0; b < 8; b++) {
    if (bins[b] == 0) continue;
    float p = (float)bins[b] / (float)total;
    entropy -= p * log2f(p);
}
```

**Execution time**: Measured via `clock_gettime(CLOCK_MONOTONIC)` delta between consecutive `afl_custom_fuzz` calls, smoothed with an exponential moving average (α = 0.01):
```c
float delta_us = (now.tv_sec - m->last_exec_time.tv_sec) * 1e6f
               + (now.tv_nsec - m->last_exec_time.tv_nsec) / 1000.0f;
m->avg_exec_time_us = EMA_ALPHA * delta_us + (1.0f - EMA_ALPHA) * m->avg_exec_time_us;
```

**Coverage velocity**: Computed from a 1000-entry ring buffer of edge counts:
```c
m->edge_ring[m->ring_idx] = total_edges;
m->ring_idx = (m->ring_idx + 1) % VELOCITY_WINDOW;
// velocity = (newest - oldest) / window_size
uint32_t oldest = m->edge_ring[m->ring_idx]; // Next slot = oldest entry
float velocity = (float)(total_edges - oldest) / (float)VELOCITY_WINDOW;
```

**Pre-normalization**: Some features are normalized in C before writing to SHM to keep them in [0, 1]:
- `entropy / 3.0` (max theoretical entropy for 8 bins is log₂(8) = 3.0)
- `hit_mean / 255.0` (max hit count per byte)
- `hit_std / 255.0`
- `log1p(avg_exec_time_us) / log1p(100000)` (microseconds, log-compressed)
- `min(velocity / 0.1, 1.0)` (clipped)

### 9.2 Python Model (`scripts/models/m3_0.py`)

The Python side mirrors the SHM layout and applies complementary normalization:

```python
STATE_SIZE = 13
SHM_SIZE   = 128

def build_state(d, train_steps):
    te = max(float(d["total_edges"]), 1.0)
    return np.array([
        d["total_edges"] / 65536.0,                               # 0: fraction of map covered
        d["cold_edges"] / 65536.0,                                # 1: fraction unexplored
        d["hot_edges"] / te,                                      # 2: hot ratio
        d["warm_edges"] / te,                                     # 3: warm ratio
        d["cool_edges"] / te,                                     # 4: cool ratio
        d["entropy"],                                             # 5: pre-normed in C
        d["hit_mean"],                                            # 6: pre-normed in C
        d["hit_std"],                                             # 7: pre-normed in C
        math.log1p(float(d["corpus_size"])) / math.log1p(10000), # 8: log-compressed
        math.log1p(float(d["crashes"])) / math.log1p(1000),      # 9: log-compressed
        min(float(d["new_edges"]), 100.0) / 100.0,               # 10: clipped + normalized
        d["exec_time"],                                           # 11: pre-normed in C
        d["velocity"],                                            # 12: pre-normed in C
    ], dtype=np.float32)
```

**Normalization split rationale**: Features with well-defined ranges (entropy, hit counts, timing) are normalized in C to avoid transmitting large floats through SHM. Features requiring `log1p` or dynamic scaling (corpus_size, crashes, new_edges) are normalized in Python where `math.log1p` is more convenient.

### 9.3 Neural Network Architecture

The network architecture is shared across all models via `scripts/models/common.py`:

**DQN variant**:
- Input: 13 features
- Hidden layers: [128, 128, 64] with ReLU activation
- Output: 47 action Q-values
- Training: Double DQN with target network (soft update τ = 0.005)
- Replay buffer: 100,000 transitions
- Batch size: 128
- Learning rate: 1×10⁻⁴ (Adam)
- Discount factor: γ = 0.99
- Exploration: ε-greedy, ε decays from 1.0 to 0.01 over training

**Contextual Bandit variant**:
- Input: 13 features
- Hidden layers: [128, 128, 64] with ReLU
- Two output heads: μ (mean) and log σ² (log-variance) for each of 47 actions
- Action selection: Thompson sampling — sample from N(μᵢ, σᵢ²) for each action, pick argmax
- Training: Negative log-likelihood loss on observed rewards
- No replay buffer (online updates)
- No discount factor (single-step rewards)

**Reward function** (same for both):
```python
def compute_reward(new_edges, crashes):
    return float(new_edges) + 10.0 * float(crashes)
```

**Why [128, 128, 64]?** This architecture has ~22K parameters for a 13-dimensional input. It's deliberately small to avoid overfitting in the sparse-reward fuzzing regime. The same architecture is used for M1_0 (12-dim input), ensuring that performance differences are attributable to features, not capacity.

---

## 10. Phase 5 — Training and Evaluation

### 10.1 Experiment Design

The experiment trains three model variants on `xml005_buggy` and evaluates them on two targets:

| Variant | Model | Algorithm | Training Target | Training Steps |
|---------|-------|-----------|----------------|---------------|
| M3_0 DQN | M3_0 (13-dim differential) | Double DQN | xml005_buggy | 500,000 |
| M3_0 Bandit | M3_0 (13-dim differential) | Contextual Bandit (Thompson) | xml005_buggy | 500,000 |
| M1_0 | M1_0 (12-dim edge stability) | Double DQN | xml005_buggy | 500,000 |

| Evaluation Target | Purpose | Runs per Variant |
|-------------------|---------|-----------------|
| xml005_buggy | In-distribution (same target as training) | 5 |
| xml017_buggy | Transfer (different CVE, same codebase) | 5 |

A **vanilla AFL++ baseline** (no custom mutator, no RL) is also run 5 times on each evaluation target.

**Why 5 runs?** Fuzzing has inherent randomness (random seed selection, mutation randomness). Five runs provide enough samples to compute meaningful means and standard deviations for comparison, while keeping total compute time manageable (~12 hours total).

**Why `--no-plateau`?** The plateau detector (which stops training early when coverage saturates) is disabled to ensure all variants train for exactly 500K steps, making the comparison fair.

### 10.2 Training Configuration

```bash
# M3_0 DQN training
bash scripts/run_model.sh \
    --model-id m3_0 \
    --train-steps 500000 \
    --eval-steps 500000 \
    --target experiments/differential/targets/xml005_buggy/target \
    --seeds experiments/differential/seeds \
    --exp-dir experiments/differential/results/m3_0_dqn \
    --no-plateau \
    --algorithm dqn
```

The `run_model.sh` script:
1. Compiles `src/mutator_m3_0.c` into `bin/mutator_m3_0.so`
2. Launches `rl_server.py` in the background (creates SHM, waits for states)
3. Launches `afl-fuzz` with `AFL_CUSTOM_MUTATOR_LIBRARY=bin/mutator_m3_0.so AFL_CUSTOM_MUTATOR_ONLY=1`
4. Waits for the RL server to complete its training steps
5. Kills AFL++, saves `fuzzer_stats`
6. Runs an eval phase: loads the saved checkpoint, runs AFL++ for 500K more steps with the frozen policy

### 10.3 Evaluation Protocol

For each evaluation run:
1. Copy the trained checkpoint (`.pt` file) to a fresh directory
2. Run `rl_server.py` in eval mode (no gradient updates, ε = 0.01 for near-greedy action selection)
3. Run AFL++ with the custom mutator against the evaluation target
4. Record final coverage from `fuzzer_stats`

The baseline runs AFL++ with its default mutation scheduling (no custom mutator):
```bash
AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
    afl-fuzz -i seeds -o afl_out -x libxml2.dict \
    -E 500000 -- target @@
```

### 10.4 Trained Model Persistence

All trained checkpoints are saved:
- `experiments/differential/results/m3_0_dqn/bin/rl_m3_0.pt` — M3_0 DQN weights
- `experiments/differential/results/m3_0_bandit/bin/rl_m3_0.pt` — M3_0 Bandit weights
- `experiments/differential/results/m1_0_compare/bin/rl_m1_0.pt` — M1_0 DQN weights

These can be reloaded for further evaluation or analysis.

---

## 11. Results

### 11.1 Raw Coverage Numbers

**xml005_buggy (in-distribution — CVE-2017-5130):**

| Variant | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Std | 95% CI |
|---------|-------|-------|-------|-------|-------|------|-----|--------|
| M3_0 DQN | 3,945 | 3,940 | 4,010 | 3,927 | 3,962 | **3,957** | 31.2 | ±27.4 |
| M3_0 Bandit | 3,629 | 3,677 | 3,644 | 3,674 | 3,633 | 3,651 | 22.0 | ±19.3 |
| M1_0 (DQN) | 3,608 | 3,631 | 3,570 | 3,528 | 3,591 | 3,586 | 38.4 | ±33.7 |
| Baseline AFL++ | 4,320 | 4,030 | 4,324 | 4,273 | 4,302 | **4,250** | 125.6 | ±110.2 |

**xml017_buggy (transfer — CVE-2016-1762):**

| Variant | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Std | 95% CI |
|---------|-------|-------|-------|-------|-------|------|-----|--------|
| M3_0 DQN | 3,910 | 3,876 | 3,905 | 3,913 | 3,893 | **3,899** | 15.0 | ±13.2 |
| M3_0 Bandit | 3,603 | 3,588 | 3,574 | 3,601 | 3,625 | 3,598 | 19.1 | ±16.8 |
| M1_0 (DQN) | 3,588 | 3,577 | 3,540 | 3,557 | 3,572 | 3,567 | 18.2 | ±16.0 |
| Baseline AFL++ | 3,973 | 4,303 | 4,298 | 4,314 | 3,982 | **4,174** | 173.2 | ±152.1 |

### 11.2 Pairwise Comparisons

| Comparison | xml005 Δ | xml005 % | xml017 Δ | xml017 % |
|------------|----------|----------|----------|----------|
| **M3_0 DQN vs M1_0** | +371 edges | **+10.3%** | +332 edges | **+9.3%** |
| M3_0 Bandit vs M1_0 | +65 edges | +1.8% | +31 edges | +0.9% |
| M3_0 DQN vs Baseline | −293 edges | −6.9% | −275 edges | −6.6% |
| M3_0 Bandit vs Baseline | −599 edges | −14.1% | −576 edges | −13.8% |

### 11.3 Variance Analysis

| Variant | xml005 CV | xml017 CV | Interpretation |
|---------|----------|----------|----------------|
| M3_0 DQN | 0.79% | 0.38% | **Most stable** — consistent learned behavior |
| M3_0 Bandit | 0.60% | 0.53% | Stable — Thompson sampling reduces variance |
| M1_0 | 1.07% | 0.51% | Moderate — similar architecture, different features |
| Baseline AFL++ | 2.96% | 4.15% | **Most variable** — random mutation scheduling |

The coefficient of variation (CV = std/mean) reveals that RL models produce much more consistent results than the baseline. M3_0 DQN on xml017 has CV = 0.38%, meaning runs differ by less than 15 edges. The baseline varies by up to 300 edges between runs.

---

## 12. Analysis and Discussion

### 12.1 Differential Features Outperform Hand-Designed Features

M3_0 DQN outperforms M1_0 by 10.3% on in-distribution and 9.3% on transfer. This validates the central hypothesis: empirically-derived features from differential analysis capture more relevant information about fuzzing state than hand-designed edge stability metrics.

The improvement is consistent across both targets and all runs (no overlap in confidence intervals between M3_0 DQN and M1_0), suggesting a real effect rather than noise.

### 12.2 Transfer Generalization Holds

The 9.3% improvement on xml017 (unseen during training) is only slightly smaller than the 10.3% on xml005 (training target). This supports the generalization argument: edge heat distribution, entropy, and timing features capture structural properties of code exploration that are not target-specific.

The transfer gap (9.3% vs 10.3%) is small enough to be within noise. A larger study with more diverse targets would be needed to quantify the true generalization penalty.

### 12.3 Policy Collapse: The Baseline Gap

Despite outperforming M1_0, all RL variants underperform vanilla AFL++ by 7-14%. The primary cause is **policy collapse**: during evaluation, the DQN agent converges to a near-deterministic policy favoring action 10 (ARITH_SUB2LE — 2-byte little-endian arithmetic subtraction).

This is a known problem in RL-guided fuzzing:
1. **Sparse rewards**: New edge discoveries become rare as coverage saturates, providing almost no learning signal
2. **Temporal credit assignment**: The DQN's γ = 0.99 discount factor attributes reward to recent actions, but in fuzzing, the causal chain from mutation to new edge can span many executions
3. **Evaluation greediness**: ε = 0.01 during eval means the agent almost always picks its highest-Q action, losing mutation diversity

AFL++'s built-in scheduling uses a rich set of heuristics (power schedules, queue culling, splice mutations, deterministic/havoc stages) that maintain mutation diversity throughout the campaign. The RL agent replaces all of this with a single argmax over 47 actions.

### 12.4 Bandit Underperformance

The contextual bandit (Thompson sampling) performed worse than DQN:
- xml005: 3,651 vs 3,957 (−7.7%)
- xml017: 3,598 vs 3,899 (−7.7%)

**Why?** Thompson sampling maintains uncertainty estimates and samples from posterior distributions, which should encourage exploration. However, it lacks temporal credit assignment entirely — it treats each step independently, unable to learn that a sequence of mutations leads to new coverage several steps later. In the sparse-reward fuzzing regime, this is a significant disadvantage.

The bandit's exploration also appears less effective than ε-greedy: sampling from a narrow Gaussian around the posterior mean produces less diversity than uniformly random exploration with probability ε.

### 12.5 Implications for Future Work

The results suggest a clear path forward:

1. **Address policy collapse**: Entropy regularization (add an entropy bonus to the reward to penalize narrow action distributions), or action diversity constraints (minimum usage rate per action).

2. **Hybrid scheduling**: Use the RL agent to modulate AFL++'s existing scheduler rather than replacing it entirely. The agent could adjust power schedules, mutation stage weights, or queue prioritization.

3. **Longer-horizon credit assignment**: Replace DQN with an algorithm better suited to sparse, delayed rewards — e.g., reward shaping based on coverage velocity, or intrinsic motivation (curiosity-driven exploration).

4. **More training data**: Train on multiple targets simultaneously to prevent overfitting to a single target's coverage landscape.

---

## 13. Reproducing This Experiment

### 13.1 Full Reproduction (All Phases)

```bash
# Phase 0: Prerequisites
# Install AFL++, FuzzBench, Python venv with PyTorch, clang-18

# Phase 1: Build targets
cd experiments/differential
bash build/build_libxml2_targets.sh

# Phase 2: Telemetry collection (~10 hours)
bash run_differential_campaigns.sh --duration 36000

# Phase 3: Analysis
python3 scripts/analysis/differential_analysis.py \
    --telemetry-dir experiments/differential/telemetry \
    --output-dir experiments/differential/analysis

# Phase 4: M3_0 implementation (already done in src/mutator_m3_0.c + scripts/models/m3_0.py)

# Phase 5: Train + Evaluate (~12 hours)
bash scripts/run_m3_0_experiment.sh --train-steps 500000 --eval-steps 500000 --eval-runs 5
```

### 13.2 Evaluation Only (Using Saved Checkpoints)

```bash
# Reuse trained models from experiments/differential/results/
# Copy checkpoints and run eval-only:
bash scripts/run_model.sh \
    --model-id m3_0 \
    --eval-only \
    --target experiments/differential/targets/xml017_buggy/target \
    --seeds experiments/differential/seeds \
    --exp-dir /path/to/eval/output \
    --algorithm dqn
```

### 13.3 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AFL_ROOT` | `~/packages/AFLplusplus` | Path to AFL++ installation |
| `CC` | `clang-18` | C compiler for mutator compilation |

---

## 14. File Inventory

### Source Code

| File | Lines | Description |
|------|-------|-------------|
| `src/mutator_m3_0.c` | ~480 | RL mutator with 13-dim differential features |
| `src/mutator_telemetry.c` | ~700 | Data collection mutator (Phase 2) |
| `src/mutator_m0_0.c` | ~450 | Reference mutator (3-dim baseline features) |
| `scripts/models/m3_0.py` | ~105 | M3_0 state vector definition and SHM parsing |
| `scripts/models/m1_0.py` | ~105 | M1_0 state vector (comparison model) |
| `scripts/models/common.py` | ~350 | DQN agent, Bandit agent, replay buffer, reward |
| `scripts/rl_server.py` | ~200 | Unified RL training/eval server |
| `scripts/run_model.sh` | ~220 | Train+eval pipeline for any model |
| `scripts/run_m3_0_experiment.sh` | ~175 | Full M3_0 experiment orchestration |

### Documentation

| File | Description |
|------|-------------|
| `docs/differential_fuzzing_experiment_plan.md` | Master experiment plan (Phases 0–4) |
| `docs/m3_0_feature_derivation.md` | Feature selection rationale with data tables |
| `docs/experiment_3_verification_and_next_steps.md` | Implementation specification |
| `docs/experiment_3_full_report.md` | This document |

### Analysis Outputs

| File | Description |
|------|-------------|
| `experiments/differential/analysis/m3_0_feature_spec.json` | Authoritative 13-feature JSON spec |
| `experiments/differential/analysis/feature_importance_report.json` | Raw Mann-Whitney U / A12 results |
| `experiments/differential/analysis/summary.md` | Divergence points, top mutations, feature rankings |
| `experiments/differential/analysis/ANALYSIS_METHODOLOGY.md` | Statistical methods documentation |

### Data

| Directory | Contents |
|-----------|----------|
| `experiments/differential/telemetry/` | 12 coverage_dynamics CSVs, 12 mutation_attribution CSVs, ~40K bitmap snapshots |
| `experiments/differential/seeds/` | 38 XML seed files from FuzzBench |
| `experiments/differential/dictionaries/` | `libxml2.dict` for dictionary-based mutations |
| `experiments/differential/targets/` | 4 compiled libxml2 binaries |

### Results

| Directory | Contents |
|-----------|----------|
| `experiments/differential/results/m3_0_dqn/` | M3_0 DQN training outputs + checkpoint |
| `experiments/differential/results/m3_0_bandit/` | M3_0 Bandit training outputs + checkpoint |
| `experiments/differential/results/m1_0_compare/` | M1_0 training outputs + checkpoint |
| `experiments/differential/results/eval_xml005_buggy/` | 5 eval runs × 4 variants |
| `experiments/differential/results/eval_xml017_buggy/` | 5 eval runs × 4 variants |

---

*End of report.*
