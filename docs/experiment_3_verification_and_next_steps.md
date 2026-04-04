# Experiment 3: Differential Fuzzing — Verification & Next Steps

## Purpose

This document records the complete state of the differential fuzzing experiment
(Experiment 3), verifies every asset and decision, and provides exact
instructions for the one remaining implementation task: regenerating the M3_0
C mutator and Python model from the feature specification.

---

## 1. Experiment Summary

### Goal

Build an RL model (M3_0) whose state features are informed by differential
analysis of buggy vs fixed libxml2 versions. The features should capture
structural properties of code exploration that generalize to finding bugs
in unseen targets.

### Phases Completed

| Phase | Status | Key Output |
|-------|--------|-----------|
| 0: Prerequisites | Done | AFL++ v4.41a, FuzzBench, clang-18, Python venv |
| 1: Build 4 targets | Done | xml005_buggy/fixed, xml017_buggy/fixed (all ASAN) |
| 2: Telemetry collection | Done | 9.7h campaign, 12 telemetry runs, 40K snapshots |
| 3: Differential analysis | Done | 13-feature spec in `m3_0_feature_spec.json` |
| 4: M3_0 implementation | **INCOMPLETE** | Placeholder files still use M1_1 features |

---

## 2. Provenance — Every Asset Traced to Source

### Harness

- **File**: `experiments/differential/build/harness.cc`
- **Source**: `~/fuzzbench/benchmarks/libxml2_xml/target.cc`
- **SHA256**: `bd91b6d126d6cd7215cf2752657eaa08268c75d0344f4a415ef374b75d951510`
- **Modifications**: NONE. File is byte-identical to FuzzBench original. Portability
  (`uint8_t`, `size_t` types) handled via `-include cstdint -include cstddef`
  compiler flags in `build_libxml2_targets.sh`.

### Seeds

- **Directory**: `experiments/differential/seeds/`
- **Source**: `libxml2/test/*.xml` from FuzzBench-pinned commit `c7260a47`
- **Count**: 38 XML files, 188KB total
- **Files**: attrib.xml, badcomment.xml, bigentname.xml, boundaries1.xml,
  cdata-2-byte-UTF-8.xml, cdata-3-byte-UTF-8.xml, cdata-4-byte-UTF-8.xml,
  comment.xml, comment2.xml, comment3.xml, comment4.xml, comment5.xml,
  comment6.xml, defattr.xml, defattr2.xml, ebcdic_566012.xml, emptycdata.xml,
  ent_738805.xml, eve.xml, icu_parse_test.xml, intsubset.xml, japancrlf.xml,
  nsclean.xml, pi.xml, pi2.xml, slashdot.xml, text-4-byte-UTF-16-BE-offset.xml,
  text-4-byte-UTF-16-BE.xml, text-4-byte-UTF-16-LE-offset.xml,
  text-4-byte-UTF-16-LE.xml, title.xml, utf16bebom.xml, utf16bom.xml,
  utf16lebom.xml, utf8bom.xml, wap.xml, winblanks.xml, wml.xml

### Dictionary

- **File**: `experiments/differential/dictionaries/libxml2.dict`
- **Source**: `libxml2/fuzz/xml.dict` from FuzzBench-pinned commit `c7260a47`
- **SHA256**: `1a6c8d151a20c505a4ab2cd1be7e7616baafe0c23426ab289835836aac14665a`
- **Entries**: 89

### Targets

| Target | Git Tag | Full Commit Hash | Binary SHA256 |
|--------|---------|-----------------|---------------|
| xml005_buggy | v2.9.4 | bdec2183f34b37ee89ae1d330c6ad2bb4d76605f | 68350249f6c98f1387e72c59d08d362ad8d3e6ec67e663b2785bba5c835a678f |
| xml005_fixed | v2.9.5 | 2960178fe8f9fe690b7f8c1c49093ff54bb56934 | 8bad88b84befa390d7fe1d5d0677c713ad0f7316e88201248dbacad81a77d880 |
| xml017_buggy | v2.9.3 | 6657afe83a38278f124ace71dc85f60420beb2d5 | 64e03b1708e62f50ba64d60a77e0e26440ee92b6695a638c266be2286e490630 |
| xml017_fixed | v2.9.4 | bdec2183f34b37ee89ae1d330c6ad2bb4d76605f | 7a4980df6428474826c31eb3e307c22fc34ba7d63f7396996f295434a5179485 |

**Build flags** (all 4 targets identical):
```
CC=afl-clang-fast  CXX=afl-clang-fast++  AFL_USE_ASAN=1
CFLAGS="-g -O2"  CXXFLAGS="-g -O2 -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13"
./autogen.sh --prefix=... --disable-shared --without-debug --without-ftp --without-http --without-legacy --without-python
```

**Harness link step**:
```
AFL_USE_ASAN=1 $CXX $CXXFLAGS -include cstdint -include cstddef \
    -I$INSTALL/include/libxml2 harness.cc $INSTALL/lib/libxml2.a \
    $AFL_ROOT/libAFLDriver.a -lz -llzma -lstdc++ -o target
```

### CVE Details

**CVE-2017-5130** (xml005): Integer overflow in `xmlMemoryStrdup` (`xmlmemory.c`).
Size parameter overflows, causing undersized malloc and heap buffer overflow on strcpy.
Fix commit: `897dffbae322b46b83f99a607d527058a72c51ed`. NVD CVSS 8.8.

**CVE-2016-1762** (xml017): Heap buffer overread in `xmlNextChar` (`parserInternals.c`).
Triggered during internal subset parsing with specific multi-byte UTF-8 sequences.
Fix commit: `a7a94612aa3b16779e2c74e1fa353b5d9786c602`. NVD CVSS 8.1.

---

## 3. Telemetry Campaign Results

### Campaign Configuration

- Duration: 9.7 hours (campaign killed after coverage saturation confirmed)
- 12 telemetry runs: 4 targets x 3 random seeds
- Mutator: `src/mutator_telemetry.c` — uniform random over 47 AFL++ mutations
- Compiled with plain `clang-18` (NOT `afl-clang-fast` — custom mutator .so must
  not contain AFL++ instrumentation symbols like `__afl_area_ptr`)
- `AFL_CUSTOM_MUTATOR_ONLY=1` — telemetry mutator controls all mutations
- 15 parallel jobs on 16-core machine

### Data Collected

| Data Type | Files | Rows Per File | Columns |
|-----------|-------|---------------|---------|
| Coverage dynamics CSV | 12 | 26K-38K | 17 (total_edges, hot/warm/cool/cold, entropy, mean, std, etc.) |
| Mutation attribution CSV | 12 | 26K-38K | 142 (47 mutations x 3 metrics: count, new_edges, crashes) |
| Bitmap snapshots (.bin) | 40,415 | -- | 65,536 bytes each (cumulative hit counts) |
| Total disk | -- | -- | ~3.4 GB |

### Coverage Saturation

All 4 targets reached full saturation (edge discovery rate < 3% of initial rate):

| Target | Final Edges (mean, 3 seeds) | Discovery Rate at End |
|--------|---------------------------|----------------------|
| xml005_buggy | 5,432 | 2.8% of initial |
| xml005_fixed | 5,194 | 0.9% of initial |
| xml017_buggy | 5,790 | 0.1% of initial |
| xml017_fixed | 5,505 | 1.0% of initial |

### Crash Results

| Target | Crashes (seeds 1, 2, 3) | Total | CVE Confirmed? |
|--------|------------------------|-------|----------------|
| xml005_buggy (v2.9.4) | [1, 0, 277] | 278 | YES — clean differential (fixed = 0) |
| xml005_fixed (v2.9.5) | [0, 0, 0] | 0 | -- |
| xml017_buggy (v2.9.3) | [81, 16, 0] | 97 | Partial — fixed also crashes (87) |
| xml017_fixed (v2.9.4) | [74, 0, 13] | 87 | Other bugs in v2.9.4 |

### Differential Signal

| CVE Pair | Buggy Edges | Fixed Edges | Delta | Divergence Point |
|----------|-------------|-------------|-------|-----------------|
| xml005 | 5,432 +/- 64 | 5,194 +/- 38 | +238 | 3,486 execs |
| xml017 | 5,790 +/- 31 | 5,505 +/- 41 | +285 | 238,453 execs |

Consistent across all 3 seeds for both pairs.

### Baseline Data Gap

Only 3/12 baseline runs completed (xml005_buggy seeds 1-3 with ~6.5K rows each).
The other 9 baselines were queued behind telemetry jobs on the 15-job limit and
hadn't started when the campaign was killed. This is not critical for feature
selection but means the M3_0 evaluation experiment (`run_m3_0_experiment.sh`)
will need to run its own baselines.

---

## 4. Feature Selection Results

### Statistical Method

- Test: Mann-Whitney U (two-sided) at 5 timepoints per CVE pair (10%, 25%, 50%, 75%, 90% of total execs)
- Effect size: Vargha-Delaney A12 (probability that a random buggy observation > random fixed observation)
- Correction: Bonferroni (alpha = 0.05 / 65 tests = 0.000769)
- Result: 0 Bonferroni-significant features (expected with n=3 per group)
- Fallback: Rank by mean |A12 - 0.5| across all 10 observations (both CVE pairs x 5 timepoints)

### Feature Rankings

| Rank | Feature | Mean |A12-0.5| | Medium/Large Effects | Description |
|------|---------|---------------------|---------------------|-------------|
| 1 | total_edges | 0.389 | 9/10 | Discovered edges normalized by MAP_SIZE |
| 2 | cold_edges | 0.267 | 8/10 | Never-hit edges (unexplored frontier) |
| 3 | corpus_size | 0.244 | 7/10 | AFL++ queue entries (log-normalized) |
| 4 | hot_edges | 0.244 | 8/10 | Edges hit >128 times (execution hotspots) |
| 5 | cool_edges | 0.233 | 8/10 | Edges hit 1-7 times (discovery frontier) |
| 6 | avg_exec_time | 0.222 | 9/10 | Per-execution time (code depth proxy) |
| 7 | edge_hit_mean | 0.211 | 7/10 | Mean hit count across edges |
| 8 | warm_edges | 0.200 | 7/10 | Edges hit 8-128 times (transition zone) |
| 9 | edge_hit_std | 0.200 | 6/10 | Hit count std dev (uneven exploration) |
| 10 | edge_entropy | 0.189 | 4/10 | Shannon entropy of hit distribution |
| 11 | crashes | 0.133 | 6/10 | Unique crashes (direct bug signal) |
| 12 | edge_discovery_rate | 0.000 | 0/10 | **EXCLUDED** — zero discriminative power |
| 13 | edge_hit_max | 0.000 | 0/10 | **EXCLUDED** — zero discriminative power |

Features 12 and 13 were excluded. Two non-discriminative but functionally
essential features were added:
- **new_edges** (A12=0.0): Primary reward signal for the RL agent
- **coverage_velocity** (A12=0.0): Exploration vs exploitation temporal signal

### Raw Feature Deltas (buggy vs fixed at saturation)

**XML005 (CVE-2017-5130, integer overflow):**

| Feature | Buggy | Fixed | Delta | Delta % |
|---------|-------|-------|-------|---------|
| total_edges | 5,432 | 5,194 | +238 | +4.6% |
| hot_edges | 745 | 988 | -243 | -24.6% |
| warm_edges | 2,874 | 2,616 | +259 | +9.9% |
| cool_edges | 31,184 | 45,484 | -14,301 | -31.4% |
| cold_edges | 30,733 | 16,448 | +14,285 | +86.8% |
| edge_entropy | 1.4 | 0.6 | +0.7 | +120.5% |
| edge_hit_mean | 23.4 | 8.9 | +14.5 | +163.6% |
| edge_hit_std | 48.7 | 35.7 | +13.0 | +36.3% |
| corpus_size | 6,281 | 6,456 | -175 | -2.7% |
| crashes | 93 | 0 | +93 | -- |
| avg_exec_time_us | 445M | 343M | +102M | +29.6% |

**XML017 (CVE-2016-1762, heap overread):**

| Feature | Buggy | Fixed | Delta | Delta % |
|---------|-------|-------|-------|---------|
| total_edges | 5,790 | 5,505 | +285 | +5.2% |
| hot_edges | 773 | 720 | +52 | +7.3% |
| warm_edges | 3,141 | 2,999 | +142 | +4.7% |
| cool_edges | 44,918 | 45,816 | -898 | -2.0% |
| cold_edges | 16,704 | 16,000 | +704 | +4.4% |
| edge_entropy | 0.7 | 0.6 | +0.0 | +5.9% |
| edge_hit_mean | 8.5 | 8.1 | +0.4 | +5.1% |
| edge_hit_std | 34.2 | 32.9 | +1.3 | +4.0% |
| corpus_size | 6,696 | 6,468 | +228 | +3.5% |
| crashes | 32 | 29 | +3 | +11.5% |
| avg_exec_time_us | 445M | 450M | -5M | -1.2% |

### Mutation Effectiveness Differential

**XML005 — mutations most effective on buggy:**

| Mutation | Buggy Gain Rate | Fixed Gain Rate | Delta |
|----------|----------------|----------------|-------|
| H_ARITH32- (havoc 32-bit subtract) | 0.000108 | 0.000063 | +0.000045 |
| H_ARITH16+ (havoc 16-bit add) | 0.000066 | 0.000032 | +0.000033 |
| ARITH-4LE (det. 32-bit subtract LE) | 0.000084 | 0.000056 | +0.000028 |

Arithmetic mutations dominate — they directly manipulate numeric values that
trigger the integer overflow.

**XML017 — mutations most effective on buggy:**

| Mutation | Buggy Gain Rate | Fixed Gain Rate | Delta |
|----------|----------------|----------------|-------|
| DICT_UINS (dictionary user insert) | 0.000822 | 0.000601 | +0.000220 |
| H_ARITH16- (havoc 16-bit subtract) | 0.000137 | 0.000072 | +0.000065 |

Dictionary insertions dominate — they introduce XML structure tokens that
exercise deeper parsing paths where the overread occurs.

---

## 5. M3_0 Feature Specification

**File**: `experiments/differential/analysis/m3_0_feature_spec.json`

### 13-Dimensional State Vector

| Index | Name | SHM Offset | Format | Normalization | C Implementation |
|-------|------|-----------|--------|---------------|-----------------|
| 0 | total_edges_norm | 4 | u32 | / 65536 | `count_coverage(afl)` (reuse from m0_0) |
| 1 | cold_edges_norm | 8 | u32 | / 65536 | `MAP_SIZE - count_nonzero(cumulative_map)` |
| 2 | hot_edges_norm | 12 | u32 | / total_edges | `count(cumulative_map[i] > 128)` |
| 3 | warm_edges_norm | 16 | u32 | / total_edges | `count(8 <= cumulative_map[i] <= 128)` |
| 4 | cool_edges_norm | 20 | u32 | / total_edges | `count(1 <= cumulative_map[i] <= 7)` |
| 5 | edge_entropy | 24 | f32 | / 3.0 | Shannon entropy over 8 power-of-2 bins |
| 6 | edge_hit_mean | 28 | f32 | / 255.0 | `sum(cumulative_map[nonzero]) / count_nonzero` |
| 7 | edge_hit_std | 32 | f32 | / 255.0 | `sqrt(E[x^2] - E[x]^2)` over nonzero entries |
| 8 | corpus_size_norm | 36 | u32 | log1p / log1p(10000) | `afl->queued_items` |
| 9 | crashes_norm | 40 | u32 | log1p / log1p(1000) | `afl->total_crashes` |
| 10 | new_edges_norm | 44 | u32 | min(n,100) / 100 | `coverage - prev_coverage` |
| 11 | avg_exec_time_norm | 48 | f32 | log1p / log1p(100000) | EMA of `clock_gettime()` deltas |
| 12 | coverage_velocity | 52 | f32 | / 0.1 | Ring buffer of last 1000 total_edges, derivative |

### SHM Layout (128 bytes)

```
Offset  Size  Field
------  ----  -----
0       4     state_seq (u32) — monotonic counter, RELEASE store
4       4     total_edges (u32, raw)
8       4     cold_edges (u32, raw)
12      4     hot_edges (u32, raw)
16      4     warm_edges (u32, raw)
20      4     cool_edges (u32, raw)
24      4     edge_entropy (f32, pre-normalized / 3.0)
28      4     edge_hit_mean (f32, pre-normalized / 255.0)
32      4     edge_hit_std (f32, pre-normalized / 255.0)
36      4     corpus_size (u32, raw)
40      4     crashes (u32, raw)
44      4     new_edges (u32, raw)
48      4     avg_exec_time (f32, pre-normalized log1p / log1p(100000))
52      4     coverage_velocity (f32, pre-normalized / 0.1)
56-63   8     padding
64      4     action_seq (u32) — monotonic counter, ACQUIRE load
68      4     action (i32) — 0..46
72-127  56    padding
```

### DQN Architecture

- Input: 13
- Hidden: [128, 128, 64] (ReLU)
- Output: 47 actions
- Matches M1_0/M1_1 architecture (best performers in Experiment 2)

---

## 6. Current State — The Placeholder Problem

### What exists but is WRONG

**`src/mutator_m3_0.c`** — This is a copy of `mutator_m1_1.c` with renamed
identifiers. It computes M1_1's visited-edge stability features (enabled/disabled
counts, stability ratio) using a completely different data structure. It does NOT
compute any of the 13 features in the spec. SHM size is 256 (spec says 128).

**`scripts/models/m3_0.py`** — Same problem. Uses M1_1's `build_state()` which
reads stability metrics from SHM and normalizes them. Does not read any of the
spec features.

### What exists and is CORRECT

| File | Status |
|------|--------|
| `scripts/models/common.py` — ContextualBanditAgent | Ready (line 171+) |
| `scripts/rl_server.py` — `--algorithm dqn\|bandit` flag | Ready (line 49) |
| `scripts/run_model.sh` — `--algorithm` passthrough | Ready |
| `scripts/models/__init__.py` — `m3_0` in MODEL_IDS | Ready |
| `scripts/run_m3_0_experiment.sh` — full evaluation pipeline | Ready |
| `experiments/differential/analysis/m3_0_feature_spec.json` | Ready |

---

## 7. Implementation Instructions — Regenerate M3_0

### Task: Rewrite 2 files to match the spec

### File 1: `src/mutator_m3_0.c`

**Delete the entire current file and write a new one.** Do NOT try to modify the M1_1 code — the data structures are fundamentally different.

**Reuse from `src/mutator_m0_0.c`** (copy verbatim):
- Interesting value tables (`MYINTERESTING_8`, `MYINTERESTING_16`, `MYINTERESTING_32`)
- Byte-swap helpers (`bswap16`, `bswap32`)
- Dictionary helpers (`dict_overwrite`, `dict_insert`, `pick_user_extra`, `pick_auto_extra`)
- `apply_mutation()` — the full 47-action switch statement (lines 208-407)
- `count_coverage()` — virgin_bits scanning with 8-byte chunk skip (lines 151-168)

**New data structures needed in the mutator struct**:
```c
typedef struct my_mutator {
    afl_state_t *afl;
    uint8_t     *mutated_buf;
    int          shm_fd;
    void        *shm;

    // Coverage tracking
    uint32_t     prev_coverage;
    uint32_t     prev_crashes;
    uint32_t     state_seq;
    uint32_t     last_action_seq;

    // Cumulative bitmap — max(hit_count) per edge across all execs
    uint8_t      cumulative_map[65536];

    // Coverage velocity ring buffer
    uint32_t     edge_ring[1000];
    uint32_t     ring_idx;
    int          ring_full;   // 0 until 1000 entries written

    // Execution time tracking
    struct timespec last_exec_time;
    float        avg_exec_time_us;  // exponential moving average
} my_mutator_t;
```

**New `shm_push_state()` logic** (called in `afl_custom_fuzz` before action selection):
```
1. coverage = count_coverage(afl)
2. new_edges = coverage - prev_coverage
3. crashes = afl->total_crashes

4. Update cumulative_map:
   const uint8_t *trace = afl->shm.map  (or fsrv.trace_bits)
   for i in 0..65535: cumulative_map[i] = max(cumulative_map[i], trace[i])

5. Scan cumulative_map to compute:
   - total_nz = count of nonzero entries
   - hot = count where val > 128
   - warm = count where 8 <= val <= 128
   - cool = count where 1 <= val <= 7
   - cold = 65536 - total_nz
   - sum_hits, sum_sq_hits (for mean/std)
   - 8-bin histogram: [1], [2], [3-4], [5-8], [9-16], [17-32], [33-64], [65+]
     (for entropy computation)

6. Compute entropy:
   H = 0; for each bin with count > 0: p = count/total_nz; H -= p * log2f(p)
   entropy_norm = H / 3.0  (max entropy with 8 bins = log2(8) = 3.0)

7. Compute mean/std:
   mean = sum_hits / max(total_nz, 1)
   var = sum_sq_hits / max(total_nz, 1) - mean*mean
   std = sqrtf(max(var, 0))
   mean_norm = mean / 255.0
   std_norm = std / 255.0

8. Corpus size = afl->queued_items

9. Execution time:
   clock_gettime(CLOCK_MONOTONIC, &now)
   delta_us = (now - last_exec_time) in microseconds
   avg_exec_time_us = 0.99 * avg_exec_time_us + 0.01 * delta_us  (EMA)
   last_exec_time = now
   exec_time_norm = log1p(avg_exec_time_us) / log1p(100000.0)

10. Coverage velocity:
    edge_ring[ring_idx % 1000] = coverage
    ring_idx++
    if ring_idx >= 1000: ring_full = 1
    if ring_full:
        oldest = edge_ring[ring_idx % 1000]
        velocity = (float)(coverage - oldest) / 1000.0f
        velocity_norm = velocity / 0.1f
    else:
        velocity_norm = 0.0f

11. Write all 13 values to SHM at spec offsets
12. Increment state_seq with __ATOMIC_RELEASE
```

**SHM constants**:
```c
#define SHM_PATH         "/tmp/rl_shm_m3_0"
#define SHM_SIZE         128
#define OFF_STATE_SEQ    0
#define OFF_TOTAL_EDGES  4
#define OFF_COLD_EDGES   8
#define OFF_HOT_EDGES    12
#define OFF_WARM_EDGES   16
#define OFF_COOL_EDGES   20
#define OFF_ENTROPY      24
#define OFF_HIT_MEAN     28
#define OFF_HIT_STD      32
#define OFF_CORPUS_SIZE  36
#define OFF_CRASHES      40
#define OFF_NEW_EDGES    44
#define OFF_EXEC_TIME    48
#define OFF_VELOCITY     52
#define OFF_ACTION_SEQ   64
#define OFF_ACTION       68
```

**Compilation**: `clang-18 -O2 -shared -fPIC -g -I ~/packages/AFLplusplus/include -o bin/mutator_m3_0.so src/mutator_m3_0.c -lm`

Do NOT use `afl-clang-fast`. Custom mutator .so must not contain `__afl_area_ptr`.

### File 2: `scripts/models/m3_0.py`

**Delete the current contents and write from scratch.**

```python
"""Model M3_0 — Differential-informed coverage distribution features (13-dim)."""

import struct, math
import numpy as np

STATE_SIZE      = 13
SHM_SIZE        = 128
SHM_PATH        = "/tmp/rl_shm_m3_0"
MODEL_PATH_DEFAULT = "rl_m3_0.pt"
LABEL           = "M3_0"
HIDDEN_LAYERS   = [128, 128, 64]

# SHM offsets — must match mutator_m3_0.c
STATE_SEQ_OFF    = 0
TOTAL_EDGES_OFF  = 4
COLD_EDGES_OFF   = 8
HOT_EDGES_OFF    = 12
WARM_EDGES_OFF   = 16
COOL_EDGES_OFF   = 20
ENTROPY_OFF      = 24
HIT_MEAN_OFF     = 28
HIT_STD_OFF      = 32
CORPUS_SIZE_OFF  = 36
CRASHES_OFF      = 40
NEW_EDGES_OFF    = 44
EXEC_TIME_OFF    = 48
VELOCITY_OFF     = 52
ACTION_SEQ_OFF   = 64
ACTION_OFF       = 68

CSV_EXTRA_HEADER = ",cold_edges,hot_edges,warm_edges,cool_edges,entropy,hit_mean,hit_std,corpus_size,exec_time,velocity"

MAP_SIZE = 65536.0


def shm_read(shm, shm_size):
    shm.seek(0); raw = shm.read(shm_size)
    return {
        "state_seq":     struct.unpack_from("=I", raw, STATE_SEQ_OFF)[0],
        "total_edges":   struct.unpack_from("=I", raw, TOTAL_EDGES_OFF)[0],
        "cold_edges":    struct.unpack_from("=I", raw, COLD_EDGES_OFF)[0],
        "hot_edges":     struct.unpack_from("=I", raw, HOT_EDGES_OFF)[0],
        "warm_edges":    struct.unpack_from("=I", raw, WARM_EDGES_OFF)[0],
        "cool_edges":    struct.unpack_from("=I", raw, COOL_EDGES_OFF)[0],
        "entropy":       struct.unpack_from("=f", raw, ENTROPY_OFF)[0],
        "hit_mean":      struct.unpack_from("=f", raw, HIT_MEAN_OFF)[0],
        "hit_std":       struct.unpack_from("=f", raw, HIT_STD_OFF)[0],
        "corpus_size":   struct.unpack_from("=I", raw, CORPUS_SIZE_OFF)[0],
        "crashes":       struct.unpack_from("=I", raw, CRASHES_OFF)[0],
        "new_edges":     struct.unpack_from("=I", raw, NEW_EDGES_OFF)[0],
        "exec_time":     struct.unpack_from("=f", raw, EXEC_TIME_OFF)[0],
        "velocity":      struct.unpack_from("=f", raw, VELOCITY_OFF)[0],
    }


def build_state(d, train_steps):
    te = max(float(d["total_edges"]), 1.0)
    return np.array([
        d["total_edges"] / MAP_SIZE,                              # 0: total_edges_norm
        d["cold_edges"] / MAP_SIZE,                               # 1: cold_edges_norm
        d["hot_edges"] / te,                                      # 2: hot_edges_norm
        d["warm_edges"] / te,                                     # 3: warm_edges_norm
        d["cool_edges"] / te,                                     # 4: cool_edges_norm
        d["entropy"],                                             # 5: pre-normalized in C
        d["hit_mean"],                                            # 6: pre-normalized in C
        d["hit_std"],                                             # 7: pre-normalized in C
        math.log1p(float(d["corpus_size"])) / math.log1p(10000), # 8: corpus_size_norm
        math.log1p(float(d["crashes"])) / math.log1p(1000),      # 9: crashes_norm
        min(float(d["new_edges"]), 100.0) / 100.0,               # 10: new_edges_norm
        d["exec_time"],                                           # 11: pre-normalized in C
        d["velocity"],                                            # 12: pre-normalized in C
    ], dtype=np.float32)


def zero_state_data():
    return {"total_edges": 0, "cold_edges": 65536, "hot_edges": 0,
            "warm_edges": 0, "cool_edges": 0, "entropy": 0.0,
            "hit_mean": 0.0, "hit_std": 0.0, "corpus_size": 0,
            "crashes": 0, "new_edges": 0, "exec_time": 0.0,
            "velocity": 0.0, "coverage": 0}


def csv_extra_fields(d, args):
    return (f",{d['cold_edges']},{d['hot_edges']},{d['warm_edges']},"
            f"{d['cool_edges']},{d['entropy']:.4f},{d['hit_mean']:.4f},"
            f"{d['hit_std']:.4f},{d['corpus_size']},{d['exec_time']:.4f},"
            f"{d['velocity']:.4f}")


def log_extra(d, args):
    return (f"hot={d['hot_edges']} warm={d['warm_edges']} "
            f"cool={d['cool_edges']} ent={d['entropy']:.3f} "
            f"vel={d['velocity']:.4f}")


def exit_summary(d, step, cov, cr, epsilon, tag):
    pass
```

### Normalization Strategy

Features 5, 6, 7, 11, 12 are pre-normalized in C (entropy / 3.0, mean / 255.0,
std / 255.0, exec_time via log1p, velocity / 0.1). These arrive in Python as
floats in [0, 1] range and are passed through directly.

Features 0, 1 are normalized in Python (/ MAP_SIZE).
Features 2, 3, 4 are normalized in Python (/ total_edges).
Features 8, 9 are normalized in Python (log1p scaling).
Feature 10 is normalized in Python (min-cap + / 100).

This split exists because C can efficiently compute entropy/mean/std during the
cumulative_map scan (single pass), while count-based features are simpler to
normalize in Python where the denominator (total_edges) is already available.

### Verification After Implementation

1. `clang-18 -O2 -shared -fPIC -g -I ~/packages/AFLplusplus/include -o bin/mutator_m3_0.so src/mutator_m3_0.c -lm` — must compile cleanly
2. `nm bin/mutator_m3_0.so | grep "U.*__afl"` — must produce NO output
3. `python3 -c "import sys; sys.path.insert(0,'scripts'); import models.m3_0 as m; print(m.STATE_SIZE, m.SHM_SIZE, m.LABEL)"` — must print `13 128 M3_0`
4. `python3 -c "import sys; sys.path.insert(0,'scripts'); from models.m3_0 import build_state, zero_state_data; import numpy as np; s = build_state(zero_state_data(), 500000); print(s.shape, s.dtype)"` — must print `(13,) float32`

---

## 8. Known Issues

1. **Baseline data incomplete**: Only 3/12 baseline runs finished. The M3_0
   evaluation experiment will run its own baselines.

2. **xml017 crash signal is noisy**: Both buggy and fixed versions crash.
   Differential features come from coverage, not crashes. Documented.

3. **Statistical power**: n=3 per group. Bonferroni makes nothing significant.
   Ranking uses effect size (A12). Publishable but must acknowledge limitation.

4. **`afl->shm.map` vs `afl->fsrv.trace_bits`**: The cumulative map update
   needs `afl->shm.map` (the shared trace bitmap). In some AFL++ versions this
   is `afl->fsrv.trace_bits`. Check `afl-fuzz.h` for the correct field. The
   existing telemetry mutator uses `afl->shm.map` (line 660 of mutator_telemetry.c).

---

## 9. Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `experiments/differential/analysis/m3_0_feature_spec.json` | Feature specification | Complete |
| `docs/m3_0_feature_derivation.md` | Feature rationale with data | Complete |
| `docs/differential_fuzzing_experiment_plan.md` | Original experiment plan (10 issues fixed) | Complete |
| `src/mutator_m3_0.c` | C mutator | **REWRITE NEEDED** |
| `scripts/models/m3_0.py` | Python model module | **REWRITE NEEDED** |
| `scripts/models/common.py` | ContextualBanditAgent | Ready |
| `scripts/rl_server.py` | --algorithm flag | Ready |
| `scripts/run_model.sh` | --algorithm passthrough | Ready |
| `scripts/run_m3_0_experiment.sh` | Evaluation pipeline | Ready |
| `src/mutator_m0_0.c` | Source for apply_mutation() and helpers to copy | Reference |
| `src/mutator_telemetry.c` | Reference for cumulative_map pattern | Reference |
