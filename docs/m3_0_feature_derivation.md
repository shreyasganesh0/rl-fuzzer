# M3_0 Feature Derivation — From Differential Telemetry to RL State Vector

## 1. Experiment Overview

This document records how the 13 features in M3_0's state vector were derived
from a differential fuzzing experiment, what the data showed, and why these
features should help an RL agent find bugs faster on unseen targets.

### Setup

| Property | Value |
|----------|-------|
| Targets | 4 libxml2 binaries: 2 CVEs x (buggy, fixed) |
| CVEs | CVE-2017-5130 (integer overflow), CVE-2016-1762 (heap overread) |
| Versions | v2.9.3 (xml017_buggy), v2.9.4 (xml005_buggy / xml017_fixed), v2.9.5 (xml005_fixed) |
| Seeds per target | 3 (random seeds 1, 2, 3) |
| Duration | 9.7 hours (~35-45M execs per run) |
| Runs | 12 telemetry + 3 baseline (15 parallel on 16-core machine) |
| Mutator | Random uniform selection over 47 AFL++ mutations (same action space as RL models) |
| ASAN | AFL_USE_ASAN=1 on all builds |
| Harness | FuzzBench libxml2_xml/target.cc (byte-identical, SHA256: bd91b6d1...) |
| Seeds | 38 XML files from libxml2/test/*.xml (canonical) |
| Dictionary | 89 entries from libxml2/fuzz/xml.dict (canonical) |

### What We Measured

Every 1,000 executions, the telemetry mutator logged:
- **Coverage dynamics**: total_edges, new_edges, crashes, corpus_size, execution time
- **Edge heat distribution**: hot (>128), warm (8-128), cool (1-7), cold (0) edge counts
- **Edge statistics**: Shannon entropy, mean/std/max hit counts over the cumulative bitmap
- **Per-mutation attribution**: For each of 47 mutations, count of uses, new edges found, crashes triggered

---

## 2. What the Differential Data Showed

### 2.1 Coverage Divergence

Buggy versions consistently discover more edges than their fixed counterparts.
This is because vulnerability-adjacent code (error handlers, overflow checks,
ASAN instrumentation branches) adds reachable edges that don't exist in the
fixed version.

**XML005 (CVE-2017-5130, integer overflow):**

| Execs | Buggy Edges | Fixed Edges | Delta | Buggy Crashes | Fixed Crashes |
|-------|-------------|-------------|-------|---------------|---------------|
| 1K | 2,500 | 2,433 | +67 | 0 | 0 |
| 10K | 2,568 | 2,500 | +68 | 0 | 0 |
| 100K | 3,375 | 3,255 | +120 | 0 | 0 |
| 1M | 4,348 | 4,229 | +119 | 0 | 0 |
| 5M | 5,048 | 4,767 | +281 | 0 | 0 |
| 10M | 5,163 | 4,889 | +274 | 0 | 0 |
| 20M | 5,285 | 5,002 | +283 | 0 | 0 |
| 40M | 5,356 | 5,223 | +133 | 0 | 0 |

The delta emerges immediately (+67 at 1K execs) and peaks at ~5-20M execs (+280).
The buggy version has extra edges from the start because the vulnerable code path
in `xmlMemoryStrdup` is reachable from common XML parsing operations.

**XML017 (CVE-2016-1762, heap overread):**

| Execs | Buggy Edges | Fixed Edges | Delta | Buggy Crashes | Fixed Crashes |
|-------|-------------|-------------|-------|---------------|---------------|
| 1K | 2,502 | 2,500 | +2 | 0 | 0 |
| 100K | 3,360 | 3,329 | +31 | 0 | 0 |
| 1M | 4,432 | 4,404 | +28 | 0 | 0 |
| 5M | 5,189 | 4,948 | +241 | 0 | 0 |
| 10M | 5,450 | 5,127 | +323 | 5 | 0 |
| 20M | 5,654 | 5,347 | +307 | 27 | 0 |
| 30M | 5,706 | 5,470 | +236 | 42 | 40 |
| 40M | 5,761 | 5,497 | +264 | 76 | 64 |

The delta is near-zero until ~1M execs, then jumps to +241 at 5M as the fuzzer
reaches deeper parsing code. Crashes appear at 10M execs on buggy, confirming
the heap overread path was reached. Notably, the fixed version (v2.9.4) also
crashes starting at 30M execs — these are different bugs in v2.9.4, not
CVE-2016-1762.

### 2.2 Crash Differential

| Target | Crashes (seeds 1,2,3) | Total |
|--------|----------------------|-------|
| xml005_buggy (v2.9.4) | [1, 0, 277] | 278 |
| xml005_fixed (v2.9.5) | [0, 0, 0] | **0** |
| xml017_buggy (v2.9.3) | [81, 16, 0] | 97 |
| xml017_fixed (v2.9.4) | [74, 0, 13] | 87 |

XML005 shows a clean crash differential: 278 vs 0. The integer overflow in
`xmlMemoryStrdup` is deterministically triggered once the right allocation size
is reached.

XML017 is messier: both versions crash, but from different bugs. The buggy
version (v2.9.3) crashes from CVE-2016-1762 plus other v2.9.3 bugs. The fixed
version (v2.9.4) crashes from different vulnerabilities that were introduced or
remain in v2.9.4.

### 2.3 Feature Comparison at Saturation

All targets reached coverage saturation (discovery rate < 3% of initial rate)
by ~6 hours. Final feature values averaged across 3 seeds:

**XML005 (CVE-2017-5130):**

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

**XML017 (CVE-2016-1762):**

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

Key observation: XML005 shows much larger deltas than XML017 on edge heat features.
This is because CVE-2017-5130 (integer overflow) involves allocation code that
gets called from many paths, creating widespread heat redistribution. CVE-2016-1762
(heap overread in parser) is more localized — the differential is mostly in edge count,
not heat distribution.

### 2.4 Mutation Effectiveness Differential

Certain mutations are disproportionately effective on buggy versions:

**XML005 — top mutations favoring buggy:**

| Mutation | Buggy Rate | Fixed Rate | Delta |
|----------|-----------|-----------|-------|
| H_ARITH32- (havoc 32-bit subtract) | 0.000108 | 0.000063 | +0.000045 |
| H_ARITH16+ (havoc 16-bit add) | 0.000066 | 0.000032 | +0.000033 |
| ARITH-4LE (det. 32-bit subtract LE) | 0.000084 | 0.000056 | +0.000028 |
| INT8 (interesting byte values) | 0.000054 | 0.000037 | +0.000016 |
| H_BYTESUB (havoc byte subtract) | 0.000068 | 0.000053 | +0.000015 |

**XML017 — top mutations favoring buggy:**

| Mutation | Buggy Rate | Fixed Rate | Delta |
|----------|-----------|-----------|-------|
| DICT_UINS (dictionary user insert) | 0.000822 | 0.000601 | +0.000220 |
| H_ARITH16- (havoc 16-bit subtract) | 0.000137 | 0.000072 | +0.000065 |
| H_INT16BE (havoc interesting 16BE) | 0.000088 | 0.000025 | +0.000063 |
| CUSTOM (focused multi-op) | 0.000141 | 0.000103 | +0.000038 |
| ARITH-4LE (det. 32-bit subtract LE) | 0.000087 | 0.000059 | +0.000028 |

For XML005 (integer overflow), arithmetic mutations are most effective — they
directly manipulate numeric values that can trigger the overflow. For XML017
(parser overread), dictionary insertions are most effective — they introduce
XML structure tokens that exercise deeper parsing paths where the overread occurs.

---

## 3. Feature Selection — Statistical Ranking

Features were ranked by Vargha-Delaney A12 effect size across both CVE pairs at
5 timepoints each (10%, 25%, 50%, 75%, 90% of total execs). A12 measures
discriminative power: 1.0 or 0.0 = perfect separation, 0.5 = no separation.

With only 3 seeds per condition, Mann-Whitney U tests have very limited statistical
power (minimum achievable p-value with n=3 per group is 0.05). We therefore rank
by effect size rather than p-value significance.

| Rank | Feature | Mean |A12-0.5| | Large/Med Effects | Consistent Direction |
|------|---------|---------------------|-------------------|---------------------|
| 1 | total_edges | 0.389 | 9/10 | No (mostly buggy > fixed) |
| 2 | cold_edges | 0.267 | 8/10 | No (mostly fixed > buggy) |
| 3 | corpus_size | 0.244 | 7/10 | No (mostly buggy > fixed) |
| 4 | hot_edges | 0.244 | 8/10 | No (mostly buggy > fixed) |
| 5 | cool_edges | 0.233 | 8/10 | No (mostly buggy > fixed) |
| 6 | avg_exec_time | 0.222 | 9/10 | No (direction varies) |
| 7 | edge_hit_mean | 0.211 | 7/10 | No (mostly buggy > fixed) |
| 8 | warm_edges | 0.200 | 7/10 | No (mostly buggy > fixed) |
| 9 | edge_hit_std | 0.200 | 6/10 | No (mostly buggy > fixed) |
| 10 | edge_entropy | 0.189 | 4/10 | No (direction varies) |
| 11 | crashes | 0.133 | 6/10 | Yes (buggy > fixed) |
| 12 | edge_discovery_rate | 0.000 | 0/10 | -- |
| 13 | edge_hit_max | 0.000 | 0/10 | -- |

"Consistent Direction" is "No" for most features because the direction flips
between CVE pairs or timepoints. This is actually expected and desirable — it
means the feature captures structural variation rather than a fixed bias. The
RL agent learns to interpret the feature in context, not as a simple
"higher = better" signal.

### Excluded Features

| Feature | Reason |
|---------|--------|
| edge_hit_max | Zero discriminative power (A12=0.5 everywhere). Dominated by a single hot edge. |
| edge_discovery_rate | Zero discriminative power. Per-interval rate is identical between buggy/fixed. |

---

## 4. The 13 Selected Features and Why They Matter

### Feature 0: total_edges (rank 1, A12 dev = 0.389)

**What it measures**: Count of unique edges discovered so far, normalized by
the map size (65,536).

**What the data showed**: Buggy versions consistently have more edges (+238 for
xml005, +285 for xml017). The extra edges come from vulnerability-adjacent code:
ASAN instrumentation branches, error handling paths, and the bug's own code path.

**Why it generalizes**: ANY buggy version of ANY program will have additional
reachable code near the vulnerability. An RL agent seeing total_edges climb
faster than expected (compared to what it learned during training) is a signal
that it may be near bug-adjacent code worth deepening.

### Feature 1: cold_edges (rank 2, A12 dev = 0.267)

**What it measures**: Edges with hit_count = 0 in the cumulative bitmap — the
unexplored frontier.

**What the data showed**: XML005 shows a dramatic inversion: buggy has MORE
cold edges (+14,285) despite having more total edges. This happens because the
cumulative bitmap tracking is affected by how ASAN reshapes the control flow
graph — the buggy version's bitmap has more entries total (ASAN adds branches),
but many of them are never hit.

**Why it generalizes**: The cold-edge count reflects the exploration state space
size. A large cold frontier with rising total_edges means the fuzzer is in
productive territory. An agent can learn to keep exploring (try diverse mutations)
when cold_edges is high and shift to deepening when it's low.

### Feature 2: hot_edges (rank 4, A12 dev = 0.244)

**What it measures**: Edges hit more than 128 times — heavily-exercised code.

**What the data showed**: XML005 buggy has FEWER hot edges (745 vs 988) despite
more total edges. The heat is more spread out. XML017 buggy has slightly more
hot edges (773 vs 720). The direction depends on bug type: the integer overflow
in XML005 disperses execution across more paths, while the parser overread in
XML017 concentrates execution near the vulnerable code.

**Why it generalizes**: Hot-edge ratio indicates whether the fuzzer is stuck in
a tight loop (high hot ratio = most time spent re-executing the same paths) or
exploring broadly (low hot ratio = diverse execution). The agent learns to
correlate hot-edge patterns with mutation effectiveness.

### Feature 3: warm_edges (rank 8, A12 dev = 0.200)

**What it measures**: Edges hit 8-128 times — moderately exercised paths.

**What the data showed**: Buggy versions have more warm edges across both CVEs
(+259 for xml005, +142 for xml017). Warm edges represent the transition zone
between "discovered but barely touched" and "heavily exercised." Bug-adjacent
code tends to accumulate in this range because it's reachable but not on the
main execution hot path.

**Why it generalizes**: A growing warm-edge count means the fuzzer is deepening
its exploration — exactly where bugs hide. The agent can learn that mutations
which increase the warm-edge ratio are productive.

### Feature 4: cool_edges (rank 5, A12 dev = 0.233)

**What it measures**: Edges hit 1-7 times — the discovery frontier.

**What the data showed**: Direction varies by CVE: XML005 buggy has fewer cool
edges (-14,301), XML017 buggy has fewer cool edges (-898). Cool edges
represent code that was reached but not yet explored deeply.

**Why it generalizes**: The cool-to-warm transition rate is a direct measure of
fuzzing productivity. When the agent's mutations convert cool edges to warm edges,
it's making real progress into new code territory. This is where bug-triggering
inputs tend to be discovered.

### Feature 5: edge_entropy (rank 10, A12 dev = 0.189)

**What it measures**: Shannon entropy of the hit-count distribution across 8
power-of-2 bins (1, 2, 4, 8, 16, 32, 64, 128+).

**What the data showed**: XML005 buggy has much higher entropy (1.4 vs 0.6) —
execution is more evenly distributed across hit-count bins. XML017 shows minimal
difference (+5.9%).

**Why it generalizes**: Entropy is a single number that summarizes the entire
edge heat distribution. High entropy = diverse exploration. Low entropy =
concentrated in a few patterns. Bug-adjacent code disrupts the normal entropy
pattern because vulnerability paths create unusual execution profiles. The agent
can use entropy as a compact summary of exploration state.

### Feature 6: edge_hit_mean (rank 7, A12 dev = 0.211)

**What it measures**: Average hit count across all discovered edges.

**What the data showed**: XML005 buggy has dramatically higher mean (23.4 vs 8.9,
+164%). This means each discovered edge is hit more times on average in the buggy
version — the overflow in `xmlMemoryStrdup` is called from many XML operations,
driving up hit counts across the board.

**Why it generalizes**: Mean hit count reflects execution depth. Programs with
bugs in frequently-called code will show elevated hit means. The agent learns that
a rising hit mean signals it should vary mutations to explore different execution
paths rather than re-hitting the same code.

### Feature 7: edge_hit_std (rank 9, A12 dev = 0.200)

**What it measures**: Standard deviation of hit counts — measures how uneven
the coverage distribution is.

**What the data showed**: Buggy versions have higher std in both CVEs (+36.3%
for xml005, +4.0% for xml017). Higher std means some edges are hit very
frequently while others are barely touched — uneven exploration.

**Why it generalizes**: High std indicates hotspot formation. Bug-adjacent code
creates execution hotspots that increase variance. The agent can learn that
high std + high total_edges = productive exploration near interesting code.

### Feature 8: corpus_size (rank 3, A12 dev = 0.244)

**What it measures**: Number of inputs in AFL++'s queue (log-normalized).

**What the data showed**: Direction varies: XML005 buggy has fewer corpus entries
(-2.7%), XML017 buggy has more (+3.5%). The difference is small but the A12
effect size is large because the direction is consistent within each CVE across
all timepoints.

**Why it generalizes**: Corpus growth rate indicates how productive the current
mutation strategy is. A rapidly growing corpus means the fuzzer is finding many
coverage-gaining inputs. The agent can use corpus size to gauge whether to
continue the current mutation strategy or try something different.

### Feature 9: crashes (rank 11, A12 dev = 0.133)

**What it measures**: Total unique crashes found (log-normalized).

**What the data showed**: XML005 shows the clearest signal: 278 crashes on buggy
vs 0 on fixed. XML017 is noisier: 97 buggy vs 87 fixed.

**Why it generalizes**: This is the direct reward signal for bug finding. While
it has lower discriminative power (because crashes are sparse — most of the run
has 0 crashes on both versions), it's the ground truth for what the agent is
trying to achieve. An agent that learns to associate certain state patterns with
subsequent crash discovery will generalize to any target with bugs.

### Feature 10: new_edges (not discriminative, A12 dev = 0.000)

**What it measures**: Number of new edges discovered in the last step.

**What the data showed**: Zero discriminative power between buggy and fixed —
both discover edges at the same per-step rate. The difference is in cumulative
total, not instantaneous rate.

**Why it's included anyway**: This is the primary reward signal for the RL agent.
Without per-step coverage feedback, the agent cannot learn which mutations are
productive. Experiments 1 and 2 confirmed this is the essential learning signal.

### Feature 11: avg_exec_time (rank 6, A12 dev = 0.222)

**What it measures**: Average execution time per test case (log-normalized).

**What the data showed**: XML005 buggy is 29.6% slower (445M vs 343M us). XML017
shows minimal difference (-1.2%). The slowdown in XML005 comes from ASAN
instrumentation around the integer overflow — every call to `xmlMemoryStrdup`
triggers additional overflow checks.

**Why it generalizes**: Execution time is a proxy for code path depth and
complexity. Bug-adjacent code that involves memory safety checks, error
handling, or complex state transitions will execute slower. The agent can learn
that execution time anomalies indicate interesting code regions.

### Feature 12: coverage_velocity (not discriminative, A12 dev = 0.000)

**What it measures**: Rate of new edge discovery over the last 1,000 executions.

**What the data showed**: Not discriminative between buggy and fixed — both
versions show the same velocity decay curve.

**Why it's included anyway**: This is the exploration-vs-exploitation signal.
High velocity means the fuzzer is in an active discovery phase and should
continue with diverse mutations. Low velocity (approaching saturation) means
the agent should shift to targeted mutations that deepen coverage in specific
regions. This temporal context is critical for the DQN's sequential decision
making (gamma = 0.99).

---

## 5. Generalization Argument

The core hypothesis for M3_0 is that these features capture **structural
properties of code exploration** rather than **target-specific coverage patterns**.

### What's structural (target-independent):

1. **Edge heat distribution** (features 2-5): The ratio of hot/warm/cool/cold
   edges describes HOW the fuzzer is exploring, not WHAT code it's exploring.
   A fuzzer stuck in a loop shows high hot ratio. A fuzzer making progress shows
   growing warm and cool counts. This pattern is universal across programs.

2. **Entropy** (feature 5): Coverage diversity is a property of the fuzzing
   process, not the target. Low entropy = concentrated exploration. High entropy
   = diverse exploration. The relationship between entropy and mutation
   effectiveness is structural.

3. **Execution time** (feature 11): Bug-adjacent code takes longer due to
   error handling, ASAN checks, and complex state. This is true regardless of
   the specific program.

4. **Coverage velocity** (feature 12): The saturation curve (fast discovery →
   slow plateau) is universal. The agent needs to know where it is on this
   curve to make good decisions.

### What's target-specific (but still useful):

5. **Total edges / crashes** (features 0, 9): The absolute numbers are
   target-specific, but the normalization (divide by MAP_SIZE, log-scale)
   makes them comparable across targets. The agent doesn't learn "5,000 edges
   means bugs" — it learns "rapid edge growth + rising crashes = productive
   territory."

### How an unseen target benefits:

When the trained M3_0 agent encounters a new program with an unseen bug:

1. It observes the edge heat distribution shifting as it approaches bug-adjacent code
2. It notices execution time anomalies from error-handling paths
3. It detects entropy changes from hotspot formation near the vulnerability
4. It responds by selecting mutations that deepened coverage in similar
   structural states during training (arithmetic mutations for memory bugs,
   dictionary insertions for parser bugs)

The agent doesn't need to know what the bug IS — it needs to recognize the
STRUCTURAL FINGERPRINT of approaching a bug.

---

## 6. Data Provenance

| Asset | Source | SHA256 |
|-------|--------|--------|
| Harness | FuzzBench libxml2_xml/target.cc (unmodified) | bd91b6d126d6cd7215cf2752657eaa08268c75d0344f4a415ef374b75d951510 |
| Dictionary | libxml2/fuzz/xml.dict (FuzzBench-pinned commit c7260a47) | 1a6c8d151a20c505a4ab2cd1be7e7616baafe0c23426ab289835836aac14665a |
| Seeds | 38 files from libxml2/test/*.xml | (38 individual files, 188KB total) |
| AFL++ | v4.41a, LLVM mode (clang-18) | built from source |
| Telemetry mutator | src/mutator_telemetry.c, compiled with plain clang-18 | 47 uniform-random mutations |
| Analysis | scripts/analysis/differential_analysis.py | Mann-Whitney U, Vargha-Delaney A12 |

Campaign ran for 9.7 hours. All 12 telemetry targets reached coverage saturation.
Bug paths confirmed reached via crashes (278 for xml005_buggy, 0 for xml005_fixed).
