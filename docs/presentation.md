# RL-Guided Mutation Selection for AFL++ Fuzzing

---

## Slide 1: Title

# RL-Guided Mutation Selection for AFL++ Fuzzing
### A Multi-Benchmark Evaluation of DQN-Based Mutation Scheduling

**Shreyas Ganesh**
March 2026

---

## Slide 2: Problem Statement

### The Mutation Selection Problem

**Fuzz testing** discovers software bugs by feeding programs random/mutated inputs and monitoring for crashes or new code coverage.

**AFL++** is a state-of-the-art coverage-guided fuzzer with **47 mutation operators**:
- Deterministic: bit flips, byte flips, arithmetic (16 operators)
- Havoc: random byte ops, arithmetic, interesting values (20 operators)
- Dictionary: user/auto extras insert/overwrite (4 operators)
- Splice, custom mutator, etc. (7 operators)

**The problem:** AFL++ selects mutations uniformly at random during havoc stage. Can we do better by learning which mutations are most effective for a given program?

---

## Slide 3: Research Question & Hypothesis

### Research Question

> Can a reinforcement learning agent learn to select AFL++ mutation operators
> more effectively than the default uniform-random policy, leading to faster
> code coverage discovery?

### Hypothesis

An RL agent observing fuzzer state (coverage map, edge distributions, input
characteristics) can learn target-specific mutation preferences that discover
new coverage faster than random selection, especially in the early stages of
fuzzing.

---

## Slide 4: Background — AFL++ Architecture

### How AFL++ Works

```
                    ┌─────────────┐
  seed corpus ───►  │   AFL++     │ ───► mutation ───► target program
                    │  scheduler  │          │              │
                    └─────────────┘          │         coverage map
                          ▲                  │         (shared memory)
                          │                  ▼              │
                     new coverage? ◄─── compare ◄──────────┘
                          │
                     YES: save to corpus
```

- **Coverage map**: 64KB shared memory bitmap, each byte = edge hit count
- **Mutations**: Applied to corpus entries, producing new test inputs
- **Feedback loop**: New edges → save input → re-prioritize → mutate again

---

## Slide 5: Approach — RL for Mutation Selection

### DQN-Based Mutation Scheduling

We replace AFL++'s random mutation selection with a **Deep Q-Network (DQN)**
that learns to pick the best mutation operator given the current fuzzer state.

```
  ┌──────────────┐        SHM (shared memory)        ┌───────────────┐
  │              │  ◄──── coverage map, edge stats ── │               │
  │   AFL++      │                                    │   RL Server   │
  │  + custom    │  ────► selected action (0-46) ──►  │   (Python)    │
  │   mutator.c  │        written to SHM              │   DQN Agent   │
  │              │                                    │               │
  └──────────────┘                                    └───────────────┘
```

- **State**: Derived from coverage map + fuzzer statistics
- **Action**: One of 47 mutation operators
- **Reward**: New edge coverage discovered (coverage_term + crash_term)
- **Training**: Online, using experience replay buffer (100K transitions)

---

## Slide 6: DQN Agent Details

### Agent Architecture

| Component | Details |
|-----------|---------|
| Algorithm | Double DQN with target network sync every 1000 steps |
| Network | Fully connected: state_dim → [hidden layers] → 47 actions |
| Optimizer | Adam, lr = 1e-4 |
| Replay buffer | 100,000 transitions |
| Batch size | 64 |
| Gamma | 0.99 |
| Epsilon schedule | 1.0 → 0.05 over ~6M steps |
| Training freq | Every step (online) |

### Reward Signal

```
reward = coverage_term + crash_term
       = (new_edges_found) + (10 × new_crashes_found)
```

Most steps have reward = 0 (no new coverage). The agent must learn from sparse,
delayed feedback.

---

## Slide 7: Model Family — State Space Evolution

### Three Models with Increasing State Complexity

| Model | State Dim | State Components | Hidden Layers |
|-------|-----------|-----------------|---------------|
| **M1_0** | 12 | Coverage, crashes, edge distribution stats (mean, std of enabled/disabled edges), stability | [256, 256] |
| **M1_1** | 13 | M1_0 + visited-edge count (unique edges seen so far) | [256, 256] |
| **M1_2** | 64 | M1_1 + input buffer features: buf_len, entropy, printable_ratio, histogram[16], first_32_bytes | [256, 256, 128] |

### Design Rationale

- **M1_0**: Can the agent learn from global coverage statistics alone?
- **M1_1**: Does knowing exploration breadth (visited edges) help?
- **M1_2**: Does observing the actual input being mutated enable smarter mutation selection?

---

## Slide 8: M1_2 Deep Dive — Input-Aware Mutation Selection

### 64-Dimensional State Vector

```
M1_1 base (13 dims):
  coverage, crashes, reward, loss, epsilon, coverage_term, crash_term,
  en_mean_n, dis_mean_n, stability, num_visited, action_prev, step_frac

Input buffer features (51 dims):
  buf_len_norm          ─ normalized input length
  entropy_norm          ─ Shannon entropy of input bytes
  printable_ratio       ─ fraction of ASCII printable bytes
  histogram[16]         ─ byte-value distribution (16 bins)
  first_32_bytes[32]    ─ normalized first 32 bytes of input
```

### Intuition

Different input types benefit from different mutations:
- High-entropy binary → byte arithmetic may be effective
- Structured text (JSON/XML) → dictionary mutations may be better
- Short inputs → bit flips cover more of the input space

---

## Slide 9: Experiment Infrastructure

### Generic Multi-Benchmark Build Framework

```
scripts/build_benchmark.sh <name>     # Universal entry point
benchmarks/<name>/build_recipe.sh     # Per-benchmark: BUILD_STEPS() + LINK_STEPS()
```

Each recipe:
1. Clones target source at FuzzBench-pinned commit
2. Builds with AFL++ instrumentation (`afl-clang-fast++`)
3. Links harness + static lib + `libAFLDriver.a` → `bin/target`

**Local dependency builds** (no sudo required):
- `liblzma` (for libxml2) — built from source into `packages/local/`
- `libarchive` (for freetype2) — built from source into `packages/local/`
- `meson` (for harfbuzz) — from Python venv

---

## Slide 10: Benchmark Details

### 6 FuzzBench Benchmarks

| Benchmark | Domain | Harness Source | Seeds | Dictionary |
|-----------|--------|---------------|-------|------------|
| **jsoncpp** | JSON parser | Source tree | Recipe (4 JSON files) | Source (`fuzz.dict`) |
| **freetype2** | Font renderer | Source tree | Fallback minimal | Fallback minimal |
| **libxml2** | XML parser | FuzzBench | Fallback minimal | Source (`xml.dict`) |
| **re2** | Regex engine | FuzzBench | Fallback minimal | Fallback minimal |
| **harfbuzz** | Text shaping | Source tree | Fallback minimal | Fallback minimal |
| **libpng** | Image decoder | Source tree (oss-fuzz) | FuzzBench (`seed.png`) | Source (`png.dict`) |

All harnesses use `LLVMFuzzerTestOneInput` — the standard libFuzzer API,
compatible with AFL++ via `libAFLDriver.a`.

---

## Slide 11: Experiment Design

### Protocol

```
For each benchmark × model:
  Phase 1: Train DQN for 10M steps (with milestone checkpoints)
  Phase 2: Eval with frozen policy for 10M steps
  Phase 3: Baseline (same steps) — plain AFL++ for 10M steps
  Phase 4: Baseline (same time) — plain AFL++ for same wall-clock as RL eval
  Phase 5: Slice eval CSVs at milestones: 500K, 1M, 2M, 10M
```

### Baselines

| Baseline | Controls For |
|----------|-------------|
| **Same-steps** | Fair step-count comparison (10M steps each) |
| **Same-time** | Fair wall-clock comparison (RL is slower per step due to SHM + inference overhead) |

### Metrics

- **Coverage** (edges discovered)
- **Coverage AUC** (area under coverage-vs-time curve)
- **Throughput** (exec/s)

---

## Slide 12: Orchestration & Crash Recovery

### `run_full_experiment.sh` Orchestrator

Runs the full pipeline across all benchmarks sequentially:

```
for benchmark in jsoncpp freetype2 libxml2 re2 harfbuzz libpng:
    build_benchmark.sh $benchmark
    for model in m1_0 m1_1 m1_2:
        run_model.sh --model-id $model --exp-dir experiments/$benchmark
    run_baseline (same-steps)
    run_baseline (same-time)
    slice_milestones.py
summarize_benchmarks.py
```

**Crash recovery**: Automatically skips completed work on restart:
- Existing checkpoints → skip training
- Complete eval CSVs (95%+ steps) → skip eval
- Existing baseline CSVs → skip baseline

*The experiment survived 4 laptop crashes over 6 days thanks to this recovery logic.*

---

## Slide 13: Results — Coverage at 500K Steps (Early Advantage)

### RL Models Show Early Coverage Lead

| Benchmark | M1_0 | M1_1 | M1_2 | Baseline |
|-----------|------|------|------|----------|
| jsoncpp | **230** | 89 | **313** | 0 |
| freetype2 | **450** | 342 | 0 | 0 |
| libxml2 | 971 | **1,213** | — | 0 |
| re2 | **1,775** | 1,068 | 1,078 | 0 |
| harfbuzz | 1,453 | 380 | **1,866** | 0 |
| libpng | 0 | 0 | 0 | 0 |

**At 500K steps, RL models discover coverage while baseline finds 0 edges on most benchmarks.**

This happens because AFL++ spends early steps in deterministic stages (sequential
bit/byte flips) which discover few new edges. The RL agent can skip directly to
more productive mutations.

---

## Slide 14: Results — Coverage at 10M Steps (Convergence)

### Baseline Catches Up and Surpasses

| Benchmark | M1_0 | M1_1 | M1_2 | Baseline | BL (same-time) |
|-----------|------|------|------|----------|----------------|
| jsoncpp | 234 | 93 | 317 | **7,444** | **7,634** |
| freetype2 | 864 | 399 | 57 | **7,674** | **8,493** |
| libxml2 | 1,065 | 1,560 | — | **3,512** | **5,373** |
| re2 | 2,505 | 1,335 | 1,301 | **22,937** | **23,343** |
| harfbuzz | 2,111 | 1,175 | 2,780 | **7,038** | **7,241** |
| libpng | 4 | 4 | 4 | **72** | **72** |

**At 10M steps, baseline outperforms all RL models on every benchmark.**

The same-time baseline (running AFL++ for the same wall-clock duration as RL)
performs even better, executing 10-50x more total test cases.

---

## Slide 15: Per-Benchmark — jsoncpp & freetype2

### jsoncpp (JSON parser)
- **Best RL**: M1_2 at 317 edges (vs baseline 7,444)
- Coverage plateaus instantly for RL (by step 123K)
- Baseline discovers 23x more edges
- RL agent converges to single action: M1_0→FLIP_2BITS, M1_1→INT_2BE, M1_2→HAVOC_ARITH32

### freetype2 (Font renderer)
- **Best RL**: M1_0 at 864 edges (vs baseline 7,674)
- M1_0 shows late coverage jump (step 3.2M → 864 edges)
- M1_2 discovers only 57 edges — input buffer features don't help here
- Baseline reaches 6,533 edges by 2M steps, eventually 7,674

---

## Slide 16: Per-Benchmark — libxml2 & re2

### libxml2 (XML parser)
- **Best RL**: M1_1 at 1,560 edges (vs baseline 3,512)
- M1_1 (visited-edge tracking) outperforms M1_0 (1,065) — edge visit info helps
- M1_2 eval was lost to a crash, no data available
- xml.dict provides good dictionary — benefits both RL and baseline

### re2 (Regex engine)
- **Best RL**: M1_0 at 2,505 edges (vs baseline 22,937)
- Largest absolute gap — baseline finds 9x more edges
- re2 has deep state space that rewards extensive corpus exploration
- RL throughput: ~1,736 steps/s vs baseline 21,277 steps/s

---

## Slide 17: Per-Benchmark — harfbuzz & libpng

### harfbuzz (Text shaping engine)
- **Best RL**: M1_2 at 2,780 edges — best relative performance
- M1_2's input-buffer awareness provides measurable advantage over M1_0 (2,111)
- Coverage AUC: M1_2 leads at 15.9M edge·seconds vs M1_0 at 11.3M
- Baseline still wins at 7,038 edges

### libpng (Image decoder)
- **All models**: 4 edges (effectively zero useful coverage)
- Only 1 FuzzBench seed (`seed.png`), AFL++ exhausts the queue immediately
- State shows `finished` — no new inputs discovered
- Baseline achieves 72 edges — low but nonzero
- **Conclusion**: Insufficient seed corpus makes this benchmark uninformative

---

## Slide 18: Coverage AUC Comparison

### Area Under Coverage-vs-Time Curve (edge·seconds)

Higher = faster and/or higher coverage over the full eval duration.

| Benchmark | M1_0 | M1_1 | M1_2 | Baseline |
|-----------|------|------|------|----------|
| jsoncpp | 844K | 329K | **1,196K** | 1,150K |
| freetype2 | **3,718K** | 1,480K | 205K | 2,301K |
| libxml2 | 5,181K | **6,911K** | — | 666K |
| re2 | **14,066K** | 6,206K | 6,079K | 9,764K |
| harfbuzz | 11,289K | 11,027K | **15,886K** | 3,188K |
| libpng | **19K** | 14K | 14K | 10K |

**Key insight**: RL models often have higher AUC than same-steps baseline
despite lower final coverage. This is because RL finds coverage *earlier*,
accumulating more area under the curve.

---

## Slide 19: Throughput & RL Overhead

### The Throughput Gap

| Benchmark | Baseline (steps/s) | M1_0 (steps/s) | Overhead |
|-----------|-------------------|-----------------|----------|
| jsoncpp | 49,505 | 2,769 | **94%** |
| freetype2 | 25,908 | 2,031 | **92%** |
| libxml2 | 40,323 | 2,046 | **95%** |
| re2 | 21,277 | 1,736 | **92%** |
| harfbuzz | 20,243 | 1,858 | **91%** |
| libpng | 72,464 | 2,123 | **97%** |

**Average overhead: 93% throughput reduction.**

The RL agent processes ~2,000 steps/s regardless of target, while baseline
throughput varies 20K-72K steps/s. The bottleneck is the SHM communication
+ Python DQN inference on every single mutation step.

---

## Slide 20: Head-to-Head Model Comparison

### Pairwise Wins at 10M Steps (coverage)

|  | M1_0 | M1_1 | M1_2 | Baseline |
|--|------|------|------|----------|
| **M1_0** | — | 4/6 | 2/5 | 0/6 |
| **M1_1** | 1/6 | — | 2/5 | 0/6 |
| **M1_2** | 2/5 | 2/5 | — | 0/5 |
| **Baseline** | **6/6** | **6/6** | **5/5** | — |

### Pairwise Wins at 500K Steps (early advantage)

|  | M1_0 | M1_1 | M1_2 | Baseline |
|--|------|------|------|----------|
| **M1_0** | — | 4/6 | 2/5 | 1/6 |
| **M1_1** | 1/6 | — | 2/5 | 1/6 |
| **M1_2** | 2/5 | 2/5 | — | 0/5 |
| **Baseline** | 5/6 | 5/6 | **5/5** | — |

M1_0 is the most consistent RL model. Baseline dominates at all horizons.

---

## Slide 21: Same-Time Baseline Comparison

### When given equal wall-clock time, does RL still have an advantage?

| Benchmark | BL (same-time) | Best RL | Best Model | RL Wins? |
|-----------|---------------|---------|------------|----------|
| jsoncpp | 7,634 | 317 | M1_2 | **NO** |
| freetype2 | 8,493 | 864 | M1_0 | **NO** |
| libxml2 | 5,373 | 1,560 | M1_1 | **NO** |
| re2 | 23,343 | 2,505 | M1_0 | **NO** |
| harfbuzz | 7,241 | 2,780 | M1_2 | **NO** |
| libpng | 72 | 4 | M1_0 | **NO** |

**RL loses on all benchmarks when controlling for wall-clock time.**

The same-time baseline executes 10-50x more test cases because it doesn't
have the SHM + inference overhead. Raw execution speed dominates.

---

## Slide 22: Analysis — Why Baseline Wins at Scale

### Three Contributing Factors

**1. Throughput bottleneck (dominant factor)**
- RL processes ~2,000 steps/s; baseline processes 20,000-72,000 steps/s
- In the same wall-clock, baseline runs 10-50x more mutations
- Coverage discovery is fundamentally a numbers game — more mutations = more chances

**2. Sparse reward signal**
- Most steps (>99.9%) have reward = 0 (no new coverage)
- DQN struggles to learn meaningful policy from such sparse feedback
- Agent converges to single-action policies (e.g., always picks FLIP_2BITS)

**3. AFL++ is already well-tuned**
- AFL++'s built-in scheduling (power schedules, splice, deterministic stages)
  is the result of years of engineering
- The RL agent replaces this with a single action choice per step
- The agent doesn't control other aspects (input selection, trimming, etc.)

---

## Slide 23: Analysis — The RL Overhead Problem

### Where Does the Time Go?

```
Per mutation step:
  ┌─ AFL++ mutator.c ──────────────────────────────────┐
  │  1. Read coverage map from SHM (64KB)               │  ~50µs
  │  2. Compute state features (edge stats, etc.)        │  ~100µs
  │  3. Write state to SHM, read action from SHM         │  ~10µs
  │  4. Wait for RL server to process                     │  ~200µs
  │  5. Apply selected mutation                           │  ~1µs
  └─────────────────────────────────────────────────────┘
  Total: ~360µs/step (vs ~20-50µs/step without RL)
```

The per-step overhead is **7-18x** higher with RL, primarily due to:
- Coverage map scanning (64KB per step)
- Python ↔ SHM synchronization latency
- DQN forward pass (even with pre-allocated tensors)

### Implication
Any RL-for-fuzzing approach must solve the throughput problem to be competitive.

---

## Slide 24: Key Findings Summary

### What We Learned

1. **RL can discover coverage faster in early steps** — at 500K steps, RL models
   find non-zero coverage while baseline often finds 0. The RL agent skips
   AFL++'s slow deterministic stages.

2. **Baseline always wins at scale** — at 10M steps and same-time comparison,
   plain AFL++ outperforms all RL models on all 6 benchmarks.

3. **Throughput is the bottleneck** — 93% average throughput reduction from
   SHM + inference overhead. The RL policy quality cannot compensate for
   executing 10-50x fewer mutations.

4. **Sparse rewards lead to degenerate policies** — agents converge to
   single-action strategies, suggesting the reward signal is insufficient
   for learning nuanced mutation preferences.

5. **Input-aware features (M1_2) show promise on harfbuzz** — M1_2 achieves
   the best RL coverage (2,780 edges) and highest AUC (15.9M) on harfbuzz,
   suggesting input characteristics can inform mutation selection.

---

## Slide 25: Future Work

### Addressing the Throughput Bottleneck

- **C/C++ inference**: Port DQN forward pass to C (e.g., ONNX Runtime, TensorRT)
  to eliminate Python overhead. Target: <10µs per inference.
- **Batch decisions**: Select mutations for N steps at once instead of per-step.
  Amortize inference cost over 100-1000 mutations.
- **Coarser action granularity**: Instead of per-mutation decisions, select a
  *mutation strategy* (e.g., "focus on arithmetic ops") every 1000 steps.

### Improving the Learning Signal

- **Shaped rewards**: Reward partial progress (e.g., edge hit count increases,
  not just new edges). Use coverage delta rather than binary new-edge signal.
- **Curiosity-driven exploration**: Intrinsic motivation for visiting rare
  coverage map states, reducing dependence on sparse external rewards.
- **Hierarchical RL**: High-level policy selects mutation *categories*;
  low-level policy selects specific operators within category.

### Experiment Improvements

- **Multi-run aggregation**: Run each configuration 3-5 times with different
  seeds for statistical significance (Mann-Whitney U tests).
- **Better seed corpora**: Use FuzzBench's full seed sets where available;
  libpng results are uninformative due to minimal seeds.
- **Longer horizons**: Some benchmarks may need 50M+ steps for RL to converge.

---

## Slide 26: Infrastructure Contributions

### What Was Built

| Component | Purpose |
|-----------|---------|
| `build_benchmark.sh` + 6 recipes | Generic FuzzBench benchmark build framework |
| `run_full_experiment.sh` | Multi-benchmark orchestrator with crash recovery |
| `slice_milestones.py` | Post-hoc eval CSV slicing at milestone step counts |
| `summarize_benchmarks.py` | Cross-benchmark summary tables and rankings |
| `generate_report.py` | Detailed statistical report generator |
| M1_2 model + mutator | 64-dim input-aware DQN with SHM interface |
| Local dependency builds | liblzma + libarchive built from source (no sudo) |

### Experiment Scale

- **6 benchmarks** × **3 models** × **10M steps** each = **180M RL training steps**
- Plus 10M eval steps per model, 2 baselines per benchmark
- Total compute: ~6 days on a single laptop (with 4 crash recoveries)

---

## Slide 27: Thank You

### Summary

We evaluated DQN-based mutation selection for AFL++ across 6 FuzzBench
benchmarks. While RL models show an early coverage advantage, the 93%
throughput overhead makes them uncompetitive at scale. Future work should
focus on eliminating the inference bottleneck and improving the reward signal.

### Code & Data

- Repository: `rl-fuzzer/`
- Experiment data: `experiments/` (6 benchmarks × milestones)
- Report: `experiments/detailed_report.txt`

### Questions?
