# MuoFuzz — Interview Prep Summary

Authoritative snapshot of the rl-fuzzer repo as of 2026-04-24. Facts are cited to files in the repo. Where documentation disagrees with code, the section is marked `[VERIFY]`.

## 1. Current system architecture

The system is a two-process design: an AFL++ fuzzer process loads a C shared library (custom mutator) that communicates over mmap'd shared memory with a Python RL server. There has been no architectural change away from this topology since the original release (README.md, docs/experiment_3_full_report.md §4.1).

**Processes**

- AFL++ with `AFL_CUSTOM_MUTATOR_LIBRARY=bin/mutator_<model>.so` and `AFL_CUSTOM_MUTATOR_ONLY=1`, so every mutation opportunity routes through the plugin (docs/differential_fuzzing_experiment_plan.md §2).
- `scripts/rl_server.py` launched in a sibling process by `scripts/run_model.sh`. Single-process server; no multi-worker architecture (scripts/rl_server.py:31).

**Shared memory layout (M3_0, the current model)**

The SHM is a 128-byte mmap'd file at `/tmp/rl_shm_m3_0`. Byte offsets (src/mutator_m3_0.c:65-80):

| Off | Size | Type | Field | Direction |
|----:|:----:|:----:|:------|:----------|
| 0   | 4    | u32  | state_seq         | C → Py (release store) |
| 4   | 4    | u32  | total_edges       | C → Py |
| 8   | 4    | u32  | cold_edges        | C → Py |
| 12  | 4    | u32  | hot_edges         | C → Py |
| 16  | 4    | u32  | warm_edges        | C → Py |
| 20  | 4    | u32  | cool_edges        | C → Py |
| 24  | 4    | f32  | edge_entropy      | C → Py (pre-normalized /3.0) |
| 28  | 4    | f32  | edge_hit_mean     | C → Py (pre-normalized /255.0) |
| 32  | 4    | f32  | edge_hit_std      | C → Py (pre-normalized /255.0) |
| 36  | 4    | u32  | corpus_size       | C → Py |
| 40  | 4    | u32  | crashes           | C → Py |
| 44  | 4    | u32  | new_edges         | C → Py |
| 48  | 4    | f32  | avg_exec_time     | C → Py (pre-normalized log1p/log1p(100000)) |
| 52  | 4    | f32  | coverage_velocity | C → Py (pre-normalized /0.1, clipped) |
| 56–63 | 8  | —    | padding (cache-line split) | — |
| 64  | 4    | u32  | action_seq        | Py → C (release store) |
| 68  | 4    | i32  | action            | Py → C (0..46) |
| 72–127 | 56 | —   | padding | — |

Earlier models use different SHM sizes and feature layouts: M0_0 128 B (3-dim), M1_0/M1_1 256 B, M1_2 512 B, M2 1024 B (README.md §Models).

**Synchronization primitives**

- C increments `state_seq` with `__atomic_store_n(..., __ATOMIC_RELEASE)` after writing all 13 feature fields (src/mutator_m3_0.c:305).
- C reads `action_seq` with `__atomic_load_n(..., __ATOMIC_ACQUIRE)` in a spin loop with a 100 µs `nanosleep` between polls (src/mutator_m3_0.c:308-320, SPIN_NS = 100000 at line 82).
- Python busy-polls `state_seq` with a 100 µs `time.sleep(0.0001)` (scripts/rl_server.py:105-109), then writes `action` followed by `action_seq` (scripts/models/common.py:266-268).
- No ring buffer or queue was ever added on top of this; the protocol is unchanged from the original release-store/acquire-load pattern (docs/experiment_3_full_report.md §4.2).

**Mutator-side state**

`my_mutator_t` in `src/mutator_m3_0.c:120-142` carries a 65,536-byte `cumulative_map` (max-merged trace_bits across all executions), a 1000-entry `edge_ring` for coverage-velocity computation, and an EMA of per-execution wall time seeded from `clock_gettime(CLOCK_MONOTONIC)`.

**Deltas from the original two-process AFL++-plus-Python-DQN design**

- Contextual bandit agent added alongside DQN as a second algorithm (scripts/models/common.py:189-259).
- `--algorithm {dqn,bandit}` flag added to the unified RL server (scripts/rl_server.py:49-51).
- `_skip` variants added that keep the same SHM and mutator but train the DQN every N steps instead of every step (`--train-freq`) to cut the ~1050 µs train_step from the hot loop (CHANGES_SKIP_MODELS.md §Approach).
- Milestone checkpointing was added (`--milestones 500000,1000000,...`) for long multi-benchmark runs (scripts/rl_server.py:46-48, docs/experiment_2_multi_benchmark_10m.md §Infrastructure).
- A telemetry-only mutator (`src/mutator_telemetry.c`, no SHM to Python) was added for the differential-fuzzing data collection campaign; it runs under `AFL_CUSTOM_MUTATOR_ONLY=1` with uniform random action selection and logs CSV + cumulative-bitmap snapshots (docs/experiment_3_verification_and_next_steps.md §3).

## 2. Model variants — authoritative list

All models share the same 47-action discrete action space (scripts/models/common.py:9-33, enforced by `assert ACTION_SIZE == 47`). Action indices map to AFL++ mutation primitives: 0–5 bit/byte flips, 6–15 deterministic arithmetic, 16–20 interesting values, 21–40 havoc single-ops, 41–44 dictionary ops, 45 CUSTOM_MUTATOR (focused multi-op), 46 HAVOC (stacked random) (src/mutator_m3_0.c:376-549, docs/experiment_3_full_report.md §4.3).

**Standard models**

| ID    | State dim | State features | Hidden layers | SHM | Mutator source |
|-------|:---------:|----------------|---------------|-----|----------------|
| M0_0  | 3  | `[coverage_n, new_edges_n, crashes_n]` | [128,128,64]   | 128 B | src/mutator_m0_0.c |
| M1_0  | 12 | coverage + crashes + edge-stability distribution stats (en/dis mean, std, max, nonzero, stability) over all 65536 edges | [128,128,64] | 256 B | src/mutator_m1_0.c |
| M1_1  | 13 | M1_0 features normalized over visited edges + visited-edge count | [128,128,64] | 256 B | src/mutator_m1_1.c |
| M1_2  | 64 | M1_1 + input-buffer features: buf_len, entropy, printable_ratio, histogram[16], first_32_bytes | [256,256,128] | 512 B | src/mutator_m1_2.c |
| M2    | 97 | Per-mutator trace-bit magnitudes (47 enabled + 47 disabled averages + 3 base) | [256,256,128] | 1024 B | src/mutator_m2.c |
| M3_0  | 13 | Differential-informed: total_edges_n, cold/hot/warm/cool_edges_n, edge_entropy, hit_mean, hit_std, corpus_size_n, crashes_n, new_edges_n, avg_exec_time_n, coverage_velocity | [128,128,64] | 128 B | src/mutator_m3_0.c |

Sources: README.md table §Models; docs/experiment_2_multi_benchmark_10m.md §Models Tested; experiments/differential/analysis/m3_0_feature_spec.json; scripts/models/m3_0.py lines 13-81.

**Skip / ablation variants**

Registered in `scripts/models/__init__.py`:

```
MODEL_IDS = ["m0_0", "m1_0", "m1_1", "m1_2", "m2", "m3_0",
             "m0_0_skip", "m1_0_skip", "m1_1_skip", "m1_2_skip", "m2_skip"]
```

Skip variants share the same state/SHM/mutator as their parent but train every 4 steps instead of every step (`--train-freq 4`); action selection still runs every step. They exist to cut the ~1050 µs per-step train_step cost (CHANGES_SKIP_MODELS.md). No `m3_0_skip` entry exists.

**Algorithm variants on M3_0**

M3_0 has two algorithm variants over the same state vector (scripts/rl_server.py:49-51, 77-80):

- M3_0 DQN — Double DQN with target network, replay buffer, ε-greedy (scripts/models/common.py:95-168).
- M3_0 Bandit — Neural contextual bandit with Thompson sampling: two heads for per-action mean and log-variance, loss = negative log-likelihood of observed reward (scripts/models/common.py:171-259). No replay buffer, no discount factor, no temporal credit assignment.

**How many distinct models have been trained and evaluated?**

- Trained and evaluated on jsoncpp at 500K steps, 5 eval runs (Experiment 1, 2026-03-05): 8 models — M0_0, M1_0, M1_1, M2 plus their `_skip` counterparts (docs/experiment_1_jsoncpp_500k.md §Overview).
- Trained and evaluated on 6 benchmarks at 10M steps, 1 eval run (Experiment 2, March 2026): 3 models — M1_0, M1_1, M1_2 (docs/experiment_2_multi_benchmark_10m.md §Overview).
- Trained and evaluated on libxml2 xml005_buggy and xml017_buggy at 500K steps, 5 eval runs (Experiment 3, 2026-04-04): 3 variants — M3_0 DQN, M3_0 Bandit, M1_0 (the "comparison" baseline) (docs/experiment_3_full_report.md §10.1).

**"N of M models collapsed" enumeration**

From Experiment 1 (docs/experiment_1_jsoncpp_500k.md §Eval Action Degeneration), 6 of 8 models fully collapsed to a single action at eval. Per-model collapsed action:

| Model       | Dominant action at eval | % |
|-------------|-------------------------|---|
| M0_0        | #36 HAVOC_MUT_ARITH32BE            | 100.0% |
| M1_0        | #15 DET_ARITH_SUB_FOUR_BIG         | 53.7% (not collapsed) |
| M1_1        | #41 DICTIONARY_USER_EXTRAS_OVER    | 100.0% |
| M2          | #42 DICTIONARY_USER_EXTRAS_INSERT  | 100.0% |
| M0_0_skip   | #2  DET_FLIP_FOUR_BITS             | 100.0% |
| M1_0_skip   | #45 CUSTOM_MUTATOR                 | 41.8% (not collapsed) |
| M1_1_skip   | #11 DET_ARITH_SUB_TWO_BIG          | 86.7% |
| M2_skip     | #45 CUSTOM_MUTATOR                 | 100.0% |

In Experiment 2 (docs/experiment_2_multi_benchmark_10m.md §Per-Benchmark Analysis, jsoncpp), all three models collapsed on jsoncpp: M1_0→#1, M1_1→#18, M1_2→#35. Collapse actions varied per benchmark and are not fully tabulated across all 6 targets in the docs.

In Experiment 3 (verified directly from `experiments/differential/results/eval_*`), aggregated over 5 eval runs per variant:

| Variant       | Target         | Top actions (count of 25,000 logged steps) |
|---------------|----------------|---------------------------------------------|
| M3_0 DQN      | xml005_buggy   | #40 HAVOC_MUT_FLIP8 (9,641), #45 CUSTOM_MUTATOR (6,699), #15 DET_ARITH_SUB_FOUR_BIG (4,752) — diverse |
| M3_0 DQN      | xml017_buggy   | #40 (11,007), #45 (5,666), #15 (4,576) — diverse |
| M3_0 Bandit   | xml005_buggy   | #10 DET_ARITH_SUB_TWO_LE (25,000, **100% collapse**) |
| M3_0 Bandit   | xml017_buggy   | #10 (25,000, **100% collapse**) |
| M1_0 (compare)| xml005_buggy   | #1 DET_FLIP_TWO_BITS (23,255), #29 HAVOC_MUT_ARITH16BE_ (1,745) — 93.0% to #1 |
| M1_0 (compare)| xml017_buggy   | #1 (23,262), #29 (1,738) — 93.0% to #1 |

## 3. Current training configuration

**Training step budget per run**

- Default in the server: `DEFAULT_TRAIN_STEPS = 500_000` (scripts/models/common.py:51).
- Experiment 1 used 500K train + 500K eval, 5 eval runs per model, with plateau early-stopping firing around 350K–362K (docs/experiment_1_jsoncpp_500k.md §Training Results).
- Experiment 2 used 10M train + 10M eval, 1 eval run per model, `--no-plateau`, with milestone snapshots at 500K/1M/2M/10M (docs/experiment_2_multi_benchmark_10m.md §Shared Hyperparameters).
- Experiment 3 (current) uses 500K train + 500K eval, 5 eval runs, `--no-plateau` (scripts/run_m3_0_experiment.sh:21-23, 77; docs/experiment_3_full_report.md §10.1).

**Reward function (verbatim from code)**

`scripts/models/common.py:55-58`:

```python
def compute_reward(cov, pcov, cr, pcr):
    ct = float(cov - pcov)
    xt = (math.log1p(cr) - math.log1p(pcr)) * 1000.0
    return ct + xt - STEP_COST, {"coverage_term": ct, "crash_term": xt}
```

`STEP_COST = 0.0` (scripts/models/common.py:48). So the reward is `(coverage - prev_coverage) + 1000 * (log1p(crashes) - log1p(prev_crashes))`. One new edge contributes +1.0; each crash contributes a log-scaled bonus multiplied by 1000. The README states the same formula (README.md §Reward Function).

`[VERIFY]` docs/experiment_3_full_report.md §9.3 says the reward is `new_edges + 10 * crashes`, and presentation.md §Slide 6 says `(new_edges_found) + (10 × new_crashes_found)`. These do not match the code in `scripts/models/common.py`. Conflicting files: `scripts/models/common.py:55-58` (code) vs `docs/experiment_3_full_report.md` §9.3 and `docs/presentation.md` §Slide 6 (docs). The README and the running server use the log-scaled formula above.

**Epsilon schedule, replay buffer, target sync**

From `scripts/models/common.py`:

- `EPSILON_START = 1.0`, `EPSILON_MIN = 0.05` (lines 41-42).
- Epsilon decays linearly over `decay_steps = int(train_steps * 0.6)` (scripts/rl_server.py:80 passes `decay_steps=int(args.train_steps * 0.6)`; scripts/models/common.py:120-122 linearly interpolates from EPSILON_START to EPSILON_MIN over that budget). At 500K steps this means ε hits the floor at 300K, so the last 200K of training is near-greedy with 5% exploration.
- `REPLAY_SIZE = 100_000` transitions (line 39).
- `TARGET_SYNC = 1000` steps — target network is hard-copied from online network every 1000 gradient steps (lines 40, 148-150).
- `BATCH_SIZE = 128`, `GAMMA = 0.99`, `LEARNING_RATE = 1e-4` (Adam with weight_decay=1e-5), `ENTROPY_COEF = 0.01`, `GRAD_CLIP = 10.0` (lines 36-44, 105-106, 142-146).
- During eval, `ε = 0.0` (scripts/models/common.py:108 sets `self.epsilon = 0.0 if eval_mode else EPSILON_START`). Note: docs/experiment_3_full_report.md §10.3 says "ε = 0.01 for near-greedy action selection"; the code sets it to 0.0. `[VERIFY]` — conflict between docs/experiment_3_full_report.md §10.3 and scripts/models/common.py:108. Code-path behaviour is fully greedy at eval.

**Fixes status**

- **Double-normalization fix**: No applied-or-not status in the repo. `[VERIFY]` — no file in `docs/` or `src/` uses the term "double-normalization"; the current M3_0 splits normalization explicitly between C and Python (entropy/hit_mean/hit_std/exec_time/velocity normalized in C; total/cold/hot/warm/cool/corpus/crashes/new_edges normalized in Python, each field normalized exactly once, docs/experiment_3_verification_and_next_steps.md §7 "Normalization Strategy"). If a previous model had double-normalization and the question refers to it, I cannot locate a commit or file in this repo confirming the fix; flagged for user confirmation.
- **Potential-based reward shaping**: NOT applied. The reward function is still `Δcoverage + 1000·Δlog1p(crashes)` (scripts/models/common.py:55-58). Entropy bonus in the loss is present (`loss = td_loss - ENTROPY_COEF * entropy`, scripts/models/common.py:144) but that is a policy entropy bonus on the loss, not reward shaping.
- **Training budget increase from 50K to 500K–1M steps**: 500K is the current default (`DEFAULT_TRAIN_STEPS = 500_000`, scripts/models/common.py:51). The 10M budget used in Experiment 2 was a superset, used once and then rolled back to 500K for Experiment 3 (docs/experiment_3_full_report.md §10.1; scripts/run_m3_0_experiment.sh:21). The original 50K default is gone from the code.

## 4. Experimental results — current numbers only

Most recent experiment run timestamp: 2026-04-04, approximately 10:42 to 13:27 local time, from directory mtimes under `experiments/differential/results/` (listed in `experiments/differential/results/m3_0_dqn/`, `m3_0_bandit/`, `m1_0_compare/`, and `eval_xml005_buggy/`, `eval_xml017_buggy/`).

All current numbers are from Experiment 3 (docs/experiment_3_full_report.md §11). Experiment 1 and 2 numbers are historical; they are not the current authoritative numbers.

**xml005_buggy — in-distribution (AFL ground truth `edges_found` from `fuzzer_stats`, 5 runs, 500K eval steps)**

| Variant          | edges (mean) | std   | min   | max   | throughput (exec/s mean) | crashes |
|------------------|-------------:|------:|------:|------:|-------------------------:|--------:|
| M3_0 DQN         | 3,934.8      | 20.5  | 3,899 | 3,948 | 3,297                    | 0       |
| M3_0 Bandit      | 3,619.4      | 18.7  | 3,599 | 3,641 | 3,531                    | 0       |
| M1_0 (compare)   | 3,551.6      | 54.8  | 3,489 | 3,626 | 2,904                    | 0       |
| Baseline AFL++   | 4,249.8      | 125.5 | 4,030 | 4,324 | 12,006                   | 0       |

(Aggregated from `experiments/differential/results/eval_xml005_buggy/*/run_*/plots/*/fuzzer_stats_eval.txt` and `experiments/differential/results/eval_xml005_buggy/baseline/run_*/afl_out/default/fuzzer_stats`.)

**xml017_buggy — transfer (AFL ground truth, 5 runs, 500K eval steps)**

| Variant          | edges (mean) | std   | min   | max   | throughput (exec/s mean) | crashes |
|------------------|-------------:|------:|------:|------:|-------------------------:|--------:|
| M3_0 DQN         | 3,822.8      | 82.9  | 3,718 | 3,905 | 3,370                    | 0       |
| M3_0 Bandit      | 3,587.2      | 10.7  | 3,571 | 3,597 | 3,907                    | 0       |
| M1_0 (compare)   | 3,469.6      | 12.4  | 3,463 | 3,490 | 3,225                    | 0       |
| Baseline AFL++   | 4,174.0      | 173.2 | 3,973 | 4,314 | 14,114                   | 0       |

(From `experiments/differential/results/eval_xml017_buggy/*/run_*/...`.)

Caveat marked in docs/experiment_3_full_report.md §11 and docs/presentation/presenter_guide.md §Slide 22: the RL-CSV `coverage` column for xml017 M3_0 DQN reported a mean of 3,899 ± 15, inflated by a `count_coverage` buffer overread bug that has since been fixed in `src/mutator_m3_0.c:158-177` (now uses `afl->fsrv.map_size` instead of `afl->total_bitmap_size`; see docs/static_analysis.md BUG 1 status "Fix"). The AFL-ground-truth numbers above (3,822.8 ± 82.9) are authoritative. The same report also notes that runs 2 and 3 for xml017 M3_0 DQN were truncated to ~120 s instead of 180 s due to CPU scheduling contention, which explains the elevated std.

**Action entropy at eval / majority-action percentage**

From raw action-column aggregation over all 5 runs × 500K steps (CSVs sampled every 100 steps → 25,000 action samples per variant per target):

| Variant       | Target        | Top-1 action       | Top-1 %  | Top-3 actions (counts) | Effective distinct actions (>1% usage) |
|---------------|---------------|--------------------|---------:|------------------------|---------------------------------------:|
| M3_0 DQN      | xml005_buggy  | #40 HAVOC_MUT_FLIP8 | 38.6%    | #40, #45, #15          | 5 |
| M3_0 DQN      | xml017_buggy  | #40                 | 44.0%    | #40, #45, #15          | 5 |
| M3_0 Bandit   | xml005_buggy  | #10 DET_ARITH_SUB_TWO_LE | 100.0% | #10 only              | 1 (fully collapsed) |
| M3_0 Bandit   | xml017_buggy  | #10                 | 100.0% | #10 only                | 1 (fully collapsed) |
| M1_0          | xml005_buggy  | #1 DET_FLIP_TWO_BITS | 93.0%   | #1, #29                 | 2 |
| M1_0          | xml017_buggy  | #1                  | 93.1%  | #1, #29                  | 2 |

(Derived by `awk` aggregation over `experiments/differential/results/eval_*/*/run_*/plots/*/rl_metrics_*_eval.csv`.)

Exact action entropy in bits is not precomputed in any repo artifact. `[VERIFY]` — if the interviewer asks for an entropy number, compute from the distributions above; for M3_0 Bandit entropy = 0 bits; for M1_0, H ≈ 0.37 bits; for M3_0 DQN, H is clearly higher (multi-modal distribution with ~5 non-trivial actions).

## 5. Root cause analysis — as currently documented

**Throughput bottleneck**

Per-step overhead breakdown (docs/experiment_2_multi_benchmark_10m.md §Where the Time Goes, also docs/presentation.md §Slide 23):

| Component                                           | Time |
|-----------------------------------------------------|-----:|
| Read coverage map from SHM / virgin_bits (64 KB)     | ~50 µs |
| Compute state features (edge stats, entropy, etc.)   | ~100 µs |
| Write state to SHM, read action back                 | ~10 µs |
| Wait for RL server + DQN forward pass                | ~200 µs |
| Apply selected mutation                              | ~1 µs |
| **Total with RL**                                    | **~360 µs/step** |
| **Total without RL (baseline AFL++)**                | ~20–50 µs/step |

The server-side breakdown from `CHANGES_SKIP_MODELS.md §Problem Statement` is more pessimistic and finer-grained: full RL server loop ≈ 1,290 µs/step, dominated by `agent.train_step()` at ~1,050 µs (≈81%) and action selection at ~50 µs; the remaining ~190 µs is SHM read + state build + reward + SHM write. This is why the `_skip` variants exist (train_freq=4 amortizes train_step across 4 steps → ~355 µs/step). Across 6 FuzzBench targets, RL averaged ~2,000 steps/s while plain AFL++ averaged 19K–64K execs/s, a 93% throughput reduction (docs/experiment_2_multi_benchmark_10m.md §Throughput Comparison).

**Policy collapse — current written explanation**

From docs/experiment_3_full_report.md §12.3 and docs/presentation/presenter_guide.md §Slide 26:

1. Coverage saturates after ~50–100K executions → reward becomes sparse.
2. Reward = new_edges + crash log delta; most steps yield 0.
3. Small Q-value differences compound; target network soft-updates reinforce the argmax action.
4. ε-greedy decays to 0.05 during training and to 0.0/0.01 at eval (the code uses 0.0; docs claim 0.01 — see §3 `[VERIFY]`).
5. The RL agent fully replaces AFL++'s power schedules, splice, and deterministic/havoc staging; once argmax locks, diversity collapses.

**Which models resisted collapse, and collapsed-action per model**

- Experiment 1 (jsoncpp, 500K eval): M1_0 (53.7% on #15 DET_ARITH_SUB_FOUR_BIG) and M1_0_skip (41.8% on #45 CUSTOM_MUTATOR) resisted; all other 6 fully collapsed — see §2 table (docs/experiment_1_jsoncpp_500k.md §Eval Action Degeneration).
- Experiment 2 (6 benchmarks, 10M eval, 1 run): all three models (M1_0, M1_1, M1_2) fully collapsed on jsoncpp; collapse actions varied per benchmark and are not exhaustively tabulated (docs/experiment_2_multi_benchmark_10m.md §jsoncpp notes M1_0→#1, M1_1→#18, M1_2→#35).
- Experiment 3 (libxml2, 500K eval, 5 runs): M3_0 DQN resisted collapse (top-1 ≈ 40%, ≥5 non-trivial actions), M3_0 Bandit fully collapsed to #10 DET_ARITH_SUB_TWO_LE on both targets, M1_0 nearly collapsed at 93% on #1 DET_FLIP_TWO_BITS (verified directly from eval CSVs, §4 above). docs/experiment_3_full_report.md §12.3 claims M3_0 DQN converged to action #10 — that is contradicted by the actual CSV data in the repo. `[VERIFY]` — conflict between docs/experiment_3_full_report.md §12.3 and `experiments/differential/results/eval_xml005_buggy/m3_0_dqn/run_*/plots/m3_0/rl_metrics_m3_0_eval.csv`. The CSV data shows top-1 = #40, not #10.

## 6. M* contextual bandit — status

**Status: implemented and evaluated.**

- Implementation: `ContextualBanditAgent` in scripts/models/common.py:171-259.
  - Architecture: `BanditNet` (common.py:171-186) is a two-head network. Trunk is the same [128, 128, 64] MLP with ReLU as the DQN; two linear heads output per-action `mean` (47-d) and `logvar` (47-d).
  - Action selection: Thompson sampling — `std = exp(0.5 * logvar); sample = mean + std * N(0,1); argmax(sample)` (common.py:210-218). In eval mode, action is `argmax(mean)`.
  - Training: negative log-likelihood of observed scalar reward under `N(mean_a, exp(logvar_a))`: `nll = 0.5 * (logvar_a + (r - mean_a)² / exp(logvar_a))`. No replay buffer, no discount, trains on the single most recent `(s, a, r)` pending transition (common.py:220-243).
  - Exploration field `self.epsilon = EPSILON_MIN = 0.05` is set but never read by `select_action` — documented as cosmetic in docs/static_analysis.md BUG 5.

- Evaluation: docs/experiment_3_full_report.md §11.1 and data in `experiments/differential/results/eval_*/m3_0_bandit/run_*/`. On xml005_buggy the bandit reached 3,619 ± 19 edges (mean of 5 runs, AFL ground truth); on xml017_buggy 3,587 ± 11. Both targets saw 100% action collapse to #10 DET_ARITH_SUB_TWO_LE. DQN beat Bandit by ≈ 8.7% (xml005) and ≈ 6.6% (xml017) at the same feature set.

- Selection rationale and ablation note are spelled out in `experiments/differential/analysis/m3_0_feature_spec.json` ("algorithm_note": "DQN preferred over contextual bandit … Bandit should be tested as an ablation"). The ablation has been run.

## 7. Magma differential fuzzing — status

**Status: implemented and evaluated.**

Two CVEs from the Magma benchmark's libxml2 entries have been selected and built as buggy/fixed pairs (docs/experiment_3_verification_and_next_steps.md §2):

| ID      | CVE          | Class                | libxml2 buggy tag / commit | libxml2 fixed tag / commit | Location |
|---------|--------------|----------------------|----------------------------|----------------------------|----------|
| xml005  | CVE-2017-5130 | Integer overflow    | v2.9.4 / bdec2183f34b37ee89ae1d330c6ad2bb4d76605f | v2.9.5 / 2960178fe8f9fe690b7f8c1c49093ff54bb56934 | `xmlmemory.c:xmlMemStrdupLoc()` |
| xml017  | CVE-2016-1762 | Heap buffer overread | v2.9.3 / 6657afe83a38278f124ace71dc85f60420beb2d5 | v2.9.4 / bdec2183f34b37ee89ae1d330c6ad2bb4d76605f | `parserInternals.c:xmlNextChar()` |

Harness, seeds, and dictionary are byte-identical copies from FuzzBench's `libxml2_xml` benchmark (harness SHA256 `bd91b6d126d6cd7215cf2752657eaa08268c75d0344f4a415ef374b75d951510`, dict SHA256 `1a6c8d151a20c505a4ab2cd1be7e7616baafe0c23426ab289835836aac14665a`, 38 seed files from `libxml2/test/*.xml`). Binaries compiled with `afl-clang-fast++`, static-linked against the respective `libxml2.a`, ASAN via `AFL_USE_ASAN=1`, portability via compiler flags `-include cstdint -include cstddef` (docs/experiment_3_verification_and_next_steps.md §2; experiments/differential/build/build_libxml2_targets.sh).

**Baseline telemetry collected on buggy/fixed pairs**

Campaign ran 2026-04-04 for 20.9/24.0 hours over 24 parallel runs on 15 cores (12 instrumented telemetry + 12 vanilla baseline), experiments/differential/telemetry/STATUS.md. All 24 runs reported `status: done`.

Saturated coverage at the end of the 9.7 h telemetry runs (docs/m3_0_feature_derivation.md §2.3, experiments/differential/analysis/summary.md):

| Target         | Final edges (mean ± std of 3 seeds) | ~Total execs | Crashes (3 seeds) |
|----------------|-------------------------------------|-------------:|------------------:|
| xml005_buggy   | 5,371 ± 25.7                        | ~41.5 M       | [1, 0, 277] = 278 |
| xml005_fixed   | 5,165 ± 9.4                         | ~37.9 M       | [0, 0, 0] = 0     |
| xml017_buggy   | 5,784 ± 31.6                        | ~41.5 M       | [43, 6, 0] = 49*  |
| xml017_fixed   | 5,488 ± 41.2                        | ~41.0 M       | [34, 0, 8] = 42*  |

\* Note: `experiments/differential/telemetry/STATUS.md` lists xml017_buggy crashes as [43, 6, 0] = 49 total and xml017_fixed as [34, 0, 8] = 42. `docs/m3_0_feature_derivation.md §2.2` reports [81, 16, 0] = 97 and [74, 0, 13] = 87. `[VERIFY]` — conflict between STATUS.md (campaign runtime log) and m3_0_feature_derivation.md (analysis doc). The STATUS.md values likely reflect the actual `saved_crashes` counter; the derivation doc may be aggregating `unique_crashes` from a different snapshot.

Only 3 of the 12 baseline (vanilla AFL++, no custom mutator) runs captured meaningful execs; the other 9 report `312 execs` — consistent with being queued behind the telemetry jobs and never starting before the 24 h deadline (STATUS.md Baseline Runs table; docs/experiment_3_verification_and_next_steps.md §Baseline Data Gap). The M3_0 training/eval experiment runs its own baselines for comparison (see §4).

**Preliminary results (after M3_0 was implemented and trained)**

See §4. On the in-distribution target (xml005_buggy), M3_0 DQN improves over M1_0 by ~10.8% (3,934.8 vs 3,551.6 AFL ground-truth edges). On transfer (xml017_buggy), by ~10.2% (3,822.8 vs 3,469.6). Baseline AFL++ still beats all RL variants by 7.3–15.1%. Neither buggy target produced `saved_crashes > 0` in the 180-second, 500K-step eval window, despite the telemetry campaign producing 278 crashes on xml005_buggy at 9.7 hours — the eval window is too short for rare-path bugs.

## 8. Known limitations and open problems

Documented in `docs/static_analysis.md` (static review of M3_0 code), `docs/experiment_3_full_report.md §12.5`, and the presenter guide.

**Bugs known in current M3_0 implementation (docs/static_analysis.md):**

| # | Severity | File | Summary | Status |
|---|----------|------|---------|--------|
| 1 | CRITICAL | src/mutator_m3_0.c:157 | `count_coverage` used `afl->total_bitmap_size` (accumulator) instead of `afl->fsrv.map_size` — buffer overread of up to ~6.8 MB | **Fixed** (now reads `afl->fsrv.map_size`, src/mutator_m3_0.c:164) |
| 2 | HIGH | src/mutator_m3_0.c:501 | HAVOC stack depth was 2^(1+4)=32 via AFL default; telemetry mutator used 2^(1+9)=512, so training-data semantics differed from inference | **Fixed** (src/mutator_m3_0.c:62-63 now `#define HAVOC_STACK_POW2 9`) |
| 3 | INFO | src/mutator_m3_0.c:209,214 | Edge with hit=128 is classified "warm" by heat counts but "hot-range" (bin 7) by entropy; telemetry mutator has the same boundary so it's consistent | Leave |
| 4 | LOW | src/mutator_m3_0.c:224 | `cold_edges` uses `sz` not `MAP_SIZE`; combined with #1 when `fsrv.map_size < MAP_SIZE` this under-counts cold | Leave per docs (note: src/mutator_m3_0.c:231 now uses `MAP_SIZE - nonzero`, so effectively fixed) |
| 5 | LOW | scripts/models/common.py:203 | Bandit `epsilon` field is cosmetic, never read by `select_action` | Leave |
| 6 | MEDIUM | src/mutator_m3_0.c:157,186 | `total_edges` counts from `afl->virgin_bits` but heat/entropy count from `cumulative_map` — different edge sets | Leave (design) |
| 7 | LOW | scripts/run_m3_0_experiment.sh:39 | No existence guard on the dict file (subsequent silent-skip in run_model.sh) | **Fixed** per script line 42: `[[ -f "$DICT" ]] || exit 1` |
| 8 | COSMETIC | scripts/run_model.sh:193 | `AFL_AUTORESUME=1` set on an eval dir that's been wiped | Leave |

**What is actively being worked on this week/month**

The presenter guide (dated implicitly to 2026-04-04, matching the experiment commits) and docs/experiment_3_verification_and_next_steps.md list "next steps" that are design/documentation only, not coded in the repo:

- Entropy regularization on the reward (λ · H(π) added to `R`) — proposed but not implemented.
- Action diversity constraints / raising eval ε above 0.0 — proposed, not coded.
- Hybrid scheduling: RL modulates AFL++ power schedules instead of replacing the mutator — proposed only.
- Longer-horizon credit assignment / PPO / intrinsic motivation — proposed.
- Multi-target training and ≥10 seeds per telemetry run for statistical significance — proposed.

No uncommitted work-in-progress files beyond `docs/presentation/`, `docs/static_analysis.md`, and the differential `experiments/differential/results/` and `telemetry/` trees (git status at session start).

**Threats to validity (from the paper-adjacent docs and presenter guide)**

- n=3 seeds per telemetry target → Mann-Whitney U cannot reach Bonferroni-corrected significance (min p=0.05 with n=3; α/65 ≈ 0.00077). Ranking uses A12 effect size instead of p-values (docs/experiment_3_verification_and_next_steps.md §4.1, docs/presentation/presenter_guide.md §Slide 13).
- xml017 crash signal is noisy — both buggy (v2.9.3) and fixed (v2.9.4) versions crash under ASAN, from different bugs. Coverage-only differential is clean; crash-only differential is not (docs/m3_0_feature_derivation.md §2.2; summary.md).
- Wall-time vs step-count comparison: RL eval is controlled at 500K steps (~180 s); baseline runs at 500K execs (~35–50 s). Baseline would execute ~2.3× more mutations in the same wall-time budget and its gap would widen (docs/presenter_guide.md §Slide 20, §Q&A "Is the comparison with baseline AFL++ fair?").
- `total_edges` and heat counts measure different edge sets (virgin_bits vs cumulative_map), see static_analysis.md BUG 6.
- Baseline telemetry gap: only 3/12 vanilla baseline runs captured full data in the 24 h differential campaign (STATUS.md).
- libpng is uninformative: only 1 seed, AFL++ exhausts the queue immediately; all RL models got 4 edges (docs/experiment_2_multi_benchmark_10m.md §libpng).

## 9. Related work cited

No BibTeX file exists in the repo (no `*.bib` found). `docs/` and `presenter_guide.md` mention only the following papers/systems by name, without per-paper citations or quoted contributions:

- **NEUZZ** — mentioned in docs/presentation/presenter_guide.md §Slide 26 Q&A as prior RL-for-fuzzing work that observed similar policy-collapse issues. No other description in repo.
- **RLFuzz** — mentioned in the same Q&A block as another RL-guided fuzzer experiencing policy collapse. No other description in repo.
- **MOPT** — mentioned in the same Q&A block as prior work on mutation scheduling that observed similar stagnation. No other description in repo.
- **AlphaGo** — referenced as an analogy for hybrid scheduling (neural network guiding a classical search engine), docs/presentation/presenter_guide.md §Q&A "What would a hybrid scheduling approach look like".
- **AFL++** — the fuzzer being extended (repository: github.com/AFLplusplus/AFLplusplus; built from source at `~/packages/AFLplusplus`, docs/differential_fuzzing_experiment_plan.md §Phase 0). No paper citation in repo.
- **FuzzBench** — Google's fuzzer benchmark suite, used for standardized harnesses, seeds, and dictionaries (`~/fuzzbench`, docs/BUILDING_BENCHMARKS.md). No paper citation.
- **Magma** — mentioned as the source of the CVE selection for Experiment 3 (docs/differential_fuzzing_experiment_plan.md §Context and §Phase 1; docs/experiment_3_full_report.md §6.1). No paper citation.

There is no `.docx` or `paper_draft.md` file in this repo; only the presentation deck at `docs/presentation/experiment_3_review.pptx` (binary, not parsed) and the Markdown presenter guide at `docs/presentation/presenter_guide.md`. If a fuller related-work section exists, it is not checked into this repo. `[VERIFY]` — the task prompt asked for "Most recent paper draft (.docx or .md)"; the closest artifact is `docs/experiment_3_full_report.md`, which does not have a formal references section.

## 10. File inventory

Absolute paths (all resolved, verified as present):

- **Most recent paper draft (MD proxy, no .docx exists)**: `/home/shreyasganesh/projects/rl-fuzzer/docs/experiment_3_full_report.md`
- **Most recent presentation (.pptx)**: `/home/shreyasganesh/projects/rl-fuzzer/docs/presentation/experiment_3_review.pptx` (binary, accompanied by `/home/shreyasganesh/projects/rl-fuzzer/docs/presentation/presenter_guide.md` and `/home/shreyasganesh/projects/rl-fuzzer/docs/presentation/build_slides.py`)
- **Most recent results CSV (a representative sample; there are 5 × 3 = 15 per target)**: `/home/shreyasganesh/projects/rl-fuzzer/experiments/differential/results/eval_xml005_buggy/m3_0_dqn/run_1/plots/m3_0/rl_metrics_m3_0_eval.csv` (timestamp 2026-04-04)
- **BibTeX file**: None in repo. `[VERIFY]`
- **Main mutator.c**: `/home/shreyasganesh/projects/rl-fuzzer/src/mutator_m3_0.c` (current model). All six model mutators: `src/mutator_m0_0.c`, `mutator_m1_0.c`, `mutator_m1_1.c`, `mutator_m1_2.c`, `mutator_m2.c`, `mutator_m3_0.c`. Telemetry-only collector: `src/mutator_telemetry.c`. Target program source (for self-contained builds): `src/target.c`.
- **Main Python agent file**: `/home/shreyasganesh/projects/rl-fuzzer/scripts/models/common.py` (contains `DQNAgent` and `ContextualBanditAgent`). Per-model config modules in `scripts/models/m0_0.py`, `m1_0.py`, `m1_1.py`, `m1_2.py`, `m2.py`, `m3_0.py`.
- **Main training loop file**: `/home/shreyasganesh/projects/rl-fuzzer/scripts/rl_server.py`. Shell entry point that orchestrates mutator compilation + server + AFL++: `/home/shreyasganesh/projects/rl-fuzzer/scripts/run_model.sh`. Experiment 3 driver: `/home/shreyasganesh/projects/rl-fuzzer/scripts/run_m3_0_experiment.sh`.

---

Prepared from direct inspection of the repo at `/home/shreyasganesh/projects/rl-fuzzer` on 2026-04-24.
