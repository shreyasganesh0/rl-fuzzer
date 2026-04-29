# Interview Walkthrough Plan — xAI SWE Specialist (Human Data / Coding Evals)

**Project shown**: RL-guided AFL++ fuzzer (`rl-fuzzer/`) — specifically the M3_0 differential-informed variant.
**Target window**: 5–10 minutes of code, 15–30 minute total slot.
**Strategy**: Open 3 files. Never scroll. Every visible line is defensible.

---

## Thesis sentence (memorize)

> "The question in RL-guided fuzzing is *what should the agent observe*. Prior models in this repo guessed. M3_0 measures — features are derived from differential analysis of buggy vs. fixed libxml2, not from intuition."

This is the frame that gives every code region below a reason to exist.

---

## Tier 1 — Must-show (3 files, ~4 min if only these are shown)

### T1-A — `src/mutator_m3_0.c:288-320` — SHM release/acquire protocol
**Defensibility: 5/5**

**Why worth showing**: The IPC backbone. A senior engineer will instantly look for memory-ordering discipline in a C↔Python SHM scheme; you have it, correctly, with named atomics. This is the piece most likely to earn unprompted respect.

**Walkthrough script** (what to say while this is on screen):
- "The C mutator writes all 13 feature fields first, then publishes a monotonic `state_seq` with `__ATOMIC_RELEASE`. Python reads `state_seq` first — only accepting data that's paired with a sequence number it hasn't seen."
- "Release pairs with acquire on the action side. Python writes the action, then bumps `action_seq`; the C spin-wait `__atomic_load_n(..., __ATOMIC_ACQUIRE)` is what guarantees the action value is visible before the load returns."
- "Sentinel-counter design means we don't need a mutex. AFL++ runs the forkserver at tens of thousands of execs per second — a futex wake per step would dominate."
- "Spin loop uses `nanosleep(100µs)` so we don't burn a core; 100µs was chosen because it's below AFL++'s typical per-exec time, so we never add measurable latency."
- "Zero allocations on this hot path — the SHM is mmap'd once in `afl_custom_init`."

**Follow-up questions to expect**:
1. "What if `state_seq` wraps?" → u32, 4B rollover. Comparison is `!=`, not `<`, so wrap is safe as long as only one write is in flight. In practice we'd hit the heat death of the universe first at ~100k execs/sec.
2. "Why not a semaphore or eventfd?" → Per-step syscall cost. Spin + `nanosleep` keeps the fast path in userspace; we only enter the kernel on the sleep.
3. "Is the volatile qualifier on `u32_at` doing work given the atomic builtins?" → Honest answer: it's belt-and-suspenders. `__atomic_*` provides the ordering; `volatile` just prevents LTO from caching the load across iterations. I'd remove one or the other in cleanup.

**Weak spots to flag proactively**:
- `SPIN_NS = 100000` is a magic number (line 82) — be ready to defend "0.1 ms ≈ median AFL++ exec latency on this target."
- The spin loop has no timeout/abort — if the Python side dies, the C side blocks forever. Known limitation; process tree is managed by `run_model.sh` which kills both together.

---

### T1-B — `scripts/models/common.py:128-151` — DQN train_step (Double DQN + entropy regularization)
**Defensibility: 5/5**

**Why worth showing**: 24 lines that encode three deliberate, non-default choices: (1) Double DQN decoupling, (2) entropy regularization on the online Q-distribution, (3) grad clipping. Each one has a "rejected alternative" story tied directly to an observed failure mode (policy collapse) documented in the experiment report.

**Walkthrough script**:
- "Standard Double DQN: online net picks `argmax` on the next state; target net evaluates that argmax. Decouples selection from valuation, which reduces the classic DQN overestimation bias."
- "The unusual bit is line 144: `loss = td_loss - ENTROPY_COEF * entropy`. Softmax over the online Q's, then a Shannon entropy bonus. This is a direct response to what I found in evaluation — the DQN collapses to near-deterministic action selection (mostly action 10, `ARITH_SUB2LE`), which hurts coverage because fuzzing rewards mutation diversity."
- "Grad clip at 10.0 because the reward includes a `1000 × log1p(crashes)` term — one crash in a batch can produce a huge TD error."
- "`weight_decay=1e-5` on Adam. Small 22K-parameter net, sparse reward, long training — slight L2 prevents the net from memorizing transient reward spikes."
- "Target sync every 1000 training steps — hard copy, not soft update. Simpler, one less hyperparameter."

**Follow-up questions to expect**:
1. "Did entropy regularization actually fix policy collapse?" → Honest: it reduced it but didn't eliminate it. In the final eval the agent still favored a handful of arithmetic mutations. Real fix is probably hybrid scheduling (RL modulates AFL++'s scheduler rather than replacing it).
2. "Why `argmax` on the online net for bootstrap, not target net?" → That's the Double DQN correction. If you bootstrap with the target net's own argmax you get the overestimation you're trying to avoid.
3. "Why MSE and not Huber?" → Honest answer: I started with MSE and never A/B'd Huber. On a small net with gradient clipping the difference is usually marginal, but Huber would be my first cleanup.

**Weak spots to flag proactively**:
- `ENTROPY_COEF = 0.01` (line 43) — magic number, never swept. Defensible as "small bonus, doesn't dominate TD loss," but an interviewer could push.
- The replay buffer sampling (line 130) calls `random.sample` with no prioritization. Prioritized replay is the obvious next step.

---

### T1-C — `scripts/models/m3_0.py:65-81` — The 13-dim differential state vector
**Defensibility: 4/5**

**Why worth showing**: This is the payoff. The SHM protocol and training loop exist to transport and consume *these* 13 numbers. Each is annotated. Every feature was ranked by a Vargha-Delaney A12 effect size between buggy and fixed libxml2 runs — the feature selection has an empirical justification behind it.

**Walkthrough script**:
- "This is the state vector the DQN sees. Thirteen features, chosen by measuring which ones diverged between buggy and fixed libxml2 runs — ranked by Vargha-Delaney A12, not statistical significance, because fuzzing campaigns are expensive and n=3 per group can't clear Bonferroni."
- "Features 2-4 are edge heat ratios — fraction of reached edges that are hot (>128 hits cumulative), warm (8-128), cool (1-7). A vulnerability creates new reachable paths which shifts the heat distribution; this was the highest-A12 class of features after raw coverage."
- "Feature 5 is Shannon entropy over 8 power-of-2 hit-count bins — a compact summary of 'how peaked vs. flat is the execution profile.'"
- "Features 10 and 12 — `new_edges` and `coverage_velocity` — have A12 near 0.5, meaning they don't distinguish buggy from fixed. I kept them anyway because they're the RL signal: `new_edges` is the immediate reward, `velocity` tells the agent whether it's in early exploration or late saturation. Not a bug discriminator, but a necessary learning signal."
- "Normalization is split across the boundary: features with tight ranges (entropy, hit-count stats) are normed in C before the SHM write to avoid transmitting unbounded floats; features needing `log1p` are normed in Python."

**Follow-up questions to expect**:
1. "Features 0 and 1 look redundant — `cold_edges = MAP_SIZE - total_edges`." → Correct, they're linearly dependent after normalization. Not fatal for a 22K-param net — at worst one weight goes to zero. But if I were minimizing the state I'd drop `cold_edges`.
2. "How did you validate the features actually generalize?" → Trained on CVE-2017-5130 (xml005), evaluated on CVE-2016-1762 (xml017 — different CVE class, same codebase). M3_0 still beat M1_0 by 9.3% on transfer vs. 10.3% in-distribution. Small transfer gap.
3. "What happens if another target has totally different coverage scale?" → These are normalized ratios (hot/total, entropy ÷ log₂(8)), not raw counts, so the absolute coverage scale drops out. The brittleness is `log1p(corpus_size)/log1p(10000)` — if corpus blows past 10K the feature saturates. For xml that ceiling is fine.

**Weak spots to flag proactively**:
- Line 76: `math.log1p(d["corpus_size"]) / math.log1p(10000)` — the 10000 is a hand-picked ceiling.
- Line 77: `math.log1p(d["crashes"]) / math.log1p(1000)` — same.
- Line 78: `min(new_edges, 100.0) / 100.0` — hard clip at 100, chosen because we rarely see more than a few new edges per step. Flag this before they do.

---

## Tier 2 — Show if asked (4-5 locations)

### T2-A — `src/mutator_m3_0.c:197-229` — Single-pass edge classification + entropy bins
**Why**: If asked "how are the features actually computed?" — one O(MAP_SIZE) loop does heat classification, sum/sum-sq accumulation, and the 8-bin entropy histogram together. No separate passes. 33 lines.

**Say**: "MAP_SIZE is 65,536 and this runs on every fuzz iteration, so I fused the passes. Edge heat thresholds (>128 hot, ≥8 warm) were picked from the telemetry data — 128 is where the cumulative-max saturates for hot loop edges."

**Follow-ups**: "Why power-of-2 bins?" (dynamic range of AFL++'s bucketed hit-count — a byte where 1/2/4/8/16/32/64/128 are the bucket boundaries, so the bins match AFL++'s own discretization). "Max entropy 3.0?" (log₂(8), normalize to [0,1]).

**Weak spot**: The inner `if/else if` chain (216-228) could be a lookup table. Micro-opt, not a bug.

### T2-B — `src/mutator_m3_0.c:158-177` — `count_coverage()` with defensive OOB comment
**Why**: The comment itself (lines 160-163) is the feature. It documents a real AFL++ footgun — `afl->total_bitmap_size` grows to millions and reading `virgin_bits` that far is OOB. Shows I've actually debugged into AFL++ internals.

**Say**: "Every RL mutator in this repo had a version of this bug at some point. `total_bitmap_size` sounds like what you want — it's not. It's the accumulator of per-queue-entry coverage sizes. The right source is `fsrv.map_size`."

**Follow-ups**: "How did you find this?" (ASAN crash in the RL server's state-read path; traced it back to the C side reading past the bitmap). "Why the 8-byte chunked scan?" (common case — whole 8-byte chunk is 0xFF, skip the inner loop. About 3x faster on late-saturation runs where most of virgin_bits is set.)

### T2-C — `scripts/models/common.py:171-218` — BanditNet (two-head) + Thompson sampling
**Why**: The alternative algorithm. Shows I didn't just reach for DQN — I implemented a contextual bandit with a two-head network (μ, log σ²) and Thompson sampling, ran the full eval, and *it lost*. That negative result is more credible than a single-algorithm story.

**Say**: "Two heads: per-action mean and log-variance. Thompson sampling is `sample ~ N(μ, exp(logvar)) per action, pick argmax`. It has the same agent interface as DQNAgent so `rl_server.py` can swap them with one arg. The bandit underperformed DQN by 7.7% on both targets — I think because it has no temporal credit assignment, and in fuzzing the causal distance from mutation to new-edge can be many steps."

**Follow-ups**: "Why log-variance, not variance directly?" (unconstrained output; variance always positive after `exp`). "Why NLL loss, not MSE?" (NLL penalizes overconfidence — if the bandit is certain about a bad action the loss blows up appropriately).

**Weak spot**: Line 208 `_pending = None` — the bandit holds exactly one pending transition. If `train_step` isn't called before the next `remember`, the pending one is dropped. In practice rl_server.py calls them in lockstep so it's fine, but it's fragile.

### T2-D — `scripts/rl_server.py:100-122` — The inference/train loop core
**Why**: The Python mirror of T1-A. 23 lines showing the acquire-side of the SHM protocol and the (remember → train_step → select_action → write) ordering.

**Say**: "Busy-polls `state_seq` with a 100µs sleep — same rationale as the C spin-wait, different direction. One `remember` per step, one `train_step` per step modulated by `--train-freq`. Action is always written within microseconds of reading the state."

**Follow-up**: "What if the Python side falls behind?" (AFL++ is bounded by its own exec speed; if Python is slower, AFL++ blocks on `shm_wait_action` — the whole system degrades to Python's rate, no dropped states.)

---

## Tier 3 — Mention verbally, do NOT open

- **Telemetry mutator** (`src/mutator_telemetry.c`): "I also built a telemetry-only mutator that runs with uniform random action selection and no SHM IPC — that's how the 12 baseline runs that produced the differential dataset were collected."
- **The analysis pipeline** (`scripts/analysis/differential_analysis.py`): "The Mann-Whitney U + A12 analysis over 40K bitmap snapshots is in its own pipeline — it ranked the 13 features out of a candidate pool of ~20."
- **Four prior models** (M0_0, M1_0, M1_1, M2 in `scripts/models/`): "M3_0 is the 5th model variant; the earlier four each tried a different hand-designed state representation and plateaued around 3,500–3,600 edges. M3_0 reaches 3,957."
- **Results**: "Vanilla AFL++ still beats all RL variants by 7% — primarily policy collapse. M3_0 DQN beat M1_0 (the prior-best RL) by 10.3% in-distribution and 9.3% on transfer."

---

## Narrative arcs

### 10-minute arc
1. **(0:00–0:15) Problem, verbal**. "RL-guided fuzzers need a state representation. Prior attempts in this project used hand-designed features and plateaued. I ran a differential analysis — fuzzed buggy vs. fixed libxml2 with identical infrastructure and measured which features diverge."
2. **(0:15–3:30) Open T1-A** (`mutator_m3_0.c:288-320`). SHM release/acquire protocol. Sets up "this is a real C↔Python system" framing.
3. **(3:30–6:30) Open T1-B** (`common.py:128-151`). DQN train_step + entropy regularization. The algorithmic choice story + the policy-collapse-driven addition.
4. **(6:30–9:00) Open T1-C** (`m3_0.py:65-81`). The 13-dim state vector. "These specific features were ranked by A12 effect size from the differential analysis."
5. **(9:00–9:30) Close, verbal**. "What I'd change: hybrid scheduler instead of argmax-over-47. RL modulates AFL++'s native scheduling, keeps mutation diversity. Results showed vanilla AFL++ still wins by 7% because of policy collapse — that's the next experiment."
6. **(9:30–end) Q&A**.

### 5-minute arc (cut T1-C + skip the bandit context in T1-B)
1. **(0:00–0:15) Problem, verbal** (same).
2. **(0:15–2:30) Open T1-A**. SHM protocol. Cut the wrap-around discussion if short on time.
3. **(2:30–4:30) Open T1-B**. DQN train_step. Lead with the entropy-regularization point; skip the grad clip and weight decay asides.
4. **(4:30–5:00) Close**. "Features came from differential analysis — can walk through `m3_0.py` if we have time. Main limitation is policy collapse despite entropy reg."

---

## Screenshot prep (fallback if screen share fails)

Take these the morning of. Store in `~/interview_backup/`. Both PNG (for slides/messaging) and plain text (for pasting).

| # | What to capture | Command | Illustrates | Format |
|---|---|---|---|---|
| 1 | T1-A region | `sed -n '288,320p' src/mutator_m3_0.c` | SHM release/acquire — the headline code | PNG with syntax highlighting (e.g. VS Code screenshot) |
| 2 | T1-B region | `sed -n '128,151p' scripts/models/common.py` | DQN train_step w/ entropy reg | PNG with syntax highlighting |
| 3 | T1-C region | `sed -n '65,81p' scripts/models/m3_0.py` | 13-dim state vector with per-feature comments | PNG with syntax highlighting |
| 4 | Results table from report | Screenshot of `docs/experiment_3_full_report.md` §11.1 table rendered in a markdown viewer | M3_0 DQN 3,957 vs M1_0 3,586 vs baseline 4,250 — the quantitative punchline | PNG |
| 5 | SHM layout diagram | Screenshot of §4.2 of `docs/experiment_3_full_report.md` (the offset/size/type/field table) | The IPC contract in one picture — grounds T1-A visually | PNG |

Keep a plain-text copy of #1–#3 too; a terminal paste works even on video-feed-only calls.

---

## Before-interview cleanup checklist

Decide per-item whether to fix or flag. None are bugs blocking the demo, but you should know they exist before an interviewer points at them.

### `src/mutator_m3_0.c`
- **Line 62-63**: `#undef HAVOC_STACK_POW2` followed by `#define HAVOC_STACK_POW2 9`. The comment explains why (match telemetry mutator for training-data consistency). Defensible, but flag proactively before they ask "why are you redefining AFL++ internals?"
- **Line 82**: `#define SPIN_NS 100000` — magic number. Comment `/* 0.1 ms */` is present; fine.
- **Line 83-84**: `VELOCITY_WINDOW 1000`, `EMA_ALPHA 0.01f` — no comment. Defensible ("1000-step window ≈ 10-second wall-clock; EMA α=0.01 gives ~100-step effective window") but add one-line rationale if you have time.
- **Line 613**: `uint32_t sz = m->afl->total_bitmap_size;` in the first-call path. **Actual inconsistency**: the comment at line 160-163 explicitly warns against using `total_bitmap_size` as an OOB hazard, yet this line uses it for the seed pass. On first call `total_bitmap_size` is usually 0 or small, so the cumulative_map seeding is often a no-op. Either fix this to `fsrv.map_size` or be ready to explain: "first-call seed path — if `total_bitmap_size` is undersized we just pick it up on the next pass through `shm_push_state`." Honest answer is "I missed updating this site; good catch."
- **Lines 371-374**: `#define NEED / RND / RPOS / RDELTA` in the middle of a function. Defensible ("scoped to this switch"), but a purist reviewer might prefer `static inline` helpers.
- **Line 400 & others**: Unaligned `*(uint16_t*)(mb+pos)` and `*(uint32_t*)(mb+pos)` writes. On x86_64 this is fine; on strict-alignment architectures this UBs. Flag if asked about portability.

### `scripts/models/common.py`
- **Line 43**: `ENTROPY_COEF = 0.01` — magic, never swept. Flag proactively.
- **Line 44**: `GRAD_CLIP = 10.0` — magic. Defensible (reward has `1000 × log1p(crashes)` spike).
- **Lines 45-48**: `MAX_COVERAGE`, `MAX_NEW_EDGES`, `MAX_CRASHES`, `STEP_COST`. Used by older models (m0_0, m1_*, m2); M3_0 overrides with its own constants inside `m3_0.py`. Not dead code — but not documented as such.
- **Line 208**: `self._pending = None` comment — fine but a one-liner on why only one transition is held would preempt a question.

### `scripts/models/m3_0.py`
- **Lines 76-78**: three magic ceilings (10000, 1000, 100) inside `build_state`. Add per-line comments if time permits: `# corpus rarely exceeds 10K queue items on xml targets`.

### `scripts/rl_server.py`
- **Line 84**: `shm_write_action(shm, 46, 1, ...)` — hardcoded 46. It's the HAVOC action (AFL++ default-safe fallback). Add a named constant `DEFAULT_ACTION = 46` if cleaning up.
- **Line 109**: `time.sleep(0.0001)` — 100µs literal; pair with T1-A's `SPIN_NS` as a named constant (`POLL_SLEEP_S = 1e-4`).

### Repo-level
- `ls /home/shreyasganesh/` visible in any shared-screen terminal would reveal the home path. Before the call, `cd` into the repo and run commands with relative paths, or use a clean prompt (`PS1='\W$ '`).
- `muofuzz_summary_for_interview_prep.md` at repo root — don't share this screen. Rename to `.local-notes.md` or move out of the repo.
- `.venv/` in git status — harmless but visible; add to `.gitignore` if not already.

---

## Required output acknowledgement

**Absolute path**: `/home/shreyasganesh/projects/rl-fuzzer/interview_walkthrough_plan.md`

**Tier 1 selections with defensibility scores**:
1. `src/mutator_m3_0.c:288-320` — SHM release/acquire protocol — **5/5**
2. `scripts/models/common.py:128-151` — DQN train_step + entropy regularization — **5/5**
3. `scripts/models/m3_0.py:65-81` — 13-dim differential state vector — **4/5**
