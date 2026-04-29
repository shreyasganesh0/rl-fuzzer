# Static Analysis: Expectations vs Reality

**Scope**: `src/mutator_m3_0.c`, `scripts/models/m3_0.py`, `scripts/rl_server.py`,
           `scripts/models/common.py`, `scripts/run_model.sh`, `scripts/run_m3_0_experiment.sh`

---

## BUG 1 — `count_coverage` reads past `virgin_bits` buffer  [CRITICAL]

**File**: `src/mutator_m3_0.c:157`

**Expectation**: `sz` = size of `virgin_bits` = `MAP_SIZE` (65536 bytes). The loop
iterates over exactly the coverage bitmap.

**Reality**: `afl->total_bitmap_size` is an **accumulator** — AFL++ adds
`q->bitmap_size` (= non-zero byte count of that input's trace, via `count_bytes`)
for every queue entry that is calibrated or rescored:
```c
afl->total_bitmap_size += q->bitmap_size;   // afl-fuzz-run.c:699
```
With ~2278 corpus entries each covering ~3000 edges,
`total_bitmap_size ≈ 6.8 million` — two orders of magnitude larger than
`virgin_bits` (65536 bytes). The loop silently reads ~6.8 MB past the end of
`virgin_bits`.

**Why it hasn't crashed**: By luck of heap layout, `virgin_bits`, `virgin_tmout`,
and `virgin_crash` are all `memset(0xFF, MAP_SIZE)`. The overread traverses these
buffers (which are also 0xFF) and produces an approximately correct edge count,
because 0xFF means "never hit" and is skipped by the counter.

**Correct field**: `afl->fsrv.map_size` — the actual trace bitmap size for this target.

---

## BUG 2 — `HAVOC_STACK_POW2` mismatch between telemetry and RL mutator  [HIGH]

**File**: `src/mutator_m3_0.c:501`

**Expectation**: HAVOC (action 46) in the RL mutator applies the same number of
stacked operations as HAVOC did during the telemetry collection campaigns used to
derive features.

**Reality**:
- `mutator_telemetry.c` (used for feature derivation):
  ```c
  #undef  HAVOC_STACK_POW2
  #define HAVOC_STACK_POW2  9    // stack up to 2^(1+9) = 512 ops
  ```
- `mutator_m3_0.c` (no explicit definition — inherits from AFL++ `config.h`):
  ```c
  #define HAVOC_STACK_POW2 4U   // stack up to 2^(1+4) = 32 ops
  ```

HAVOC in the RL mutator applies at most **32 operations**; HAVOC in the telemetry
mutator applied at most **512 operations**. The RL agent learned Q-values for
action 46 under the telemetry semantics, but at runtime it executes a much milder
variant.

---

## BUG 3 — Hot/entropy boundary off-by-one (consistent between telemetry and m3_0 — leave as-is)

**File**: `src/mutator_m3_0.c:209,214`

**Note**: This is identical in both `mutator_telemetry.c` and `mutator_m3_0.c`, so
the feature derivation and the RL implementation agree. Listed for awareness.

An edge with cumulative hit count `v = 128` is classified as:
- **Warm** in the heat count (`v > 128` is the hot threshold → 128 is not hot)
- **Bin 7 (hot-range)** in the entropy histogram (`v >= 128` → 128 goes to bins[7])

The comment says warm = `8 <= v <= 128`, hot = `v > 128`. The entropy bins treat
128 as part of the hot-range bucket. This is internally consistent between
derivation and inference so no fix is needed, but it means `hot_edges` and the
entropy's hot-range bin can disagree for edges at exactly 128.

---

## BUG 4 — `cold_edges` uses scan size `sz` instead of `MAP_SIZE`  [LOW]

**File**: `src/mutator_m3_0.c:224`

**Expectation**: `cold_edges = MAP_SIZE - nonzero` (unscanned edges = unexplored
code).

**Reality**:
```c
uint32_t sz = m->afl->total_bitmap_size;
if (sz > MAP_SIZE) sz = MAP_SIZE;          // clips, but sz may be < MAP_SIZE
...
uint32_t cold_edges = (sz > nonzero) ? (sz - nonzero) : 0;
```
If `afl->total_bitmap_size` were ever *less than* MAP_SIZE (unlikely in practice
but possible during initialization), `cold_edges` would be under-counted.
Combined with Bug 1 being fixed (using `fsrv.map_size`), this should also use
`fsrv.map_size` directly.

---

## BUG 5 — Bandit `epsilon` is cosmetic — never influences action selection  [LOW]

**File**: `scripts/models/common.py:203`

**Expectation**: The bandit agent's `epsilon` field tracks exploration, analogous
to the DQN.

**Reality**:
```python
# ContextualBanditAgent.__init__
self.epsilon = 0.0 if eval_mode else EPSILON_MIN  # = 0.05

# select_action — epsilon is NEVER READ:
std = (logvar * 0.5).exp()
sample = mean + std * torch.randn_like(std)   # Thompson sampling, not ε-greedy
return int(sample.argmax(1).item())
```
The bandit uses Thompson sampling, which ignores `epsilon` entirely. The field is
set to `EPSILON_MIN = 0.05` at init and never changes. It is logged to CSV and
passed to the plateau detector, creating the false impression that the bandit has
5% exploration throughout training. The plateau detector checks:
```python
if eps > EPSILON_MIN + 0.01: return False    # 0.05 > 0.06 → False → never blocks
```
This means the plateau detector for bandit training will never be inhibited by the
epsilon condition. With `--no-plateau` in the experiment, this has no effect, but
it's misleading.

---

## BUG 6 — `count_coverage` vs heat features use different coverage sources  [MEDIUM]

**File**: `src/mutator_m3_0.c:157–170, 186`

**Expectation**: `total_edges`, `cold_edges`, and the heat distribution all count
the same set of edges.

**Reality**:
- `total_edges` = `count_coverage(afl)` → reads `afl->virgin_bits` (AFL++'s
  cumulative "ever hit" bitmap, updated by AFL++ for all corpus calibrations)
- Heat distribution (hot/warm/cool/cold), entropy, mean, std → computed from
  `m->cumulative_map`, which is a max-merge of `afl->fsrv.trace_bits` only from
  executions that went through `afl_custom_fuzz`

AFL++ updates `virgin_bits` during queue calibration and trimming phases, which
happen between calls to `afl_custom_fuzz`. So `total_edges` can be higher than
`nonzero` in `cumulative_map`, and `cold_edges = MAP_SIZE - nonzero` can be
larger than `MAP_SIZE - total_edges`. The state vector has a small internal
inconsistency: dimension 0 (`total_edges`) and dimension 1 (`cold_edges`) measure
different sets of edges.

This is minor in practice — the divergence is typically a few hundred edges at
most — but is worth documenting.

---

## BUG 7 — No dictionary existence guard in experiment script  [LOW]

**File**: `scripts/run_m3_0_experiment.sh:39`

**Expectation**: The experiment fails early with a clear error if the dictionary
file is missing.

**Reality**: The script defines `DICT` but never checks whether the file exists:
```bash
DICT="$EXP_DIR/dictionaries/libxml2.dict"
# No [[ -f "$DICT" ]] || exit 1 guard
```
`run_model.sh` silently skips the `-x` flag if `$DICT` is not found
(`[[ -f "$DICT" ]] && DICT_FLAG="-x $DICT"`), so RL runs would silently proceed
without the dictionary. The baseline uses `$DICT` directly in the `afl-fuzz`
invocation, where a missing file would cause AFL++ to error but that error is
redirected to `afl.log`, so it's also silent.

---

## BUG 8 — `AFL_AUTORESUME=1` set unnecessarily for eval runs  [COSMETIC]

**File**: `scripts/run_model.sh:193`

**Expectation**: AFL++ starts fresh for each eval run.

**Reality**: `AFL_AUTORESUME=1` is set, but the eval output dir is wiped
immediately before (`rm -rf "$AFL_EVAL_DIR"`). The flag has no effect since
there is nothing to resume. No functional impact, but misleading.

---

## Summary Table

| # | Severity | File | Description | Status |
|---|----------|------|-------------|--------|
| 1 | CRITICAL | `mutator_m3_0.c:157` | `count_coverage` uses `total_bitmap_size` (accumulator) — buffer overread | **Fix** |
| 2 | HIGH | `mutator_m3_0.c:501` | `HAVOC_STACK_POW2` inherits AFL value (4, max 32 ops) vs telemetry (9, max 512 ops) | **Fix** |
| 3 | INFO | `mutator_m3_0.c:209,214` | Hot boundary v=128 inconsistent between heat count and entropy bin — but consistent with telemetry | Leave |
| 4 | LOW | `mutator_m3_0.c:224` | `cold_edges` uses `sz` not `MAP_SIZE` directly | **Fix with #1** |
| 5 | LOW | `common.py:203` | Bandit epsilon is cosmetic, never drives action selection | Leave |
| 6 | MEDIUM | `mutator_m3_0.c:157,186` | `total_edges` and heat features count different edge sets | Leave (design) |
| 7 | LOW | `run_m3_0_experiment.sh:39` | No dict existence guard | **Fix** |
| 8 | COSMETIC | `run_model.sh:193` | `AFL_AUTORESUME=1` on wiped eval dir | Leave |
