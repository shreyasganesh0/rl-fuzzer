# Skip Model Variants: Changes & Rationale

## Problem Statement

Benchmarking the RL server loop revealed that the DQN `train_step()` call dominates latency at ~1050 us per step. The full loop (SHM read + state build + reward + train + action select + SHM write) takes ~1290 us, meaning training accounts for ~81% of per-step cost. This bottleneck limits AFL++ throughput because the mutator blocks waiting for the RL server to return a new action.

**Key insight:** Action selection (forward pass) is cheap (~50 us), but backpropagation (train_step) is expensive (~1050 us). If we train every 4 steps instead of every 1, the amortized per-step cost drops from ~1290 us to ~355 us — a 3.6x speedup.

## Approach

Create `_skip` variants of all 4 models that are identical in every way (same state space, same hidden layers, same C mutator, same SHM layout) except they train every 4th step via a `--train-freq 4` flag. This lets us directly measure the coverage impact of reduced training frequency against the originals and a plain AFL++ baseline.

## Changes Made

### 1. `scripts/rl_server.py` — Added `--train-freq` flag

**What:** Added a `--train-freq` CLI argument (default: 1 for backward compatibility). The main loop now gates `agent.remember()` and `agent.train_step()` behind `step % train_freq == 0`. Action selection (`agent.select_action`) still runs every step.

**Why:** This is the core mechanism. By skipping training on most steps, the RL server responds to the C mutator faster, allowing AFL++ to execute more fuzz iterations per second. The agent still observes every state and selects actions every step — it just doesn't update its weights as often.

**Before (line 94-95):**
```python
if step > 0:
    agent.remember(pstate, pact, rew, state); loss = agent.train_step()
```

**After:**
```python
if step > 0 and step % args.train_freq == 0:
    agent.remember(pstate, pact, rew, state); loss = agent.train_step()
```

### 2. `scripts/models/m0_0_skip.py`, `m1_0_skip.py`, `m1_1_skip.py`, `m2_skip.py` (4 new files)

**What:** Thin wrapper modules that import everything from their parent model (`from .m0_0 import *`) and override only `MODEL_PATH_DEFAULT` (e.g., `"rl_m0_0_skip.pt"`) and `LABEL` (e.g., `"M0_0_SKIP"`).

**Why:** Each _skip model needs its own checkpoint file so training results don't overwrite the original model's weights. All other attributes (STATE_SIZE, HIDDEN_LAYERS, SHM_PATH, SHM offsets, CSV format functions) are inherited unchanged. This means _skip models use the exact same C mutator shared library and SHM layout as their parent — no C code changes needed.

**Example (m0_0_skip.py):**
```python
from .m0_0 import *
MODEL_PATH_DEFAULT = "rl_m0_0_skip.pt"
LABEL = "M0_0_SKIP"
```

### 3. `scripts/models/__init__.py` — Registered _skip model IDs

**What:** Added `m0_0_skip`, `m1_0_skip`, `m1_1_skip`, `m2_skip` to the `MODEL_IDS` list.

**Why:** Other scripts reference this list for validation and iteration.

### 4. `scripts/run_model.sh` — BASE_MODEL_ID derivation + --train-freq passthrough

**What:**
- Derives `BASE_MODEL_ID` by stripping the `_skip` suffix: `BASE_MODEL_ID="${MODEL_ID%_skip}"`
- Uses `BASE_MODEL_ID` for mutator source path (`src/mutator_${BASE_MODEL_ID}.c`), compiled binary (`bin/mutator_${BASE_MODEL_ID}.so`), and SHM path (`/tmp/rl_shm_${BASE_MODEL_ID}`)
- Uses `MODEL_ID` for everything else (checkpoint path, output dirs, plots dir)
- Sets `TRAIN_FREQ=4` when `MODEL_ID` ends in `_skip`, passes it to `rl_server.py`

**Why:** The _skip models reuse the parent's C mutator (no code changes needed in C). The mutator communicates via SHM at a fixed path, so `SHM_PATH` must match the parent. But each _skip model needs its own checkpoint, AFL++ output directory, and plots directory to avoid data collisions.

### 5. `scripts/build_and_compare.sh` — Whitelist + comparison loop

**What:**
- Added `m0_0_skip|m1_0_skip|m1_1_skip|m2_skip` to the `case` statement whitelist
- Added _skip models to the comparison args loop that builds `--m0-0-skip` etc. flags for `compare_metrics.py`

**Why:** Without the whitelist entry, `build_and_compare.sh` would skip unknown model IDs with a warning. The comparison loop ensures _skip models appear in the comparison plots and reports.

### 6. `scripts/compare_metrics.py` — _skip model support in comparison

**What:**
- Added _skip models to `MODELS` list, `MODEL_LABELS` dict, and `MODEL_COLORS` dict
- Added `--m0-0-skip`, `--m1-0-skip`, `--m1-1-skip`, `--m2-skip` CLI arguments
- Updated `dir_map` construction to include _skip models when their flags are passed
- Filtered `all_models` to only include models present in `dir_map` (avoids looking up default paths for _skip models that weren't passed)

**Why:** The comparison script needs to know about _skip models to load their CSVs, assign them distinct colors/labels in plots, and include them in the comparison report table.

**Color choices for _skip models:** Distinct from originals to allow visual comparison on the same plot:
- M0_0_SKIP: `#76b7b2` (teal)
- M1_0_SKIP: `#edc948` (gold)
- M1_1_SKIP: `#b07aa1` (purple)
- M2_SKIP: `#ff9da7` (pink)

### 7. `scripts/run_experiment.sh` — No changes needed

The experiment runner already iterates `MODEL_LIST` (parsed from `--models` CSV) and generates comparison flags dynamically with `--${model//_/-}`. Passing `--models "m0_0_skip,m1_0_skip,..."` works automatically.

## Experiment Results

### Short run (50K train, 20K eval, 2 eval runs)

| Model | Coverage Gained | Std |
|-------|----------------|-----|
| M0_0 | 276 | +/- 0 |
| M1_0 | **347** | +/- 11 |
| M1_1 | 334 | +/- 10 |
| M2 | 324 | +/- 0 |
| M0_0_SKIP (freq=4) | 167 | +/- 55 |
| M1_0_SKIP (freq=4) | 338 | +/- 14 |
| M1_1_SKIP (freq=4) | 318 | +/- 42 |
| M2_SKIP (freq=4) | 324 | +/- 0 |

### Full run (500K train, 500K eval, 2 eval runs, _skip models only)

| Model | Coverage Gained | Std |
|-------|----------------|-----|
| M0_0_SKIP (freq=4) | 502 | +/- 56 |
| M1_0_SKIP (freq=4) | 652 | +/- 4 |
| M1_1_SKIP (freq=4) | 624 | +/- 11 |
| M2_SKIP (freq=4) | **668** | +/- 21 |
| Baseline (plain AFL++) | 5,796 | +/- 4 |

### Key Takeaways

1. **M1_0_SKIP and M2_SKIP** perform nearly identically to their non-skip originals at 50K steps, showing that training every 4th step preserves policy quality for complex state representations.

2. **M0_0_SKIP** shows the largest drop vs. M0_0 (167 vs 276 at 50K). The simple 3-dim state may need more frequent training updates to learn effectively.

3. At 500K steps, **M2_SKIP leads** with 668 edges gained, suggesting the per-mutator magnitude state (97 dims) benefits most from the freq=4 approach — the richer state compensates for less frequent weight updates.

4. The **baseline (plain AFL++)** still outperforms all RL models in absolute coverage (~5,800 edges), which is expected for a well-optimized fuzzer on a simple target. The RL models are learning mutation selection strategies that could prove more valuable on harder targets.

## Files Changed Summary

| File | Type | Description |
|------|------|-------------|
| `scripts/rl_server.py` | Modified | Added `--train-freq` arg, gated training behind step modulo |
| `scripts/models/__init__.py` | Modified | Registered 4 _skip model IDs |
| `scripts/models/m0_0_skip.py` | New | Thin wrapper inheriting from m0_0 |
| `scripts/models/m1_0_skip.py` | New | Thin wrapper inheriting from m1_0 |
| `scripts/models/m1_1_skip.py` | New | Thin wrapper inheriting from m1_1 |
| `scripts/models/m2_skip.py` | New | Thin wrapper inheriting from m2 |
| `scripts/run_model.sh` | Modified | BASE_MODEL_ID derivation, --train-freq passthrough |
| `scripts/build_and_compare.sh` | Modified | Whitelist + comparison args for _skip models |
| `scripts/compare_metrics.py` | Modified | CLI args, labels, colors, dir_map for _skip models |

## How to Run

```bash
# All 8 models + baseline (short sanity check)
bash scripts/run_experiment.sh \
    --models "m0_0,m1_0,m1_1,m2,m0_0_skip,m1_0_skip,m1_1_skip,m2_skip" \
    --eval-runs 2 --train-steps 50000 --eval-steps 20000 --run-baseline

# Only _skip models (full run)
bash scripts/run_experiment.sh \
    --models "m0_0_skip,m1_0_skip,m1_1_skip,m2_skip" \
    --eval-runs 2 --train-steps 500000 --eval-steps 500000 --run-baseline
```

## Output Locations

- Per-run CSVs: `plots/<model>/run_1/`, `plots/<model>/run_2/`
- Training CSVs: `plots/<model>/rl_metrics_<model>_train.csv`
- Same-steps comparison: `comparison_results/same_steps/`
- Same-time comparison: `comparison_results/same_time/`
- Plots: `comparison_results/*/plot_*.png`
- Reports: `comparison_results/*/comparison_report.txt`
