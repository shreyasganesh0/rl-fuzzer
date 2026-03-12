# RL-Guided AFL++ Fuzzer

This is a research fuzzing framework that integrates **AFL++** with **Deep Q-Network (DQN)** agents to learn and optimise mutation strategies at runtime. Instead of applying mutations randomly, It treats mutation selection as a sequential decision problem and learns which of 47 AFL++ mutation primitives is most likely to discover new code coverage for a given execution context.

The project benchmarks **4 DQN model variants** (differing in state representation complexity) against a **plain AFL++ baseline**, producing comparison plots and statistical reports.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Models](#models)
3. [Architecture](#architecture)
4. [Action Space](#action-space)
5. [Reward Function](#reward-function)
6. [IPC: Shared Memory Protocol](#ipc-shared-memory-protocol)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Output Structure](#output-structure)
11. [Metrics and Analysis](#metrics-and-analysis)

---

## System Overview

Each model variant consists of two cooperating processes that communicate through a memory-mapped file:

| Component | Language | Role |
|---|---|---|
| **Custom Mutator** (`src/mutator_m*.c`) | C | AFL++ plugin; collects execution state, writes to SHM, reads action, executes the chosen mutation |
| **RL Server** (`scripts/rl_server.py`) | Python / PyTorch | Hosts the DQN; reads state from SHM, computes actions, runs the training loop, writes CSV metrics |

```
┌──────────────────────────────────────────────────────┐
│  AFL++ Process                                        │
│                                                       │
│  afl_custom_fuzz() — mutator_m*.c                     │
│    1. Collect coverage, edges, crashes from trace_bits │
│    2. Write state to SHM  (RELEASE store)        ─────┤──→  /tmp/rl_shm_<model_id>
│    3. Poll SHM for action (ACQUIRE load)         ←────┤──   (mmap file)
│    4. Apply mutation primitive                        │
└──────────────────────────────────────────────────────┘
                         ↑↓
┌──────────────────────────────────────────────────────┐
│  RL Server — rl_server.py --model-id <id>             │
│    1. Poll SHM for new state  (state_seq sentinel)    │
│    2. Build state vector (model-specific)              │
│    3. DQN forward pass → choose action                 │
│    4. Write action to SHM (action_seq sentinel)        │
│    5. Store (s, a, r, s') → replay buffer → backprop   │
└──────────────────────────────────────────────────────┘
```

---

## Models

All 4 models share the same 47-action DQN architecture and training loop, differing only in **state representation** and **SHM layout**:

| Model | State Dims | SHM Size | DQN Architecture | State Description |
|---|---|---|---|---|
| **M0_0** | 3 | 128 B | 3→128→128→64→47 | Basic: `[coverage_n, new_edges_n, crashes_n]` |
| **M1_0** | 12 | 256 B | 12→128→128→64→47 | Edge stability distribution over all 65536 edges |
| **M1_1** | 13 | 256 B | 13→128→128→64→47 | Edge stability over visited edges only + visit count |
| **M2** | 97 | 1024 B | 97→256→256→128→47 | Per-mutator trace-bit magnitudes (47 enabled + 47 disabled averages) |

Each model's configuration is defined in a self-contained module under `scripts/models/m*.py`, exporting SHM layout constants, `build_state()`, `shm_read()`, and CSV column definitions.

---

## Action Space

The agent selects from 47 discrete mutation primitives that map directly to AFL++'s internal mutator IDs:

- **Deterministic stages** (16): bit flips (1/2/4 bits, 1/2/4 bytes), arithmetic add/sub (8/16/32-bit, LE/BE)
- **Interesting values** (5): boundary constants (8/16/32-bit, LE/BE)
- **Havoc mutations** (18): random bit flips, arithmetic, byte operations
- **Dictionary operations** (4): user/auto extras overwrite/insert
- **Meta** (2): custom mutator, full havoc
- **Total: 47 actions** (enforced by `assert ACTION_SIZE == 47`)

---

## Reward Function

```
reward = (coverage_now - coverage_prev) + (log1p(crashes_now) - log1p(crashes_prev)) * 1000
```

Coverage deltas are measured in raw edge counts (one new edge = +1.0 reward). Crash discovery provides a large bonus through log-scaled crash count deltas. No step cost penalty is applied (`STEP_COST = 0.0`).

---

## IPC: Shared Memory Protocol

Each model uses its own SHM file at `/tmp/rl_shm_<model_id>`. The C mutator writes execution state (coverage, edges, crashes, and model-specific features) and the Python server reads state, computes an action, and writes it back. Synchronisation uses monotonically incrementing sequence numbers with GCC atomic builtins (`__ATOMIC_RELEASE` / `__ATOMIC_ACQUIRE`).

---

## Installation

### Prerequisites

| Dependency | Purpose |
|---|---|
| AFL++ | Fuzzing engine |
| Python 3.8+ | RL server runtime |
| PyTorch | DQN implementation |
| NumPy | Numerical operations |
| Matplotlib | Comparison plots |
| LLVM / Clang | Mutator compilation and AFL++ instrumentation |

```bash
# Create virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib

# Set AFL++ path
export AFL_ROOT=~/packages/AFLplusplus
```

### Build Target

Build the instrumented target binary (jsoncpp-based):

```bash
bash scripts/build_jsoncpp.sh
```

This produces `bin/target` and sets up `inputs/` and `dictionaries/`.

---

## Usage

### Run a Single Model

```bash
# Train + eval for one model
bash scripts/run_model.sh --model-id m0_0 --train-steps 50000 --eval-steps 20000

# Eval only (requires existing checkpoint at bin/rl_m0_0.pt)
bash scripts/run_model.sh --model-id m2 --eval-only --eval-steps 20000
```

Options for `run_model.sh`:

| Flag | Default | Description |
|---|---|---|
| `--model-id ID` | (required) | Model: `m0_0`, `m1_0`, `m1_1`, `m2` |
| `--train-steps N` | 500000 | Training step limit |
| `--eval-steps N` | 50000 | Evaluation step limit |
| `--afl-dir DIR` | `$AFL_ROOT` | AFL++ installation directory |
| `--target PATH` | `bin/target` | Instrumented target binary |
| `--seeds DIR` | `inputs/` | Seed corpus directory |
| `--no-build` | | Skip mutator recompilation |
| `--eval-only` | | Skip training phase |
| `--no-plateau` | | Disable coverage-plateau early stopping |

### Run All Models + Comparison

```bash
# Full benchmark: train + eval all 4 models, then compare
bash scripts/build_and_compare.sh --train-steps 50000 --eval-steps 20000

# Include a plain AFL++ baseline
bash scripts/build_and_compare.sh --train-steps 50000 --eval-steps 20000 --run-baseline

# Compare only (models already ran)
bash scripts/build_and_compare.sh --compare-only
```

### Multi-Run Experiment (Statistical)

```bash
# Train once, eval N times, produce mean +/- std comparison
bash scripts/run_experiment.sh --eval-runs 5 --eval-steps 20000
```

### RL Server Directly (Advanced)

```bash
python3 scripts/rl_server.py --model-id m0_0 --mode train --train-steps 50000
python3 scripts/rl_server.py --model-id m2   --mode eval  --eval-steps 20000 --model bin/rl_m2.pt
```

---

## Project Structure

```
rl-fuzzer/
├── src/
│   ├── mutator_m0_0.c          # AFL++ custom mutator — basic state (128B SHM)
│   ├── mutator_m1_0.c          # AFL++ custom mutator — edge stability (256B SHM)
│   ├── mutator_m1_1.c          # AFL++ custom mutator — visited-edge stability (256B SHM)
│   └── mutator_m2.c            # AFL++ custom mutator — per-action magnitudes (1024B SHM)
├── scripts/
│   ├── rl_server.py             # Unified RL server entry point (--model-id)
│   ├── run_model.sh             # Unified train+eval shell runner (--model-id)
│   ├── models/
│   │   ├── __init__.py          # MODEL_IDS = ["m0_0", "m1_0", "m1_1", "m2"]
│   │   ├── common.py            # Shared: DQN, DQNAgent, ReplayBuffer, PlateauDetector,
│   │   │                        #   compute_reward, ACTION_COLUMNS, hyperparameters
│   │   ├── m0_0.py              # M0_0 config: STATE_SIZE=3, shm_read, build_state
│   │   ├── m1_0.py              # M1_0 config: STATE_SIZE=12
│   │   ├── m1_1.py              # M1_1 config: STATE_SIZE=13
│   │   └── m2.py                # M2 config: STATE_SIZE=97
│   ├── build_and_compare.sh     # Orchestrator: run all models + comparison
│   ├── run_experiment.sh        # Multi-run experiment (train once, eval N times)
│   ├── compare_metrics.py       # 4-way comparison, plots, and report generation
│   └── build_jsoncpp.sh         # One-time target build script
├── inputs/                      # Seed corpus (generated by build_jsoncpp.sh)
├── dictionaries/                # AFL++ dictionaries
└── bin/                         # Build outputs (mutator .so, target, .pt checkpoints)
```

### Adding a New Model

1. Create `scripts/models/m_new.py` implementing the module interface:
   - `STATE_SIZE`, `SHM_SIZE`, `SHM_PATH`, `MODEL_PATH_DEFAULT`, `LABEL`, `HIDDEN_LAYERS`
   - `STATE_SEQ_OFF`, `ACTION_OFF`, `ACTION_SEQ_OFF`
   - `CSV_EXTRA_HEADER`
   - `shm_read(shm, shm_size) -> dict`
   - `build_state(d, train_steps) -> np.ndarray`
   - `zero_state_data() -> dict`
   - `csv_extra_fields(d, args) -> str`
   - `log_extra(d, args) -> str`
   - `exit_summary(d, step, cov, cr, epsilon, tag) -> None`
2. Add the ID to `MODEL_IDS` in `scripts/models/__init__.py`
3. Create a corresponding `src/mutator_m_new.c` with matching SHM layout
4. Run: `bash scripts/run_model.sh --model-id m_new`

---

## Output Structure

```
plots/<model_id>/
  rl_metrics_<model_id>_train.csv    # Training metrics (every 100 steps)
  rl_metrics_<model_id>_eval.csv     # Eval metrics
  fuzzer_stats_train.txt             # AFL++ fuzzer_stats snapshot
  fuzzer_stats_eval.txt

comparison_results/
  comparison_report.txt              # Mean +/- std table (if --multi-run)
  comparison_summary.json
  plot_coverage_eval_steps.png
  plot_coverage_eval_time.png
  plot_coverage_bar_eval.png
  plot_throughput_eval.png
  plot_coverage_per_sec_eval.png
```

---

## Metrics and Analysis

Training and eval metrics are written to CSV every 100 steps. The base columns shared by all models:

| Column | Description |
|---|---|
| `step` | Global step counter |
| `reward` | Reward received |
| `coverage_term` | Coverage delta component of reward |
| `crash_term` | Crash delta component of reward |
| `loss` | DQN TD loss |
| `epsilon` | Current exploration rate |
| `coverage` | AFL++ edge coverage count |
| `crashes` | Total crashes |
| `action` | Action chosen by the agent |
| `elapsed_seconds` | Wall-clock time since start |

Model-specific extra columns:

| Model | Extra Columns |
|---|---|
| M0_0 | (none) |
| M1_0 | `en_mean_n`, `dis_mean_n`, `stability` |
| M1_1 | `num_visited`, `stability` |
| M2 | `mean_avg_en`, `mean_avg_dis`, `top_en_action`, `top_dis_action`, `nonzero_mag_frac` |

Generate comparison plots and reports:

```bash
python3 scripts/compare_metrics.py \
    --m0-0 plots/m0_0 --m1-0 plots/m1_0 --m1-1 plots/m1_1 --m2 plots/m2 \
    --out comparison_results/ --phase eval
```
