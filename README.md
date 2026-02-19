# Neuro-Symbolic Fuzzer with Deep Reinforcement Learning

MuoFuzz is a research fuzzing framework that integrates **AFL++** with a **Deep Q-Network (DQN)** agent to learn and optimise mutation strategies at runtime. Unlike traditional fuzzers that apply mutations randomly or through fixed deterministic schedules, MuoFuzz treats mutation selection as a sequential decision problem and learns which mutation primitive is most likely to discover new code coverage for a given execution context.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Key Design Decisions](#key-design-decisions)
4. [State Vector](#state-vector)
5. [Action Space](#action-space)
6. [Reward Function](#reward-function)
7. [IPC: Shared Memory Protocol](#ipc-shared-memory-protocol)
8. [Edge Probability Tables](#edge-probability-tables)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Project Structure](#project-structure)
12. [Metrics and Analysis](#metrics-and-analysis)

---

## System Overview

MuoFuzz replaces AFL++'s stochastic mutation scheduler with a trained DQN agent that observes the fuzzer's state at each execution step and selects the mutation primitive most likely to produce new coverage. The agent learns through a standard online reinforcement learning loop: it selects an action, observes the coverage delta, receives a reward, and updates its neural network weights accordingly.

The system consists of two cooperating processes:

| Component | Language | Role |
|---|---|---|
| **Custom Mutator** (`src/mutator.c`) | C | AFL++ plugin; collects execution state, communicates with the RL agent, executes the chosen mutation |
| **RL Brain** (`scripts/rl_server.py`) | Python / PyTorch | Hosts the DQN; computes actions from state observations; runs the training loop |

The two processes share state through a **memory-mapped file** (`/tmp/muofuzz_shm`), eliminating socket overhead and avoiding the need for a separate IPC daemon.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  AFL++ Process                                             │
│                                                            │
│   ┌───────────────────────────────────────────┐            │
│   │  afl_custom_fuzz() — mutator.c            │            │
│   │                                           │            │
│   │  1. Read trace_bits  →  find edge_id      │            │
│   │  2. Count coverage, crashes               │            │
│   │  3. Write state to SHM  (RELEASE store)   │────────────┤──→  /tmp/muofuzz_shm
│   │  4. Poll SHM for action (ACQUIRE load)    │←───────────┤──   (128-byte mmap)
│   │  5. Apply mutation primitive              │            │
│   └───────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────┘
                                                      ↑↓
┌─────────────────────────────────────────────────────────────┐
│  RL Brain — rl_server.py                                    │
│                                                             │
│   1. Poll SHM for new state  (state_seq sentinel)           │
│   2. Lookup edge_id in enable/disable probability tables    │
│   3. Build 19-element state vector                          │
│   4. DQN forward pass  →  choose action                     │
│   5. Write action to SHM  (action_seq sentinel)             │
│   6. Store (s, a, r, s') in replay buffer                   │
│   7. Sample mini-batch  →  backprop  →  update weights      │
└─────────────────────────────────────────────────────────────┘
```

### Why `AFL_CUSTOM_MUTATOR_ONLY=1` is required

AFL++ normally runs its own deterministic stages (bit flips, arithmetic, dictionary) before invoking the custom mutator. If those stages are active, any coverage gain they produce is attributed to whatever action the RL agent last chose — a false association. Setting `AFL_CUSTOM_MUTATOR_ONLY=1` ensures all mutations are exclusively delegated to the custom mutator, making the reward signal causally attributable to the agent's decisions.

---

## State Vector

The RL agent observes a 19-dimensional state vector at each fuzzing step:

| Index | Feature | Description |
|---|---|---|
| 0 | `edge_id_norm` | Most recently discovered (or hottest) edge, normalised to [0, 1] over 65,536 possible edges |
| 1 | `coverage_pct` | Fraction of the AFL++ bitmap covered so far |
| 2 | `new_edges_norm` | Number of new edges discovered in this step, clipped at 100 and normalised |
| 3 | `crashes_log` | Total crash count, log-normalised |
| 4–10 | `enable_prob[0..6]` | Per-action probability that applying that mutation to the current edge will **enable** a new edge (from CSV lookup) |
| 11–17 | `disable_prob[0..6]` | Per-action probability that applying that mutation to the current edge will **disable** an existing edge (from CSV lookup) |
| 18 | `prev_action_norm` | The action chosen in the previous step, normalised to [0, 1] |

The enable/disable probability features replace the previous mock static analysis (`constraints.json`). They are derived from empirical data in `edge_enable_prob.csv` and `edge_disable_prob.csv`, which record, for each AFL++ edge ID and each AFL++ mutator type, the observed probability of that mutation transitioning the program into or out of that edge.

### Edge ID Extraction

The current edge is identified by scanning `afl->fsrv.trace_bits` (the bitmap written by the last execution) against `afl->virgin_bits` (the set of ever-seen edges). The mutator uses a two-pass heuristic:

1. **Priority 1 — Newly discovered edge:** Scan from the highest index downward for any edge that is hit in `trace_bits` but was previously unseen in `virgin_bits` (i.e., `virgin_bits[i] == 0xFF`). This is the most informative signal: the current input reached a new path.
2. **Priority 2 — Hottest edge:** If no new edge exists, return the index with the highest hit count in `trace_bits`. This characterises the code path being exercised even when no new coverage is found.

---

## Action Space

The agent operates over 7 discrete mutation primitives:

| Action | Name | Description | Primary Use Case |
|---|---|---|---|
| 0 | Arithmetic Increment | Increments a random byte by one | Bypassing loop counters and off-by-one conditions |
| 1 | Arithmetic Decrement | Decrements a random byte by one | Bypassing loop counters and off-by-one conditions |
| 2 | Interesting 8-bit | Replaces a byte with a boundary value (−128, 0, 127, etc.) | Triggering 8-bit integer overflow/underflow |
| 3 | Interesting 32-bit | Overwrites four bytes with a boundary int32 value | Triggering 32-bit integer overflow and signed/unsigned edge cases |
| 4 | Dictionary Insertion | Inserts or overwrites with a token from the loaded dictionary | Satisfying magic byte and header checks |
| 5 | Delete Bytes | Removes a random subsequence of bytes | Satisfying length/size constraints |
| 6 | Havoc | Stacked random bit flips, byte replacements, and inversions | General exploration of the state space |

---

## Reward Function

At each step the agent receives:

```
reward = (Δcoverage × 100) + (Δcrashes × 10,000) − 0.1
```

The −0.1 term is a small living penalty that discourages the agent from taking actions that produce no new information. An additional +20 bonus is added when new coverage is found *and* the current edge has non-zero enable probabilities in the lookup table — rewarding the agent for exploiting structural knowledge about the program.

---

## IPC: Shared Memory Protocol

The shared memory file at `/tmp/muofuzz_shm` is 128 bytes divided into two regions:

**State region (bytes 0–63) — written by C, read by Python:**

| Offset | Type | Field | Notes |
|---|---|---|---|
| 0 | `uint32_t` | `state_seq` | Sequence sentinel; C **release-stores** this last |
| 4 | `uint32_t` | `edge_id` | Current edge |
| 8 | `uint32_t` | `coverage` | Bitmap coverage count |
| 12 | `uint32_t` | `new_edges` | Delta new edges this step |
| 16 | `uint32_t` | `crashes` | Total crash count |
| 24 | `uint64_t` | `total_execs` | Total executions |

**Action region (bytes 64–127) — written by Python, read by C:**

| Offset | Type | Field | Notes |
|---|---|---|---|
| 64 | `uint32_t` | `action_seq` | Sequence sentinel; Python writes this last |
| 68 | `int32_t` | `action` | Chosen action (0–6) |

**Synchronisation** uses GCC atomic builtins (`__atomic_store_n` / `__atomic_load_n`) with `__ATOMIC_RELEASE` and `__ATOMIC_ACQUIRE` memory orders. This is necessary on AArch64 (ARM64), which has a weakly ordered memory model and does not guarantee that plain stores become visible to other cores in program order. No mutex or semaphore is required: the monotonically incrementing sequence sentinel provides the necessary happens-before relationship between writer and reader.

---

## Edge Probability Tables

Two CSV files provide the per-edge, per-mutator empirical transition probabilities used to populate the state vector:

| File | Content |
|---|---|
| `edge_enable_prob.csv` | `P(edge E becomes active | mutator M is applied)` |
| `edge_disable_prob.csv` | `P(edge E becomes inactive | mutator M is applied)` |

Rows are indexed by integer edge ID. Columns correspond to the 20 AFL++ mutator types (e.g., `DET_ARITH_ADD_ONE`, `HAVOC_MUT_FLIPBIT`). The RL server aggregates the columns most relevant to each of the 7 custom actions using a predefined mapping and computes their mean as the feature value.

These files must be present in the project root directory (same level as `scripts/`) when the RL brain is launched. The shell script will warn if they are missing; the agent will still run but without edge-probability features (all zeros).

---

## Installation

### Prerequisites

| Dependency | Purpose |
|---|---|
| AFL++ | Fuzzing engine |
| Python 3.8+ | RL brain runtime |
| PyTorch | DQN implementation |
| NumPy, Pandas | Numerical operations and CSV loading |
| LLVM / Clang | Compilation and AFL++ instrumentation |

```bash
pip install torch numpy pandas
```

### Environment Setup

```bash
export AFL_ROOT=~/AFLplusplus   # adjust to your AFL++ installation path
```

### Edge Probability CSVs

Copy your edge probability CSV files to the project root:

```bash
cp /path/to/edge_enable_prob.csv  .
cp /path/to/edge_disable_prob.csv .
```

---

## Usage

All build, initialisation, and launch steps are automated by the provided shell script:

```bash
chmod +x scripts/run_muofuzz.sh
./scripts/run_muofuzz.sh
```

The script performs the following steps in order:

1. Removes stale build artefacts, output directories, and the shared memory file.
2. Creates the initial seed input and dictionary.
3. Copies edge probability CSVs to the working directory (if `EDGE_CSV_DIR` is set).
4. Compiles the target binary using `afl-clang-fast`.
5. Compiles the custom mutator as a shared library.
6. Starts the RL brain (`rl_server.py`) in the background and waits 2 seconds for it to initialise the shared memory file.
7. Launches `afl-fuzz` with `AFL_CUSTOM_MUTATOR_ONLY=1` and the compiled mutator library.

### Manual Launch (two terminals)

If you prefer to run components separately:

**Terminal 1 — RL brain:**
```bash
cd <project_root>
python3 scripts/rl_server.py
```

**Terminal 2 — AFL++:**
```bash
export AFL_CUSTOM_MUTATOR_LIBRARY=bin/rl_mutator.dylib
export AFL_CUSTOM_MUTATOR_ONLY=1
afl-fuzz -i inputs -o outputs -x dictionaries/target.dict -- bin/target
```

> **Note:** The RL brain must be started first so that the shared memory file exists before the mutator attempts to map it.

---

## Project Structure

```
muofuzz/
├── src/
│   ├── mutator.c           # AFL++ custom mutator (IPC, edge detection, mutation execution)
│   └── target.c            # Reference vulnerable target for benchmarking
├── scripts/
│   ├── rl_server.py        # RL brain: DQN, replay buffer, state builder, SHM protocol
│   ├── plot_metrics.py     # Training metrics visualisation
│   └── run_muofuzz.sh      # Automated build and launch script
├── inputs/
│   └── seed.txt            # Initial seed corpus
├── dictionaries/
│   └── target.dict         # AFL++ token dictionary
├── edge_enable_prob.csv    # Per-edge enable probability table (must be provided)
├── edge_disable_prob.csv   # Per-edge disable probability table (must be provided)
├── rl_metrics.csv          # Generated at runtime: per-step training log
└── plots/                  # Generated by plot_metrics.py
```

---

## Metrics and Analysis

Training metrics are written to `rl_metrics.csv` every 100 steps:

| Column | Description |
|---|---|
| `step` | Global step counter |
| `reward` | Reward received at this step |
| `loss` | DQN MSE loss for the training batch |
| `epsilon` | Current exploration rate |
| `coverage` | AFL++ bitmap coverage count |
| `crashes` | Total crashes discovered |
| `action` | Action chosen by the agent |
| `edge_id` | Edge ID from the current execution |

Generate training plots with:

```bash
python3 scripts/plot_metrics.py
```

This produces `plots/training_health.png` (reward, loss, epsilon over time) and `plots/policy_analysis.png` (action distribution and per-edge policy trajectory).
