# RL-Fuzzer (MuoFuzz)

**A Neuro-Symbolic Fuzzer using Deep Reinforcement Learning.**

**RL-Fuzzer** is an enhanced fuzzing framework that integrates **AFL++** with a Deep Reinforcement Learning (DRL) agent. Unlike traditional fuzzers that rely on stochastic (random) mutation scheduling, RL-Fuzzer employs a Deep Q-Network (DQN) to dynamically optimize mutation strategies based on real-time feedback from the target application.

## 🔬 System Overview

The system replaces the standard non-deterministic mutation operators in AFL++ with an intelligent agent capable of learning context-aware strategies. The architecture consists of two primary components communicating via Inter-Process Communication (IPC):

1. **The Environment (Mutator):** A C-based custom mutator for AFL++ that observes the fuzzing state and executes actions.
2. **The Agent (RL Server):** A Python-based neural network that evaluates the state and selects the optimal mutation strategy.

## ✨ Key Features

* **Neuro-Symbolic Architecture:** Combines symbolic execution concepts with neural network-based decision making.
* **Closed-Loop Optimization:** Establishes a real-time feedback loop where mutation efficacy directly influences future policy decisions.
* **Hierarchical Reward Function:** Implements a tiered reward structure that prioritizes code coverage expansion and crash discovery.
* **Seamless Integration:** implemented as a dynamic library compliant with the AFL++ custom mutator API.

## ⚙️ Methodology

The fuzzing process follows a standard Reinforcement Learning cycle:

1. **State Observation:** The mutator (`mutator.c`) extracts the current input state, including input hash, global coverage bitmap density, and crash history.
2. **Policy Inference:** The RL Server (`rl_server.py`) processes the state vector through a DQN to select the action with the highest expected Q-value.
3. **Action Execution:** The mutator applies the selected transformation (e.g., Dictionary Insertion, Arithmetic Mutation) to the input seed.
4. **Feedback & Learning:** The environment calculates the reward based on execution results (coverage delta). The agent updates its neural weights to reinforce successful strategies.

## 🚀 Installation and Usage

### 1. Prerequisites

Ensure the following dependencies are installed:

* **AFL++**: The core fuzzer engine.
* **Python 3.8+**: With `torch` and `numpy` libraries.
* **LLVM/Clang**: For compilation and instrumentation.

```bash
# Install Python dependencies
pip install torch numpy
```
2. Environment Configuration

Define the AFL_ROOT environment variable to point to your AFL++ installation directory.

```
export AFL_ROOT=~/AFLplusplus  # Adjust path as necessary
```

3. Execution

The run_muofuzz.sh script automates the build process, server initialization, and fuzzer execution.

```
chmod +x scripts/run_muofuzz.sh
./scripts/run_muofuzz.sh
```

## Runtime Behavior:

The RL Server initializes and binds to a Unix Domain Socket.
AFL++ compiles the target and connects to the server.
Training metrics are logged to rl_metrics.csv for post-run analysis.

## Project Structure

- Core Components (src/)

    - mutator.c: The AFL++ custom mutator library. Handles IPC with the Python server and executes mutation primitives.

    - target.c: A reference vulnerable application containing nested conditional logic (Magic Bytes, Integer Comparisons, Size Constraints) to benchmark the agent's path-finding capabilities.

- Control Logic (scripts/)

    - rl_server.py: Implements the Deep Q-Network (DQN), experience replay buffer, and training loop.

    - mock_analysis.py: A utility script that simulates static analysis to provide auxiliary state features to the model.

    - plot_metrics.py: Visualization tool for analyzing reward convergence and loss over time.

## Action Space (Mutation Strategies)

The agent operates within a discrete action space consisting of 7 mutation primitives:

1. Arithmetic (+)

Byte-level increment operation
Bypassing loop counters and inequalities


2. Arithmetic (-)

Byte-level decrement operation
Bypassing loop counters and inequalities


3. Interesting 8

Substitution with boundary int8 values
Triggering integer overflows/underflows

4. Interesting 32

Substitution with boundary int32 values
Triggering large integer overflows

5. Dictionary

Token injection from a provided dictionary
Satisfying magic byte/header checks

6. Delete Bytes

Removal of data sequences
Satisfying file size constraints

7. Havoc

Randomized bit-flipping and stacking
General state space exploration

## Analysis

Training performance can be monitored via the generated CSV logs.

Metric Logging: The system logs step, reward, loss, and epsilon values to rl_metrics.csv.

Visualization: Generate training plots using the provided utility:

```
python3 scripts/plot_metrics.py

```
