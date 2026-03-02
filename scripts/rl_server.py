# [sysrel]
#
# Two modes:
#   train  (default) — ε-greedy exploration, replay buffer, gradient updates,
#                      plateau detection, auto-stop, checkpoint on exit.
#   eval             — loads checkpoint, ε=0 (pure greedy), no weight updates,
#                      separate metrics file, runs until AFL++ is killed.
#
# Usage:
#   # Train until plateau:
#   python3 scripts/rl_server.py --mode train --disable-csv edgeDisablingMutator.csv
#
#   # Evaluate a saved model:
#   python3 scripts/rl_server.py --mode eval  --model rl_dqn.pt \
#           --disable-csv edgeDisablingMutator.csv --eval-steps 20000
#
# IPC: Shared memory file at SHM_PATH.
#
# SHM layout (128 bytes total):
#
#   Offset  0..63  — STATE REGION (C writes, Python reads)
#     [0]   state_seq    u32  — sequence sentinel; C release-stores this LAST
#     [4]   edge_id      u32  — most recently new/hot edge from trace_bits
#     [8]   coverage     u32  — total bitmap coverage count
#     [12]  new_edges    u32  — delta new edges since last step
#     [16]  crashes      u32  — total crash count
#     [20]  _pad         u32
#     [24]  total_execs  u64
#     [32]  _pad         32 bytes (alignment / false-share guard)
#
#   Offset 64..127 — ACTION REGION (Python writes, C reads)
#     [64]  action_seq   u32  — sentinel; Python writes this LAST
#     [68]  action       i32  — chosen action (0..ACTION_SIZE-1)
#     [72]  _pad         56 bytes
#
# State vector (98 elements):
#   [0]      edge_id  / MAX_EDGE_ID
#   [1]      coverage / MAX_COVERAGE
#   [2]      min(new_edges, MAX_NEW_EDGES) / MAX_NEW_EDGES
#   [3]      log1p(crashes) / log1p(MAX_CRASHES)
#   [4..50]  disable_prob[0..46]   — per-action disable probability for edge_id
#   [51..97] one-hot prev_action   — 47-element vector
#
# Reward function (potential-based shaping):
#   phi(s)          = (coverage / MAX_COVERAGE) * 100
#   coverage_term   = phi(s') - phi(s)
#   crash_term      = (log1p(crashes') - log1p(crashes)) * 1000
#   disable_penalty = 5.0 * disable_prob[edge_id][prev_action]
#   reward          = coverage_term + crash_term + disable_penalty - 0.1
#
# DQN (Double DQN):
#   Architecture : Linear(98→256) → ReLU → Linear(256→256) → ReLU
#                  → Linear(256→128) → ReLU → Linear(128→47)
#   Gamma        : 0.99
#   LR           : 1e-4
#   Batch        : 64
#   Replay buffer: 50 000 transitions
#   Target sync  : every 500 training steps
#   Epsilon      : 1.0 → 0.05 decayed linearly over 10 000 steps
#   Grad clip    : max_norm = 10.0
#
# Plateau detection (train mode only):
#   Tracks coverage in a sliding window of PLATEAU_WINDOW steps.
#   A plateau is declared when the max coverage gain over that window
#   is below PLATEAU_MIN_DELTA AND epsilon has finished decaying
#   (i.e. the agent is fully exploiting, not still exploring).
#   On plateau: saves checkpoint, prints summary, exits cleanly.
#   run_rl.sh's EXIT trap then kills afl-fuzz automatically.

import mmap
import struct
import os
import sys
import time
import math
import signal
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Configuration ─────────────────────────────────────────────────────────────

SHM_PATH     = "/tmp/rl_shm"
SHM_SIZE     = 128
METRICS_FILE       = "rl_metrics.csv"
EVAL_METRICS_FILE  = "rl_metrics_eval.csv"
MODEL_PATH   = os.environ.get("RL_MODEL_PATH", "rl_dqn.pt")
DISABLE_CSV  = os.environ.get("RL_DISABLE_CSV", "edgeDisablingMutator.csv")

# SHM byte offsets — identical to mutator.c
STATE_SEQ_OFF   = 0
EDGE_ID_OFF     = 4
COVERAGE_OFF    = 8
NEW_EDGES_OFF   = 12
CRASHES_OFF     = 16
TOTAL_EXECS_OFF = 24
ACTION_SEQ_OFF  = 64
ACTION_OFF      = 68

# ── Action / CSV column names ─────────────────────────────────────────────────

ACTION_COLUMNS = [
    "DET_FLIP_ONE_BIT",          #  0
    "DET_FLIP_TWO_BITS",         #  1
    "DET_FLIP_FOUR_BITS",        #  2
    "DET_FLIP_ONE_BYTE",         #  3
    "DET_FLIP_TWO_BYTES",        #  4
    "DET_FLIP_FOUR_BYTES",       #  5
    "DET_ARITH_ADD_ONE",         #  6
    "DET_ARITH_SUB_ONE",         #  7
    "DET_ARITH_ADD_TWO_LE",      #  8
    "DET_ARITH_SUB_TWO_LE",      #  9
    "DET_ARITH_ADD_TWO_BIG",     # 10
    "DET_ARITH_SUB_TWO_BIG",     # 11
    "DET_ARITH_ADD_FOUR_LE",     # 12
    "DET_ARITH_SUB_FOUR_LE",     # 13
    "DET_ARITH_ADD_FOUR_BIG",    # 14
    "DET_ARITH_SUB_FOUR_BIG",    # 15
    "INTERESTING_BYTE",          # 16
    "INTERESTING_TWO_BYTES_LE",  # 17
    "INTERESTING_TWO_BYTES_BIG", # 18
    "INTERESTING_FOUR_BYTES_LE", # 19
    "INTERESTING_FOUR_BYTES_BIG",# 20
    "HAVOC_MUT_FLIPBIT",         # 21
    "HAVOC_MUT_INTERESTING8",    # 22
    "HAVOC_MUT_INTERESTING16",   # 23
    "HAVOC_MUT_INTERESTING16BE", # 24
    "HAVOC_MUT_INTERESTING32",   # 25
    "HAVOC_MUT_INTERESTING32BE", # 26
    "HAVOC_MUT_ARITH8_",         # 27
    "HAVOC_MUT_ARITH8",          # 28
    "HAVOC_MUT_ARITH16_",        # 29
    "HAVOC_MUT_ARITH16BE_",      # 30
    "HAVOC_MUT_ARITH16",         # 31
    "HAVOC_MUT_ARITH16BE",       # 32
    "HAVOC_MUT_ARITH32_",        # 33
    "HAVOC_MUT_ARITH32BE_",      # 34
    "HAVOC_MUT_ARITH32",         # 35
    "HAVOC_MUT_ARITH32BE",       # 36
    "HAVOC_MUT_RAND8",           # 37
    "HAVOC_MUT_BYTEADD",         # 38
    "HAVOC_MUT_BYTESUB",         # 39
    "HAVOC_MUT_FLIP8",           # 40
    "DICTIONARY_USER_EXTRAS_OVER",   # 41
    "DICTIONARY_USER_EXTRAS_INSERT", # 42
    "DICTIONARY_AUTO_EXTRAS_OVER",   # 43
    "DICTIONARY_AUTO_EXTRAS_INSERT", # 44
    "CUSTOM_MUTATOR",            # 45
    "HAVOC",                     # 46
]

ACTION_SIZE = len(ACTION_COLUMNS)   # 47
assert ACTION_SIZE == 47

# ── RL hyper-parameters ───────────────────────────────────────────────────────

STATE_SIZE    = 4 + ACTION_SIZE + ACTION_SIZE   # 98
BATCH_SIZE    = 64
GAMMA         = 0.99
LEARNING_RATE = 1e-4
REPLAY_SIZE   = 50_000
TARGET_SYNC   = 500
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY_STEPS = 10_000
GRAD_CLIP     = 10.0

# Normalisation constants
MAX_EDGE_ID   = 16384
MAX_COVERAGE  = 65536.0
MAX_NEW_EDGES = 100.0
MAX_CRASHES   = 1000.0

# Reward weights
W_DISABLE_PENALTY = 5.0
STEP_COST         = 0.1

# ── Plateau detection parameters ──────────────────────────────────────────────
#
# The agent is considered to have plateaued when BOTH conditions hold:
#
#   1. Epsilon is at or near minimum (exploration is done — we're in pure
#      exploitation mode, so any remaining stagnation is real, not just the
#      agent still learning what to try).
#
#   2. The best coverage seen in the last PLATEAU_WINDOW steps has not
#      improved by more than PLATEAU_MIN_DELTA edges over the oldest
#      coverage value in that same window.
#
# Tuning guidance:
#   PLATEAU_WINDOW     — larger = more patient; 2000 steps ≈ a few minutes
#                        of fuzzing at ~500 exec/sec with 1 RL step/exec.
#   PLATEAU_MIN_DELTA  — 1 means "even a single new edge resets the clock".
#                        Raise to 2-3 if you want to ignore marginal gains.
#   PLATEAU_GRACE_STEPS — don't start checking until the replay buffer is
#                         full and epsilon has had time to decay; protects
#                         against false early plateau on the first cycle.

PLATEAU_WINDOW      = 5000   # steps of coverage history to inspect
PLATEAU_MIN_DELTA   = 1      # minimum new edges needed to NOT be a plateau
PLATEAU_GRACE_STEPS = 25000  # don't check before this many steps


# ── Plateau detector ──────────────────────────────────────────────────────────

class PlateauDetector:
    """
    Maintains a sliding window of coverage observations.
    Call update(coverage, epsilon, step) every RL step.
    Returns True (plateau) when both conditions above are met.
    """

    def __init__(self, window: int = PLATEAU_WINDOW,
                 min_delta: int = PLATEAU_MIN_DELTA,
                 grace: int = PLATEAU_GRACE_STEPS):
        self._window    = window
        self._min_delta = min_delta
        self._grace     = grace
        self._history   = deque(maxlen=window)   # rolling coverage values
        self._triggered = False

    def update(self, coverage: int, epsilon: float, step: int) -> bool:
        """Return True if plateau condition is met (only fires once)."""
        if self._triggered:
            return True

        self._history.append(coverage)

        # Grace period: don't evaluate until we have enough data
        if step < self._grace or len(self._history) < self._window:
            return False

        # Condition 1: epsilon must be at or near minimum
        if epsilon > EPSILON_MIN + 0.01:
            return False

        # Condition 2: coverage hasn't grown meaningfully in the window
        oldest = self._history[0]
        best   = max(self._history)
        if best - oldest <= self._min_delta:
            self._triggered = True
            return True

        return False

    def reset(self):
        self._history.clear()
        self._triggered = False


# ── Disable probability table ─────────────────────────────────────────────────

class DisableTable:
    """
    Loads edgeDisablingMutator.csv and provides O(1) lookup of the
    47-element disable probability vector for a given edge_id.
    Unknown edge_ids return all-zeros (conservative: no penalty).
    """

    def __init__(self, path: str):
        self._rows: dict[int, np.ndarray] = {}
        if not os.path.exists(path):
            print(f"[!] Warning: disable CSV not found at '{path}' — "
                  f"disable features will be zero.")
            return
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(int)
        cols_present = [c for c in ACTION_COLUMNS if c in df.columns]
        missing      = [c for c in ACTION_COLUMNS if c not in df.columns]
        if missing:
            print(f"[!] Warning: {len(missing)} CSV columns missing: {missing[:5]}…")
        df = df.reindex(columns=ACTION_COLUMNS, fill_value=0.0)
        for eid, row in df.iterrows():
            self._rows[int(eid)] = row.values.astype(np.float32)
        print(f"[+] DisableTable: loaded {len(self._rows)} edges from '{path}'")

    def get(self, edge_id: int) -> np.ndarray:
        return self._rows.get(edge_id, np.zeros(ACTION_SIZE, dtype=np.float32))


# ── State builder ─────────────────────────────────────────────────────────────

def build_state(edge_id: int, coverage: int, new_edges: int, crashes: int,
                prev_action: int, disable_table: DisableTable) -> np.ndarray:
    """
    Construct the 98-element state vector.

    Layout:
      [0]      edge_id / MAX_EDGE_ID
      [1]      coverage / MAX_COVERAGE
      [2]      min(new_edges, MAX_NEW_EDGES) / MAX_NEW_EDGES
      [3]      log1p(crashes) / log1p(MAX_CRASHES)
      [4..50]  disable_prob[0..46]  for current edge_id   (47 values)
      [51..97] one-hot encoding of prev_action            (47 values)
    """
    dis = disable_table.get(edge_id)

    one_hot = np.zeros(ACTION_SIZE, dtype=np.float32)
    if 0 <= prev_action < ACTION_SIZE:
        one_hot[prev_action] = 1.0

    state = np.concatenate([
        np.array([
            edge_id / MAX_EDGE_ID,
            coverage / MAX_COVERAGE,
            min(new_edges, MAX_NEW_EDGES) / MAX_NEW_EDGES,
            math.log1p(crashes) / math.log1p(MAX_CRASHES),
        ], dtype=np.float32),
        dis,
        one_hot,
    ])
    assert len(state) == STATE_SIZE, f"State size mismatch: {len(state)}"
    return state


# ── Reward function ───────────────────────────────────────────────────────────

def compute_reward(coverage: int, prev_coverage: int,
                   crashes: int, prev_crashes: int,
                   edge_id: int, prev_action: int,
                   disable_table: DisableTable) -> tuple[float, dict]:
    phi_curr = (coverage      / MAX_COVERAGE) * 100.0
    phi_prev = (prev_coverage / MAX_COVERAGE) * 100.0
    coverage_term = phi_curr - phi_prev

    crash_delta = math.log1p(crashes) - math.log1p(prev_crashes)
    crash_term  = crash_delta * 1000.0

    dis = disable_table.get(edge_id)
    disable_prob    = float(dis[prev_action]) if 0 <= prev_action < ACTION_SIZE else 0.0
    disable_penalty = W_DISABLE_PENALTY * disable_prob

    reward = coverage_term + crash_term + disable_penalty - STEP_COST

    components = {
        "coverage_term":   coverage_term,
        "crash_term":      crash_term,
        "disable_penalty": disable_penalty,
        "step_cost":       -STEP_COST,
    }
    return reward, components


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self._buf.append((state, action, reward, next_state))

    def sample(self, n: int):
        return random.sample(self._buf, n)

    def __len__(self):
        return len(self._buf)


# ── DQN architecture ──────────────────────────────────────────────────────────

class DQN(nn.Module):
    """4-layer MLP: 98 → 256 → 256 → 128 → 47"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, ACTION_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── DQN Agent (Double DQN) ────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN: online net selects action, target net evaluates Q-value.
    In eval mode, epsilon is forced to 0 and train_step() is a no-op.
    """

    def __init__(self, eval_mode: bool = False):
        self.eval_mode = eval_mode
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online    = DQN().to(self.device)
        self.target    = DQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.buffer    = ReplayBuffer(REPLAY_SIZE)

        # In eval mode we hard-set epsilon to 0 (pure greedy) regardless of
        # what the checkpoint stored — we are done exploring.
        self.epsilon       = 0.0 if eval_mode else EPSILON_START
        self._train_steps  = 0
        self._total_steps  = 0

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy (train) or greedy (eval)."""
        if not self.eval_mode and random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online(s).argmax(dim=1).item())

    def _decay_epsilon(self):
        frac = min(self._total_steps / EPSILON_DECAY_STEPS, 1.0)
        self.epsilon = EPSILON_START + frac * (EPSILON_MIN - EPSILON_START)

    def remember(self, state, action, reward, next_state):
        if self.eval_mode:
            return   # never write to buffer during eval
        self.buffer.push(state, action, reward, next_state)
        self._total_steps += 1
        self._decay_epsilon()

    def train_step(self) -> float:
        """One gradient update. No-op in eval mode. Returns loss."""
        if self.eval_mode:
            return 0.0
        if len(self.buffer) < BATCH_SIZE:
            return 0.0

        batch  = self.buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        s  = torch.FloatTensor(np.array(states)).to(self.device)
        ns = torch.FloatTensor(np.array(next_states)).to(self.device)
        a  = torch.LongTensor(np.array(actions)).to(self.device)
        r  = torch.FloatTensor(np.array(rewards)).to(self.device)

        # Double DQN target
        with torch.no_grad():
            best_next_actions = self.online(ns).argmax(dim=1)
            target_q          = self.target(ns).gather(
                                    1, best_next_actions.unsqueeze(1)).squeeze(1)
            y                 = r + GAMMA * target_q

        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss   = nn.functional.mse_loss(q_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % TARGET_SYNC == 0:
            self.target.load_state_dict(self.online.state_dict())
            print(f"[i] Target network synced at train step {self._train_steps}")

        return loss.item()

    def save(self, path: str):
        torch.save({
            "online":       self.online.state_dict(),
            "target":       self.target.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "epsilon":      self.epsilon,
            "train_steps":  self._train_steps,
            "total_steps":  self._total_steps,
        }, path)
        print(f"[+] Model saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[i] No checkpoint at {path} — starting fresh.")
            return
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optimizer"])
        # In eval mode, always override epsilon to 0 regardless of checkpoint
        if not self.eval_mode:
            self.epsilon = ck.get("epsilon", EPSILON_START)
        self._train_steps = ck.get("train_steps", 0)
        self._total_steps = ck.get("total_steps", 0)
        print(f"[+] Loaded checkpoint from {path}  "
              f"(stored ε={ck.get('epsilon', '?'):.3f}, "
              f"steps={self._total_steps}, "
              f"running ε={self.epsilon:.3f})")


# ── Shared-memory helpers ─────────────────────────────────────────────────────

def create_shm() -> mmap.mmap:
    fd = open(SHM_PATH, "w+b")
    fd.write(b'\x00' * SHM_SIZE)
    fd.flush()
    shm = mmap.mmap(fd.fileno(), SHM_SIZE)
    fd.close()
    return shm


def shm_read_state(shm: mmap.mmap) -> tuple:
    shm.seek(0)
    raw = shm.read(64)
    state_seq, edge_id, coverage, new_edges, crashes, _pad, total_execs = \
        struct.unpack_from("=IIIIII Q", raw, 0)
    return state_seq, edge_id, coverage, new_edges, crashes, total_execs


def shm_write_action(shm: mmap.mmap, action: int, action_seq: int):
    shm.seek(ACTION_OFF)
    shm.write(struct.pack("=i", action))
    shm.seek(ACTION_SEQ_OFF)
    shm.write(struct.pack("=I", action_seq))
    shm.flush()


# ── Training summary ──────────────────────────────────────────────────────────

def print_train_summary(step: int, coverage: int, crashes: int,
                        agent: DQNAgent, reason: str):
    print("")
    print("=" * 60)
    print(f"  training stopped — {reason}")
    print(f"  Total RL steps    : {step:,}")
    print(f"  Total train steps : {agent._train_steps:,}")
    print(f"  Final epsilon     : {agent.epsilon:.4f}")
    print(f"  Final coverage    : {coverage}")
    print(f"  Total crashes     : {crashes}")
    print("=" * 60)
    print("")


# ── Eval summary ─────────────────────────────────────────────────────────────

def print_eval_summary(step: int, coverage_start: int, coverage_end: int,
                       crashes: int):
    print("")
    print("=" * 60)
    print("   eval run complete")
    print(f"  Eval steps        : {step:,}")
    print(f"  Coverage start    : {coverage_start}")
    print(f"  Coverage end      : {coverage_end}")
    print(f"  Coverage gained   : {coverage_end - coverage_start}")
    print(f"  Crashes found     : {crashes}")
    print("=" * 60)
    print("")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL server")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="train: explore+learn+plateau-stop | "
                             "eval: greedy, no training, fixed steps")
    parser.add_argument("--disable-csv", default=DISABLE_CSV,
                        help="Path to edgeDisablingMutator.csv")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to save/load DQN checkpoint (.pt)")
    parser.add_argument("--eval-steps", type=int, default=20_000,
                        help="(eval mode) number of RL steps to run then exit")
    parser.add_argument("--print-map", action="store_true",
                        help="Print action index → CSV column mapping and exit")
    args = parser.parse_args()

    if args.print_map:
        print(f"{'Idx':>4}  {'CSV column'}")
        print("-" * 50)
        for i, col in enumerate(ACTION_COLUMNS):
            print(f"  {i:>2}  {col}")
        return

    is_eval = (args.mode == "eval")

    if is_eval:
        print(f"[+] Mode: EVAL  (ε=0, no weight updates, {args.eval_steps:,} steps)")
    else:
        print(f"[+] Mode: TRAIN  (ε-greedy, replay+gradient, plateau detection)")
        print(f"[+] Plateau config: window={PLATEAU_WINDOW} steps, "
              f"min_delta={PLATEAU_MIN_DELTA} edge(s), "
              f"grace={PLATEAU_GRACE_STEPS} steps")

    disable_table = DisableTable(args.disable_csv)
    shm           = create_shm()
    agent         = DQNAgent(eval_mode=is_eval)
    agent.load(args.model)

    plateau = PlateauDetector() if not is_eval else None

    # Write initial action so the C side doesn't spin on boot
    shm_write_action(shm, 46, 1)   # HAVOC, action_seq=1

    # Open the appropriate metrics file
    metrics_path = EVAL_METRICS_FILE if is_eval else METRICS_FILE
    with open(metrics_path, "w") as f:
        f.write("step,reward,cov_term,crash_term,disable_penalty,"
                "loss,epsilon,coverage,crashes,action,edge_id\n")

    print(f"[+] RL brain ready.  SHM={SHM_PATH}  "
          f"STATE_SIZE={STATE_SIZE}  ACTION_SIZE={ACTION_SIZE}")
    if is_eval:
        print(f"[+] Metrics → {EVAL_METRICS_FILE}")
    else:
        print(f"[+] Metrics → {METRICS_FILE}")

    prev_coverage  = 0
    prev_crashes   = 0
    prev_action    = 46
    prev_state     = build_state(0, 0, 0, 0, 46, disable_table)
    last_state_seq = 0
    action_seq     = 1
    step           = 0
    save_every     = 1000
    coverage_at_eval_start = None
    stop_reason    = "user interrupt"

    try:
        while True:
            # ── Eval step limit ──────────────────────────────────────────────
            if is_eval and step >= args.eval_steps:
                stop_reason = f"eval step limit ({args.eval_steps:,} steps)"
                break

            # ── Poll for new state ───────────────────────────────────────────
            while True:
                shm.seek(STATE_SEQ_OFF)
                cur_seq = struct.unpack("=I", shm.read(4))[0]
                if cur_seq != last_state_seq:
                    break
                time.sleep(0.0001)

            last_state_seq = cur_seq
            _, edge_id, coverage, new_edges, crashes, total_execs = \
                shm_read_state(shm)

            if is_eval and coverage_at_eval_start is None:
                coverage_at_eval_start = coverage

            # ── Build state ──────────────────────────────────────────────────
            state = build_state(edge_id, coverage, new_edges, crashes,
                                prev_action, disable_table)

            # ── Compute reward ───────────────────────────────────────────────
            reward, comps = compute_reward(
                coverage, prev_coverage, crashes, prev_crashes,
                edge_id, prev_action, disable_table)

            # ── Store transition & train (no-op in eval) ─────────────────────
            loss = 0.0
            if step > 0:
                agent.remember(prev_state, prev_action, reward, state)
                loss = agent.train_step()

            # ── Choose and send action ───────────────────────────────────────
            action     = agent.select_action(state)
            action_seq += 1
            shm_write_action(shm, action, action_seq)

            # ── Logging ─────────────────────────────────────────────────────
            if step % 100 == 0:
                with open(metrics_path, "a") as f:
                    f.write(
                        f"{step},{reward:.4f},"
                        f"{comps['coverage_term']:.4f},{comps['crash_term']:.4f},"
                        f"{comps['disable_penalty']:.4f},"
                        f"{loss:.6f},{agent.epsilon:.4f},"
                        f"{coverage},{crashes},{action},{edge_id}\n"
                    )
                mode_tag = "EVAL" if is_eval else "TRAIN"
                print(
                    f"[{mode_tag}:{step:>6}] edge={edge_id:<5} cov={coverage:<5} "
                    f"new={new_edges:<3} crash={crashes}  "
                    f"act={action:<2} ({ACTION_COLUMNS[action]:<32}) "
                    f"ε={agent.epsilon:.3f} r={reward:+.2f} loss={loss:.5f}"
                )

            # ── Periodic checkpoint (train only) ─────────────────────────────
            if not is_eval and step > 0 and step % save_every == 0:
                agent.save(args.model)

            # ── Plateau detection (train only) ───────────────────────────────
            if not is_eval and plateau is not None:
                if plateau.update(coverage, agent.epsilon, step):
                    stop_reason = "coverage plateau detected"
                    break

            prev_state    = state
            prev_action   = action
            prev_coverage = coverage
            prev_crashes  = crashes
            step         += 1

    except KeyboardInterrupt:
        stop_reason = "user interrupt (Ctrl-C)"

    finally:
        # Always save on exit in train mode
        if not is_eval:
            agent.save(args.model)
            print_train_summary(step, coverage if step > 0 else 0,
                                crashes if step > 0 else 0,
                                agent, stop_reason)
        else:
            cov_start = coverage_at_eval_start or 0
            cov_end   = coverage if step > 0 else 0
            print_eval_summary(step, cov_start, cov_end,
                               crashes if step > 0 else 0)

        print(f"[+] rl_server.py exiting ({stop_reason})")
        # Exiting here triggers run_rl.sh's EXIT trap, which kills afl-fuzz


if __name__ == "__main__":
    main()
