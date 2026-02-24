# [sysrel]
# rl_server.py — MuoFuzz RL Brain
#
# IPC: Shared memory file at SHM_PATH (no sockets).  Layout unchanged from
# original — only the action range (0..46) and state vector have grown.
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
# Synchronisation (no semaphore, no mutex) — unchanged:
#   C   → writes state fields → release-stores state_seq
#   Py  → polls state_seq; when changed reads state, computes action
#   Py  → writes action → writes action_seq (sentinel last)
#   C   → acquire-loads action_seq until changed → reads action
#
# State vector (98 elements):
#   [0]      edge_id  / MAX_EDGE_ID        (normalised scalar)
#   [1]      coverage / MAX_COVERAGE       (normalised scalar)
#   [2]      min(new_edges, MAX_NEW_EDGES) / MAX_NEW_EDGES
#   [3]      log1p(crashes) / log1p(MAX_CRASHES)
#   [4..50]  disable_prob[0..46]   — per-action disable probability for edge_id
#   [51..97] one-hot prev_action   — 47-element vector
#
# Reward function (potential-based shaping):
#   phi(s)          = (coverage / MAX_COVERAGE) * 100
#   coverage_term   = phi(s') - phi(s)          # can be negative if coverage stalls
#   crash_term      = (log1p(crashes') - log1p(crashes)) * 1000
#   disable_penalty = -5.0 * disable_prob[edge_id][prev_action]
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

import mmap
import struct
import os
import time
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Configuration ─────────────────────────────────────────────────────────────

SHM_PATH     = "/tmp/muofuzz_shm"
SHM_SIZE     = 128
METRICS_FILE = "rl_metrics.csv"
MODEL_PATH   = os.environ.get("MUOFUZZ_MODEL_PATH", "muofuzz_dqn.pt")
DISABLE_CSV  = os.environ.get("MUOFUZZ_DISABLE_CSV", "edgeDisablingMutator.csv")

# SHM byte offsets — identical to original / mutator.c
STATE_SEQ_OFF   = 0
EDGE_ID_OFF     = 4
COVERAGE_OFF    = 8
NEW_EDGES_OFF   = 12
CRASHES_OFF     = 16
TOTAL_EXECS_OFF = 24
ACTION_SEQ_OFF  = 64
ACTION_OFF      = 68

# ── Action / CSV column names (index → name must match mutator.c exactly) ─────

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

# State: 4 scalars + 47 disable probs + 47 one-hot prev_action = 98
STATE_SIZE    = 4 + ACTION_SIZE + ACTION_SIZE   # 98
BATCH_SIZE    = 64
GAMMA         = 0.99
LEARNING_RATE = 1e-4
REPLAY_SIZE   = 50_000
TARGET_SYNC   = 500     # training steps between target-network copies
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY_STEPS = 10_000   # linear decay to EPSILON_MIN over N steps
GRAD_CLIP     = 10.0

# Normalisation constants
MAX_EDGE_ID   = 65536.0
MAX_COVERAGE  = 65536.0
MAX_NEW_EDGES = 100.0
MAX_CRASHES   = 1000.0

# Reward weights
W_DISABLE_PENALTY = 5.0     # multiplied by disable_prob of chosen action
STEP_COST         = 0.1     # small negative reward per step to encourage progress

# ── Disable probability table ─────────────────────────────────────────────────

class DisableTable:
    """
    Loads edgeDisablingMutator.csv and provides O(1) lookup of the
    47-element disable probability vector for a given edge_id.

    CSV format (index column = edge_id, 47 named mutator columns).
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
        # Reorder columns to match ACTION_COLUMNS order (defensive)
        cols_present = [c for c in ACTION_COLUMNS if c in df.columns]
        missing      = [c for c in ACTION_COLUMNS if c not in df.columns]
        if missing:
            print(f"[!] Warning: {len(missing)} CSV columns missing: {missing[:5]}…")
        df = df.reindex(columns=ACTION_COLUMNS, fill_value=0.0)
        for eid, row in df.iterrows():
            self._rows[int(eid)] = row.values.astype(np.float32)
        print(f"[+] DisableTable: loaded {len(self._rows)} edges from '{path}'")

    def get(self, edge_id: int) -> np.ndarray:
        """Return 47-element float32 array; zeros for unknown edges."""
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

    One-hot for prev_action: actions have no ordinal relationship, so encoding
    as a single float would imply incorrect ordering — one-hot is exact.
    """
    dis = disable_table.get(edge_id)   # shape (47,)

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
    """
    Potential-based coverage shaping + crash bonus + disable penalty.

    phi(s) = (coverage / MAX_COVERAGE) * 100
    Shaping: phi(s') - phi(s)   — smooth gradient even when coverage stalls

    Crash term: log-scaled to prevent first crash from dominating.

    Disable penalty: the RL agent chose prev_action on this edge; if the
    disable table says that action is harmful here, penalise it.
    """
    phi_curr = (coverage      / MAX_COVERAGE) * 100.0
    phi_prev = (prev_coverage / MAX_COVERAGE) * 100.0
    coverage_term = phi_curr - phi_prev

    crash_delta = math.log1p(crashes) - math.log1p(prev_crashes)
    crash_term  = crash_delta * 1000.0

    dis = disable_table.get(edge_id)
    disable_prob    = float(dis[prev_action]) if 0 <= prev_action < ACTION_SIZE else 0.0
    disable_penalty = -W_DISABLE_PENALTY * disable_prob

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
    Prevents Q-value overestimation that plain DQN suffers.
    """

    def __init__(self):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online  = DQN().to(self.device)
        self.target  = DQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)
        self.buffer    = ReplayBuffer(REPLAY_SIZE)

        self.epsilon       = EPSILON_START
        self._train_steps  = 0
        self._total_steps  = 0

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online(s).argmax(dim=1).item())

    def _decay_epsilon(self):
        """Linear decay from EPSILON_START to EPSILON_MIN over EPSILON_DECAY_STEPS."""
        frac = min(self._total_steps / EPSILON_DECAY_STEPS, 1.0)
        self.epsilon = EPSILON_START + frac * (EPSILON_MIN - EPSILON_START)

    def remember(self, state, action, reward, next_state):
        self.buffer.push(state, action, reward, next_state)
        self._total_steps += 1
        self._decay_epsilon()

    def train_step(self) -> float:
        """One gradient update. Returns loss (0.0 if buffer too small)."""
        if len(self.buffer) < BATCH_SIZE:
            return 0.0

        batch  = self.buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        s  = torch.FloatTensor(np.array(states)).to(self.device)
        ns = torch.FloatTensor(np.array(next_states)).to(self.device)
        a  = torch.LongTensor(np.array(actions)).to(self.device)
        r  = torch.FloatTensor(np.array(rewards)).to(self.device)

        # Double DQN target:
        #   online net selects the best next action
        #   target net evaluates its Q-value
        with torch.no_grad():
            best_next_actions = self.online(ns).argmax(dim=1)
            target_q          = self.target(ns).gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
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
        self.epsilon      = ck.get("epsilon",     EPSILON_START)
        self._train_steps = ck.get("train_steps", 0)
        self._total_steps = ck.get("total_steps", 0)
        print(f"[+] Loaded checkpoint from {path}  "
              f"(ε={self.epsilon:.3f}, steps={self._total_steps})")


# ── Shared-memory helpers — identical to original ─────────────────────────────

def create_shm() -> mmap.mmap:
    """Create or truncate the shared memory file and mmap it."""
    fd = open(SHM_PATH, "w+b")
    fd.write(b'\x00' * SHM_SIZE)
    fd.flush()
    shm = mmap.mmap(fd.fileno(), SHM_SIZE)
    fd.close()   # mmap keeps the mapping alive
    return shm


def shm_read_state(shm: mmap.mmap) -> tuple:
    """Return (state_seq, edge_id, coverage, new_edges, crashes, total_execs)."""
    shm.seek(0)
    raw = shm.read(64)
    state_seq, edge_id, coverage, new_edges, crashes, _pad, total_execs = \
        struct.unpack_from("=IIIIII Q", raw, 0)
    return state_seq, edge_id, coverage, new_edges, crashes, total_execs


def shm_write_action(shm: mmap.mmap, action: int, action_seq: int):
    """Write action data first, sentinel last (matches C acquire/release protocol)."""
    shm.seek(ACTION_OFF)
    shm.write(struct.pack("=i", action))
    shm.seek(ACTION_SEQ_OFF)
    shm.write(struct.pack("=I", action_seq))
    shm.flush()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MuoFuzz RL server")
    parser.add_argument("--disable-csv", default=DISABLE_CSV,
                        help="Path to edgeDisablingMutator.csv")
    parser.add_argument("--model",       default=MODEL_PATH,
                        help="Path to save/load DQN checkpoint (.pt)")
    parser.add_argument("--print-map",   action="store_true",
                        help="Print action index → CSV column mapping and exit")
    args = parser.parse_args()

    if args.print_map:
        print(f"{'Idx':>4}  {'CSV column'}")
        print("-" * 50)
        for i, col in enumerate(ACTION_COLUMNS):
            print(f"  {i:>2}  {col}")
        return

    disable_table = DisableTable(args.disable_csv)
    shm           = create_shm()
    agent         = DQNAgent()
    agent.load(args.model)

    # Write initial action so C side doesn't spin on boot
    shm_write_action(shm, 46, 1)   # action=HAVOC, action_seq=1

    with open(METRICS_FILE, "w") as f:
        f.write("step,reward,cov_term,crash_term,disable_penalty,"
                "loss,epsilon,coverage,crashes,action,edge_id\n")

    print(f"[+] MuoFuzz RL brain ready.  SHM={SHM_PATH}  "
          f"STATE_SIZE={STATE_SIZE}  ACTION_SIZE={ACTION_SIZE}")

    prev_coverage  = 0
    prev_crashes   = 0
    prev_action    = 46   # HAVOC
    prev_state     = build_state(0, 0, 0, 0, 46, disable_table)
    last_state_seq = 0
    action_seq     = 1
    step           = 0
    save_every     = 1000   # checkpoint interval

    while True:
        # ── Poll for new state from the mutator ──────────────────────────────
        while True:
            shm.seek(STATE_SEQ_OFF)
            cur_seq = struct.unpack("=I", shm.read(4))[0]
            if cur_seq != last_state_seq:
                break
            time.sleep(0.0001)

        last_state_seq = cur_seq
        _, edge_id, coverage, new_edges, crashes, total_execs = shm_read_state(shm)

        # ── Build state ──────────────────────────────────────────────────────
        state = build_state(edge_id, coverage, new_edges, crashes,
                            prev_action, disable_table)

        # ── Compute reward ───────────────────────────────────────────────────
        reward, comps = compute_reward(
            coverage, prev_coverage, crashes, prev_crashes,
            edge_id, prev_action, disable_table)

        # ── Store transition & train ─────────────────────────────────────────
        loss = 0.0
        if step > 0:
            agent.remember(prev_state, prev_action, reward, state)
            loss = agent.train_step()

        # ── Choose and send action ───────────────────────────────────────────
        action     = agent.select_action(state)
        action_seq += 1
        shm_write_action(shm, action, action_seq)

        # ── Logging ─────────────────────────────────────────────────────────
        if step % 100 == 0:
            with open(METRICS_FILE, "a") as f:
                f.write(
                    f"{step},{reward:.4f},"
                    f"{comps['coverage_term']:.4f},{comps['crash_term']:.4f},"
                    f"{comps['disable_penalty']:.4f},"
                    f"{loss:.6f},{agent.epsilon:.4f},"
                    f"{coverage},{crashes},{action},{edge_id}\n"
                )
            print(
                f"[{step:>6}] edge={edge_id:<5} cov={coverage:<5} "
                f"new={new_edges:<3} crash={crashes}  "
                f"act={action:<2} ({ACTION_COLUMNS[action]:<32}) "
                f"ε={agent.epsilon:.3f} r={reward:+.2f} loss={loss:.5f}"
            )

        if step > 0 and step % save_every == 0:
            agent.save(args.model)

        prev_state    = state
        prev_action   = action
        prev_coverage = coverage
        prev_crashes  = crashes
        step         += 1


if __name__ == "__main__":
    main()
