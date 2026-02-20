# [sysrel]
# rl_server.py — MuoFuzz RL Brain
#
# IPC: Shared memory file at SHM_PATH (no sockets).
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
# Synchronisation (no semaphore, no mutex):
#   C   → writes state fields → release-stores state_seq (via __atomic_store_n RELEASE)
#   Py  → polls state_seq; when it changes reads state, computes action
#   Py  → writes action → writes action_seq (plain store; GIL makes it safe)
#   C   → acquire-loads action_seq until it changes, then reads action

import mmap
import struct
import os
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Configuration ────────────────────────────────────────────────────────────
SHM_PATH     = "/tmp/muofuzz_shm"
SHM_SIZE     = 128          # bytes
METRICS_FILE = "rl_metrics.csv"
ENABLE_CSV   = "edge_enable_prob.csv"
DISABLE_CSV  = "edge_disable_prob.csv"

# State indices
STATE_SEQ_OFF    = 0        # offsets into the mmap buffer (bytes)
EDGE_ID_OFF      = 4
COVERAGE_OFF     = 8
NEW_EDGES_OFF    = 12
CRASHES_OFF      = 16
TOTAL_EXECS_OFF  = 24       # u64

ACTION_SEQ_OFF   = 64
ACTION_OFF       = 68

# struct format strings
STATE_FMT  = "=IIIIIII Q 32x"   # state_seq, edge_id, cov, new_edges, crashes, _pad, _pad | total_execs | 32 pad
ACTION_FMT = "=I i 56x"          # action_seq, action | 56 pad

# RL hyper-parameters
STATE_SIZE    = 19   # see build_state() for breakdown
ACTION_SIZE   = 7
BATCH_SIZE    = 64
GAMMA         = 0.99
LEARNING_RATE = 1e-3

# Maximum expected values for normalisation
MAX_EDGE_ID   = 65536.0
MAX_COVERAGE  = 65536.0
MAX_NEW_EDGES = 100.0        # clip and normalise
MAX_CRASHES   = 100.0        # log-normalised anyway

# How each of the 7 custom actions maps to AFL++ mutator column names.
# Probabilities for unmapped actions (4=Dict) fall back to the mean over all cols.
ACTION_MUTATOR_MAP = {
    0: ["DET_ARITH_ADD_ONE",   "HAVOC_MUT_BYTEADD",    "HAVOC_MUT_ARITH8_"],
    1: ["DET_ARITH_SUB_ONE",   "HAVOC_MUT_BYTESUB",    "HAVOC_MUT_ARITH8"],
    2: ["DET_FLIP_ONE_BIT",    "DET_FLIP_ONE_BYTE"],
    3: ["DET_FLIP_TWO_BITS",   "DET_FLIP_FOUR_BITS",
        "DET_ARITH_ADD_TWO_LE","DET_ARITH_ADD_TWO_BIG"],
    4: None,    # Dictionary — no direct AFL++ match; will use row mean
    5: ["DET_ARITH_SUB_TWO_LE","DET_ARITH_SUB_TWO_BIG"],
    6: ["HAVOC_MUT_FLIPBIT",   "HAVOC_MUT_ARITH16_",   "HAVOC_MUT_ARITH16BE_",
        "HAVOC_MUT_ARITH16",   "HAVOC_MUT_ARITH16BE"],
}

# ── Edge-probability tables ───────────────────────────────────────────────────

def load_edge_tables():
    """Load enable/disable probability CSVs into dicts keyed by int edge_id.
    Each value is a Series of per-mutator probabilities (20 columns)."""
    tables = {"enable": {}, "disable": {}}
    for kind, path in [("enable", ENABLE_CSV), ("disable", DISABLE_CSV)]:
        if not os.path.exists(path):
            print(f"[!] Warning: {path} not found — edge probability features will be zero.")
            continue
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(int)
        tables[kind] = {eid: row for eid, row in df.iterrows()}
        print(f"[+] Loaded {kind} probabilities for {len(tables[kind])} edges from {path}.")
    return tables


def edge_probs_for_action(tables, edge_id, kind):
    """Return a 7-element list of mean probabilities (one per custom action)
    for a given edge_id and kind='enable'|'disable'.
    Falls back to zeros for unknown edges."""
    row = tables[kind].get(edge_id)
    if row is None:
        return [0.0] * ACTION_SIZE

    probs = []
    for action_idx in range(ACTION_SIZE):
        cols = ACTION_MUTATOR_MAP[action_idx]
        if cols is None:
            # Dictionary action: use mean over all columns
            val = float(row.mean())
        else:
            # Average probabilities of the mapped columns that exist in the CSV
            valid = [row[c] for c in cols if c in row.index]
            val = float(np.mean(valid)) if valid else 0.0
        probs.append(val)
    return probs


# ── DQN ──────────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SIZE),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = DQN().to(self.device)
        self.target  = DQN().to(self.device)   # target network for stable Q-targets
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory  = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.995
        self.update_target_every = 200
        self._train_steps = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.model(s).argmax().item())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        s  = torch.FloatTensor(np.array(states)).to(self.device)
        ns = torch.FloatTensor(np.array(next_states)).to(self.device)
        a  = torch.LongTensor(np.array(actions)).to(self.device)
        r  = torch.FloatTensor(np.array(rewards)).to(self.device)

        with torch.no_grad():
            max_q_next = self.target(ns).max(dim=1).values
            targets = r + GAMMA * max_q_next

        predictions = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        loss = nn.functional.mse_loss(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self._train_steps += 1
        if self._train_steps % self.update_target_every == 0:
            self.target.load_state_dict(self.model.state_dict())

        return loss.item()


# ── State builder ─────────────────────────────────────────────────────────────

def build_state(edge_id, coverage, new_edges, crashes, prev_action, tables):
    """Construct the 19-element state vector.

    Layout:
      [0]     edge_id       normalised to [0,1]
      [1]     coverage_pct  fraction of MAX_COVERAGE
      [2]     new_edges     clipped & normalised
      [3]     crashes       log-normalised
      [4..10] enable_prob   per custom action (7 values)
      [11..17] disable_prob per custom action (7 values)
      [18]    prev_action   normalised to [0,1]
    """
    en = edge_probs_for_action(tables, edge_id, "enable")
    di = edge_probs_for_action(tables, edge_id, "disable")

    state = [
        edge_id / MAX_EDGE_ID,
        coverage / MAX_COVERAGE,
        min(new_edges, MAX_NEW_EDGES) / MAX_NEW_EDGES,
        math.log1p(crashes) / math.log1p(MAX_CRASHES),
        *en,
        *di,
        prev_action / float(ACTION_SIZE - 1),
    ]
    assert len(state) == STATE_SIZE, f"State size mismatch: {len(state)} != {STATE_SIZE}"
    return np.array(state, dtype=np.float32)


# ── Shared-memory helpers ─────────────────────────────────────────────────────

def create_shm():
    """Create or truncate the shared memory file and mmap it."""
    fd = open(SHM_PATH, "w+b")
    fd.write(b'\x00' * SHM_SIZE)
    fd.flush()
    shm = mmap.mmap(fd.fileno(), SHM_SIZE)
    fd.close()   # fd can close; mmap keeps the mapping alive
    return shm


def shm_read_state(shm):
    """Read the state region from the mmap.
    Returns (state_seq, edge_id, coverage, new_edges, crashes, total_execs)."""
    shm.seek(0)
    raw = shm.read(64)
    # u32 state_seq, u32 edge_id, u32 cov, u32 new_edges, u32 crashes, u32 _pad, u64 total_execs
    state_seq, edge_id, coverage, new_edges, crashes, _pad, total_execs = struct.unpack_from(
        "=IIIIII Q", raw, 0
    )
    return state_seq, edge_id, coverage, new_edges, crashes, total_execs


def shm_write_action(shm, action, action_seq):
    """Write the action region. Writes action first, then action_seq (the sentinel)."""
    shm.seek(ACTION_OFF)
    shm.write(struct.pack("=i", action))      # write data first
    shm.seek(ACTION_SEQ_OFF)
    shm.write(struct.pack("=I", action_seq))  # write sentinel last (Python GIL is our 'release')
    shm.flush()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tables = load_edge_tables()
    shm    = create_shm()
    agent  = Agent()

    # Write an initial action so the C side doesn't spin forever on boot
    shm_write_action(shm, 6, 1)   # action=Havoc, action_seq=1

    with open(METRICS_FILE, "w") as f:
        f.write("step,reward,loss,epsilon,coverage,crashes,action,edge_id\n")

    print(f"[+] MuoFuzz RL brain ready. SHM at {SHM_PATH}. STATE_SIZE={STATE_SIZE}.")

    prev_coverage    = 0
    prev_crashes     = 0
    prev_action      = 6            # start with Havoc
    prev_state       = build_state(0, 0, 0, 0, 6, tables)

    last_state_seq   = 0
    action_seq       = 1            # starts at 1 (already written above)
    step             = 0

    while True:
        # ── Poll for new state from the mutator ──────────────────────────────
        # Spin until state_seq changes (C release-stored it after writing data)
        while True:
            shm.seek(STATE_SEQ_OFF)
            cur_seq = struct.unpack("=I", shm.read(4))[0]
            if cur_seq != last_state_seq:
                break
            time.sleep(0.0001)   # ~0.1 ms back-off; avoids burning a full core

        last_state_seq = cur_seq
        _, edge_id, coverage, new_edges, crashes, total_execs = shm_read_state(shm)

        # ── Build state & reward ─────────────────────────────────────────────
        state = build_state(edge_id, coverage, new_edges, crashes, prev_action, tables)

        d_cov   = coverage - prev_coverage
        d_crash = crashes  - prev_crashes
        reward  = (d_cov * 100.0) + (d_crash * 10_000.0) - 0.1

        # Bonus: if we gained new edges AND the edge table had useful probabilities
        en_probs = edge_probs_for_action(tables, edge_id, "enable")
        if d_cov > 0 and any(p > 0 for p in en_probs):
            reward += 20.0

        # ── Store & train ────────────────────────────────────────────────────
        loss = 0.0
        if step > 0:
            agent.remember(prev_state, prev_action, reward, state)
            loss = agent.train()

        # ── Choose action ────────────────────────────────────────────────────
        action     = agent.act(state)
        action_seq += 1
        shm_write_action(shm, action, action_seq)

        # ── Log every 100 steps ──────────────────────────────────────────────
        if step % 100 == 0:
            with open(METRICS_FILE, "a") as f:
                f.write(
                    f"{step},{reward:.4f},{loss:.6f},{agent.epsilon:.4f},"
                    f"{coverage},{crashes},{action},{edge_id}\n"
                )
            print(
                f"[{step:>6}] edge={edge_id:<5} cov={coverage:<5} "
                f"new={new_edges:<3} crash={crashes} "
                f"act={action} ε={agent.epsilon:.3f} loss={loss:.5f}"
            )

        prev_state    = state
        prev_action   = action
        prev_coverage = coverage
        prev_crashes  = crashes
        step         += 1


if __name__ == "__main__":
    main()
