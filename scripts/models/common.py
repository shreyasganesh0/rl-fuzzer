"""Shared infrastructure for all RL fuzzer models."""

import mmap, struct, os, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from collections import deque

# ── Action labels (must match mutator_m*.c) ──────────────────────────────────
ACTION_COLUMNS = [
    "DET_FLIP_ONE_BIT","DET_FLIP_TWO_BITS","DET_FLIP_FOUR_BITS",
    "DET_FLIP_ONE_BYTE","DET_FLIP_TWO_BYTES","DET_FLIP_FOUR_BYTES",
    "DET_ARITH_ADD_ONE","DET_ARITH_SUB_ONE",
    "DET_ARITH_ADD_TWO_LE","DET_ARITH_SUB_TWO_LE",
    "DET_ARITH_ADD_TWO_BIG","DET_ARITH_SUB_TWO_BIG",
    "DET_ARITH_ADD_FOUR_LE","DET_ARITH_SUB_FOUR_LE",
    "DET_ARITH_ADD_FOUR_BIG","DET_ARITH_SUB_FOUR_BIG",
    "INTERESTING_BYTE","INTERESTING_TWO_BYTES_LE","INTERESTING_TWO_BYTES_BIG",
    "INTERESTING_FOUR_BYTES_LE","INTERESTING_FOUR_BYTES_BIG",
    "HAVOC_MUT_FLIPBIT","HAVOC_MUT_INTERESTING8",
    "HAVOC_MUT_INTERESTING16","HAVOC_MUT_INTERESTING16BE",
    "HAVOC_MUT_INTERESTING32","HAVOC_MUT_INTERESTING32BE",
    "HAVOC_MUT_ARITH8_","HAVOC_MUT_ARITH8",
    "HAVOC_MUT_ARITH16_","HAVOC_MUT_ARITH16BE_",
    "HAVOC_MUT_ARITH16","HAVOC_MUT_ARITH16BE",
    "HAVOC_MUT_ARITH32_","HAVOC_MUT_ARITH32BE_",
    "HAVOC_MUT_ARITH32","HAVOC_MUT_ARITH32BE",
    "HAVOC_MUT_RAND8","HAVOC_MUT_BYTEADD","HAVOC_MUT_BYTESUB","HAVOC_MUT_FLIP8",
    "DICTIONARY_USER_EXTRAS_OVER","DICTIONARY_USER_EXTRAS_INSERT",
    "DICTIONARY_AUTO_EXTRAS_OVER","DICTIONARY_AUTO_EXTRAS_INSERT",
    "CUSTOM_MUTATOR","HAVOC",
]
ACTION_SIZE = len(ACTION_COLUMNS)
assert ACTION_SIZE == 47

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE          = 128
GAMMA               = 0.99
LEARNING_RATE       = 1e-4
REPLAY_SIZE         = 100_000
TARGET_SYNC         = 1000
EPSILON_START       = 1.0
EPSILON_MIN         = 0.05
ENTROPY_COEF        = 0.01
GRAD_CLIP           = 10.0
MAX_COVERAGE        = 65536.0
MAX_NEW_EDGES       = 100.0
MAX_CRASHES         = 1000.0
STEP_COST           = 0.0
PLATEAU_WINDOW      = 10_000
PLATEAU_MIN_DELTA   = 1
DEFAULT_TRAIN_STEPS = 500_000

import math

def compute_reward(cov, pcov, cr, pcr):
    ct = float(cov - pcov)
    xt = (math.log1p(cr) - math.log1p(pcr)) * 1000.0
    return ct + xt - STEP_COST, {"coverage_term": ct, "crash_term": xt}


class PlateauDetector:
    def __init__(self, grace_steps=350_000):
        self._h = deque(maxlen=PLATEAU_WINDOW); self._triggered = False
        self._grace = grace_steps
    def update(self, cov, eps, step):
        if self._triggered: return True
        self._h.append(cov)
        if step < self._grace or len(self._h) < PLATEAU_WINDOW: return False
        if eps > EPSILON_MIN + 0.01: return False
        if max(self._h) - self._h[0] <= PLATEAU_MIN_DELTA:
            self._triggered = True; return True
        return False


class ReplayBuffer:
    def __init__(self, cap): self._b = deque(maxlen=cap)
    def push(self, *t): self._b.append(t)
    def sample(self, n): return random.sample(self._b, n)
    def __len__(self): return len(self._b)


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super().__init__()
        layers = []
        prev = state_size
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_size))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, state_size, hidden_layers, label,
                 eval_mode=False, decay_steps=50_000):
        self.eval_mode = eval_mode
        self._decay_steps = max(decay_steps, 1)
        self._label = label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = DQN(state_size, ACTION_SIZE, hidden_layers).to(self.device)
        self.target = DQN(state_size, ACTION_SIZE, hidden_layers).to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE,
                                    weight_decay=1e-5)
        self.buffer = ReplayBuffer(REPLAY_SIZE)
        self.epsilon = 0.0 if eval_mode else EPSILON_START
        self._train_steps = 0; self._total_steps = 0

    def select_action(self, state):
        if not self.eval_mode and random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): return int(self.online(s).argmax(1).item())

    def _decay(self):
        f = min(self._total_steps / self._decay_steps, 1.0)
        self.epsilon = EPSILON_START + f * (EPSILON_MIN - EPSILON_START)

    def remember(self, s, a, r, ns):
        if self.eval_mode: return
        self.buffer.push(s, a, r, ns); self._total_steps += 1; self._decay()

    def train_step(self):
        if self.eval_mode or len(self.buffer) < BATCH_SIZE: return 0.0
        states, actions, rewards, nexts = zip(*self.buffer.sample(BATCH_SIZE))
        s  = torch.FloatTensor(np.array(states)).to(self.device)
        ns = torch.FloatTensor(np.array(nexts)).to(self.device)
        a  = torch.LongTensor(np.array(actions)).to(self.device)
        r  = torch.FloatTensor(np.array(rewards)).to(self.device)
        with torch.no_grad():
            best = self.online(ns).argmax(1)
            tq   = self.target(ns).gather(1, best.unsqueeze(1)).squeeze(1)
            y    = r + GAMMA * tq
        q_online = self.online(s)
        qp       = q_online.gather(1, a.unsqueeze(1)).squeeze(1)
        td_loss  = nn.functional.mse_loss(qp, y)
        probs    = torch.softmax(q_online, dim=1)
        entropy  = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
        loss     = td_loss - ENTROPY_COEF * entropy
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), GRAD_CLIP)
        self.optimizer.step(); self._train_steps += 1
        if self._train_steps % TARGET_SYNC == 0:
            self.target.load_state_dict(self.online.state_dict())
            print(f"[i] {self._label} target sync @ {self._train_steps}")
        return td_loss.item()

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
                    "train_steps": self._train_steps, "total_steps": self._total_steps}, path)
        print(f"[+] {self._label} saved \u2192 {path}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"[i] {self._label}: no checkpoint \u2014 fresh."); return
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"]); self.target.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if not self.eval_mode: self.epsilon = ck.get("epsilon", EPSILON_START)
        self._train_steps = ck.get("train_steps", 0)
        self._total_steps = ck.get("total_steps", 0)
        print(f"[+] {self._label} loaded {path}  (\u03b5={self.epsilon:.3f}, steps={self._total_steps})")


def create_shm(shm_path, shm_size):
    fd = open(shm_path, "w+b"); fd.write(b"\x00" * shm_size); fd.flush()
    shm = mmap.mmap(fd.fileno(), shm_size); fd.close(); return shm

def shm_write_action(shm, action, seq, action_off, action_seq_off):
    shm.seek(action_off); shm.write(struct.pack("=i", action))
    shm.seek(action_seq_off); shm.write(struct.pack("=I", seq)); shm.flush()
