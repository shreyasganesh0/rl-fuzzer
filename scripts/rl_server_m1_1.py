#!/usr/bin/env python3
"""
rl_server_m1_1.py  —  RL server / Model M1_1

Visited-edge stability distribution state.

Same SHM layout as M1_0, but the C side reports total_edges = num_visited
(edges where enabled[i]+disabled[i]>0) instead of MAP_SIZE=65536.
This removes the majority-zero bias in M1_0's distributions.

State (13):
  [0..11]  same as M1_0 but denominator T = num_visited (not 65536)
  [12]     num_visited_norm = log1p(num_visited) / log1p(65536)
           — tells the agent how much of the code it has reached so far

DQN:  Linear(13→128→128→64→47)
SHM:  256 B  /tmp/rl_shm_m1_1  (identical layout to M1_0)
"""

import mmap, struct, os, time, math, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from collections import deque

# ── SHM layout (same offsets as M1_0) ────────────────────────────────────────
SHM_PATH        = "/tmp/rl_shm_m1_1"
SHM_SIZE        = 256
MODEL_PATH      = os.environ.get("RL_M1_1_MODEL_PATH", "rl_m1_1.pt")

STATE_SEQ_OFF   = 0
COVERAGE_OFF    = 4
NEW_EDGES_OFF   = 8
CRASHES_OFF     = 12
TOTAL_EXECS_OFF = 24
N_NZ_EN_OFF     = 32
N_NZ_DIS_OFF    = 36
MAX_EN_OFF      = 40
MAX_DIS_OFF     = 44
SUM_EN_OFF      = 48
SUM_SQ_EN_OFF   = 56
SUM_DIS_OFF     = 64
SUM_SQ_DIS_OFF  = 72
SUM_STAB_OFF    = 80
TOTAL_EDGES_OFF = 84   # = num_visited for M1_1
STEP_COUNT_OFF  = 88
ACTION_SEQ_OFF  = 128
ACTION_OFF      = 132

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

# ── Hyperparameters ───────────────────────────────────────────────────────────
STATE_SIZE          = 13
BATCH_SIZE          = 128
GAMMA               = 0.99
LEARNING_RATE       = 1e-4
REPLAY_SIZE         = 100_000
TARGET_SYNC         = 1000
EPSILON_START       = 1.0
EPSILON_MIN         = 0.05
# Epsilon decays over 60% of train_steps so the agent has a full 40%
# exploitation phase before training ends. Set dynamically in main().
ENTROPY_COEF        = 0.01   # penalise action collapse during training
GRAD_CLIP           = 10.0
MAX_COVERAGE        = 65536.0
MAX_NEW_EDGES       = 100.0
MAX_CRASHES         = 1000.0
STEP_COST           = 0.1
PLATEAU_WINDOW      = 10_000
PLATEAU_MIN_DELTA   = 1
# PLATEAU_GRACE_STEPS is now set to 70% of train_steps dynamically in main()
DEFAULT_TRAIN_STEPS = 500_000
_LOG_MAP_SIZE       = math.log1p(65536.0)


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
    """13 → 128 → 128 → 64 → 47"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, ACTION_SIZE),
        )
    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, eval_mode=False, decay_steps=50_000):
        self.eval_mode = eval_mode
        self._decay_steps = max(decay_steps, 1)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online    = DQN().to(self.device)
        self.target    = DQN().to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.optimizer    = optim.Adam(self.online.parameters(), lr=LEARNING_RATE,
                                       weight_decay=1e-5)
        self.buffer       = ReplayBuffer(REPLAY_SIZE)
        self.epsilon      = 0.0 if eval_mode else EPSILON_START
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
            print(f"[i] M1_1 target sync @ {self._train_steps}")
        return td_loss.item()

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
                    "train_steps": self._train_steps, "total_steps": self._total_steps}, path)
        print(f"[+] M1_1 saved → {path}")

    def load(self, path):
        if not os.path.exists(path): print("[i] M1_1: no checkpoint — fresh."); return
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"]); self.target.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if not self.eval_mode: self.epsilon = ck.get("epsilon", EPSILON_START)
        self._train_steps = ck.get("train_steps", 0)
        self._total_steps = ck.get("total_steps", 0)
        print(f"[+] M1_1 loaded {path}  (ε={self.epsilon:.3f}, steps={self._total_steps})")


def create_shm():
    fd = open(SHM_PATH, "w+b"); fd.write(b"\x00" * SHM_SIZE); fd.flush()
    shm = mmap.mmap(fd.fileno(), SHM_SIZE); fd.close(); return shm

def shm_read(shm):
    shm.seek(0); raw = shm.read(SHM_SIZE)
    return {
        "state_seq":   struct.unpack_from("=I", raw, STATE_SEQ_OFF)[0],
        "coverage":    struct.unpack_from("=I", raw, COVERAGE_OFF)[0],
        "new_edges":   struct.unpack_from("=I", raw, NEW_EDGES_OFF)[0],
        "crashes":     struct.unpack_from("=I", raw, CRASHES_OFF)[0],
        "n_nz_en":     struct.unpack_from("=I", raw, N_NZ_EN_OFF)[0],
        "n_nz_dis":    struct.unpack_from("=I", raw, N_NZ_DIS_OFF)[0],
        "max_en":      struct.unpack_from("=I", raw, MAX_EN_OFF)[0],
        "max_dis":     struct.unpack_from("=I", raw, MAX_DIS_OFF)[0],
        "sum_en":      struct.unpack_from("=Q", raw, SUM_EN_OFF)[0],
        "sum_sq_en":   struct.unpack_from("=Q", raw, SUM_SQ_EN_OFF)[0],
        "sum_dis":     struct.unpack_from("=Q", raw, SUM_DIS_OFF)[0],
        "sum_sq_dis":  struct.unpack_from("=Q", raw, SUM_SQ_DIS_OFF)[0],
        "sum_stab":    struct.unpack_from("=f", raw, SUM_STAB_OFF)[0],
        "num_visited": struct.unpack_from("=I", raw, TOTAL_EDGES_OFF)[0],
    }

def shm_write_action(shm, action, seq):
    shm.seek(ACTION_OFF); shm.write(struct.pack("=i", action))
    shm.seek(ACTION_SEQ_OFF); shm.write(struct.pack("=I", seq)); shm.flush()

def build_state(d, train_steps):
    """13-element state using visited-only denominator."""
    nv  = max(float(d["num_visited"]), 1.0)
    S   = max(float(train_steps), 1.0)
    me  = d["sum_en"]  / nv;  md  = d["sum_dis"]  / nv
    ve  = max(0.0, d["sum_sq_en"]  / nv - me ** 2)
    vd  = max(0.0, d["sum_sq_dis"] / nv - md ** 2)
    return np.array([
        d["coverage"]  / MAX_COVERAGE,
        min(d["new_edges"], MAX_NEW_EDGES) / MAX_NEW_EDGES,
        math.log1p(d["crashes"]) / math.log1p(MAX_CRASHES),
        me               / S,
        math.sqrt(ve)    / S,
        d["max_en"]      / S,
        d["n_nz_en"]     / nv,
        md               / S,
        math.sqrt(vd)    / S,
        d["max_dis"]     / S,
        d["n_nz_dis"]    / nv,
        d["sum_stab"]    / nv,
        math.log1p(float(d["num_visited"])) / _LOG_MAP_SIZE,  # [12]
    ], dtype=np.float32)

def compute_reward(cov, pcov, cr, pcr):
    ct = (cov / MAX_COVERAGE - pcov / MAX_COVERAGE) * 100.0
    xt = (math.log1p(cr) - math.log1p(pcr)) * 1000.0
    return ct + xt - STEP_COST, {"coverage_term": ct, "crash_term": xt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",        choices=["train","eval"], default="train")
    ap.add_argument("--model",       default=MODEL_PATH)
    ap.add_argument("--eval-steps",  type=int, default=20_000)
    ap.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS,
                    help="Step limit AND MAX_COUNT normaliser")
    ap.add_argument("--results-dir", default=".")
    args = ap.parse_args()

    is_eval = args.mode == "eval"
    tag     = "EVAL" if is_eval else "TRAIN"
    os.makedirs(args.results_dir, exist_ok=True)
    mpath = os.path.join(args.results_dir,
                         f"rl_metrics_m1_1_{'eval' if is_eval else 'train'}.csv")

    print(f"[+] M1_1  mode={tag}  state={STATE_SIZE}  actions={ACTION_SIZE}  "
          f"train_steps/MAX_COUNT={args.train_steps}")

    shm = create_shm(); agent = DQNAgent(is_eval, decay_steps=int(args.train_steps * 0.6)); agent.load(args.model)
    plateau = PlateauDetector(grace_steps=int(args.train_steps * 0.7)) if not is_eval else None
    shm_write_action(shm, 46, 1)

    zero_d = {"coverage":0,"new_edges":0,"crashes":0,"n_nz_en":0,"n_nz_dis":0,
              "max_en":0,"max_dis":0,"sum_en":0,"sum_sq_en":0,"sum_dis":0,
              "sum_sq_dis":0,"sum_stab":0.0,"num_visited":1}
    with open(mpath, "w") as f:
        f.write("step,reward,coverage_term,crash_term,loss,epsilon,"
                "coverage,crashes,action,num_visited,stability\n")

    print(f"[+] M1_1 ready.  metrics={mpath}")

    pcov=0; pcr=0; pact=46; pstate=build_state(zero_d, args.train_steps)
    last_seq=0; aseq=1; step=0; cov_start=None
    stop="user interrupt"; cov=0; cr=0

    try:
        while True:
            if is_eval and step >= args.eval_steps:      stop="eval limit";  break
            if not is_eval and step >= args.train_steps: stop="train limit"; break

            while True:
                shm.seek(STATE_SEQ_OFF); cur=struct.unpack("=I",shm.read(4))[0]
                if cur != last_seq: break
                time.sleep(0.0001)

            last_seq=cur; d=shm_read(shm)
            cov=d["coverage"]; ne=d["new_edges"]; cr=d["crashes"]
            nv=d["num_visited"]
            if is_eval and cov_start is None: cov_start=cov

            state=build_state(d, args.train_steps)
            rew, comps = compute_reward(cov, pcov, cr, pcr)
            loss=0.0
            if step > 0:
                agent.remember(pstate, pact, rew, state); loss=agent.train_step()

            act=agent.select_action(state); aseq+=1; shm_write_action(shm, act, aseq)

            if step % 100 == 0:
                nv_f = max(float(nv), 1.0)
                stb  = d["sum_stab"] / nv_f
                with open(mpath,"a") as f:
                    f.write(f"{step},{rew:.4f},{comps['coverage_term']:.4f},"
                            f"{comps['crash_term']:.4f},{loss:.6f},{agent.epsilon:.4f},"
                            f"{cov},{cr},{act},{nv},{stb:.4f}\n")
                print(f"[M1_1-{tag}:{step:>7}] cov={cov:<5} ne={ne:<3} cr={cr} "
                      f"act={act:<2}({ACTION_COLUMNS[act]:<30}) "
                      f"ε={agent.epsilon:.3f} r={rew:+.2f} loss={loss:.5f} "
                      f"visited={nv} stab={stb:.3f}")

            if not is_eval and step>0 and step%1000==0: agent.save(args.model)
            if not is_eval and plateau and plateau.update(cov, agent.epsilon, step):
                stop="coverage plateau"; break

            pstate=state; pact=act; pcov=cov; pcr=cr; step+=1

    except KeyboardInterrupt: stop="user interrupt"
    finally:
        if not is_eval: agent.save(args.model)
        print(f"\n{'='*58}\n  M1_1 {tag} done — {stop}")
        print(f"  steps={step:,}  cov={cov}  crashes={cr}  ε={agent.epsilon:.4f}")
        print(f"{'='*58}\n")

if __name__ == "__main__": main()
