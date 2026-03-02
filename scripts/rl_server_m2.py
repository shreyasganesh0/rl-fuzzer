#!/usr/bin/env python3
"""
rl_server_m2.py  —  RL server / Model M2

Per-mutator trace-bit magnitude state.

For each of the 47 actions the C mutator maintains running sums:
  enabled_mag[a]  = total trace bits active across all calls to action a
  disabled_mag[a] = total trace bits that dropped out after action a
  calls[a]        = times action a was invoked

The C side writes avg_enabled_mag[a]  = enabled_mag[a]  / calls[a] / 65536
                  avg_disabled_mag[a] = disabled_mag[a] / calls[a] / 65536
already normalised to [0,1].

State (97):
  [0]      coverage_n               = coverage / 65536
  [1]      new_edges_n              = min(new_edges,100) / 100
  [2]      crashes_n                = log1p(crashes) / log1p(1000)
  [3..49]  avg_enabled_mag_n[47]    (pre-normalised by C)
  [50..96] avg_disabled_mag_n[47]   (pre-normalised by C)

DQN:  Linear(97→256→256→128→47)  (wider net for larger state)
SHM:  1024 B  /tmp/rl_shm_m2
"""

import mmap, struct, os, time, math, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from collections import deque

# ── SHM layout ────────────────────────────────────────────────────────────────
SHM_PATH        = "/tmp/rl_shm_m2"
SHM_SIZE        = 1024
MODEL_PATH      = os.environ.get("RL_M2_MODEL_PATH", "rl_m2.pt")

STATE_SEQ_OFF   = 0
COVERAGE_OFF    = 4
NEW_EDGES_OFF   = 8
CRASHES_OFF     = 12
TOTAL_EXECS_OFF = 24
AVG_EN_OFF      = 32    # f32[47]  188 bytes
AVG_DIS_OFF     = 220   # f32[47]  188 bytes
ACTION_SEQ_OFF  = 512
ACTION_OFF      = 516

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
STATE_SIZE          = 97   # 3 + 47 + 47
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
    """97 → 256 → 256 → 128 → 47  (wider to handle 94 magnitude features)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, ACTION_SIZE),
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
            print(f"[i] M2 target sync @ {self._train_steps}")
        return td_loss.item()

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
                    "train_steps": self._train_steps, "total_steps": self._total_steps}, path)
        print(f"[+] M2 saved → {path}")

    def load(self, path):
        if not os.path.exists(path): print("[i] M2: no checkpoint — fresh."); return
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"]); self.target.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if not self.eval_mode: self.epsilon = ck.get("epsilon", EPSILON_START)
        self._train_steps = ck.get("train_steps", 0)
        self._total_steps = ck.get("total_steps", 0)
        print(f"[+] M2 loaded {path}  (ε={self.epsilon:.3f}, steps={self._total_steps})")


def create_shm():
    fd = open(SHM_PATH, "w+b"); fd.write(b"\x00" * SHM_SIZE); fd.flush()
    shm = mmap.mmap(fd.fileno(), SHM_SIZE); fd.close(); return shm

def shm_read(shm):
    shm.seek(0); raw = shm.read(SHM_SIZE)
    coverage    = struct.unpack_from("=I", raw, COVERAGE_OFF)[0]
    new_edges   = struct.unpack_from("=I", raw, NEW_EDGES_OFF)[0]
    crashes     = struct.unpack_from("=I", raw, CRASHES_OFF)[0]
    state_seq   = struct.unpack_from("=I", raw, STATE_SEQ_OFF)[0]
    # 47 floats each for enabled and disabled average magnitudes
    avg_en  = list(struct.unpack_from(f"={ACTION_SIZE}f", raw, AVG_EN_OFF))
    avg_dis = list(struct.unpack_from(f"={ACTION_SIZE}f", raw, AVG_DIS_OFF))
    return state_seq, coverage, new_edges, crashes, avg_en, avg_dis

def shm_write_action(shm, action, seq):
    shm.seek(ACTION_OFF); shm.write(struct.pack("=i", action))
    shm.seek(ACTION_SEQ_OFF); shm.write(struct.pack("=I", seq)); shm.flush()

def build_state(cov, ne, cr, avg_en, avg_dis):
    """97-element state: 3 base metrics + 47 enabled mags + 47 disabled mags."""
    base = [
        cov / MAX_COVERAGE,
        min(ne, MAX_NEW_EDGES) / MAX_NEW_EDGES,
        math.log1p(cr) / math.log1p(MAX_CRASHES),
    ]
    # avg_en / avg_dis are already normalised by C (divided by MAP_SIZE)
    # clip to [0,1] as a safety net against floating point drift
    en_n  = [min(max(v, 0.0), 1.0) for v in avg_en]
    dis_n = [min(max(v, 0.0), 1.0) for v in avg_dis]
    return np.array(base + en_n + dis_n, dtype=np.float32)

def compute_reward(cov, pcov, cr, pcr):
    ct = (cov / MAX_COVERAGE - pcov / MAX_COVERAGE) * 100.0
    xt = (math.log1p(cr) - math.log1p(pcr)) * 1000.0
    return ct + xt - STEP_COST, {"coverage_term": ct, "crash_term": xt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",        choices=["train","eval"], default="train")
    ap.add_argument("--model",       default=MODEL_PATH)
    ap.add_argument("--eval-steps",  type=int, default=20_000)
    ap.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    ap.add_argument("--results-dir", default=".")
    args = ap.parse_args()

    is_eval = args.mode == "eval"
    tag     = "EVAL" if is_eval else "TRAIN"
    os.makedirs(args.results_dir, exist_ok=True)
    mpath = os.path.join(args.results_dir,
                         f"rl_metrics_m2_{'eval' if is_eval else 'train'}.csv")

    print(f"[+] M2  mode={tag}  state={STATE_SIZE}  actions={ACTION_SIZE}  "
          f"train_steps={args.train_steps}")
    print(f"[+] M2  net: 97→256→256→128→47")

    shm = create_shm(); agent = DQNAgent(is_eval, decay_steps=int(args.train_steps * 0.6)); agent.load(args.model)
    plateau = PlateauDetector(grace_steps=int(args.train_steps * 0.7)) if not is_eval else None
    shm_write_action(shm, 46, 1)

    zero_en  = [0.0] * ACTION_SIZE
    zero_dis = [0.0] * ACTION_SIZE
    with open(mpath, "w") as f:
        f.write("step,reward,coverage_term,crash_term,loss,epsilon,"
                "coverage,crashes,action,"
                "mean_avg_en,mean_avg_dis,top_en_action,top_dis_action\n")

    print(f"[+] M2 ready.  metrics={mpath}")

    pcov=0; pcr=0; pact=46
    pstate=build_state(0, 0, 0, zero_en, zero_dis)
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

            last_seq=cur
            _, cov, ne, cr, avg_en, avg_dis = shm_read(shm)
            if is_eval and cov_start is None: cov_start=cov

            state=build_state(cov, ne, cr, avg_en, avg_dis)
            rew, comps = compute_reward(cov, pcov, cr, pcr)
            loss=0.0
            if step > 0:
                agent.remember(pstate, pact, rew, state); loss=agent.train_step()

            act=agent.select_action(state); aseq+=1; shm_write_action(shm, act, aseq)

            if step % 100 == 0:
                mean_en  = sum(avg_en)  / ACTION_SIZE
                mean_dis = sum(avg_dis) / ACTION_SIZE
                top_en   = max(range(ACTION_SIZE), key=lambda i: avg_en[i])
                top_dis  = max(range(ACTION_SIZE), key=lambda i: avg_dis[i])
                with open(mpath,"a") as f:
                    f.write(f"{step},{rew:.4f},{comps['coverage_term']:.4f},"
                            f"{comps['crash_term']:.4f},{loss:.6f},{agent.epsilon:.4f},"
                            f"{cov},{cr},{act},"
                            f"{mean_en:.4f},{mean_dis:.4f},{top_en},{top_dis}\n")
                print(f"[M2-{tag}:{step:>7}] cov={cov:<5} ne={ne:<3} cr={cr} "
                      f"act={act:<2}({ACTION_COLUMNS[act]:<30}) "
                      f"ε={agent.epsilon:.3f} r={rew:+.2f} loss={loss:.5f} "
                      f"Σen={mean_en:.3f} Σdis={mean_dis:.3f} "
                      f"top_en={ACTION_COLUMNS[top_en][:16]} "
                      f"top_dis={ACTION_COLUMNS[top_dis][:16]}")

            if not is_eval and step>0 and step%1000==0: agent.save(args.model)
            if not is_eval and plateau and plateau.update(cov, agent.epsilon, step):
                stop="coverage plateau"; break

            pstate=state; pact=act; pcov=cov; pcr=cr; step+=1

    except KeyboardInterrupt: stop="user interrupt"
    finally:
        if not is_eval: agent.save(args.model)
        # Print per-action magnitude summary on exit
        print(f"\n{'='*58}\n  M2 {tag} done — {stop}")
        print(f"  steps={step:,}  cov={cov}  crashes={cr}  ε={agent.epsilon:.4f}")
        if avg_en and any(v > 0 for v in avg_en):
            print("\n  Top-5 highest avg enabled magnitude:")
            ranked = sorted(range(ACTION_SIZE), key=lambda i: avg_en[i], reverse=True)
            for i in ranked[:5]:
                print(f"    [{i:>2}] {ACTION_COLUMNS[i]:<34} en={avg_en[i]:.4f}  dis={avg_dis[i]:.4f}")
        print(f"{'='*58}\n")

if __name__ == "__main__": main()
