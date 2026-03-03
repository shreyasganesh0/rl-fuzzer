"""Model M2 — per-mutator trace-bit magnitude state (97 dims)."""

import struct, math
import numpy as np
from .common import ACTION_SIZE, ACTION_COLUMNS, MAX_COVERAGE, MAX_NEW_EDGES, MAX_CRASHES

STATE_SIZE      = 97   # 3 + 47 + 47
SHM_SIZE        = 1024
SHM_PATH        = "/tmp/rl_shm_m2"
MODEL_PATH_DEFAULT = "rl_m2.pt"
LABEL           = "M2"
HIDDEN_LAYERS   = [256, 256, 128]

STATE_SEQ_OFF   = 0
COVERAGE_OFF    = 4
NEW_EDGES_OFF   = 8
CRASHES_OFF     = 12
TOTAL_EXECS_OFF = 24
AVG_EN_OFF      = 32    # f32[47]  188 bytes
AVG_DIS_OFF     = 220   # f32[47]  188 bytes
ACTION_SEQ_OFF  = 512
ACTION_OFF      = 516

CSV_EXTRA_HEADER = ",mean_avg_en,mean_avg_dis,top_en_action,top_dis_action,nonzero_mag_frac"


def shm_read(shm, shm_size):
    shm.seek(0); raw = shm.read(shm_size)
    return {
        "state_seq":  struct.unpack_from("=I", raw, STATE_SEQ_OFF)[0],
        "coverage":   struct.unpack_from("=I", raw, COVERAGE_OFF)[0],
        "new_edges":  struct.unpack_from("=I", raw, NEW_EDGES_OFF)[0],
        "crashes":    struct.unpack_from("=I", raw, CRASHES_OFF)[0],
        "avg_en":     list(struct.unpack_from(f"={ACTION_SIZE}f", raw, AVG_EN_OFF)),
        "avg_dis":    list(struct.unpack_from(f"={ACTION_SIZE}f", raw, AVG_DIS_OFF)),
    }


def build_state(d, train_steps):
    base = [
        d["coverage"] / MAX_COVERAGE,
        min(d["new_edges"], MAX_NEW_EDGES) / MAX_NEW_EDGES,
        math.log1p(d["crashes"]) / math.log1p(MAX_CRASHES),
    ]
    en_n  = [min(max(v, 0.0), 1.0) for v in d["avg_en"]]
    dis_n = [min(max(v, 0.0), 1.0) for v in d["avg_dis"]]
    return np.array(base + en_n + dis_n, dtype=np.float32)


def zero_state_data():
    return {"coverage": 0, "new_edges": 0, "crashes": 0,
            "avg_en": [0.0] * ACTION_SIZE, "avg_dis": [0.0] * ACTION_SIZE}


def csv_extra_fields(d, args):
    avg_en  = d["avg_en"]
    avg_dis = d["avg_dis"]
    mean_en  = sum(avg_en)  / ACTION_SIZE
    mean_dis = sum(avg_dis) / ACTION_SIZE
    top_en   = max(range(ACTION_SIZE), key=lambda i: avg_en[i])
    top_dis  = max(range(ACTION_SIZE), key=lambda i: avg_dis[i])
    nz_frac  = sum(1 for v in avg_en if v > 1e-4) / ACTION_SIZE
    return f",{mean_en:.4f},{mean_dis:.4f},{top_en},{top_dis},{nz_frac:.4f}"


def log_extra(d, args):
    avg_en  = d["avg_en"]
    avg_dis = d["avg_dis"]
    mean_en  = sum(avg_en)  / ACTION_SIZE
    mean_dis = sum(avg_dis) / ACTION_SIZE
    nz_frac  = sum(1 for v in avg_en if v > 1e-4) / ACTION_SIZE
    top_en   = max(range(ACTION_SIZE), key=lambda i: avg_en[i])
    top_dis  = max(range(ACTION_SIZE), key=lambda i: avg_dis[i])
    return (f"\u03a3en={mean_en:.3f} \u03a3dis={mean_dis:.3f} "
            f"nz={nz_frac:.2f} "
            f"top_en={ACTION_COLUMNS[top_en][:16]} "
            f"top_dis={ACTION_COLUMNS[top_dis][:16]}")


def exit_summary(d, step, cov, cr, epsilon, tag):
    avg_en  = d.get("avg_en", [])
    avg_dis = d.get("avg_dis", [])
    if avg_en and any(v > 0 for v in avg_en):
        final_nz = sum(1 for v in avg_en if v > 1e-4) / ACTION_SIZE
        print(f"  nonzero_mag_frac={final_nz:.3f}  "
              f"({int(final_nz*ACTION_SIZE)}/{ACTION_SIZE} mutator slots populated)")
        print("\n  Top-5 highest avg enabled magnitude:")
        ranked = sorted(range(ACTION_SIZE), key=lambda i: avg_en[i], reverse=True)
        for i in ranked[:5]:
            print(f"    [{i:>2}] {ACTION_COLUMNS[i]:<34} en={avg_en[i]:.4f}  dis={avg_dis[i]:.4f}")
