"""Model M1_0 — full edge stability distribution state (12 dims)."""

import struct, math
import numpy as np
from .common import MAX_COVERAGE, MAX_NEW_EDGES, MAX_CRASHES

STATE_SIZE      = 12
SHM_SIZE        = 256
SHM_PATH        = "/tmp/rl_shm_m1_0"
MODEL_PATH_DEFAULT = "rl_m1_0.pt"
LABEL           = "M1_0"
HIDDEN_LAYERS   = [128, 128, 64]

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
TOTAL_EDGES_OFF = 84
STEP_COUNT_OFF  = 88
ACTION_SEQ_OFF  = 128
ACTION_OFF      = 132

CSV_EXTRA_HEADER = ",en_mean_n,dis_mean_n,stability"


def shm_read(shm, shm_size):
    shm.seek(0); raw = shm.read(shm_size)
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
        "total_edges": struct.unpack_from("=I", raw, TOTAL_EDGES_OFF)[0],
    }


def build_state(d, train_steps):
    T = max(float(d["total_edges"]), 1.0)
    S = max(float(train_steps), 1.0)
    me = d["sum_en"] / T;  md = d["sum_dis"] / T
    ve = max(0.0, d["sum_sq_en"] / T - me ** 2)
    vd = max(0.0, d["sum_sq_dis"] / T - md ** 2)
    return np.array([
        d["coverage"]  / MAX_COVERAGE,
        min(d["new_edges"], MAX_NEW_EDGES) / MAX_NEW_EDGES,
        math.log1p(d["crashes"]) / math.log1p(MAX_CRASHES),
        me            / S,
        math.sqrt(ve) / S,
        d["max_en"]   / S,
        d["n_nz_en"]  / T,
        md            / S,
        math.sqrt(vd) / S,
        d["max_dis"]  / S,
        d["n_nz_dis"] / T,
        d["sum_stab"] / T,
    ], dtype=np.float32)


def zero_state_data():
    return {"coverage": 0, "new_edges": 0, "crashes": 0,
            "n_nz_en": 0, "n_nz_dis": 0, "max_en": 0, "max_dis": 0,
            "sum_en": 0, "sum_sq_en": 0, "sum_dis": 0, "sum_sq_dis": 0,
            "sum_stab": 0.0, "total_edges": 1}


def csv_extra_fields(d, args):
    T   = max(float(d["total_edges"]), 1.0)
    S   = max(float(args.train_steps), 1.0)
    enm = d["sum_en"]  / T / S
    dim = d["sum_dis"] / T / S
    stb = d["sum_stab"] / T
    return f",{enm:.4f},{dim:.4f},{stb:.4f}"


def log_extra(d, args):
    T   = max(float(d["total_edges"]), 1.0)
    S   = max(float(args.train_steps), 1.0)
    enm = d["sum_en"]  / T / S
    dim = d["sum_dis"] / T / S
    stb = d["sum_stab"] / T
    return f"en={enm:.3f} dis={dim:.3f} stab={stb:.3f}"


def exit_summary(d, step, cov, cr, epsilon, tag):
    pass
