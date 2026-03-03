"""Model M1_1 — visited-edge stability distribution state (13 dims)."""

import struct, math
import numpy as np
from .common import MAX_COVERAGE, MAX_NEW_EDGES, MAX_CRASHES

STATE_SIZE      = 13
SHM_SIZE        = 256
SHM_PATH        = "/tmp/rl_shm_m1_1"
MODEL_PATH_DEFAULT = "rl_m1_1.pt"
LABEL           = "M1_1"
HIDDEN_LAYERS   = [128, 128, 64]

_LOG_MAP_SIZE   = math.log1p(65536.0)

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

CSV_EXTRA_HEADER = ",num_visited,stability"


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
        "num_visited": struct.unpack_from("=I", raw, TOTAL_EDGES_OFF)[0],
    }


def build_state(d, train_steps):
    nv = max(float(d["num_visited"]), 1.0)
    S  = max(float(train_steps), 1.0)
    me = d["sum_en"] / nv;  md = d["sum_dis"] / nv
    ve = max(0.0, d["sum_sq_en"] / nv - me ** 2)
    vd = max(0.0, d["sum_sq_dis"] / nv - md ** 2)
    return np.array([
        d["coverage"]  / MAX_COVERAGE,
        min(d["new_edges"], MAX_NEW_EDGES) / MAX_NEW_EDGES,
        math.log1p(d["crashes"]) / math.log1p(MAX_CRASHES),
        me            / S,
        math.sqrt(ve) / S,
        d["max_en"]   / S,
        d["n_nz_en"]  / nv,
        md            / S,
        math.sqrt(vd) / S,
        d["max_dis"]  / S,
        d["n_nz_dis"] / nv,
        d["sum_stab"] / nv,
        math.log1p(float(d["num_visited"])) / _LOG_MAP_SIZE,
    ], dtype=np.float32)


def zero_state_data():
    return {"coverage": 0, "new_edges": 0, "crashes": 0,
            "n_nz_en": 0, "n_nz_dis": 0, "max_en": 0, "max_dis": 0,
            "sum_en": 0, "sum_sq_en": 0, "sum_dis": 0, "sum_sq_dis": 0,
            "sum_stab": 0.0, "num_visited": 1}


def csv_extra_fields(d, args):
    nv  = d["num_visited"]
    nv_f = max(float(nv), 1.0)
    stb = d["sum_stab"] / nv_f
    return f",{nv},{stb:.4f}"


def log_extra(d, args):
    nv  = d["num_visited"]
    nv_f = max(float(nv), 1.0)
    stb = d["sum_stab"] / nv_f
    return f"visited={nv} stab={stb:.3f}"


def exit_summary(d, step, cov, cr, epsilon, tag):
    pass
