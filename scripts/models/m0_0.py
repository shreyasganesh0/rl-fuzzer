"""Model M0_0 — basic 3-dim state: [coverage_n, new_edges_n, crashes_n]."""

import struct, math
import numpy as np
from .common import MAX_COVERAGE, MAX_NEW_EDGES, MAX_CRASHES

STATE_SIZE      = 3
SHM_SIZE        = 128
SHM_PATH        = "/tmp/rl_shm_m0_0"
MODEL_PATH_DEFAULT = "rl_m0_0.pt"
LABEL           = "M0_0"
HIDDEN_LAYERS   = [128, 128, 64]

STATE_SEQ_OFF   = 0
COVERAGE_OFF    = 4
NEW_EDGES_OFF   = 8
CRASHES_OFF     = 12
TOTAL_EXECS_OFF = 24
ACTION_SEQ_OFF  = 64
ACTION_OFF      = 68

CSV_EXTRA_HEADER = ""


def shm_read(shm, shm_size):
    shm.seek(0); raw = shm.read(shm_size)
    return {
        "state_seq":   struct.unpack_from("=I", raw, STATE_SEQ_OFF)[0],
        "coverage":    struct.unpack_from("=I", raw, COVERAGE_OFF)[0],
        "new_edges":   struct.unpack_from("=I", raw, NEW_EDGES_OFF)[0],
        "crashes":     struct.unpack_from("=I", raw, CRASHES_OFF)[0],
        "total_execs": struct.unpack_from("=Q", raw, TOTAL_EXECS_OFF)[0],
    }


def build_state(d, train_steps):
    return np.array([
        d["coverage"] / MAX_COVERAGE,
        min(d["new_edges"], MAX_NEW_EDGES) / MAX_NEW_EDGES,
        math.log1p(d["crashes"]) / math.log1p(MAX_CRASHES),
    ], dtype=np.float32)


def zero_state_data():
    return {"coverage": 0, "new_edges": 0, "crashes": 0}


def csv_extra_fields(d, args):
    return ""


def log_extra(d, args):
    return ""


def exit_summary(d, step, cov, cr, epsilon, tag):
    pass
