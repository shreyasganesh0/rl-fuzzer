"""Model M1_2 — visited-edge stability + input buffer features (64 dims)."""

import struct, math
import numpy as np
from .common import MAX_COVERAGE, MAX_NEW_EDGES, MAX_CRASHES

STATE_SIZE      = 64
SHM_SIZE        = 512
SHM_PATH        = "/tmp/rl_shm_m1_2"
MODEL_PATH_DEFAULT = "rl_m1_2.pt"
LABEL           = "M1_2"
HIDDEN_LAYERS   = [256, 256, 128]

_LOG_MAP_SIZE   = math.log1p(65536.0)
_LOG_1MB        = math.log1p(1024 * 1024)

# ── SHM offsets (state region) ───────────────────────────────────────────────
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
TOTAL_EDGES_OFF = 84   # = num_visited
STEP_COUNT_OFF  = 88

# Buffer feature offsets
BUF_LEN_OFF       = 92
ENTROPY_OFF        = 96
PRINTABLE_OFF      = 100
HISTOGRAM_OFF      = 104   # u32[16]
FIRST_BYTES_OFF    = 168   # u8[32]

# Action region (at 256 for 512-byte SHM)
ACTION_SEQ_OFF  = 256
ACTION_OFF      = 260

CSV_EXTRA_HEADER = ",num_visited,stability,buf_len,entropy,printable_ratio"


def shm_read(shm, shm_size):
    shm.seek(0); raw = shm.read(shm_size)
    hist = struct.unpack_from("=16I", raw, HISTOGRAM_OFF)
    first_bytes = struct.unpack_from("=32B", raw, FIRST_BYTES_OFF)
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
        "buf_len":     struct.unpack_from("=I", raw, BUF_LEN_OFF)[0],
        "entropy":     struct.unpack_from("=f", raw, ENTROPY_OFF)[0],
        "printable":   struct.unpack_from("=f", raw, PRINTABLE_OFF)[0],
        "histogram":   hist,
        "first_bytes": first_bytes,
    }


def build_state(d, train_steps):
    nv = max(float(d["num_visited"]), 1.0)
    S  = max(float(train_steps), 1.0)
    me = d["sum_en"] / nv;  md = d["sum_dis"] / nv
    ve = max(0.0, d["sum_sq_en"] / nv - me ** 2)
    vd = max(0.0, d["sum_sq_dis"] / nv - md ** 2)

    # 13 M1_1 dims
    m1_1_dims = [
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
    ]

    # Buffer features (51 dims)
    buf_len = float(d["buf_len"])
    buf_len_norm = math.log1p(buf_len) / _LOG_1MB
    entropy_norm = d["entropy"] / 8.0
    printable_ratio = d["printable"]

    hist = d["histogram"]
    buf_total = max(buf_len, 1.0)
    hist_norm = [float(h) / buf_total for h in hist]  # 16 dims

    first_bytes_norm = [float(b) / 255.0 for b in d["first_bytes"]]  # 32 dims

    buf_dims = [buf_len_norm, entropy_norm, printable_ratio] + hist_norm + first_bytes_norm

    return np.array(m1_1_dims + buf_dims, dtype=np.float32)


def zero_state_data():
    return {"coverage": 0, "new_edges": 0, "crashes": 0,
            "n_nz_en": 0, "n_nz_dis": 0, "max_en": 0, "max_dis": 0,
            "sum_en": 0, "sum_sq_en": 0, "sum_dis": 0, "sum_sq_dis": 0,
            "sum_stab": 0.0, "num_visited": 1,
            "buf_len": 0, "entropy": 0.0, "printable": 0.0,
            "histogram": (0,) * 16, "first_bytes": (0,) * 32}


def csv_extra_fields(d, args):
    nv  = d["num_visited"]
    nv_f = max(float(nv), 1.0)
    stb = d["sum_stab"] / nv_f
    return f",{nv},{stb:.4f},{d['buf_len']},{d['entropy']:.4f},{d['printable']:.4f}"


def log_extra(d, args):
    nv  = d["num_visited"]
    nv_f = max(float(nv), 1.0)
    stb = d["sum_stab"] / nv_f
    return f"visited={nv} stab={stb:.3f} buflen={d['buf_len']} ent={d['entropy']:.2f}"


def exit_summary(d, step, cov, cr, epsilon, tag):
    pass
