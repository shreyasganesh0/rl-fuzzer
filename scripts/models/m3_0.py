"""Model M3_0 — Differential-informed coverage distribution features (13-dim).

State vector derived from differential analysis of buggy vs fixed libxml2.
Features capture edge heat distribution, entropy, execution time, and
coverage velocity — structural properties that generalise across targets.

SHM layout (128 bytes) must match src/mutator_m3_0.c.
"""

import struct, math
import numpy as np

STATE_SIZE      = 13
SHM_SIZE        = 128
SHM_PATH        = "/tmp/rl_shm_m3_0"
MODEL_PATH_DEFAULT = "rl_m3_0.pt"
LABEL           = "M3_0"
HIDDEN_LAYERS   = [128, 128, 64]

# SHM offsets — must match mutator_m3_0.c
STATE_SEQ_OFF    = 0
TOTAL_EDGES_OFF  = 4
COLD_EDGES_OFF   = 8
HOT_EDGES_OFF    = 12
WARM_EDGES_OFF   = 16
COOL_EDGES_OFF   = 20
ENTROPY_OFF      = 24
HIT_MEAN_OFF     = 28
HIT_STD_OFF      = 32
CORPUS_SIZE_OFF  = 36
CRASHES_OFF      = 40
NEW_EDGES_OFF    = 44
EXEC_TIME_OFF    = 48
VELOCITY_OFF     = 52
ACTION_SEQ_OFF   = 64
ACTION_OFF       = 68

MAP_SIZE = 65536.0

CSV_EXTRA_HEADER = (",cold_edges,hot_edges,warm_edges,cool_edges,"
                    "entropy,hit_mean,hit_std,corpus_size,exec_time,velocity")


def shm_read(shm, shm_size):
    shm.seek(0); raw = shm.read(shm_size)
    return {
        "state_seq":    struct.unpack_from("=I", raw, STATE_SEQ_OFF)[0],
        "coverage":     struct.unpack_from("=I", raw, TOTAL_EDGES_OFF)[0],
        "total_edges":  struct.unpack_from("=I", raw, TOTAL_EDGES_OFF)[0],
        "cold_edges":   struct.unpack_from("=I", raw, COLD_EDGES_OFF)[0],
        "hot_edges":    struct.unpack_from("=I", raw, HOT_EDGES_OFF)[0],
        "warm_edges":   struct.unpack_from("=I", raw, WARM_EDGES_OFF)[0],
        "cool_edges":   struct.unpack_from("=I", raw, COOL_EDGES_OFF)[0],
        "entropy":      struct.unpack_from("=f", raw, ENTROPY_OFF)[0],
        "hit_mean":     struct.unpack_from("=f", raw, HIT_MEAN_OFF)[0],
        "hit_std":      struct.unpack_from("=f", raw, HIT_STD_OFF)[0],
        "corpus_size":  struct.unpack_from("=I", raw, CORPUS_SIZE_OFF)[0],
        "crashes":      struct.unpack_from("=I", raw, CRASHES_OFF)[0],
        "new_edges":    struct.unpack_from("=I", raw, NEW_EDGES_OFF)[0],
        "exec_time":    struct.unpack_from("=f", raw, EXEC_TIME_OFF)[0],
        "velocity":     struct.unpack_from("=f", raw, VELOCITY_OFF)[0],
    }


def build_state(d, train_steps):
    te = max(float(d["total_edges"]), 1.0)
    return np.array([
        d["total_edges"] / MAP_SIZE,                              # 0: total_edges
        d["cold_edges"] / MAP_SIZE,                               # 1: cold_edges
        d["hot_edges"] / te,                                      # 2: hot ratio
        d["warm_edges"] / te,                                     # 3: warm ratio
        d["cool_edges"] / te,                                     # 4: cool ratio
        d["entropy"],                                             # 5: pre-normed in C
        d["hit_mean"],                                            # 6: pre-normed in C
        d["hit_std"],                                             # 7: pre-normed in C
        math.log1p(float(d["corpus_size"])) / math.log1p(10000), # 8: corpus_size
        math.log1p(float(d["crashes"])) / math.log1p(1000),      # 9: crashes
        min(float(d["new_edges"]), 100.0) / 100.0,               # 10: new_edges
        d["exec_time"],                                           # 11: pre-normed in C
        d["velocity"],                                            # 12: pre-normed in C
    ], dtype=np.float32)


def zero_state_data():
    return {"coverage": 0, "total_edges": 0, "cold_edges": 65536, "hot_edges": 0,
            "warm_edges": 0, "cool_edges": 0, "entropy": 0.0,
            "hit_mean": 0.0, "hit_std": 0.0, "corpus_size": 0,
            "crashes": 0, "new_edges": 0, "exec_time": 0.0,
            "velocity": 0.0}


def csv_extra_fields(d, args):
    return (f",{d['cold_edges']},{d['hot_edges']},{d['warm_edges']},"
            f"{d['cool_edges']},{d['entropy']:.4f},{d['hit_mean']:.4f},"
            f"{d['hit_std']:.4f},{d['corpus_size']},{d['exec_time']:.4f},"
            f"{d['velocity']:.4f}")


def log_extra(d, args):
    return (f"hot={d['hot_edges']} warm={d['warm_edges']} "
            f"cool={d['cool_edges']} ent={d['entropy']:.3f} "
            f"vel={d['velocity']:.4f}")


def exit_summary(d, step, cov, cr, epsilon, tag):
    pass
