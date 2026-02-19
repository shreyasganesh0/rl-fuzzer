/*
 * mutator.c — MuoFuzz custom AFL++ mutator
 * [sysrel]
 *
 * IPC: Shared memory file at SHM_PATH — no Unix sockets.
 *
 * SHM layout (128 bytes):
 *
 *   Offset  0..63  — STATE REGION (we write, Python reads)
 *     [0]   state_seq    uint32_t  — sequence sentinel; release-stored LAST
 *     [4]   edge_id      uint32_t  — most recently new/hot edge from trace_bits
 *     [8]   coverage     uint32_t  — total bitmap coverage count
 *     [12]  new_edges    uint32_t  — delta new edges since last step
 *     [16]  crashes      uint32_t
 *     [20]  _pad         uint32_t
 *     [24]  total_execs  uint64_t
 *     [32]  _pad[32]     padding / false-share guard
 *
 *   Offset 64..127 — ACTION REGION (Python writes, we read)
 *     [64]  action_seq   uint32_t  — sentinel; Python writes this LAST
 *     [68]  action       int32_t   — chosen action (0..6)
 *     [72]  _pad[56]
 *
 * Synchronisation (no semaphore):
 *   After writing all state fields, C does an __ATOMIC_RELEASE store of state_seq.
 *   Python polls state_seq; when it changes, it reads state and writes the action
 *   followed by an __ATOMIC_RELEASE (via GIL + mmap flush) store of action_seq.
 *   C polls action_seq with __ATOMIC_ACQUIRE until it changes, then reads action.
 *   This is safe on both x86 (TSO) and aarch64 (explicit barrier via GCC builtins).
 */

#include "afl-fuzz.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>

/* ── Constants ─────────────────────────────────────────────────────────────── */

#define SHM_PATH         "/tmp/muofuzz_shm"
#define SHM_SIZE         128
#define MAX_MUTATED_SIZE (1024 * 1024)

/* Byte offsets matching the Python layout */
#define OFF_STATE_SEQ    0
#define OFF_EDGE_ID      4
#define OFF_COVERAGE     8
#define OFF_NEW_EDGES    12
#define OFF_CRASHES      16
#define OFF_PAD0         20
#define OFF_TOTAL_EXECS  24
/* 32 bytes of padding at offset 32..63 */
#define OFF_ACTION_SEQ   64
#define OFF_ACTION       68

/* Spin poll interval when waiting for Python's action (nanoseconds) */
#define SPIN_NS          100000   /* 0.1 ms */

/* Interesting mutation values */
static const int32_t INTERESTING_8[]  = { -128, -1, 0, 1, 16, 32, 64, 100, 127 };
static const int32_t INTERESTING_32[] = {
    -2147483648, -100663046, -32769, 32768, 65535, 65536, 100663045, 2147483647
};

/* ── Mutator state ──────────────────────────────────────────────────────────── */

typedef struct my_mutator {
    afl_state_t *afl;
    uint8_t     *mutated_buf;

    /* shared memory */
    int    shm_fd;
    void  *shm;

    /* tracking for deltas */
    uint32_t prev_coverage;
    uint32_t prev_crashes;

    /* IPC sequence counters */
    uint32_t state_seq;       /* incremented each time we push a new state   */
    uint32_t last_action_seq; /* last action_seq we read from Python          */
} my_mutator_t;

/* ── SHM helpers ────────────────────────────────────────────────────────────── */

/* Typed accessors into the raw shm page */
static inline volatile uint32_t *shm_u32(void *base, size_t off) {
    return (volatile uint32_t *)((uint8_t *)base + off);
}
static inline volatile int32_t *shm_i32(void *base, size_t off) {
    return (volatile int32_t *)((uint8_t *)base + off);
}
static inline volatile uint64_t *shm_u64(void *base, size_t off) {
    return (volatile uint64_t *)((uint8_t *)base + off);
}

/* Write all state fields then release-store the sentinel.
 * This ensures Python always sees a consistent snapshot. */
static void shm_push_state(my_mutator_t *data,
                            uint32_t edge_id,
                            uint32_t coverage,
                            uint32_t new_edges,
                            uint32_t crashes,
                            uint64_t total_execs)
{
    void *shm = data->shm;

    *shm_u32(shm, OFF_EDGE_ID)     = edge_id;
    *shm_u32(shm, OFF_COVERAGE)    = coverage;
    *shm_u32(shm, OFF_NEW_EDGES)   = new_edges;
    *shm_u32(shm, OFF_CRASHES)     = crashes;
    *shm_u64(shm, OFF_TOTAL_EXECS) = total_execs;

    /* Release-store: all prior stores are visible before this one */
    data->state_seq++;
    __atomic_store_n(shm_u32(shm, OFF_STATE_SEQ), data->state_seq, __ATOMIC_RELEASE);
}

/* Spin-wait until Python writes a new action (action_seq changed).
 * Returns the chosen action. */
static int shm_wait_action(my_mutator_t *data)
{
    void *shm = data->shm;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = SPIN_NS };

    while (1) {
        uint32_t cur = __atomic_load_n(shm_u32(shm, OFF_ACTION_SEQ), __ATOMIC_ACQUIRE);
        if (cur != data->last_action_seq) {
            data->last_action_seq = cur;
            /* Acquire fence above ensures action value is visible now */
            return (int)(*shm_i32(shm, OFF_ACTION));
        }
        nanosleep(&ts, NULL);
    }
}

/* ── Coverage helpers ───────────────────────────────────────────────────────── */

/* Count bytes in virgin_bits that are not 0xFF (i.e. have been touched). */
static uint32_t count_coverage(afl_state_t *afl)
{
    uint32_t count = 0;
    uint32_t map_size = afl->total_bitmap_size;
    const uint8_t *virgin = afl->virgin_bits;

    for (uint32_t i = 0; i < map_size; i++) {
        if (virgin[i] != 0xFF) count++;
    }
    return count;
}

/* Find the "most interesting" edge from the last execution.
 *
 * Priority 1: the highest-index edge that was hit by the last execution
 *             AND was previously unseen (virgin_bits[i] == 0xFF before hit).
 *             This represents a newly discovered edge — the most valuable
 *             signal for the RL agent.
 *
 * Priority 2: if no new edge, return the highest-frequency edge in trace_bits.
 *             This gives the agent context about which code path is being
 *             exercised even when no new coverage is found.
 */
static uint32_t find_current_edge(afl_state_t *afl)
{
    uint32_t        map_size  = afl->total_bitmap_size;
    const uint8_t  *trace     = afl->fsrv.trace_bits; /* last execution */
    const uint8_t  *virgin    = afl->virgin_bits;

    /* Pass 1: newest (highest-index) newly covered edge */
    for (int32_t i = (int32_t)map_size - 1; i >= 1; i--) {
        if (trace[i] && virgin[i] == 0xFF) {
            return (uint32_t)i;
        }
    }

    /* Pass 2: hottest (most-hit) edge from last execution */
    uint32_t best_idx = 0, best_val = 0;
    for (uint32_t i = 1; i < map_size; i++) {
        if (trace[i] > best_val) {
            best_val = trace[i];
            best_idx = i;
        }
    }
    return best_idx;
}

/* ── AFL++ API ──────────────────────────────────────────────────────────────── */

my_mutator_t *afl_custom_init(afl_state_t *afl, unsigned int seed)
{
    srand(seed);

    my_mutator_t *data = calloc(1, sizeof(my_mutator_t));
    if (!data) { perror("calloc"); return NULL; }

    data->afl          = afl;
    data->mutated_buf  = malloc(MAX_MUTATED_SIZE);
    if (!data->mutated_buf) { perror("malloc"); free(data); return NULL; }

    /* Open / create the shared memory file */
    data->shm_fd = open(SHM_PATH, O_RDWR | O_CREAT, 0600);
    if (data->shm_fd < 0) {
        perror("[-] SHM open failed");
        data->shm = NULL;
        return data;
    }

    if (ftruncate(data->shm_fd, SHM_SIZE) < 0) {
        perror("[-] SHM ftruncate");
    }

    data->shm = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, data->shm_fd, 0);
    if (data->shm == MAP_FAILED) {
        perror("[-] SHM mmap failed");
        data->shm = NULL;
    } else {
        printf("[+] MuoFuzz mutator: shared memory mapped at %s\n", SHM_PATH);
    }

    /* Read the initial action_seq so we don't mistake 0→0 as "no change" */
    if (data->shm) {
        data->last_action_seq = __atomic_load_n(
            shm_u32(data->shm, OFF_ACTION_SEQ), __ATOMIC_ACQUIRE);
    }

    data->state_seq    = 0;
    data->prev_coverage = 0;
    data->prev_crashes  = 0;

    return data;
}

size_t afl_custom_fuzz(my_mutator_t *data,
                        uint8_t *buf, size_t buf_size,
                        uint8_t **out_buf,
                        uint8_t *add_buf, size_t add_buf_size,
                        size_t max_size)
{
    int action = 6;  /* default: Havoc */

    if (data->shm) {
        /* ── Gather state ──────────────────────────────────────────────── */
        uint32_t coverage    = count_coverage(data->afl);
        uint32_t crashes     = (uint32_t)data->afl->total_crashes;
        uint64_t total_execs = data->afl->fsrv.total_execs;
        uint32_t edge_id     = find_current_edge(data->afl);

        uint32_t new_edges = (coverage > data->prev_coverage)
                             ? (coverage - data->prev_coverage) : 0;

        /* ── Push state (release-store) ────────────────────────────────── */
        shm_push_state(data, edge_id, coverage, new_edges, crashes, total_execs);

        /* ── Wait for Python's action (acquire-load) ────────────────────── */
        action = shm_wait_action(data);

        /* ── Update deltas ──────────────────────────────────────────────── */
        data->prev_coverage = coverage;
        data->prev_crashes  = crashes;
    }

    /* ── Apply mutation ─────────────────────────────────────────────────── */
    size_t mut_len = (buf_size < MAX_MUTATED_SIZE) ? buf_size : MAX_MUTATED_SIZE;
    memcpy(data->mutated_buf, buf, mut_len);

    int pos;

    switch (action) {

        case 0: /* ARITH INC */
            if (mut_len > 0) {
                pos = rand() % (int)mut_len;
                data->mutated_buf[pos]++;
            }
            break;

        case 1: /* ARITH DEC */
            if (mut_len > 0) {
                pos = rand() % (int)mut_len;
                data->mutated_buf[pos]--;
            }
            break;

        case 2: /* INTERESTING 8-BIT */
            if (mut_len > 0) {
                pos = rand() % (int)mut_len;
                int8_t v = (int8_t)INTERESTING_8[rand() % (sizeof(INTERESTING_8)/sizeof(int32_t))];
                data->mutated_buf[pos] = (uint8_t)v;
            }
            break;

        case 3: /* INTERESTING 32-BIT */
            if (mut_len >= 4) {
                pos = rand() % (int)(mut_len - 3);
                int32_t v = INTERESTING_32[rand() % (sizeof(INTERESTING_32)/sizeof(int32_t))];
                memcpy(data->mutated_buf + pos, &v, 4);
            }
            break;

        case 4: /* DICTIONARY INSERT/OVERWRITE */
            if (data->afl->extras_cnt > 0 || data->afl->a_extras_cnt > 0) {
                int     use_user = (data->afl->extras_cnt > 0)
                                   && (data->afl->a_extras_cnt == 0 || (rand() % 2));
                uint8_t *tok     = NULL;
                uint32_t tok_len = 0;

                if (use_user) {
                    uint32_t idx = rand() % data->afl->extras_cnt;
                    tok     = data->afl->extras[idx].data;
                    tok_len = data->afl->extras[idx].len;
                } else {
                    uint32_t idx = rand() % data->afl->a_extras_cnt;
                    tok     = data->afl->a_extras[idx].data;
                    tok_len = data->afl->a_extras[idx].len;
                }

                if (tok && tok_len > 0 && mut_len + tok_len <= max_size) {
                    if (rand() % 2 == 0) { /* insert */
                        pos = rand() % (int)(mut_len + 1);
                        memmove(data->mutated_buf + pos + tok_len,
                                data->mutated_buf + pos,
                                mut_len - (size_t)pos);
                        memcpy(data->mutated_buf + pos, tok, tok_len);
                        mut_len += tok_len;
                    } else {               /* overwrite */
                        if (mut_len >= tok_len) {
                            pos = rand() % (int)(mut_len - tok_len + 1);
                            memcpy(data->mutated_buf + pos, tok, tok_len);
                        }
                    }
                }
                break;
            }
            /* fall through to Havoc if no dictionary available */

        case 5: /* DELETE BYTES */
            if (mut_len > 1) {
                uint32_t del_len = 1 + (uint32_t)(rand() % (int)(mut_len - 1));
                if (del_len >= mut_len) del_len = (uint32_t)mut_len - 1;
                pos = rand() % (int)(mut_len - del_len);
                memmove(data->mutated_buf + pos,
                        data->mutated_buf + pos + del_len,
                        mut_len - (size_t)pos - del_len);
                mut_len -= del_len;
            }
            break;

        default: /* HAVOC (action 6 and any out-of-range value) */
            for (int i = 0; i < 1 + rand() % 4; i++) {
                if (mut_len == 0) break;
                pos = rand() % (int)mut_len;
                int m = rand() % 3;
                if      (m == 0) data->mutated_buf[pos] ^= (uint8_t)(1 << (rand() % 8));
                else if (m == 1) data->mutated_buf[pos]  = (uint8_t)(rand() % 256);
                else             data->mutated_buf[pos]  = ~data->mutated_buf[pos];
            }
            break;
    }

    *out_buf = data->mutated_buf;
    return mut_len;
}

void afl_custom_deinit(my_mutator_t *data)
{
    if (data->shm && data->shm != MAP_FAILED) munmap(data->shm, SHM_SIZE);
    if (data->shm_fd > 0) close(data->shm_fd);
    free(data->mutated_buf);
    free(data);
}
