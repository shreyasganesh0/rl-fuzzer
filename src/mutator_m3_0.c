/*
 * mutator_m3_0.c  —  RL mutator / Model M3_0 (differential-informed)
 *
 * 13-dimensional state derived from differential analysis of buggy vs fixed
 * libxml2 versions.  Features capture coverage distribution, edge heat,
 * entropy, execution time, and coverage velocity — structural properties
 * that generalise across targets.
 *
 * SHM layout (128 bytes, /tmp/rl_shm_m3_0):
 *
 *   STATE REGION [0..55]
 *     [0]   state_seq        u32   monotonic, RELEASE store
 *     [4]   total_edges      u32   count_coverage(afl)
 *     [8]   cold_edges       u32   MAP_SIZE - nonzero(cumulative_map)
 *     [12]  hot_edges        u32   cumulative_map[i] > 128
 *     [16]  warm_edges       u32   8 <= cumulative_map[i] <= 128
 *     [20]  cool_edges       u32   1 <= cumulative_map[i] <= 7
 *     [24]  edge_entropy     f32   Shannon entropy / 3.0
 *     [28]  edge_hit_mean    f32   mean(nonzero) / 255.0
 *     [32]  edge_hit_std     f32   std(nonzero) / 255.0
 *     [36]  corpus_size      u32   afl->queued_items
 *     [40]  crashes          u32   afl->total_crashes
 *     [44]  new_edges        u32   coverage - prev_coverage
 *     [48]  avg_exec_time    f32   log1p(ema_us) / log1p(100000)
 *     [52]  coverage_velocity f32  (coverage - ring_oldest) / 1000 / 0.1
 *
 *   ACTION REGION [64..71]
 *     [64]  action_seq       u32   monotonic, ACQUIRE load
 *     [68]  action           i32   0..46
 *
 * Action table (47 entries, identical to all RL mutators):
 *   0-5    deterministic bit/byte flips
 *   6-15   deterministic arithmetic (1/2/4 bytes, LE + BE)
 *   16-20  interesting value substitutions
 *   21-40  havoc-style single ops
 *   41-44  dictionary token over/insert
 *   45     CUSTOM_MUTATOR (focused multi-op)
 *   46     HAVOC (stacked random)
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
#include <math.h>

/* ── Constants ────────────────────────────────────────────────────────────── */

#define SHM_PATH      "/tmp/rl_shm_m3_0"
#define SHM_SIZE      128
#undef  MAP_SIZE
#define MAP_SIZE      65536
#define MAX_MUTATED_SIZE (1024 * 1024)
#define ACTION_SIZE   47
#define ARITH_MAX     35
/* Match telemetry mutator so HAVOC action is identical to training data */
#undef  HAVOC_STACK_POW2
#define HAVOC_STACK_POW2 9

#define OFF_STATE_SEQ    0
#define OFF_TOTAL_EDGES  4
#define OFF_COLD_EDGES   8
#define OFF_HOT_EDGES    12
#define OFF_WARM_EDGES   16
#define OFF_COOL_EDGES   20
#define OFF_ENTROPY      24
#define OFF_HIT_MEAN     28
#define OFF_HIT_STD      32
#define OFF_CORPUS_SIZE  36
#define OFF_CRASHES      40
#define OFF_NEW_EDGES    44
#define OFF_EXEC_TIME    48
#define OFF_VELOCITY     52
#define OFF_ACTION_SEQ   64
#define OFF_ACTION       68

#define SPIN_NS          100000   /* 0.1 ms */
#define VELOCITY_WINDOW  1000
#define EMA_ALPHA        0.01f

/* ── Interesting value tables ─────────────────────────────────────────────── */

static const int8_t MYINTERESTING_8[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127
};
#define N8  ((int)(sizeof(MYINTERESTING_8)  / sizeof(MYINTERESTING_8[0])))

static const int16_t MYINTERESTING_16[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127,
    -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767
};
#define N16 ((int)(sizeof(MYINTERESTING_16) / sizeof(MYINTERESTING_16[0])))

static const int32_t MYINTERESTING_32[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127,
    -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767,
    -2147483648, -32769, 32768, 65535, 65536, 100663045, 2147483647
};
#define N32 ((int)(sizeof(MYINTERESTING_32) / sizeof(MYINTERESTING_32[0])))

/* ── Byte-swap helpers ────────────────────────────────────────────────────── */

static inline uint16_t bswap16(uint16_t x) {
    return (uint16_t)((x >> 8) | (x << 8));
}
static inline uint32_t bswap32(uint32_t x) {
    return ((x >> 24))
         | ((x >>  8) & 0x0000ff00u)
         | ((x <<  8) & 0x00ff0000u)
         | ((x << 24));
}

/* ── Mutator state ────────────────────────────────────────────────────────── */

typedef struct my_mutator {
    afl_state_t *afl;
    uint8_t     *mutated_buf;
    int          shm_fd;
    void        *shm;

    uint32_t     prev_coverage;
    uint32_t     state_seq;
    uint32_t     last_action_seq;
    uint32_t     step_count;

    /* Cumulative bitmap: max(hit_count) per edge across all execs */
    uint8_t      cumulative_map[MAP_SIZE];

    /* Coverage velocity ring buffer */
    uint32_t     edge_ring[VELOCITY_WINDOW];
    uint32_t     ring_idx;
    int          ring_full;

    /* Execution time tracking */
    struct timespec last_exec_time;
    float        avg_exec_time_us;
} my_mutator_t;

/* ── SHM access helpers ───────────────────────────────────────────────────── */

static inline volatile uint32_t *u32_at(void *b, size_t o) {
    return (volatile uint32_t *)((uint8_t *)b + o);
}
static inline volatile int32_t *i32_at(void *b, size_t o) {
    return (volatile int32_t *)((uint8_t *)b + o);
}
static inline volatile float *f32_at(void *b, size_t o) {
    return (volatile float *)((uint8_t *)b + o);
}

/* ── Coverage helper ──────────────────────────────────────────────────────── */

static uint32_t count_coverage(afl_state_t *afl)
{
    /* Use fsrv.map_size (the actual trace bitmap size, e.g. 65536), NOT
     * total_bitmap_size which is an accumulator of per-entry coverage sizes
     * and grows to millions of bytes — reading that far past virgin_bits is
     * an out-of-bounds read. */
    uint32_t n = 0, sz = afl->fsrv.map_size;
    const uint8_t *v = afl->virgin_bits;
    uint32_t i = 0;
    for (; i + 8 <= sz; i += 8) {
        uint64_t chunk;
        memcpy(&chunk, v + i, 8);
        if (chunk == 0xFFFFFFFFFFFFFFFFULL) continue;
        for (int j = 0; j < 8; j++)
            if (v[i + j] != 0xFF) n++;
    }
    for (; i < sz; i++)
        if (v[i] != 0xFF) n++;
    return n;
}

/*
 * shm_push_state_m3_0
 *
 * Single O(MAP_SIZE) pass over trace_bits:
 *   1. Max-merge into cumulative_map
 *   2. Classify edges into hot/warm/cool/cold
 *   3. Accumulate sum/sum_sq for mean/std
 *   4. Build 8-bin histogram for entropy
 *   5. Write all 13 features to SHM
 */
static void shm_push_state(my_mutator_t *m,
                            uint32_t coverage, uint32_t new_edges,
                            uint32_t crashes)
{
    const uint8_t *trace = m->afl->fsrv.trace_bits;
    uint32_t sz = m->afl->fsrv.map_size;
    if (sz > MAP_SIZE) sz = MAP_SIZE;

    /* ── Pass 1: update cumulative_map and collect stats ──────────────── */
    uint32_t hot = 0, warm = 0, cool = 0;
    uint32_t nonzero = 0;
    double   sum_hits = 0.0, sum_sq_hits = 0.0;
    uint32_t bins[8] = {0};  /* power-of-2 entropy bins */

    for (uint32_t i = 0; i < sz; i++) {
        uint8_t t = trace[i];
        if (t > m->cumulative_map[i])
            m->cumulative_map[i] = t;

        uint8_t v = m->cumulative_map[i];
        if (v == 0) continue;

        nonzero++;
        sum_hits   += (double)v;
        sum_sq_hits += (double)v * (double)v;

        /* Edge heat classification */
        if (v > 128)      hot++;
        else if (v >= 8)  warm++;
        else              cool++;

        /* Entropy bins: [1], [2-3], [4-7], [8-15], [16-31], [32-63], [64-127], [128+] */
        if (v >= 128)     bins[7]++;
        else if (v >= 64) bins[6]++;
        else if (v >= 32) bins[5]++;
        else if (v >= 16) bins[4]++;
        else if (v >= 8)  bins[3]++;
        else if (v >= 4)  bins[2]++;
        else if (v >= 2)  bins[1]++;
        else              bins[0]++;
    }

    uint32_t cold_edges = (MAP_SIZE > nonzero) ? (MAP_SIZE - nonzero) : 0;

    /* ── Entropy ──────────────────────────────────────────────────────── */
    float entropy_norm = 0.0f;
    if (nonzero > 0) {
        double ent = 0.0;
        for (int b = 0; b < 8; b++) {
            if (bins[b] == 0) continue;
            double p = (double)bins[b] / (double)nonzero;
            ent -= p * log2(p);
        }
        entropy_norm = (float)(ent / 3.0);  /* max entropy = log2(8) = 3 */
    }

    /* ── Mean / std ───────────────────────────────────────────────────── */
    float mean_norm = 0.0f, std_norm = 0.0f;
    if (nonzero > 0) {
        double mean = sum_hits / (double)nonzero;
        double var  = (sum_sq_hits / (double)nonzero) - (mean * mean);
        if (var < 0.0) var = 0.0;
        mean_norm = (float)(mean / 255.0);
        std_norm  = (float)(sqrt(var) / 255.0);
    }

    /* ── Corpus size ──────────────────────────────────────────────────── */
    uint32_t corpus_size = m->afl->queued_items;

    /* ── Execution time EMA ───────────────────────────────────────────── */
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double delta_us = (double)(now.tv_sec - m->last_exec_time.tv_sec) * 1e6
                    + (double)(now.tv_nsec - m->last_exec_time.tv_nsec) / 1e3;
    m->last_exec_time = now;

    if (m->step_count <= 1)
        m->avg_exec_time_us = (float)delta_us;
    else
        m->avg_exec_time_us = (1.0f - EMA_ALPHA) * m->avg_exec_time_us
                            + EMA_ALPHA * (float)delta_us;

    float exec_time_norm = (float)(log1p((double)m->avg_exec_time_us)
                                 / log1p(100000.0));

    /* ── Coverage velocity ────────────────────────────────────────────── */
    float velocity_norm = 0.0f;
    m->edge_ring[m->ring_idx % VELOCITY_WINDOW] = coverage;
    m->ring_idx++;
    if (m->ring_idx >= VELOCITY_WINDOW) m->ring_full = 1;

    if (m->ring_full) {
        uint32_t oldest = m->edge_ring[m->ring_idx % VELOCITY_WINDOW];
        float velocity = (float)(coverage - oldest) / (float)VELOCITY_WINDOW;
        velocity_norm = velocity / 0.1f;
        if (velocity_norm > 1.0f) velocity_norm = 1.0f;
        if (velocity_norm < 0.0f) velocity_norm = 0.0f;
    }

    /* ── Write to SHM ─────────────────────────────────────────────────── */
    void *s = m->shm;
    *u32_at(s, OFF_TOTAL_EDGES) = coverage;
    *u32_at(s, OFF_COLD_EDGES)  = cold_edges;
    *u32_at(s, OFF_HOT_EDGES)   = hot;
    *u32_at(s, OFF_WARM_EDGES)  = warm;
    *u32_at(s, OFF_COOL_EDGES)  = cool;
    *f32_at(s, OFF_ENTROPY)     = entropy_norm;
    *f32_at(s, OFF_HIT_MEAN)    = mean_norm;
    *f32_at(s, OFF_HIT_STD)     = std_norm;
    *u32_at(s, OFF_CORPUS_SIZE) = corpus_size;
    *u32_at(s, OFF_CRASHES)     = crashes;
    *u32_at(s, OFF_NEW_EDGES)   = new_edges;
    *f32_at(s, OFF_EXEC_TIME)   = exec_time_norm;
    *f32_at(s, OFF_VELOCITY)    = velocity_norm;

    m->state_seq++;
    __atomic_store_n(u32_at(s, OFF_STATE_SEQ), m->state_seq, __ATOMIC_RELEASE);
}

static int shm_wait_action(my_mutator_t *m)
{
    void *s = m->shm;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = SPIN_NS };
    for (;;) {
        uint32_t cur = __atomic_load_n(u32_at(s, OFF_ACTION_SEQ), __ATOMIC_ACQUIRE);
        if (cur != m->last_action_seq) {
            m->last_action_seq = cur;
            return (int)(*i32_at(s, OFF_ACTION));
        }
        nanosleep(&ts, NULL);
    }
}

/* ── Dictionary helpers ───────────────────────────────────────────────────── */

static void dict_overwrite(uint8_t *buf, size_t len, int pos,
                            const uint8_t *tok, uint32_t tlen)
{
    size_t avail = (pos < (int)len) ? len - (size_t)pos : 0;
    size_t n     = tlen < avail ? tlen : avail;
    if (n) memcpy(buf + pos, tok, n);
}

static size_t dict_insert(uint8_t *buf, size_t len, int pos,
                           const uint8_t *tok, uint32_t tlen, size_t max)
{
    size_t nlen = len + tlen;
    if (nlen > max) nlen = max;
    size_t tail = len - (size_t)pos;
    if ((size_t)pos + tail > nlen) tail = nlen - (size_t)pos;
    memmove(buf + pos + tlen, buf + pos, tail);
    memcpy(buf + pos, tok, tlen);
    return nlen;
}

static uint32_t pick_user_extra(afl_state_t *afl, const uint8_t **out) {
    if (!afl->extras_cnt) return 0;
    uint32_t i = (uint32_t)(rand() % (int)afl->extras_cnt);
    *out = afl->extras[i].data;
    return afl->extras[i].len;
}

static uint32_t pick_auto_extra(afl_state_t *afl, const uint8_t **out) {
    if (!afl->a_extras_cnt) return 0;
    uint32_t i = (uint32_t)(rand() % (int)afl->a_extras_cnt);
    *out = afl->a_extras[i].data;
    return afl->a_extras[i].len;
}

/* ── Core mutation dispatcher (47 actions, identical across all models) ───── */

static size_t apply_mutation(afl_state_t *afl,
                              uint8_t *mb, size_t mut_len, size_t max_size,
                              int action,
                              uint8_t *add_buf, size_t add_buf_size)
{
    int            pos;
    uint16_t       v16;
    uint32_t       v32, delta;
    const uint8_t *tok;
    uint32_t       tok_len;

#define NEED(n)  if (mut_len < (size_t)(n)) break
#define RND(n)   ((int)(rand() % (unsigned)(n)))
#define RPOS     ((int)(rand() % (unsigned)mut_len))
#define RDELTA   (1u + (uint32_t)RND(ARITH_MAX))

    switch (action) {

    /* deterministic bit flips */
    case 0: pos = RND((int)mut_len * 8);
            mb[pos/8] ^= (uint8_t)(1 << (pos%8)); break;
    case 1: pos = RND((int)mut_len * 8);
            mb[pos/8] ^= (uint8_t)(1 << (pos%8));
            { int p2 = (pos+1) % ((int)mut_len*8);
              mb[p2/8] ^= (uint8_t)(1 << (p2%8)); } break;
    case 2: pos = RND((int)mut_len * 8);
            for (int b = 0; b < 4; b++) {
                int bp = (pos+b) % ((int)mut_len*8);
                mb[bp/8] ^= (uint8_t)(1 << (bp%8));
            } break;
    case 3: mb[RPOS] ^= 0xFF; break;
    case 4: NEED(2); pos = RND((int)mut_len-1);
            mb[pos] ^= 0xFF; mb[pos+1] ^= 0xFF; break;
    case 5: NEED(4); pos = RND((int)mut_len-3);
            mb[pos]^=0xFF; mb[pos+1]^=0xFF; mb[pos+2]^=0xFF; mb[pos+3]^=0xFF; break;

    /* deterministic arithmetic */
    case 6:  mb[RPOS]++; break;
    case 7:  mb[RPOS]--; break;
    case 8:  NEED(2); pos=RND((int)mut_len-1); delta=RDELTA;
             *(uint16_t*)(mb+pos) = (uint16_t)(*(uint16_t*)(mb+pos) + delta); break;
    case 9:  NEED(2); pos=RND((int)mut_len-1); delta=RDELTA;
             *(uint16_t*)(mb+pos) = (uint16_t)(*(uint16_t*)(mb+pos) - delta); break;
    case 10: NEED(2); pos=RND((int)mut_len-1); delta=RDELTA;
             v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos) = bswap16((uint16_t)(v16+delta)); break;
    case 11: NEED(2); pos=RND((int)mut_len-1); delta=RDELTA;
             v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos) = bswap16((uint16_t)(v16-delta)); break;
    case 12: NEED(4); pos=RND((int)mut_len-3); delta=RDELTA;
             *(uint32_t*)(mb+pos) += delta; break;
    case 13: NEED(4); pos=RND((int)mut_len-3); delta=RDELTA;
             *(uint32_t*)(mb+pos) -= delta; break;
    case 14: NEED(4); pos=RND((int)mut_len-3); delta=RDELTA;
             v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos) = bswap32(v32+delta); break;
    case 15: NEED(4); pos=RND((int)mut_len-3); delta=RDELTA;
             v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos) = bswap32(v32-delta); break;

    /* interesting values */
    case 16: mb[RPOS] = (uint8_t)MYINTERESTING_8[RND(N8)]; break;
    case 17: NEED(2); pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos) = (uint16_t)MYINTERESTING_16[RND(N16)]; break;
    case 18: NEED(2); pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos) = bswap16((uint16_t)MYINTERESTING_16[RND(N16)]); break;
    case 19: NEED(4); pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos) = (uint32_t)MYINTERESTING_32[RND(N32)]; break;
    case 20: NEED(4); pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos) = bswap32((uint32_t)MYINTERESTING_32[RND(N32)]); break;

    /* havoc single ops */
    case 21: pos=RND((int)mut_len*8); mb[pos/8]^=(uint8_t)(128>>(pos%8)); break;
    case 22: mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)]; break;
    case 23: NEED(2); pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=(uint16_t)MYINTERESTING_16[RND(N16)]; break;
    case 24: NEED(2); pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)MYINTERESTING_16[RND(N16)]); break;
    case 25: NEED(4); pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)=(uint32_t)MYINTERESTING_32[RND(N32)]; break;
    case 26: NEED(4); pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)=bswap32((uint32_t)MYINTERESTING_32[RND(N32)]); break;
    case 27: mb[RPOS] -= (uint8_t)(1+RND(ARITH_MAX)); break;
    case 28: mb[RPOS] += (uint8_t)(1+RND(ARITH_MAX)); break;
    case 29: NEED(2); pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos) = (uint16_t)(*(uint16_t*)(mb+pos)-(1+RND(ARITH_MAX))); break;
    case 30: NEED(2); pos=RND((int)mut_len-1);
             v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)(v16-(1+RND(ARITH_MAX)))); break;
    case 31: NEED(2); pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos) = (uint16_t)(*(uint16_t*)(mb+pos)+(1+RND(ARITH_MAX))); break;
    case 32: NEED(2); pos=RND((int)mut_len-1);
             v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)(v16+(1+RND(ARITH_MAX)))); break;
    case 33: NEED(4); pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos) -= (1u+(uint32_t)RND(ARITH_MAX)); break;
    case 34: NEED(4); pos=RND((int)mut_len-3);
             v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos)=bswap32(v32-(1u+(uint32_t)RND(ARITH_MAX))); break;
    case 35: NEED(4); pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos) += (1u+(uint32_t)RND(ARITH_MAX)); break;
    case 36: NEED(4); pos=RND((int)mut_len-3);
             v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos)=bswap32(v32+(1u+(uint32_t)RND(ARITH_MAX))); break;
    case 37: mb[RPOS] = (uint8_t)(rand()&0xFF); break;
    case 38: mb[RPOS] += (uint8_t)(rand()&0x1F); break;
    case 39: mb[RPOS] -= (uint8_t)(rand()&0x1F); break;
    case 40: mb[RPOS] ^= (uint8_t)(rand()&0xFF); break;

    /* dictionary ops */
    case 41: tok_len = pick_user_extra(afl, &tok);
             if (!tok_len) { mb[RPOS]=(uint8_t)(rand()&0xFF); break; }
             dict_overwrite(mb, mut_len, RPOS, tok, tok_len); break;
    case 42: tok_len = pick_user_extra(afl, &tok);
             if (!tok_len || mut_len+tok_len > max_size) { mb[RPOS]^=0xFF; break; }
             pos = RND((int)mut_len+1);
             mut_len = dict_insert(mb, mut_len, pos, tok, tok_len, max_size); break;
    case 43: tok_len = pick_auto_extra(afl, &tok);
             if (!tok_len) { mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)]; break; }
             dict_overwrite(mb, mut_len, RPOS, tok, tok_len); break;
    case 44: tok_len = pick_auto_extra(afl, &tok);
             if (!tok_len || mut_len+tok_len > max_size) { mb[RPOS]++; break; }
             pos = RND((int)mut_len+1);
             mut_len = dict_insert(mb, mut_len, pos, tok, tok_len, max_size); break;

    /* CUSTOM_MUTATOR: focused multi-op (4-8 stacked) */
    case 45: {
        int nops = 4 + RND(5);
        for (int op = 0; op < nops && mut_len > 0; op++) {
            switch (RND(8)) {
            case 0: { int bp=RND((int)mut_len*8); mb[bp/8]^=(uint8_t)(128>>(bp%8)); } break;
            case 1: mb[RPOS] += (uint8_t)(1+RND(ARITH_MAX)); break;
            case 2: mb[RPOS] -= (uint8_t)(1+RND(ARITH_MAX)); break;
            case 3: mb[RPOS]  = (uint8_t)MYINTERESTING_8[RND(N8)]; break;
            case 4: mb[RPOS]  = (uint8_t)(rand()&0xFF); break;
            case 5: mb[RPOS] ^= 0xFF; break;
            case 6: if (mut_len>=2) { int p=RND((int)mut_len-1);
                        *(uint16_t*)(mb+p)=(uint16_t)MYINTERESTING_16[RND(N16)]; } break;
            case 7: if (mut_len>=4) { int p=RND((int)mut_len-3);
                        *(uint32_t*)(mb+p)=(uint32_t)MYINTERESTING_32[RND(N32)]; } break;
            }
        }
        break;
    }

    /* HAVOC: large stacked random */
    default:
    case 46: {
        int stack = 1 << (1 + RND(HAVOC_STACK_POW2));
        for (int op = 0; op < stack && mut_len > 0; op++) {
            switch (RND(12)) {
            case 0:  { int bp=RND((int)mut_len*8); mb[bp/8]^=(uint8_t)(128>>(bp%8)); } break;
            case 1:  mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)]; break;
            case 2:  mb[RPOS]+=(uint8_t)(1+RND(ARITH_MAX)); break;
            case 3:  mb[RPOS]-=(uint8_t)(1+RND(ARITH_MAX)); break;
            case 4:  mb[RPOS]=(uint8_t)(rand()&0xFF); break;
            case 5:  if (mut_len>2) {
                         int from=RPOS, dlen=1+RND((int)mut_len-from);
                         if ((size_t)(from+dlen)>mut_len) dlen=(int)mut_len-from;
                         memmove(mb+from, mb+from+dlen, mut_len-(size_t)from-(size_t)dlen);
                         mut_len-=(size_t)dlen;
                     } break;
            case 6:  if (mut_len>=2) {
                         int src=RPOS, dst=RPOS, cl=1+RND(8);
                         if ((size_t)(src+cl)>mut_len) cl=(int)mut_len-src;
                         if ((size_t)(dst+cl)>mut_len) cl=(int)mut_len-dst;
                         if (cl>0) memmove(mb+dst, mb+src, (size_t)cl);
                     } break;
            case 7:  { int p=RPOS, sl=1+RND(8);
                       if ((size_t)(p+sl)>mut_len) sl=(int)mut_len-p;
                       memset(mb+p, rand()&0xFF, (size_t)sl); } break;
            case 8:  if (mut_len>=2) { int p=RND((int)mut_len-1);
                         *(uint16_t*)(mb+p)=(uint16_t)MYINTERESTING_16[RND(N16)]; } break;
            case 9:  if (mut_len>=4) { int p=RND((int)mut_len-3);
                         *(uint32_t*)(mb+p)=(uint32_t)MYINTERESTING_32[RND(N32)]; } break;
            case 10: tok_len=pick_user_extra(afl,&tok);
                     if (tok_len) dict_overwrite(mb,mut_len,RPOS,tok,tok_len);
                     else mb[RPOS]^=0xAA; break;
            case 11: if (add_buf && add_buf_size>0 && mut_len>=2) {
                         int split=1+RND((int)mut_len-1);
                         int add_off=RND((int)add_buf_size);
                         int cl=(int)mut_len-split;
                         if ((size_t)(add_off+cl)>add_buf_size) cl=(int)add_buf_size-add_off;
                         if (cl>0) memcpy(mb+split, add_buf+add_off, (size_t)cl);
                     } break;
            }
        }
        break;
    }
    } /* end switch */

#undef NEED
#undef RND
#undef RPOS
#undef RDELTA

    if (mut_len == 0) mut_len = 1;
    if (mut_len > max_size) mut_len = max_size;
    return mut_len;
}

/* ── AFL++ API ────────────────────────────────────────────────────────────── */

my_mutator_t *afl_custom_init(afl_state_t *afl, unsigned int seed)
{
    srand(seed);
    my_mutator_t *m = calloc(1, sizeof(my_mutator_t));
    if (!m) { perror("calloc"); return NULL; }

    m->afl = afl;
    m->mutated_buf = malloc(MAX_MUTATED_SIZE);
    if (!m->mutated_buf) { perror("malloc"); free(m); return NULL; }

    /* cumulative_map, edge_ring zeroed by calloc */

    m->shm_fd = open(SHM_PATH, O_RDWR | O_CREAT, 0600);
    if (m->shm_fd < 0) { perror("[-] M3_0 SHM open"); m->shm = NULL; return m; }
    if (ftruncate(m->shm_fd, SHM_SIZE) < 0) perror("[-] M3_0 SHM ftruncate");

    m->shm = mmap(NULL, SHM_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, m->shm_fd, 0);
    if (m->shm == MAP_FAILED) {
        perror("[-] M3_0 SHM mmap"); m->shm = NULL;
    } else {
        printf("[+] M3_0 mutator: SHM at %s  (13-dim differential features)\n", SHM_PATH);
    }

    if (m->shm)
        m->last_action_seq = __atomic_load_n(
            u32_at(m->shm, OFF_ACTION_SEQ), __ATOMIC_ACQUIRE);

    m->state_seq     = 0;
    m->prev_coverage = 0;
    m->step_count    = 0;
    m->ring_idx      = 0;
    m->ring_full     = 0;
    m->avg_exec_time_us = 0.0f;
    clock_gettime(CLOCK_MONOTONIC, &m->last_exec_time);

    return m;
}

size_t afl_custom_fuzz(my_mutator_t *m,
                        uint8_t *buf, size_t buf_size,
                        uint8_t **out_buf,
                        uint8_t *add_buf, size_t add_buf_size,
                        size_t max_size)
{
    int action = 46; /* default: HAVOC */

    if (m->shm) {
        if (m->step_count == 0) {
            /* First call: seed the cumulative_map from initial trace.
             * Use fsrv.map_size, NOT total_bitmap_size — see count_coverage(). */
            const uint8_t *trace = m->afl->fsrv.trace_bits;
            uint32_t sz = m->afl->fsrv.map_size;
            if (sz > MAP_SIZE) sz = MAP_SIZE;
            for (uint32_t i = 0; i < sz; i++) {
                if (trace[i] > m->cumulative_map[i])
                    m->cumulative_map[i] = trace[i];
            }
            m->prev_coverage = count_coverage(m->afl);
            m->step_count = 1;
            clock_gettime(CLOCK_MONOTONIC, &m->last_exec_time);
            goto do_mutate;
        }

        uint32_t coverage  = count_coverage(m->afl);
        uint32_t crashes   = (uint32_t)m->afl->total_crashes;
        uint32_t new_edges = (coverage > m->prev_coverage)
                             ? (coverage - m->prev_coverage) : 0;

        shm_push_state(m, coverage, new_edges, crashes);
        action = shm_wait_action(m);
        if (action < 0 || action >= ACTION_SIZE) action = 46;

        m->prev_coverage = coverage;
        m->step_count++;
    }

do_mutate:;
    size_t mut_len = (buf_size < MAX_MUTATED_SIZE) ? buf_size : MAX_MUTATED_SIZE;
    memcpy(m->mutated_buf, buf, mut_len);
    *out_buf = m->mutated_buf;

    mut_len = apply_mutation(m->afl, m->mutated_buf, mut_len, max_size,
                             action, add_buf, add_buf_size);
    return mut_len;
}

void afl_custom_deinit(my_mutator_t *m)
{
    if (m->shm && m->shm != MAP_FAILED) munmap(m->shm, SHM_SIZE);
    if (m->shm_fd > 0) close(m->shm_fd);
    free(m->mutated_buf);
    free(m);
}
