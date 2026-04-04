/*
 * mutator_telemetry.c  —  Baseline telemetry mutator (random action selection)
 *
 * AFL++ custom mutator plugin (.so) that:
 *   - Selects mutations uniformly at random over 47 actions
 *   - Logs coverage dynamics and per-mutation attribution to CSV files
 *   - Saves bitmap snapshots periodically
 *   - Does NOT communicate with any Python RL server (no SHM IPC)
 *
 * Action table (47 entries, identical to RL mutators):
 *   0-5    deterministic bit/byte flips
 *   6-15   deterministic arithmetic (1/2/4 bytes, LE + BE)
 *   16-20  interesting value substitutions
 *   21-40  havoc-style single ops
 *   41-44  dictionary token over/insert
 *   45     CUSTOM_MUTATOR (focused multi-op)
 *   46     HAVOC (stacked random)
 *
 * Environment variables:
 *   TELEMETRY_CSV_DIR           directory for CSV output (required)
 *   TELEMETRY_VERSION           version label for filenames (required)
 *   TELEMETRY_LOG_INTERVAL      steps between CSV writes (default 1000)
 *   TELEMETRY_SNAPSHOT_INTERVAL steps between bitmap dumps (default 10000)
 */

#include "afl-fuzz.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/* ── Constants ────────────────────────────────────────────────────────────── */

#define MAX_MUTATED_SIZE  (1024 * 1024)
#define ACTION_SIZE       47
#define ARITH_MAX         35
#undef  HAVOC_STACK_POW2
#define HAVOC_STACK_POW2  9
#undef  MAP_SIZE
#define MAP_SIZE          65536

#define DEFAULT_LOG_INTERVAL      1000
#define DEFAULT_SNAPSHOT_INTERVAL 10000

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

/* ── Telemetry mutator state ──────────────────────────────────────────────── */

typedef struct telemetry_mutator {
    afl_state_t *afl;
    uint8_t     *mutated_buf;

    /* Telemetry state */
    uint64_t     step;              /* total executions counted by us       */
    uint64_t     interval_start;    /* step at last CSV write               */
    uint32_t     prev_coverage;     /* coverage at last step                */
    uint32_t     prev_crashes;      /* crashes at last step                 */

    /* Per-interval accumulators (reset each interval) */
    uint32_t     mut_count[ACTION_SIZE];     /* times each mutation was used     */
    uint32_t     mut_new_edges[ACTION_SIZE]; /* new edges attributed to each     */
    uint32_t     mut_crashes[ACTION_SIZE];   /* crashes attributed to each       */
    int          last_action;                /* action from previous step        */

    /* Cumulative bitmap for snapshots */
    uint8_t      cumulative_map[MAP_SIZE];

    /* File handles */
    FILE        *csv_coverage;      /* coverage_dynamics CSV    */
    FILE        *csv_mutation;      /* mutation_attribution CSV */
    char         snapshot_dir[512]; /* snapshot output directory */

    /* Timing */
    struct timespec start_time;

    /* Config */
    uint32_t     log_interval;      /* steps between CSV writes      */
    uint32_t     snapshot_interval; /* steps between bitmap dumps    */
} telemetry_mutator_t;

/* ── Coverage helper ──────────────────────────────────────────────────────── */

static uint32_t count_coverage(afl_state_t *afl)
{
    uint32_t n = 0, sz = afl->total_bitmap_size;
    const uint8_t *v = afl->virgin_bits;
    uint32_t i = 0;
    /* Skip 8-byte chunks that are all 0xFF (unvisited).  Typical benchmarks
       hit ~2-5 K edges out of 64 K, so ~93 %+ of chunks are skippable. */
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

/* ── Core mutation dispatcher (47 actions) ────────────────────────────────── */

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

/* ── Telemetry: edge heat distribution ────────────────────────────────────── */

typedef struct edge_heat {
    uint32_t hot;          /* cumulative_map[i] > 128        */
    uint32_t warm;         /* 8 <= cumulative_map[i] <= 128  */
    uint32_t cool;         /* 1 <= cumulative_map[i] <= 7    */
    uint32_t cold;         /* cumulative_map[i] == 0         */
    double   entropy;      /* Shannon entropy (log2 bins)    */
    double   hit_mean;     /* mean over nonzero edges        */
    double   hit_std;      /* stddev over nonzero edges      */
    uint8_t  hit_max;      /* max value in cumulative_map    */
} edge_heat_t;

static void compute_edge_heat(const uint8_t *cmap, uint32_t size,
                               edge_heat_t *out)
{
    memset(out, 0, sizeof(*out));

    /* Entropy bins: powers of 2 -> 1,2,4,8,16,32,64,128+ */
    uint32_t bins[8] = {0};
    double   sum = 0.0;
    double   sum_sq = 0.0;
    uint32_t nonzero = 0;

    for (uint32_t i = 0; i < size; i++) {
        uint8_t v = cmap[i];
        if (v == 0) {
            out->cold++;
        } else if (v <= 7) {
            out->cool++;
        } else if (v <= 128) {
            out->warm++;
        } else {
            out->hot++;
        }

        if (v > out->hit_max) out->hit_max = v;

        if (v > 0) {
            nonzero++;
            sum += (double)v;
            sum_sq += (double)v * (double)v;

            /* Bin by power of 2 */
            if (v >= 128)     bins[7]++;
            else if (v >= 64) bins[6]++;
            else if (v >= 32) bins[5]++;
            else if (v >= 16) bins[4]++;
            else if (v >= 8)  bins[3]++;
            else if (v >= 4)  bins[2]++;
            else if (v >= 2)  bins[1]++;
            else              bins[0]++;
        }
    }

    /* Mean and stddev over nonzero edges */
    if (nonzero > 0) {
        out->hit_mean = sum / (double)nonzero;
        double var = (sum_sq / (double)nonzero) - (out->hit_mean * out->hit_mean);
        out->hit_std = (var > 0.0) ? sqrt(var) : 0.0;
    }

    /* Shannon entropy over the bins */
    if (nonzero > 0) {
        double ent = 0.0;
        for (int b = 0; b < 8; b++) {
            if (bins[b] == 0) continue;
            double p = (double)bins[b] / (double)nonzero;
            ent -= p * log2(p);
        }
        out->entropy = ent;
    }
}

/* ── Telemetry: write coverage dynamics CSV row ───────────────────────────── */

static void write_coverage_row(telemetry_mutator_t *m)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t elapsed_us = (uint64_t)(now.tv_sec  - m->start_time.tv_sec)  * 1000000ULL
                        + (uint64_t)(now.tv_nsec - m->start_time.tv_nsec) / 1000ULL;

    uint32_t total_edges = count_coverage(m->afl);
    uint32_t crashes     = (uint32_t)m->afl->total_crashes;
    uint64_t total_execs = m->afl->fsrv.total_execs;

    uint64_t interval_execs = m->step - m->interval_start;
    uint32_t new_edges = (total_edges > m->prev_coverage)
                         ? (total_edges - m->prev_coverage) : 0;
    uint32_t new_crashes = (crashes > m->prev_crashes)
                           ? (crashes - m->prev_crashes) : 0;

    double edge_rate = (interval_execs > 0)
                       ? (double)new_edges / (double)interval_execs : 0.0;
    double avg_exec_us = (interval_execs > 0)
                         ? (double)elapsed_us / (double)total_execs : 0.0;

    /* Corpus size: queued_items from afl state */
    uint32_t corpus_size = m->afl->queued_items;

    /* Edge heat distribution */
    edge_heat_t heat;
    compute_edge_heat(m->cumulative_map, MAP_SIZE, &heat);

    fprintf(m->csv_coverage,
            "%lu,%lu,%u,%u,%.10f,%u,%u,%.2f,%u,%u,%u,%u,%u,%.6f,%.4f,%.4f,%u\n",
            (unsigned long)elapsed_us,
            (unsigned long)total_execs,
            total_edges,
            new_edges,
            edge_rate,
            crashes,
            new_crashes,
            avg_exec_us,
            corpus_size,
            heat.hot,
            heat.warm,
            heat.cool,
            heat.cold,
            heat.entropy,
            heat.hit_mean,
            heat.hit_std,
            (unsigned)heat.hit_max);
    fflush(m->csv_coverage);
}

/* ── Telemetry: write mutation attribution CSV row ────────────────────────── */

static void write_mutation_row(telemetry_mutator_t *m)
{
    uint64_t total_execs = m->afl->fsrv.total_execs;

    fprintf(m->csv_mutation, "%lu", (unsigned long)total_execs);
    for (int a = 0; a < ACTION_SIZE; a++) {
        fprintf(m->csv_mutation, ",%u,%u,%u",
                m->mut_count[a], m->mut_new_edges[a], m->mut_crashes[a]);
    }
    fprintf(m->csv_mutation, "\n");
    fflush(m->csv_mutation);
}

/* ── Telemetry: write bitmap snapshot ─────────────────────────────────────── */

static void write_snapshot(telemetry_mutator_t *m)
{
    char path[768];
    snprintf(path, sizeof(path), "%s/snapshot_%lu.bin",
             m->snapshot_dir, (unsigned long)m->afl->fsrv.total_execs);

    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(m->cumulative_map, 1, MAP_SIZE, f);
        fclose(f);
    }
}

/* ── AFL++ API: init ──────────────────────────────────────────────────────── */

telemetry_mutator_t *afl_custom_init(afl_state_t *afl, unsigned int seed)
{
    srand(seed);

    /* Read required env vars */
    const char *csv_dir = getenv("TELEMETRY_CSV_DIR");
    const char *version = getenv("TELEMETRY_VERSION");
    if (!csv_dir || !version) {
        fprintf(stderr, "[-] TELEMETRY mutator: TELEMETRY_CSV_DIR and "
                        "TELEMETRY_VERSION must be set\n");
        return NULL;
    }

    /* Read optional env vars */
    const char *log_int_str  = getenv("TELEMETRY_LOG_INTERVAL");
    const char *snap_int_str = getenv("TELEMETRY_SNAPSHOT_INTERVAL");

    telemetry_mutator_t *m = calloc(1, sizeof(telemetry_mutator_t));
    if (!m) { perror("calloc"); return NULL; }

    m->afl         = afl;
    m->mutated_buf = malloc(MAX_MUTATED_SIZE);
    if (!m->mutated_buf) { perror("malloc"); free(m); return NULL; }

    m->step           = 0;
    m->interval_start = 0;
    m->prev_coverage  = 0;
    m->prev_crashes   = 0;
    m->last_action    = -1;  /* no previous action yet */

    m->log_interval      = log_int_str  ? (uint32_t)atoi(log_int_str)
                                        : DEFAULT_LOG_INTERVAL;
    m->snapshot_interval = snap_int_str ? (uint32_t)atoi(snap_int_str)
                                        : DEFAULT_SNAPSHOT_INTERVAL;

    if (m->log_interval == 0) m->log_interval = DEFAULT_LOG_INTERVAL;
    if (m->snapshot_interval == 0) m->snapshot_interval = DEFAULT_SNAPSHOT_INTERVAL;

    memset(m->cumulative_map, 0, MAP_SIZE);
    memset(m->mut_count, 0, sizeof(m->mut_count));
    memset(m->mut_new_edges, 0, sizeof(m->mut_new_edges));
    memset(m->mut_crashes, 0, sizeof(m->mut_crashes));

    /* Create CSV directory and snapshot subdirectory */
    mkdir(csv_dir, 0755);  /* ignore error if exists */

    snprintf(m->snapshot_dir, sizeof(m->snapshot_dir),
             "%s/snapshots_%s", csv_dir, version);
    mkdir(m->snapshot_dir, 0755);

    /* Open coverage dynamics CSV */
    char path[768];
    snprintf(path, sizeof(path), "%s/coverage_dynamics_%s.csv", csv_dir, version);
    m->csv_coverage = fopen(path, "w");
    if (!m->csv_coverage) {
        fprintf(stderr, "[-] TELEMETRY: cannot open %s: %s\n",
                path, strerror(errno));
        free(m->mutated_buf); free(m); return NULL;
    }

    /* Write coverage CSV header */
    fprintf(m->csv_coverage,
            "timestamp_us,total_execs,total_edges,new_edges_this_interval,"
            "edge_discovery_rate,crashes_total,crashes_this_interval,"
            "avg_exec_time_us,corpus_size,hot_edges,warm_edges,cool_edges,"
            "cold_edges,edge_entropy,edge_hit_mean,edge_hit_std,edge_hit_max\n");
    fflush(m->csv_coverage);

    /* Open mutation attribution CSV */
    snprintf(path, sizeof(path), "%s/mutation_attribution_%s.csv", csv_dir, version);
    m->csv_mutation = fopen(path, "w");
    if (!m->csv_mutation) {
        fprintf(stderr, "[-] TELEMETRY: cannot open %s: %s\n",
                path, strerror(errno));
        fclose(m->csv_coverage);
        free(m->mutated_buf); free(m); return NULL;
    }

    /* Write mutation CSV header */
    fprintf(m->csv_mutation, "total_execs");
    for (int a = 0; a < ACTION_SIZE; a++) {
        fprintf(m->csv_mutation, ",mut_%02d_count,mut_%02d_new_edges,mut_%02d_crashes",
                a, a, a);
    }
    fprintf(m->csv_mutation, "\n");
    fflush(m->csv_mutation);

    clock_gettime(CLOCK_MONOTONIC, &m->start_time);

    printf("[+] TELEMETRY mutator initialized: version=%s log_interval=%u "
           "snapshot_interval=%u\n", version, m->log_interval, m->snapshot_interval);
    printf("[+] TELEMETRY CSV dir: %s\n", csv_dir);
    printf("[+] TELEMETRY snapshots: %s\n", m->snapshot_dir);

    return m;
}

/* ── AFL++ API: fuzz ──────────────────────────────────────────────────────── */

size_t afl_custom_fuzz(telemetry_mutator_t *m,
                        uint8_t *buf, size_t buf_size,
                        uint8_t **out_buf,
                        uint8_t *add_buf, size_t add_buf_size,
                        size_t max_size)
{
    /* ── Step 1: Read current coverage and crashes ─────────────────────── */
    uint32_t coverage = count_coverage(m->afl);
    uint32_t crashes  = (uint32_t)m->afl->total_crashes;

    /* ── Step 2: Attribute new edges/crashes to PREVIOUS action ────────── */
    if (m->last_action >= 0 && m->last_action < ACTION_SIZE) {
        uint32_t new_edges  = (coverage > m->prev_coverage)
                              ? (coverage - m->prev_coverage) : 0;
        uint32_t new_crashes = (crashes > m->prev_crashes)
                               ? (crashes - m->prev_crashes) : 0;

        m->mut_new_edges[m->last_action] += new_edges;
        m->mut_crashes[m->last_action]   += new_crashes;
    }

    m->prev_coverage = coverage;
    m->prev_crashes  = crashes;

    /* ── Step 3: Update cumulative bitmap ──────────────────────────────── */
    if (m->afl->shm.map) {
        const uint8_t *trace = m->afl->shm.map;
        uint32_t map_sz = (m->afl->total_bitmap_size < MAP_SIZE)
                          ? m->afl->total_bitmap_size : MAP_SIZE;
        for (uint32_t i = 0; i < map_sz; i++) {
            if (trace[i] > m->cumulative_map[i])
                m->cumulative_map[i] = trace[i];
        }
    }

    /* ── Step 4: Interval CSV writes ──────────────────────────────────── */
    if (m->step > 0 && (m->step % m->log_interval) == 0) {
        write_coverage_row(m);
        write_mutation_row(m);

        /* Reset per-interval accumulators */
        m->interval_start = m->step;
        memset(m->mut_count, 0, sizeof(m->mut_count));
        memset(m->mut_new_edges, 0, sizeof(m->mut_new_edges));
        memset(m->mut_crashes, 0, sizeof(m->mut_crashes));
    }

    /* ── Step 5: Snapshot writes ──────────────────────────────────────── */
    if (m->step > 0 && (m->step % m->snapshot_interval) == 0) {
        write_snapshot(m);
    }

    /* ── Step 6: Select random action, apply mutation ─────────────────── */
    int action = rand() % ACTION_SIZE;

    size_t mut_len = (buf_size < MAX_MUTATED_SIZE) ? buf_size : MAX_MUTATED_SIZE;
    memcpy(m->mutated_buf, buf, mut_len);
    *out_buf = m->mutated_buf;

    mut_len = apply_mutation(m->afl, m->mutated_buf, mut_len, max_size,
                             action, add_buf, add_buf_size);

    /* ── Step 7: Record action for attribution ────────────────────────── */
    m->mut_count[action]++;
    m->last_action = action;

    /* ── Step 8: Increment step ───────────────────────────────────────── */
    m->step++;

    return mut_len;
}

/* ── AFL++ API: deinit ────────────────────────────────────────────────────── */

void afl_custom_deinit(telemetry_mutator_t *m)
{
    if (!m) return;

    /* Final CSV flush if there is data since last interval write */
    if (m->step > m->interval_start) {
        write_coverage_row(m);
        write_mutation_row(m);
    }

    /* Final snapshot */
    write_snapshot(m);

    /* Print summary */
    uint32_t final_coverage = count_coverage(m->afl);
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed_s = (double)(now.tv_sec  - m->start_time.tv_sec)
                     + (double)(now.tv_nsec - m->start_time.tv_nsec) / 1e9;

    printf("\n[+] TELEMETRY mutator summary:\n");
    printf("    Total steps:    %lu\n", (unsigned long)m->step);
    printf("    Total execs:    %lu\n", (unsigned long)m->afl->fsrv.total_execs);
    printf("    Final coverage: %u edges\n", final_coverage);
    printf("    Total crashes:  %u\n", (uint32_t)m->afl->total_crashes);
    printf("    Wall time:      %.1f s\n", elapsed_s);
    printf("    Avg execs/sec:  %.0f\n",
           elapsed_s > 0 ? (double)m->afl->fsrv.total_execs / elapsed_s : 0.0);

    if (m->csv_coverage) fclose(m->csv_coverage);
    if (m->csv_mutation) fclose(m->csv_mutation);
    free(m->mutated_buf);
    free(m);
}
