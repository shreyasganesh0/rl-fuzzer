/*
 * mutator_m1.c  —  RL server  /  Model M1
 *
 * Extended state model.  In addition to the three base metrics (coverage,
 * new_edges, crashes) this mutator maintains a runtime table of:
 *
 *   enabled_counts[edge_id][action]  — times action was chosen at edge_id
 *                                      and coverage subsequently increased
 *   disabled_counts[edge_id][action] — times action was chosen at edge_id
 *                                      and coverage did NOT increase
 *
 * At each step the counts for the CURRENT hot edge are written into the SHM
 * so the Python DQN can build a 97-element state vector:
 *   [0..2]   coverage, new_edges, crashes (normalised)
 *   [3..49]  enabled_ratio[47]   = enabled[a] / (enabled[a]+disabled[a]+1)
 *   [50..96] experience_norm[47] = log1p(enabled[a]+disabled[a]) / log1p(MAX_EXP)
 *
 * SHM layout  (1024 bytes, path /tmp/rl_shm_m1)
 *
 *   STATE REGION  [0..511]
 *     [0]    state_seq         u32   release-stored last
 *     [4]    coverage          u32
 *     [8]    new_edges         u32
 *     [12]   crashes           u32
 *     [16]   _pad              u32
 *     [20]   _pad              u32
 *     [24]   total_execs       u64
 *     [32]   enabled_counts[47] u32[47]  (188 bytes → offsets 32..219)
 *     [220]  disabled_counts[47] u32[47] (188 bytes → offsets 220..407)
 *     [408]  _pad[104]
 *
 *   ACTION REGION [512..1023]
 *     [512]  action_seq        u32
 *     [516]  action            i32   0..46
 *     [520]  _pad[504]
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

#define SHM_PATH_M1      "/tmp/rl_shm_m1"
#define SHM_SIZE_M1      1024
#define MAX_MUTATED_SIZE (1024 * 1024)
#define ACTION_SIZE      47
#define ARITH_MAX        35

/*
 * M1_MAP_SIZE: upper bound on edge IDs tracked in the count tables.
 * AFL++ default MAP_SIZE is 65536 (2^16).  We allocate
 *   M1_MAP_SIZE * ACTION_SIZE * sizeof(uint32_t) * 2  ≈ 24 MB  on the heap.
 * Any edge_id >= M1_MAP_SIZE is clamped to slot 0 (safe fallback).
 */
#define M1_MAP_SIZE      65536

/* SHM offsets — STATE region */
#define OFF_STATE_SEQ    0
#define OFF_COVERAGE     4
#define OFF_NEW_EDGES    8
#define OFF_CRASHES      12
#define OFF_TOTAL_EXECS  24
#define OFF_ENABLED      32    /* uint32_t[47], 188 bytes */
#define OFF_DISABLED     220   /* uint32_t[47], 188 bytes */

/* SHM offsets — ACTION region */
#define OFF_ACTION_SEQ   512
#define OFF_ACTION       516

#define SPIN_NS          100000   /* 0.1 ms */

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

    int      shm_fd;
    void    *shm;

    /* Per-edge per-action outcome counts (heap allocated) */
    uint32_t *enabled_counts;    /* [M1_MAP_SIZE * ACTION_SIZE] */
    uint32_t *disabled_counts;   /* [M1_MAP_SIZE * ACTION_SIZE] */

    /* Tracking for count updates (filled at end of each fuzz call) */
    uint32_t last_edge_id;
    int      last_action;        /* -1 = no previous action yet */

    uint32_t prev_coverage;
    uint32_t prev_crashes;

    uint32_t state_seq;
    uint32_t last_action_seq;
} my_mutator_t;

/* ── SHM access helpers ───────────────────────────────────────────────────── */

static inline volatile uint32_t *u32_at(void *base, size_t off) {
    return (volatile uint32_t *)((uint8_t *)base + off);
}
static inline volatile int32_t *i32_at(void *base, size_t off) {
    return (volatile int32_t *)((uint8_t *)base + off);
}
static inline volatile uint64_t *u64_at(void *base, size_t off) {
    return (volatile uint64_t *)((uint8_t *)base + off);
}

/*
 * shm_push_state_m1 — writes the full M1 state into shared memory.
 * The enabled/disabled count rows for edge_id are copied before the
 * release-store of state_seq so Python sees a consistent snapshot.
 */
static void shm_push_state_m1(my_mutator_t *m,
                               uint32_t coverage, uint32_t new_edges,
                               uint32_t crashes,  uint64_t total_execs,
                               uint32_t edge_id)
{
    void    *s   = m->shm;
    uint32_t eid = (edge_id < M1_MAP_SIZE) ? edge_id : 0;

    *u32_at(s, OFF_COVERAGE)    = coverage;
    *u32_at(s, OFF_NEW_EDGES)   = new_edges;
    *u32_at(s, OFF_CRASHES)     = crashes;
    *u64_at(s, OFF_TOTAL_EXECS) = total_execs;

    /* Copy the 47-element enabled/disabled rows for this edge */
    const uint32_t *en  = m->enabled_counts  + (size_t)eid * ACTION_SIZE;
    const uint32_t *dis = m->disabled_counts + (size_t)eid * ACTION_SIZE;
    volatile uint32_t *shm_en  = u32_at(s, OFF_ENABLED);
    volatile uint32_t *shm_dis = u32_at(s, OFF_DISABLED);
    for (int a = 0; a < ACTION_SIZE; a++) {
        shm_en[a]  = en[a];
        shm_dis[a] = dis[a];
    }

    /* Release-store: all prior stores visible before this */
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

/* ── Coverage helpers ─────────────────────────────────────────────────────── */

static uint32_t count_coverage(afl_state_t *afl)
{
    uint32_t n = 0, sz = afl->total_bitmap_size;
    const uint8_t *v = afl->virgin_bits;
    for (uint32_t i = 0; i < sz; i++)
        if (v[i] != 0xFF) n++;
    return n;
}

/*
 * find_current_edge — highest-index new edge in this execution, or the
 * hottest edge if no new edges were found.
 */
static uint32_t find_current_edge(afl_state_t *afl)
{
    uint32_t       sz    = afl->total_bitmap_size;
    const uint8_t *trace = afl->fsrv.trace_bits;
    const uint8_t *vgn   = afl->virgin_bits;

    for (int32_t i = (int32_t)sz - 1; i >= 1; i--)
        if (trace[i] && vgn[i] == 0xFF) return (uint32_t)i;

    uint32_t best_idx = 0, best_val = 0;
    for (uint32_t i = 1; i < sz; i++)
        if (trace[i] > best_val) { best_val = trace[i]; best_idx = i; }
    return best_idx;
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

/* ── Core mutation dispatcher (identical 47-op table as M0) ──────────────── */

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

    case 0: pos=RND((int)mut_len*8); mb[pos/8]^=(uint8_t)(1<<(pos%8)); break;
    case 1: pos=RND((int)mut_len*8); mb[pos/8]^=(uint8_t)(1<<(pos%8));
            { int p2=(pos+1)%((int)mut_len*8); mb[p2/8]^=(uint8_t)(1<<(p2%8)); } break;
    case 2: pos=RND((int)mut_len*8);
            for (int b=0;b<4;b++){int bp=(pos+b)%((int)mut_len*8);mb[bp/8]^=(uint8_t)(1<<(bp%8));}break;
    case 3: mb[RPOS]^=0xFF; break;
    case 4: NEED(2);pos=RND((int)mut_len-1);mb[pos]^=0xFF;mb[pos+1]^=0xFF; break;
    case 5: NEED(4);pos=RND((int)mut_len-3);
            mb[pos]^=0xFF;mb[pos+1]^=0xFF;mb[pos+2]^=0xFF;mb[pos+3]^=0xFF; break;

    case 6:  mb[RPOS]++; break;
    case 7:  mb[RPOS]--; break;
    case 8:  NEED(2);pos=RND((int)mut_len-1);delta=RDELTA;
             *(uint16_t*)(mb+pos)=(uint16_t)(*(uint16_t*)(mb+pos)+delta); break;
    case 9:  NEED(2);pos=RND((int)mut_len-1);delta=RDELTA;
             *(uint16_t*)(mb+pos)=(uint16_t)(*(uint16_t*)(mb+pos)-delta); break;
    case 10: NEED(2);pos=RND((int)mut_len-1);delta=RDELTA;
             v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)(v16+delta)); break;
    case 11: NEED(2);pos=RND((int)mut_len-1);delta=RDELTA;
             v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)(v16-delta)); break;
    case 12: NEED(4);pos=RND((int)mut_len-3);delta=RDELTA;*(uint32_t*)(mb+pos)+=delta; break;
    case 13: NEED(4);pos=RND((int)mut_len-3);delta=RDELTA;*(uint32_t*)(mb+pos)-=delta; break;
    case 14: NEED(4);pos=RND((int)mut_len-3);delta=RDELTA;
             v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos)=bswap32(v32+delta); break;
    case 15: NEED(4);pos=RND((int)mut_len-3);delta=RDELTA;
             v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos)=bswap32(v32-delta); break;

    case 16: mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)]; break;
    case 17: NEED(2);pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=(uint16_t)MYINTERESTING_16[RND(N16)]; break;
    case 18: NEED(2);pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)MYINTERESTING_16[RND(N16)]); break;
    case 19: NEED(4);pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)=(uint32_t)MYINTERESTING_32[RND(N32)]; break;
    case 20: NEED(4);pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)=bswap32((uint32_t)MYINTERESTING_32[RND(N32)]); break;

    case 21: pos=RND((int)mut_len*8);mb[pos/8]^=(uint8_t)(128>>(pos%8)); break;
    case 22: mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)]; break;
    case 23: NEED(2);pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=(uint16_t)MYINTERESTING_16[RND(N16)]; break;
    case 24: NEED(2);pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)MYINTERESTING_16[RND(N16)]); break;
    case 25: NEED(4);pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)=(uint32_t)MYINTERESTING_32[RND(N32)]; break;
    case 26: NEED(4);pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)=bswap32((uint32_t)MYINTERESTING_32[RND(N32)]); break;
    case 27: mb[RPOS]-=(uint8_t)(1+RND(ARITH_MAX)); break;
    case 28: mb[RPOS]+=(uint8_t)(1+RND(ARITH_MAX)); break;
    case 29: NEED(2);pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=(uint16_t)(*(uint16_t*)(mb+pos)-(1+RND(ARITH_MAX))); break;
    case 30: NEED(2);pos=RND((int)mut_len-1);
             v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)(v16-(1+RND(ARITH_MAX)))); break;
    case 31: NEED(2);pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=(uint16_t)(*(uint16_t*)(mb+pos)+(1+RND(ARITH_MAX))); break;
    case 32: NEED(2);pos=RND((int)mut_len-1);
             v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)(v16+(1+RND(ARITH_MAX)))); break;
    case 33: NEED(4);pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)-=(1u+(uint32_t)RND(ARITH_MAX)); break;
    case 34: NEED(4);pos=RND((int)mut_len-3);
             v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos)=bswap32(v32-(1u+(uint32_t)RND(ARITH_MAX))); break;
    case 35: NEED(4);pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)+=(1u+(uint32_t)RND(ARITH_MAX)); break;
    case 36: NEED(4);pos=RND((int)mut_len-3);
             v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos)=bswap32(v32+(1u+(uint32_t)RND(ARITH_MAX))); break;
    case 37: mb[RPOS]=(uint8_t)(rand()&0xFF); break;
    case 38: mb[RPOS]+=(uint8_t)(rand()&0x1F); break;
    case 39: mb[RPOS]-=(uint8_t)(rand()&0x1F); break;
    case 40: mb[RPOS]^=(uint8_t)(rand()&0xFF); break;

    case 41: tok_len=pick_user_extra(afl,&tok);
             if (!tok_len){mb[RPOS]=(uint8_t)(rand()&0xFF);break;}
             dict_overwrite(mb,mut_len,RPOS,tok,tok_len); break;
    case 42: tok_len=pick_user_extra(afl,&tok);
             if (!tok_len||mut_len+tok_len>max_size){mb[RPOS]^=0xFF;break;}
             pos=RND((int)mut_len+1);
             mut_len=dict_insert(mb,mut_len,pos,tok,tok_len,max_size); break;
    case 43: tok_len=pick_auto_extra(afl,&tok);
             if (!tok_len){mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)];break;}
             dict_overwrite(mb,mut_len,RPOS,tok,tok_len); break;
    case 44: tok_len=pick_auto_extra(afl,&tok);
             if (!tok_len||mut_len+tok_len>max_size){mb[RPOS]++;break;}
             pos=RND((int)mut_len+1);
             mut_len=dict_insert(mb,mut_len,pos,tok,tok_len,max_size); break;

    case 45: {
        int nops=4+RND(5);
        for (int op=0;op<nops&&mut_len>0;op++) {
            switch(RND(8)){
            case 0:{int bp=RND((int)mut_len*8);mb[bp/8]^=(uint8_t)(128>>(bp%8));}break;
            case 1:mb[RPOS]+=(uint8_t)(1+RND(ARITH_MAX));break;
            case 2:mb[RPOS]-=(uint8_t)(1+RND(ARITH_MAX));break;
            case 3:mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)];break;
            case 4:mb[RPOS]=(uint8_t)(rand()&0xFF);break;
            case 5:mb[RPOS]^=0xFF;break;
            case 6:if(mut_len>=2){int p=RND((int)mut_len-1);*(uint16_t*)(mb+p)=(uint16_t)MYINTERESTING_16[RND(N16)];}break;
            case 7:if(mut_len>=4){int p=RND((int)mut_len-3);*(uint32_t*)(mb+p)=(uint32_t)MYINTERESTING_32[RND(N32)];}break;
            }
        }
        break;
    }

    default:
    case 46: {
        int stack=1<<(1+RND(HAVOC_STACK_POW2));
        for (int op=0;op<stack&&mut_len>0;op++) {
            switch(RND(12)){
            case 0:{int bp=RND((int)mut_len*8);mb[bp/8]^=(uint8_t)(128>>(bp%8));}break;
            case 1:mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)];break;
            case 2:mb[RPOS]+=(uint8_t)(1+RND(ARITH_MAX));break;
            case 3:mb[RPOS]-=(uint8_t)(1+RND(ARITH_MAX));break;
            case 4:mb[RPOS]=(uint8_t)(rand()&0xFF);break;
            case 5:if(mut_len>2){int from=RPOS,dlen=1+RND((int)mut_len-from);
                       if((size_t)(from+dlen)>mut_len)dlen=(int)mut_len-from;
                       memmove(mb+from,mb+from+dlen,mut_len-(size_t)from-(size_t)dlen);
                       mut_len-=(size_t)dlen;}break;
            case 6:if(mut_len>=2){int src=RPOS,dst=RPOS,cl=1+RND(8);
                       if((size_t)(src+cl)>mut_len)cl=(int)mut_len-src;
                       if((size_t)(dst+cl)>mut_len)cl=(int)mut_len-dst;
                       if(cl>0)memmove(mb+dst,mb+src,(size_t)cl);}break;
            case 7:{int p=RPOS,sl=1+RND(8);
                    if((size_t)(p+sl)>mut_len)sl=(int)mut_len-p;
                    memset(mb+p,rand()&0xFF,(size_t)sl);}break;
            case 8:if(mut_len>=2){int p=RND((int)mut_len-1);*(uint16_t*)(mb+p)=(uint16_t)MYINTERESTING_16[RND(N16)];}break;
            case 9:if(mut_len>=4){int p=RND((int)mut_len-3);*(uint32_t*)(mb+p)=(uint32_t)MYINTERESTING_32[RND(N32)];}break;
            case 10:tok_len=pick_user_extra(afl,&tok);
                    if(tok_len)dict_overwrite(mb,mut_len,RPOS,tok,tok_len);
                    else mb[RPOS]^=0xAA; break;
            case 11:if(add_buf&&add_buf_size>0&&mut_len>=2){
                        int split=1+RND((int)mut_len-1),add_off=RND((int)add_buf_size);
                        int cl=(int)mut_len-split;
                        if((size_t)(add_off+cl)>add_buf_size)cl=(int)add_buf_size-add_off;
                        if(cl>0)memcpy(mb+split,add_buf+add_off,(size_t)cl);}break;
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

    m->afl         = afl;
    m->mutated_buf = malloc(MAX_MUTATED_SIZE);
    if (!m->mutated_buf) { perror("malloc"); free(m); return NULL; }

    /* Allocate count tables (~24 MB total) */
    size_t table_sz = (size_t)M1_MAP_SIZE * ACTION_SIZE;
    m->enabled_counts  = calloc(table_sz, sizeof(uint32_t));
    m->disabled_counts = calloc(table_sz, sizeof(uint32_t));
    if (!m->enabled_counts || !m->disabled_counts) {
        perror("[-] M1 count table calloc");
        free(m->enabled_counts);
        free(m->disabled_counts);
        free(m->mutated_buf);
        free(m);
        return NULL;
    }

    m->last_action  = -1;   /* sentinel: no previous action yet */
    m->last_edge_id = 0;

    m->shm_fd = open(SHM_PATH_M1, O_RDWR | O_CREAT, 0600);
    if (m->shm_fd < 0) { perror("[-] M1 SHM open"); m->shm = NULL; return m; }
    if (ftruncate(m->shm_fd, SHM_SIZE_M1) < 0) perror("[-] M1 SHM ftruncate");

    m->shm = mmap(NULL, SHM_SIZE_M1, PROT_READ|PROT_WRITE, MAP_SHARED, m->shm_fd, 0);
    if (m->shm == MAP_FAILED) {
        perror("[-] M1 SHM mmap"); m->shm = NULL;
    } else {
        printf("[+] M1 mutator: SHM mapped at %s  (count tables: %zu MB)\n",
               SHM_PATH_M1, (table_sz * sizeof(uint32_t) * 2) >> 20);
    }

    if (m->shm)
        m->last_action_seq = __atomic_load_n(
            u32_at(m->shm, OFF_ACTION_SEQ), __ATOMIC_ACQUIRE);

    m->state_seq     = 0;
    m->prev_coverage = 0;
    m->prev_crashes  = 0;
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
        uint32_t coverage    = count_coverage(m->afl);
        uint32_t crashes     = (uint32_t)m->afl->total_crashes;
        uint64_t total_execs = m->afl->fsrv.total_execs;
        uint32_t edge_id     = find_current_edge(m->afl);
        uint32_t safe_eid    = (edge_id < M1_MAP_SIZE) ? edge_id : 0;
        uint32_t new_edges   = (coverage > m->prev_coverage)
                               ? (coverage - m->prev_coverage) : 0;

        /*
         * Update counts for the PREVIOUS (last_action, last_edge_id) pair
         * now that we can observe the outcome: did coverage increase?
         */
        if (m->last_action >= 0) {
            uint32_t last_eid = (m->last_edge_id < M1_MAP_SIZE)
                                ? m->last_edge_id : 0;
            size_t idx = (size_t)last_eid * ACTION_SIZE + (size_t)m->last_action;
            if (coverage > m->prev_coverage)
                m->enabled_counts[idx]++;
            else
                m->disabled_counts[idx]++;
        }

        /* Push extended state (including count rows for current edge) */
        shm_push_state_m1(m, coverage, new_edges, crashes, total_execs, safe_eid);

        action = shm_wait_action(m);
        if (action < 0 || action >= ACTION_SIZE) action = 46;

        /* Save context for next iteration's count update */
        m->last_edge_id  = safe_eid;
        m->last_action   = action;
        m->prev_coverage = coverage;
        m->prev_crashes  = crashes;
    }

    size_t mut_len = (buf_size < MAX_MUTATED_SIZE) ? buf_size : MAX_MUTATED_SIZE;
    memcpy(m->mutated_buf, buf, mut_len);
    *out_buf = m->mutated_buf;

    mut_len = apply_mutation(m->afl, m->mutated_buf, mut_len, max_size,
                             action, add_buf, add_buf_size);
    return mut_len;
}

void afl_custom_deinit(my_mutator_t *m)
{
    if (m->shm && m->shm != MAP_FAILED) munmap(m->shm, SHM_SIZE_M1);
    if (m->shm_fd > 0) close(m->shm_fd);
    free(m->enabled_counts);
    free(m->disabled_counts);
    free(m->mutated_buf);
    free(m);
}
