/*
 * mutator_m1_0.c  —  RL mutator / Model M1_0
 *
 * Edge stability model — full edge set.
 *
 * Per fuzz step, after AFL++ executes the target, this mutator:
 *   1. For every edge i in the 65536-entry trace_bits map:
 *        enabled[i]++  if trace[i] > 0  (edge was active this step)
 *        disabled[i]++ if trace[i] == 0 AND prev_trace[i] > 0  (edge dropped out)
 *   2. In the same O(MAP_SIZE) pass, computes distribution statistics
 *      over the entire enabled[] and disabled[] arrays.
 *   3. Writes those statistics (as raw u32/u64/f32) to SHM so Python can
 *      normalise with the known training-step budget.
 *
 * Step 0 is skipped (prev_trace is all-zero → no meaningful disabled signal).
 * A default HAVOC action is used for step 0 and prev_trace is seeded.
 *
 * SHM layout (256 bytes, /tmp/rl_shm_m1_0):
 *
 *   STATE REGION [0..127]
 *     [0]   state_seq     u32   release-stored
 *     [4]   coverage      u32
 *     [8]   new_edges     u32
 *     [12]  crashes       u32
 *     [16]  _pad          u32
 *     [20]  _pad          u32
 *     [24]  total_execs   u64
 *     [32]  n_nonzero_en  u32   count(enabled[i] > 0)
 *     [36]  n_nonzero_dis u32   count(disabled[i] > 0)
 *     [40]  max_en        u32   max(enabled[i])
 *     [44]  max_dis       u32   max(disabled[i])
 *     [48]  sum_en        u64   sum(enabled[i])
 *     [56]  sum_sq_en     u64   sum(enabled[i]^2)   for std computation
 *     [64]  sum_dis       u64   sum(disabled[i])
 *     [72]  sum_sq_dis    u64   sum(disabled[i]^2)
 *     [80]  sum_stability f32   sum(en[i]/(en[i]+dis[i]+1))
 *     [84]  total_edges   u32   = MAP_SIZE (65536) for M1_0
 *     [88]  step_count    u32   current step number
 *     [92]  _pad[36]
 *
 *   ACTION REGION [128..255]
 *     [128] action_seq    u32
 *     [132] action        i32   0..46
 *     [136] _pad[120]
 *
 * State vector built in Python (12 elements):
 *   [0]  coverage_n             = coverage / 65536
 *   [1]  new_edges_n            = min(new_edges,100) / 100
 *   [2]  crashes_n              = log1p(crashes) / log1p(1000)
 *   [3]  enabled_mean_n         = (sum_en  / total_edges) / train_steps
 *   [4]  enabled_std_n          = std(enabled)  / train_steps
 *   [5]  enabled_max_n          = max_en  / train_steps
 *   [6]  enabled_nonzero_frac   = n_nonzero_en  / total_edges
 *   [7]  disabled_mean_n        = (sum_dis / total_edges) / train_steps
 *   [8]  disabled_std_n         = std(disabled) / train_steps
 *   [9]  disabled_max_n         = max_dis / train_steps
 *   [10] disabled_nonzero_frac  = n_nonzero_dis / total_edges
 *   [11] mean_stability_ratio   = sum_stability / total_edges
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

/* ── Constants ────────────────────────────────────────────────────────────── */

#define SHM_PATH      "/tmp/rl_shm_m1_0"
#define SHM_SIZE      256
#define MAP_SIZE      65536
#define MAX_MUTATED_SIZE (1024 * 1024)
#define ACTION_SIZE   47
#define ARITH_MAX     35

/* STATE offsets */
#define OFF_STATE_SEQ    0
#define OFF_COVERAGE     4
#define OFF_NEW_EDGES    8
#define OFF_CRASHES      12
#define OFF_TOTAL_EXECS  24
#define OFF_N_NZ_EN      32
#define OFF_N_NZ_DIS     36
#define OFF_MAX_EN       40
#define OFF_MAX_DIS      44
#define OFF_SUM_EN       48
#define OFF_SUM_SQ_EN    56
#define OFF_SUM_DIS      64
#define OFF_SUM_SQ_DIS   72
#define OFF_SUM_STAB     80
#define OFF_TOTAL_EDGES  84
#define OFF_STEP_COUNT   88

/* ACTION offsets */
#define OFF_ACTION_SEQ  128
#define OFF_ACTION      132

#define SPIN_NS  100000

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

    uint32_t    *enabled;     /* [MAP_SIZE] — per-edge active counts   */
    uint32_t    *disabled;    /* [MAP_SIZE] — per-edge dropout counts  */
    uint8_t     *prev_trace;  /* [MAP_SIZE] — snapshot of last trace   */

    uint32_t     step_count;
    uint32_t     prev_coverage;
    uint32_t     state_seq;
    uint32_t     last_action_seq;
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
static inline volatile float *f32_at(void *base, size_t off) {
    return (volatile float *)((uint8_t *)base + off);
}

/*
 * shm_push_state_m1_0
 *
 * Single O(MAP_SIZE) pass:
 *   - updates enabled[i] / disabled[i] from current and previous trace
 *   - accumulates distribution statistics over the FULL edge set
 *   - writes everything to SHM atomically via state_seq release-store
 */
static void shm_push_state(my_mutator_t *m,
                            uint32_t coverage, uint32_t new_edges,
                            uint32_t crashes,  uint64_t total_execs)
{
    const uint8_t *trace = m->afl->fsrv.trace_bits;
    uint32_t       sz    = m->afl->total_bitmap_size;
    if (sz > MAP_SIZE) sz = MAP_SIZE;

    uint64_t sum_en   = 0, sum_sq_en  = 0;
    uint64_t sum_dis  = 0, sum_sq_dis = 0;
    uint32_t nz_en    = 0, nz_dis     = 0;
    uint32_t max_en   = 0, max_dis    = 0;
    float    sum_stab = 0.0f;

    for (uint32_t i = 0; i < sz; i++) {
        uint8_t cur = trace[i];
        uint8_t prv = m->prev_trace[i];

        if (cur)       m->enabled[i]++;
        if (!cur && prv) m->disabled[i]++;

        uint32_t en  = m->enabled[i];
        uint32_t dis = m->disabled[i];

        if (en) {
            nz_en++;
            sum_en    += en;
            sum_sq_en += (uint64_t)en * en;
            if (en > max_en) max_en = en;
        }
        if (dis) {
            nz_dis++;
            sum_dis    += dis;
            sum_sq_dis += (uint64_t)dis * dis;
            if (dis > max_dis) max_dis = dis;
        }
        sum_stab += (float)en / (float)(en + dis + 1u);
    }
    memcpy(m->prev_trace, trace, sz);

    void *s = m->shm;
    *u32_at(s, OFF_COVERAGE)     = coverage;
    *u32_at(s, OFF_NEW_EDGES)    = new_edges;
    *u32_at(s, OFF_CRASHES)      = crashes;
    *u64_at(s, OFF_TOTAL_EXECS)  = total_execs;
    *u32_at(s, OFF_N_NZ_EN)      = nz_en;
    *u32_at(s, OFF_N_NZ_DIS)     = nz_dis;
    *u32_at(s, OFF_MAX_EN)       = max_en;
    *u32_at(s, OFF_MAX_DIS)      = max_dis;
    *u64_at(s, OFF_SUM_EN)       = sum_en;
    *u64_at(s, OFF_SUM_SQ_EN)    = sum_sq_en;
    *u64_at(s, OFF_SUM_DIS)      = sum_dis;
    *u64_at(s, OFF_SUM_SQ_DIS)   = sum_sq_dis;
    *f32_at(s, OFF_SUM_STAB)     = sum_stab;
    *u32_at(s, OFF_TOTAL_EDGES)  = sz;
    *u32_at(s, OFF_STEP_COUNT)   = m->step_count;

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

/* ── Coverage helper ──────────────────────────────────────────────────────── */

static uint32_t count_coverage(afl_state_t *afl)
{
    uint32_t n = 0, sz = afl->total_bitmap_size;
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
    *out = afl->extras[i].data; return afl->extras[i].len;
}
static uint32_t pick_auto_extra(afl_state_t *afl, const uint8_t **out) {
    if (!afl->a_extras_cnt) return 0;
    uint32_t i = (uint32_t)(rand() % (int)afl->a_extras_cnt);
    *out = afl->a_extras[i].data; return afl->a_extras[i].len;
}

/* ── Core mutation dispatcher (47 actions, identical across all models) ───── */

static size_t apply_mutation(afl_state_t *afl,
                              uint8_t *mb, size_t mut_len, size_t max_size,
                              int action,
                              uint8_t *add_buf, size_t add_buf_size)
{
    int pos; uint16_t v16; uint32_t v32, delta;
    const uint8_t *tok; uint32_t tok_len;

#define NEED(n)  if (mut_len < (size_t)(n)) break
#define RND(n)   ((int)(rand() % (unsigned)(n)))
#define RPOS     ((int)(rand() % (unsigned)mut_len))
#define RDELTA   (1u + (uint32_t)RND(ARITH_MAX))

    switch (action) {
    case 0: pos=RND((int)mut_len*8); mb[pos/8]^=(uint8_t)(1<<(pos%8)); break;
    case 1: pos=RND((int)mut_len*8); mb[pos/8]^=(uint8_t)(1<<(pos%8));
            { int p2=(pos+1)%((int)mut_len*8); mb[p2/8]^=(uint8_t)(1<<(p2%8)); } break;
    case 2: pos=RND((int)mut_len*8);
            for(int b=0;b<4;b++){int bp=(pos+b)%((int)mut_len*8);mb[bp/8]^=(uint8_t)(1<<(bp%8));}break;
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
             v32=bswap32(*(uint32_t*)(mb+pos));*(uint32_t*)(mb+pos)=bswap32(v32+delta); break;
    case 15: NEED(4);pos=RND((int)mut_len-3);delta=RDELTA;
             v32=bswap32(*(uint32_t*)(mb+pos));*(uint32_t*)(mb+pos)=bswap32(v32-delta); break;
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
    case 30: NEED(2);pos=RND((int)mut_len-1);v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)(v16-(1+RND(ARITH_MAX)))); break;
    case 31: NEED(2);pos=RND((int)mut_len-1);
             *(uint16_t*)(mb+pos)=(uint16_t)(*(uint16_t*)(mb+pos)+(1+RND(ARITH_MAX))); break;
    case 32: NEED(2);pos=RND((int)mut_len-1);v16=bswap16(*(uint16_t*)(mb+pos));
             *(uint16_t*)(mb+pos)=bswap16((uint16_t)(v16+(1+RND(ARITH_MAX)))); break;
    case 33: NEED(4);pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)-=(1u+(uint32_t)RND(ARITH_MAX)); break;
    case 34: NEED(4);pos=RND((int)mut_len-3);v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos)=bswap32(v32-(1u+(uint32_t)RND(ARITH_MAX))); break;
    case 35: NEED(4);pos=RND((int)mut_len-3);
             *(uint32_t*)(mb+pos)+=(1u+(uint32_t)RND(ARITH_MAX)); break;
    case 36: NEED(4);pos=RND((int)mut_len-3);v32=bswap32(*(uint32_t*)(mb+pos));
             *(uint32_t*)(mb+pos)=bswap32(v32+(1u+(uint32_t)RND(ARITH_MAX))); break;
    case 37: mb[RPOS]=(uint8_t)(rand()&0xFF); break;
    case 38: mb[RPOS]+=(uint8_t)(rand()&0x1F); break;
    case 39: mb[RPOS]-=(uint8_t)(rand()&0x1F); break;
    case 40: mb[RPOS]^=(uint8_t)(rand()&0xFF); break;
    case 41: tok_len=pick_user_extra(afl,&tok);
             if(!tok_len){mb[RPOS]=(uint8_t)(rand()&0xFF);break;}
             dict_overwrite(mb,mut_len,RPOS,tok,tok_len); break;
    case 42: tok_len=pick_user_extra(afl,&tok);
             if(!tok_len||mut_len+tok_len>max_size){mb[RPOS]^=0xFF;break;}
             pos=RND((int)mut_len+1);
             mut_len=dict_insert(mb,mut_len,pos,tok,tok_len,max_size); break;
    case 43: tok_len=pick_auto_extra(afl,&tok);
             if(!tok_len){mb[RPOS]=(uint8_t)MYINTERESTING_8[RND(N8)];break;}
             dict_overwrite(mb,mut_len,RPOS,tok,tok_len); break;
    case 44: tok_len=pick_auto_extra(afl,&tok);
             if(!tok_len||mut_len+tok_len>max_size){mb[RPOS]++;break;}
             pos=RND((int)mut_len+1);
             mut_len=dict_insert(mb,mut_len,pos,tok,tok_len,max_size); break;
    case 45: {
        int nops=4+RND(5);
        for(int op=0;op<nops&&mut_len>0;op++){
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
        for(int op=0;op<stack&&mut_len>0;op++){
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
    }

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

    m->enabled    = calloc(MAP_SIZE, sizeof(uint32_t));
    m->disabled   = calloc(MAP_SIZE, sizeof(uint32_t));
    m->prev_trace = calloc(MAP_SIZE, 1);
    if (!m->enabled || !m->disabled || !m->prev_trace) {
        perror("[-] M1_0 table calloc");
        free(m->enabled); free(m->disabled); free(m->prev_trace);
        free(m->mutated_buf); free(m);
        return NULL;
    }

    m->shm_fd = open(SHM_PATH, O_RDWR | O_CREAT, 0600);
    if (m->shm_fd < 0) { perror("[-] M1_0 SHM open"); m->shm = NULL; return m; }
    if (ftruncate(m->shm_fd, SHM_SIZE) < 0) perror("[-] M1_0 SHM ftruncate");

    m->shm = mmap(NULL, SHM_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, m->shm_fd, 0);
    if (m->shm == MAP_FAILED) {
        perror("[-] M1_0 SHM mmap"); m->shm = NULL;
    } else {
        printf("[+] M1_0 mutator: SHM at %s  (edge tables: %zu KB)\n",
               SHM_PATH, (MAP_SIZE * sizeof(uint32_t) * 2 + MAP_SIZE) / 1024);
    }

    if (m->shm)
        m->last_action_seq = __atomic_load_n(
            u32_at(m->shm, OFF_ACTION_SEQ), __ATOMIC_ACQUIRE);

    m->step_count    = 0;
    m->prev_coverage = 0;
    m->state_seq     = 0;
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
        /* Step 0: seed prev_trace from current execution, skip SHM handshake */
        if (m->step_count == 0) {
            uint32_t sz = m->afl->total_bitmap_size;
            if (sz > MAP_SIZE) sz = MAP_SIZE;
            memcpy(m->prev_trace, m->afl->fsrv.trace_bits, sz);
            m->step_count = 1;
            goto do_mutate;
        }

        uint32_t coverage    = count_coverage(m->afl);
        uint32_t crashes     = (uint32_t)m->afl->total_crashes;
        uint64_t total_execs = m->afl->fsrv.total_execs;
        uint32_t new_edges   = (coverage > m->prev_coverage)
                               ? (coverage - m->prev_coverage) : 0;

        shm_push_state(m, coverage, new_edges, crashes, total_execs);
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
    free(m->enabled);
    free(m->disabled);
    free(m->prev_trace);
    free(m->mutated_buf);
    free(m);
}
