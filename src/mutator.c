/*
 * [sysrel]
 *
 * SHM layout (128 bytes) 
 *
 *   Offset  0..63  — STATE REGION (mutator.c write, Python reads)
 *     [0]   state_seq    uint32_t  — sequence sentinel; release-stored LAST
 *     [4]   edge_id      uint32_t  — most recently new/hot edge from trace_bits
 *     [8]   coverage     uint32_t  — total bitmap coverage count
 *     [12]  new_edges    uint32_t  — delta new edges since last step
 *     [16]  crashes      uint32_t
 *     [20]  _pad         uint32_t
 *     [24]  total_execs  uint64_t
 *     [32]  _pad[32]     padding / false-share guard
 *
 *   Offset 64..127 — ACTION REGION (Python writes, mutator.c read)
 *     [64]  action_seq   uint32_t  — sentinel; Python writes this LAST
 *     [68]  action       int32_t   — chosen action (0..46)
 *     [72]  _pad[56]
 *
 *   C writes state fields → __ATOMIC_RELEASE store of state_seq
 *   Python polls state_seq; reads state → writes action → stores action_seq
 *   C __ATOMIC_ACQUIRE polls action_seq until changed → reads action
 *
 * Action index → CSV column (must match rl_server.py exactly):
 *   0   DET_FLIP_ONE_BIT             21  HAVOC_MUT_FLIPBIT
 *   1   DET_FLIP_TWO_BITS            22  HAVOC_MUT_MYINTERESTING8
 *   2   DET_FLIP_FOUR_BITS           23  HAVOC_MUT_MYINTERESTING16
 *   3   DET_FLIP_ONE_BYTE            24  HAVOC_MUT_MYINTERESTING16BE
 *   4   DET_FLIP_TWO_BYTES           25  HAVOC_MUT_MYINTERESTING32
 *   5   DET_FLIP_FOUR_BYTES          26  HAVOC_MUT_MYINTERESTING32BE
 *   6   DET_ARITH_ADD_ONE            27  HAVOC_MUT_ARITH8_
 *   7   DET_ARITH_SUB_ONE            28  HAVOC_MUT_ARITH8
 *   8   DET_ARITH_ADD_TWO_LE         29  HAVOC_MUT_ARITH16_
 *   9   DET_ARITH_SUB_TWO_LE         30  HAVOC_MUT_ARITH16BE_
 *  10   DET_ARITH_ADD_TWO_BIG        31  HAVOC_MUT_ARITH16
 *  11   DET_ARITH_SUB_TWO_BIG        32  HAVOC_MUT_ARITH16BE
 *  12   DET_ARITH_ADD_FOUR_LE        33  HAVOC_MUT_ARITH32_
 *  13   DET_ARITH_SUB_FOUR_LE        34  HAVOC_MUT_ARITH32BE_
 *  14   DET_ARITH_ADD_FOUR_BIG       35  HAVOC_MUT_ARITH32
 *  15   DET_ARITH_SUB_FOUR_BIG       36  HAVOC_MUT_ARITH32BE
 *  16   MYINTERESTING_BYTE             37  HAVOC_MUT_RAND8
 *  17   MYINTERESTING_TWO_BYTES_LE     38  HAVOC_MUT_BYTEADD
 *  18   MYINTERESTING_TWO_BYTES_BIG    39  HAVOC_MUT_BYTESUB
 *  19   MYINTERESTING_FOUR_BYTES_LE    40  HAVOC_MUT_FLIP8
 *  20   MYINTERESTING_FOUR_BYTES_BIG   41  DICTIONARY_USER_EXTRAS_OVER
 *                                    42  DICTIONARY_USER_EXTRAS_INSERT
 *                                    43  DICTIONARY_AUTO_EXTRAS_OVER
 *                                    44  DICTIONARY_AUTO_EXTRAS_INSERT
 *                                    45  CUSTOM_MUTATOR
 *                                    46  HAVOC
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

/* ── Constants ──────────────────────────────────────────────────────────────── */

#define SHM_PATH         "/tmp/muofuzz_shm"
#define SHM_SIZE         128
#define MAX_MUTATED_SIZE (1024 * 1024)
#define ACTION_SIZE      47
#define ARITH_MAX        35       /* matches AFL++ ARITH_MAX */
#define HAVOC_STACK_POW2  7       /* 1<<(1..7) = 2..128 stacked ops */

/* SHM byte offsets — unchanged from original */
#define OFF_STATE_SEQ    0
#define OFF_EDGE_ID      4
#define OFF_COVERAGE     8
#define OFF_NEW_EDGES    12
#define OFF_CRASHES      16
#define OFF_PAD0         20
#define OFF_TOTAL_EXECS  24
#define OFF_ACTION_SEQ   64
#define OFF_ACTION       68

/* Spin poll interval when waiting for Python's action */
#define SPIN_NS          100000   /* 0.1 ms */

/* ── AFL++ interesting value tables (from afl-fuzz-one.c) ───────────────────── */

static const int8_t MYINTERESTING_8[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127
};
#define N8  ((int)(sizeof(MYINTERESTING_8)  / sizeof(INTERESTING_8[0])))

static const int16_t MYINTERESTING_16[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127,
    -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767
};
#define N16 ((int)(sizeof(MYINTERESTING_16) / sizeof(INTERESTING_16[0])))

static const int32_t MYINTERESTING_32[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127,
    -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767,
    -2147483648, -32769, 32768, 65535, 65536, 100663045, 2147483647
};
#define N32 ((int)(sizeof(MYINTERESTING_32) / sizeof(INTERESTING_32[0])))

/* ── Byte-swap helpers ──────────────────────────────────────────────────────── */

static inline uint16_t bswap16(uint16_t x) {
    return (uint16_t)((x >> 8) | (x << 8));
}
static inline uint32_t bswap32(uint32_t x) {
    return ((x >> 24))
         | ((x >>  8) & 0x0000ff00u)
         | ((x <<  8) & 0x00ff0000u)
         | ((x << 24));
}

/* ── Mutator state — same struct as original ────────────────────────────────── */

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

/* ── SHM helpers — identical to original ───────────────────────────────────── */

static inline volatile uint32_t *shm_u32(void *base, size_t off) {
    return (volatile uint32_t *)((uint8_t *)base + off);
}
static inline volatile int32_t *shm_i32(void *base, size_t off) {
    return (volatile int32_t *)((uint8_t *)base + off);
}
static inline volatile uint64_t *shm_u64(void *base, size_t off) {
    return (volatile uint64_t *)((uint8_t *)base + off);
}

static void shm_push_state(my_mutator_t *data,
                            uint32_t edge_id, uint32_t coverage,
                            uint32_t new_edges, uint32_t crashes,
                            uint64_t total_execs)
{
    void *shm = data->shm;
    *shm_u32(shm, OFF_EDGE_ID)     = edge_id;
    *shm_u32(shm, OFF_COVERAGE)    = coverage;
    *shm_u32(shm, OFF_NEW_EDGES)   = new_edges;
    *shm_u32(shm, OFF_CRASHES)     = crashes;
    *shm_u64(shm, OFF_TOTAL_EXECS) = total_execs;
    /* Release-store: all prior stores visible before this */
    data->state_seq++;
    __atomic_store_n(shm_u32(shm, OFF_STATE_SEQ), data->state_seq, __ATOMIC_RELEASE);
}

static int shm_wait_action(my_mutator_t *data)
{
    void *shm = data->shm;
    struct timespec ts = { .tv_sec = 0, .tv_nsec = SPIN_NS };
    while (1) {
        uint32_t cur = __atomic_load_n(shm_u32(shm, OFF_ACTION_SEQ), __ATOMIC_ACQUIRE);
        if (cur != data->last_action_seq) {
            data->last_action_seq = cur;
            return (int)(*shm_i32(shm, OFF_ACTION));
        }
        nanosleep(&ts, NULL);
    }
}

/* ── Coverage helpers — identical to original ───────────────────────────────── */

static uint32_t count_coverage(afl_state_t *afl)
{
    uint32_t count = 0;
    uint32_t map_size = afl->total_bitmap_size;
    const uint8_t *virgin = afl->virgin_bits;
    for (uint32_t i = 0; i < map_size; i++)
        if (virgin[i] != 0xFF) count++;
    return count;
}

/*
 * find_current_edge — identical to original.
 * Priority 1: highest-index newly discovered edge (virgin[i]==0xFF before hit).
 * Priority 2: hottest edge in last execution.
 */
static uint32_t find_current_edge(afl_state_t *afl)
{
    uint32_t       map_size = afl->total_bitmap_size;
    const uint8_t *trace    = afl->fsrv.trace_bits;
    const uint8_t *virgin   = afl->virgin_bits;

    for (int32_t i = (int32_t)map_size - 1; i >= 1; i--)
        if (trace[i] && virgin[i] == 0xFF) return (uint32_t)i;

    uint32_t best_idx = 0, best_val = 0;
    for (uint32_t i = 1; i < map_size; i++)
        if (trace[i] > best_val) { best_val = trace[i]; best_idx = i; }
    return best_idx;
}

/* ── Dictionary helpers ─────────────────────────────────────────────────────── */

/* Overwrite bytes at pos with tok (clamped to buf_len). */
static void dict_overwrite(uint8_t *buf, size_t buf_len, int pos,
                            const uint8_t *tok, uint32_t tok_len)
{
    size_t avail = (pos < (int)buf_len) ? buf_len - (size_t)pos : 0;
    size_t n     = tok_len < avail ? tok_len : avail;
    if (n) memcpy(buf + pos, tok, n);
}

/* Insert tok at pos, shift tail right. Returns new length (capped at max_size). */
static size_t dict_insert(uint8_t *buf, size_t buf_len, int pos,
                           const uint8_t *tok, uint32_t tok_len, size_t max_size)
{
    size_t new_len = buf_len + tok_len;
    if (new_len > max_size) new_len = max_size;
    size_t tail = buf_len - (size_t)pos;
    if ((size_t)pos + tail > new_len) tail = new_len - (size_t)pos;
    memmove(buf + pos + tok_len, buf + pos, tail);
    memcpy(buf + pos, tok, tok_len);
    return new_len;
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

/* ── AFL++ API ──────────────────────────────────────────────────────────────── */

my_mutator_t *afl_custom_init(afl_state_t *afl, unsigned int seed)
{
    srand(seed);

    my_mutator_t *data = calloc(1, sizeof(my_mutator_t));
    if (!data) { perror("calloc"); return NULL; }

    data->afl         = afl;
    data->mutated_buf = malloc(MAX_MUTATED_SIZE);
    if (!data->mutated_buf) { perror("malloc"); free(data); return NULL; }

    /* Open / create the shared memory file */
    data->shm_fd = open(SHM_PATH, O_RDWR | O_CREAT, 0600);
    if (data->shm_fd < 0) {
        perror("[-] SHM open failed");
        data->shm = NULL;
        return data;
    }
    if (ftruncate(data->shm_fd, SHM_SIZE) < 0) perror("[-] SHM ftruncate");

    data->shm = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE,
                     MAP_SHARED, data->shm_fd, 0);
    if (data->shm == MAP_FAILED) {
        perror("[-] SHM mmap failed");
        data->shm = NULL;
    } else {
        printf("[+] MuoFuzz mutator: shared memory mapped at %s  ACTION_SIZE=%d\n",
               SHM_PATH, ACTION_SIZE);
    }

    /* Read initial action_seq so we don't mistake 0→0 as "no change" */
    if (data->shm)
        data->last_action_seq = __atomic_load_n(
            shm_u32(data->shm, OFF_ACTION_SEQ), __ATOMIC_ACQUIRE);

    data->state_seq     = 0;
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
    int action = 46;   /* default: HAVOC */

    if (data->shm) {
        /* ── Gather state ──────────────────────────────────────────────── */
        uint32_t coverage    = count_coverage(data->afl);
        uint32_t crashes     = (uint32_t)data->afl->total_crashes;
        uint64_t total_execs = data->afl->fsrv.total_execs;
        uint32_t edge_id     = find_current_edge(data->afl);
        uint32_t new_edges   = (coverage > data->prev_coverage)
                               ? (coverage - data->prev_coverage) : 0;

        /* ── Push state (release-store) ────────────────────────────────── */
        shm_push_state(data, edge_id, coverage, new_edges, crashes, total_execs);

        /* ── Wait for Python's action (acquire-load) ────────────────────── */
        action = shm_wait_action(data);
        if (action < 0 || action >= ACTION_SIZE) action = 46;

        /* ── Update deltas ──────────────────────────────────────────────── */
        data->prev_coverage = coverage;
        data->prev_crashes  = crashes;
    }

    /* ── Copy input into output buffer ─────────────────────────────────── */
    size_t mut_len = (buf_size < MAX_MUTATED_SIZE) ? buf_size : MAX_MUTATED_SIZE;
    memcpy(data->mutated_buf, buf, mut_len);
    *out_buf = data->mutated_buf;

    uint8_t       *mb  = data->mutated_buf;
    int            pos;
    uint16_t       v16;
    uint32_t       v32;
    uint32_t       delta;
    const uint8_t *tok;
    uint32_t       tok_len;

/* Helpers — clean up at end of switch */
#define NEED(n)  if (mut_len < (size_t)(n)) break
#define RND(n)   ((int)(rand() % (unsigned)(n)))
#define RPOS     ((int)(rand() % (unsigned)mut_len))
#define RDELTA   (1u + (uint32_t)RND(ARITH_MAX))

    switch (action) {

    /* ─── Deterministic bit flips ───────────────────────────────────────── */

    case 0: /* DET_FLIP_ONE_BIT */
        pos = RND((int)mut_len * 8);
        mb[pos / 8] ^= (uint8_t)(1 << (pos % 8));
        break;

    case 1: /* DET_FLIP_TWO_BITS */
        pos = RND((int)mut_len * 8);
        mb[pos / 8] ^= (uint8_t)(1 << (pos % 8));
        { int p2 = (pos + 1) % ((int)mut_len * 8);
          mb[p2 / 8] ^= (uint8_t)(1 << (p2 % 8)); }
        break;

    case 2: /* DET_FLIP_FOUR_BITS */
        pos = RND((int)mut_len * 8);
        for (int b = 0; b < 4; b++) {
            int bp = (pos + b) % ((int)mut_len * 8);
            mb[bp / 8] ^= (uint8_t)(1 << (bp % 8));
        }
        break;

    case 3: /* DET_FLIP_ONE_BYTE */
        mb[RPOS] ^= 0xFF;
        break;

    case 4: /* DET_FLIP_TWO_BYTES */
        NEED(2); pos = RND((int)mut_len - 1);
        mb[pos] ^= 0xFF; mb[pos + 1] ^= 0xFF;
        break;

    case 5: /* DET_FLIP_FOUR_BYTES */
        NEED(4); pos = RND((int)mut_len - 3);
        mb[pos] ^= 0xFF; mb[pos+1] ^= 0xFF;
        mb[pos+2] ^= 0xFF; mb[pos+3] ^= 0xFF;
        break;

    /* ─── Deterministic arithmetic: byte ───────────────────────────────── */

    case 6: /* DET_ARITH_ADD_ONE */
        mb[RPOS]++;
        break;

    case 7: /* DET_ARITH_SUB_ONE */
        mb[RPOS]--;
        break;

    /* ─── Deterministic arithmetic: 16-bit ─────────────────────────────── */

    case 8: /* DET_ARITH_ADD_TWO_LE */
        NEED(2); pos = RND((int)mut_len - 1); delta = RDELTA;
        v16 = *(uint16_t *)(mb + pos);
        *(uint16_t *)(mb + pos) = (uint16_t)(v16 + delta);
        break;

    case 9: /* DET_ARITH_SUB_TWO_LE */
        NEED(2); pos = RND((int)mut_len - 1); delta = RDELTA;
        v16 = *(uint16_t *)(mb + pos);
        *(uint16_t *)(mb + pos) = (uint16_t)(v16 - delta);
        break;

    case 10: /* DET_ARITH_ADD_TWO_BIG */
        NEED(2); pos = RND((int)mut_len - 1); delta = RDELTA;
        v16 = bswap16(*(uint16_t *)(mb + pos));
        *(uint16_t *)(mb + pos) = bswap16((uint16_t)(v16 + delta));
        break;

    case 11: /* DET_ARITH_SUB_TWO_BIG */
        NEED(2); pos = RND((int)mut_len - 1); delta = RDELTA;
        v16 = bswap16(*(uint16_t *)(mb + pos));
        *(uint16_t *)(mb + pos) = bswap16((uint16_t)(v16 - delta));
        break;

    /* ─── Deterministic arithmetic: 32-bit ─────────────────────────────── */

    case 12: /* DET_ARITH_ADD_FOUR_LE */
        NEED(4); pos = RND((int)mut_len - 3); delta = RDELTA;
        v32 = *(uint32_t *)(mb + pos);
        *(uint32_t *)(mb + pos) = v32 + delta;
        break;

    case 13: /* DET_ARITH_SUB_FOUR_LE */
        NEED(4); pos = RND((int)mut_len - 3); delta = RDELTA;
        v32 = *(uint32_t *)(mb + pos);
        *(uint32_t *)(mb + pos) = v32 - delta;
        break;

    case 14: /* DET_ARITH_ADD_FOUR_BIG */
        NEED(4); pos = RND((int)mut_len - 3); delta = RDELTA;
        v32 = bswap32(*(uint32_t *)(mb + pos));
        *(uint32_t *)(mb + pos) = bswap32(v32 + delta);
        break;

    case 15: /* DET_ARITH_SUB_FOUR_BIG */
        NEED(4); pos = RND((int)mut_len - 3); delta = RDELTA;
        v32 = bswap32(*(uint32_t *)(mb + pos));
        *(uint32_t *)(mb + pos) = bswap32(v32 - delta);
        break;

    /* ─── Interesting values (deterministic) ────────────────────────────── */

    case 16: /* MYINTERESTING_BYTE */
        mb[RPOS] = (uint8_t)MYINTERESTING_8[RND(N8)];
        break;

    case 17: /* MYINTERESTING_TWO_BYTES_LE */
        NEED(2); pos = RND((int)mut_len - 1);
        *(uint16_t *)(mb + pos) = (uint16_t)MYINTERESTING_16[RND(N16)];
        break;

    case 18: /* MYINTERESTING_TWO_BYTES_BIG */
        NEED(2); pos = RND((int)mut_len - 1);
        *(uint16_t *)(mb + pos) = bswap16((uint16_t)MYINTERESTING_16[RND(N16)]);
        break;

    case 19: /* MYINTERESTING_FOUR_BYTES_LE */
        NEED(4); pos = RND((int)mut_len - 3);
        *(uint32_t *)(mb + pos) = (uint32_t)MYINTERESTING_32[RND(N32)];
        break;

    case 20: /* MYINTERESTING_FOUR_BYTES_BIG */
        NEED(4); pos = RND((int)mut_len - 3);
        *(uint32_t *)(mb + pos) = bswap32((uint32_t)MYINTERESTING_32[RND(N32)]);
        break;

    /* ─── Havoc: bit flip ───────────────────────────────────────────────── */

    case 21: /* HAVOC_MUT_FLIPBIT */
        pos = RND((int)mut_len * 8);
        mb[pos / 8] ^= (uint8_t)(128 >> (pos % 8));
        break;

    /* ─── Havoc: interesting values ─────────────────────────────────────── */

    case 22: /* HAVOC_MUT_MYINTERESTING8 */
        mb[RPOS] = (uint8_t)MYINTERESTING_8[RND(N8)];
        break;

    case 23: /* HAVOC_MUT_MYINTERESTING16 */
        NEED(2); pos = RND((int)mut_len - 1);
        *(uint16_t *)(mb + pos) = (uint16_t)MYINTERESTING_16[RND(N16)];
        break;

    case 24: /* HAVOC_MUT_MYINTERESTING16BE */
        NEED(2); pos = RND((int)mut_len - 1);
        *(uint16_t *)(mb + pos) = bswap16((uint16_t)MYINTERESTING_16[RND(N16)]);
        break;

    case 25: /* HAVOC_MUT_MYINTERESTING32 */
        NEED(4); pos = RND((int)mut_len - 3);
        *(uint32_t *)(mb + pos) = (uint32_t)MYINTERESTING_32[RND(N32)];
        break;

    case 26: /* HAVOC_MUT_MYINTERESTING32BE */
        NEED(4); pos = RND((int)mut_len - 3);
        *(uint32_t *)(mb + pos) = bswap32((uint32_t)MYINTERESTING_32[RND(N32)]);
        break;

    /* ─── Havoc: arithmetic 8-bit ───────────────────────────────────────── */

    case 27: /* HAVOC_MUT_ARITH8_ (subtract) */
        mb[RPOS] -= (uint8_t)(1 + RND(ARITH_MAX));
        break;

    case 28: /* HAVOC_MUT_ARITH8 (add) */
        mb[RPOS] += (uint8_t)(1 + RND(ARITH_MAX));
        break;

    /* ─── Havoc: arithmetic 16-bit ──────────────────────────────────────── */

    case 29: /* HAVOC_MUT_ARITH16_ (LE subtract) */
        NEED(2); pos = RND((int)mut_len - 1);
        v16 = *(uint16_t *)(mb + pos);
        *(uint16_t *)(mb + pos) = (uint16_t)(v16 - (1 + RND(ARITH_MAX)));
        break;

    case 30: /* HAVOC_MUT_ARITH16BE_ (BE subtract) */
        NEED(2); pos = RND((int)mut_len - 1);
        v16 = bswap16(*(uint16_t *)(mb + pos));
        *(uint16_t *)(mb + pos) = bswap16((uint16_t)(v16 - (1 + RND(ARITH_MAX))));
        break;

    case 31: /* HAVOC_MUT_ARITH16 (LE add) */
        NEED(2); pos = RND((int)mut_len - 1);
        v16 = *(uint16_t *)(mb + pos);
        *(uint16_t *)(mb + pos) = (uint16_t)(v16 + (1 + RND(ARITH_MAX)));
        break;

    case 32: /* HAVOC_MUT_ARITH16BE (BE add) */
        NEED(2); pos = RND((int)mut_len - 1);
        v16 = bswap16(*(uint16_t *)(mb + pos));
        *(uint16_t *)(mb + pos) = bswap16((uint16_t)(v16 + (1 + RND(ARITH_MAX))));
        break;

    /* ─── Havoc: arithmetic 32-bit ──────────────────────────────────────── */

    case 33: /* HAVOC_MUT_ARITH32_ (LE subtract) */
        NEED(4); pos = RND((int)mut_len - 3);
        v32 = *(uint32_t *)(mb + pos);
        *(uint32_t *)(mb + pos) = v32 - (1u + (uint32_t)RND(ARITH_MAX));
        break;

    case 34: /* HAVOC_MUT_ARITH32BE_ (BE subtract) */
        NEED(4); pos = RND((int)mut_len - 3);
        v32 = bswap32(*(uint32_t *)(mb + pos));
        *(uint32_t *)(mb + pos) = bswap32(v32 - (1u + (uint32_t)RND(ARITH_MAX)));
        break;

    case 35: /* HAVOC_MUT_ARITH32 (LE add) */
        NEED(4); pos = RND((int)mut_len - 3);
        v32 = *(uint32_t *)(mb + pos);
        *(uint32_t *)(mb + pos) = v32 + (1u + (uint32_t)RND(ARITH_MAX));
        break;

    case 36: /* HAVOC_MUT_ARITH32BE (BE add) */
        NEED(4); pos = RND((int)mut_len - 3);
        v32 = bswap32(*(uint32_t *)(mb + pos));
        *(uint32_t *)(mb + pos) = bswap32(v32 + (1u + (uint32_t)RND(ARITH_MAX)));
        break;

    /* ─── Havoc: byte ops ───────────────────────────────────────────────── */

    case 37: /* HAVOC_MUT_RAND8 */
        mb[RPOS] = (uint8_t)(rand() & 0xFF);
        break;

    case 38: /* HAVOC_MUT_BYTEADD */
        mb[RPOS] += (uint8_t)(rand() & 0x1F);   /* +0..31 */
        break;

    case 39: /* HAVOC_MUT_BYTESUB */
        mb[RPOS] -= (uint8_t)(rand() & 0x1F);
        break;

    case 40: /* HAVOC_MUT_FLIP8 */
        mb[RPOS] ^= (uint8_t)(rand() & 0xFF);
        break;

    /* ─── Dictionary: user extras ───────────────────────────────────────── */

    case 41: /* DICTIONARY_USER_EXTRAS_OVER */
        tok_len = pick_user_extra(data->afl, &tok);
        if (!tok_len) { mb[RPOS] = (uint8_t)(rand() & 0xFF); break; }
        dict_overwrite(mb, mut_len, RPOS, tok, tok_len);
        break;

    case 42: /* DICTIONARY_USER_EXTRAS_INSERT */
        tok_len = pick_user_extra(data->afl, &tok);
        if (!tok_len || mut_len + tok_len > max_size) { mb[RPOS] ^= 0xFF; break; }
        pos = RND((int)mut_len + 1);
        mut_len = dict_insert(mb, mut_len, pos, tok, tok_len, max_size);
        break;

    /* ─── Dictionary: auto extras ───────────────────────────────────────── */

    case 43: /* DICTIONARY_AUTO_EXTRAS_OVER */
        tok_len = pick_auto_extra(data->afl, &tok);
        if (!tok_len) { mb[RPOS] = (uint8_t)MYINTERESTING_8[RND(N8)]; break; }
        dict_overwrite(mb, mut_len, RPOS, tok, tok_len);
        break;

    case 44: /* DICTIONARY_AUTO_EXTRAS_INSERT */
        tok_len = pick_auto_extra(data->afl, &tok);
        if (!tok_len || mut_len + tok_len > max_size) { mb[RPOS]++; break; }
        pos = RND((int)mut_len + 1);
        mut_len = dict_insert(mb, mut_len, pos, tok, tok_len, max_size);
        break;

    /* ─── CUSTOM_MUTATOR: focused multi-op havoc (4–8 stacked ops) ─────── */

    case 45: /* CUSTOM_MUTATOR */ {
        int n_ops = 4 + RND(5);
        for (int op = 0; op < n_ops && mut_len > 0; op++) {
            switch (RND(8)) {
            case 0: { int bp = RND((int)mut_len * 8);
                      mb[bp/8] ^= (uint8_t)(128 >> (bp%8)); } break;
            case 1: mb[RPOS] += (uint8_t)(1 + RND(ARITH_MAX)); break;
            case 2: mb[RPOS] -= (uint8_t)(1 + RND(ARITH_MAX)); break;
            case 3: mb[RPOS]  = (uint8_t)MYINTERESTING_8[RND(N8)]; break;
            case 4: mb[RPOS]  = (uint8_t)(rand() & 0xFF); break;
            case 5: mb[RPOS] ^= 0xFF; break;
            case 6: if (mut_len >= 2) {
                        int p = RND((int)mut_len - 1);
                        *(uint16_t *)(mb+p) = (uint16_t)MYINTERESTING_16[RND(N16)];
                    } break;
            case 7: if (mut_len >= 4) {
                        int p = RND((int)mut_len - 3);
                        *(uint32_t *)(mb+p) = (uint32_t)MYINTERESTING_32[RND(N32)];
                    } break;
            }
        }
        break;
    }

    /* ─── HAVOC: large stacked random mutations (AFL++ style) ───────────── */

    default:
    case 46: /* HAVOC */ {
        int stack = 1 << (1 + RND(HAVOC_STACK_POW2));
        for (int op = 0; op < stack && mut_len > 0; op++) {
            switch (RND(12)) {
            case 0: { int bp = RND((int)mut_len * 8);
                      mb[bp/8] ^= (uint8_t)(128>>(bp%8)); } break;
            case 1: mb[RPOS]  = (uint8_t)MYINTERESTING_8[RND(N8)]; break;
            case 2: mb[RPOS] += (uint8_t)(1 + RND(ARITH_MAX)); break;
            case 3: mb[RPOS] -= (uint8_t)(1 + RND(ARITH_MAX)); break;
            case 4: mb[RPOS]  = (uint8_t)(rand() & 0xFF); break;
            case 5: /* delete block */
                if (mut_len > 2) {
                    int from = RPOS;
                    int dlen = 1 + RND((int)mut_len - from);
                    if ((size_t)(from + dlen) > mut_len) dlen = (int)mut_len - from;
                    memmove(mb+from, mb+from+dlen,
                            mut_len - (size_t)from - (size_t)dlen);
                    mut_len -= (size_t)dlen;
                } break;
            case 6: /* clone block (overwrite destination) */
                if (mut_len >= 2) {
                    int src = RPOS, dst = RPOS;
                    int cl  = 1 + RND(8);
                    if ((size_t)(src+cl) > mut_len) cl = (int)mut_len - src;
                    if ((size_t)(dst+cl) > mut_len) cl = (int)mut_len - dst;
                    if (cl > 0) memmove(mb+dst, mb+src, (size_t)cl);
                } break;
            case 7: { /* memset block with random byte */
                int p  = RPOS;
                int sl = 1 + RND(8);
                if ((size_t)(p+sl) > mut_len) sl = (int)mut_len - p;
                memset(mb+p, rand()&0xFF, (size_t)sl); } break;
            case 8: if (mut_len >= 2) {
                        int p = RND((int)mut_len - 1);
                        *(uint16_t *)(mb+p) = (uint16_t)MYINTERESTING_16[RND(N16)];
                    } break;
            case 9: if (mut_len >= 4) {
                        int p = RND((int)mut_len - 3);
                        *(uint32_t *)(mb+p) = (uint32_t)MYINTERESTING_32[RND(N32)];
                    } break;
            case 10: /* user dict overwrite */
                tok_len = pick_user_extra(data->afl, &tok);
                if (tok_len) dict_overwrite(mb, mut_len, RPOS, tok, tok_len);
                else mb[RPOS] ^= 0xAA;
                break;
            case 11: /* splice with add_buf */
                if (add_buf && add_buf_size > 0 && mut_len >= 2) {
                    int split   = 1 + RND((int)mut_len - 1);
                    int add_off = RND((int)add_buf_size);
                    int cl      = (int)mut_len - split;
                    if ((size_t)(add_off + cl) > add_buf_size)
                        cl = (int)add_buf_size - add_off;
                    if (cl > 0) memcpy(mb+split, add_buf+add_off, (size_t)cl);
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

void afl_custom_deinit(my_mutator_t *data)
{
    if (data->shm && data->shm != MAP_FAILED) munmap(data->shm, SHM_SIZE);
    if (data->shm_fd > 0) close(data->shm_fd);
    free(data->mutated_buf);
    free(data);
}
