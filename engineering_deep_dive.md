# MuoFuzz — Engineering Deep-Dive

A study document for an xAI screening interview. Tier 1 source files are
`src/mutator_m3_0.c`, `scripts/models/common.py`, `scripts/models/m3_0.py`,
`scripts/rl_server.py`. Citations use `path:line` form. Every interesting
claim is grounded in code. Sections are self-contained — each one can be read
on its own before walking through that part of the codebase out loud.

---

## 1. Architectural Overview

### 1.1 Process topology

Two long-lived OS processes communicate through a 128-byte memory-mapped file:

```
                   /tmp/rl_shm_m3_0  (mmap, MAP_SHARED, 128 B)
                   ┌───────────────────────────────────────────┐
                   │  STATE  REGION  [ 0..55]                  │
                   │    state_seq u32 + 13 features (u32/f32)  │
                   │  pad           [56..63]                   │
                   │  ACTION REGION [64..71]                   │
                   │    action_seq u32 + action i32            │
                   └───────────────────────────────────────────┘
                            ▲   write(state)        ▲ write(action)
                            │   release-store seq   │ release-store seq
   ┌────────────────────────┼─────────┐  ┌──────────┼──────────────────────┐
   │ AFL++ process           │         │  │ Python: rl_server.py            │
   │   afl-fuzz binary       │         │  │   (PyTorch + numpy)             │
   │   ↓ dlopen()            │         │  │                                 │
   │   bin/mutator_m3_0.so   │         │  │   while True:                   │
   │     afl_custom_init()   │  reads  │  │     poll state_seq (acquire)    │
   │     afl_custom_fuzz()   │  state  │  │     shm_read → 13-dim ndarray   │
   │       count_coverage()  │ ───────►│  │     compute reward (s,a,r,s')   │
   │       shm_push_state()  │         │  │     replay.push, train_step     │
   │       shm_wait_action() │ ◄────── │  │     act = agent.select_action() │
   │       apply_mutation()  │  reads  │  │     write action + bump seq     │
   │   afl_custom_deinit()   │  action │  │                                 │
   └─────────────────────────┴─────────┘  └─────────────────────────────────┘
```

The two processes are launched as siblings by `scripts/run_model.sh:145–163`:
the Python server starts first (`sleep 2` to mmap the SHM and pre-write
`action_seq=1` so the first read by C is non-blocking — `rl_server.py:84`),
then `afl-fuzz` is forked with `AFL_CUSTOM_MUTATOR_LIBRARY=$MUTATOR_SO` and
`AFL_CUSTOM_MUTATOR_ONLY=1` so AFL's vanilla mutators are bypassed entirely.

### 1.2 Per-mutation data flow

A single round through `afl_custom_fuzz` (`src/mutator_m3_0.c:601–647`) does:

1. **First-call bootstrap** (`:609–624`): seed the cumulative bitmap from the
   first trace and skip directly to mutation, because there's no prior state
   to publish or reward to assign.
2. **Coverage snapshot** (`:626–629`): `count_coverage()` walks
   `afl->virgin_bits` (chunked 64-bit-skip optimisation, `:158–177`) and
   computes the integer edge count; subtract `prev_coverage` to get
   `new_edges`.
3. **State publish** (`:631`): `shm_push_state()` writes 13 features and
   bumps `state_seq` with an `__ATOMIC_RELEASE` store (`:305`).
4. **Action wait** (`:632`): `shm_wait_action()` spin-polls `action_seq` with
   `__ATOMIC_ACQUIRE` and a 100µs `nanosleep` between polls (`:308–320`).
5. **Mutation** (`:644`): `apply_mutation()` dispatches the chosen action ID
   into a 47-arm switch (`:376–549`).
6. AFL++ executes the target on the mutated buffer; on the next call the
   loop repeats and the *previous* (state, action) pair is finally rewarded
   with the *new* coverage delta — the C side is stateless about reward, the
   Python side closes the loop in `rl_server.py:115–122`.

### 1.3 Synchronous design and what it costs

The C side blocks on every mutation. It will not return a mutated buffer
until Python has written a fresh action. This is a deliberate choice: a DQN
relies on the (s,a,r,s') tuple being correctly aligned in time, and letting
C run ahead would mean the agent receives stale states with rewards
attributed to the wrong actions (the credit-assignment problem in fuzzing
already has weak signal — burning it on a race is unrecoverable).

The throughput cost is the round-trip time of one mutation:
- **C → publish state**: a single 56-byte write region (no syscall) plus
  one release-store. Tens of nanoseconds.
- **Python → wake, compute, write action**: a `time.sleep(0.0001)`-paced
  spin (`rl_server.py:109`), a numpy build, a single 1×13 forward pass on
  CPU (the model is tiny — 13→128→128→64→47 ≈ 38 K params). Empirically
  ~100–400 µs.
- **C → wake from `nanosleep`** at 100µs spin granularity
  (`mutator_m3_0.c:82` `SPIN_NS = 100000`).

That floor — roughly 200–500 µs per mutation — caps the RL pipeline at
~2–5 K execs/sec. The vanilla AFL++ baseline on the same target hits
~50–60 K execs/sec. The IPC overhead is real and was measured during the
Experiment 2 campaign (`docs/experiment_2_multi_benchmark_10m.md`,
`scripts/benchmark_latency.py`).

The "~59K vs ~2.5K exec/s" framing in older notes is a fair rough estimate
— the gap is dominated by (a) the spin-sleep cycle on both sides and
(b) Python's per-step PyTorch/numpy overhead, not by the mmap traffic. A
production version would need to either (i) batch decisions in C (let it
buffer N actions ahead under a contextual-bandit assumption) or (ii) move
inference into the C process.

### 1.4 Why two processes — concrete alternatives

| Alternative                                  | Estimated cost                                                                  | Why rejected |
|----------------------------------------------|----------------------------------------------------------------------------------|--------------|
| CPython embedding (link `libpython` into AFL) | Initial-import cost amortised, but every call crosses GIL; PyTorch CUDA init from a forked process is fragile; ABI brittle. | AFL is GPL/AGPL, libpython is PSF; the fork model AFL uses with `fork()` from the parent fuzzer breaks down when libtorch holds CUDA contexts. Rejected on operational complexity. |
| In-process inference via libtorch / ONNX     | True zero-IPC latency (~10–50 µs forward pass on CPU). | Adds a 100+ MB native dep to the mutator `.so`, makes builds platform-specific, and forces model-format conversion at every checkpoint. Worth it for production; not worth it for a research codebase that needs to swap models nightly. |
| Unix socket / named pipe                      | One `write()` + one `read()` syscall per direction = ~2 µs uncontended.        | Strictly worse than SHM at the same effective cadence: the sync still has to be done in user space (eventfd or seqno), so SHM + sequence numbers buys you the same correctness with no syscall. |
| gRPC over loopback                            | Hundreds of µs per call from protobuf alone.                                   | Strictly dominated. |
| Same process with Python C-API and numpy buffer protocol | Lowest latency but every fork in the AFL persistent loop has to re-init Python state. | AFL's `-i`/`-o` workflow assumes the mutator can be loaded once and used across many target executions — Python init (importing torch) takes ~1 sec, a deal-breaker. |

Two processes + mmap gives **independent crash domains** (AFL crashing the
target won't take the agent down; the agent crashing won't kill AFL),
**hot-swappable models** (point Python at a new checkpoint and restart the
server while AFL keeps the corpus warm with `AFL_AUTORESUME=1`,
`run_model.sh:158`), and a wire format that is trivially inspectable from
`xxd /tmp/rl_shm_m3_0`. The throughput cost is the price paid for that
debuggability.

---

## 2. Function-by-Function Commentary (Tier 1)

### 2.1 `src/mutator_m3_0.c`

#### `bswap16` (`:108–110`), `bswap32` (`:111–116`)
Plain shift-and-mask 16/32-bit byte swaps. Inlined. Used for the BE
arithmetic actions. The implementation is hand-rolled rather than using
`__builtin_bswap16/32` — modern clang/gcc emits the same `bswap`/`rev16`
instruction either way, but the hand-rolled version avoids a portability
assumption. Acceptable.

#### `u32_at` / `i32_at` / `f32_at` (`:146–154`)
Type-punning helpers that compute `(volatile T*)((uint8_t*)base + offset)`.
The `volatile` qualifier is *not* doing the synchronisation — that's done
by the GCC atomics later — but it prevents the compiler from caching
SHM reads in a register across a function boundary. A reader using `*ptr`
will always emit a load. **Defensible:** these are necessary because the
SHM region is shared across processes; the compiler has no way to know
the bytes can change underneath it.

#### `count_coverage` (`:158–177`)
**Signature:** `static uint32_t count_coverage(afl_state_t *afl)`.
Returns the integer count of edges that AFL has seen at least once,
defined as bytes in `virgin_bits` not equal to `0xFF` (AFL's virgin
bitmap inverts the convention: `0xFF` = unseen).

**Control flow:** strided 8-byte read with `memcpy` (avoids strict-alias
UB), if the chunk is all-`0xFF` skip it, else byte-test each. Trailing
unaligned tail is handled in the second loop. The chunk skip is a real
optimisation: typical coverage on these targets sits at 2–5 K edges out
of 65 536, so ~95% of the chunks are all-`0xFF` and we cut the loop body
to a single 8-byte compare.

**The comment at `:160–163` is load-bearing:** an earlier version used
`afl->total_bitmap_size`, which AFL maintains as a *cumulative* sum of
per-corpus-entry bitmap sizes (it grows unboundedly). Reading
`virgin_bits[i]` for `i > MAP_SIZE` is an OOB read into adjacent AFL state.
The fix: use `afl->fsrv.map_size` which is the actual trace bitmap
allocation size (typically 65 536). The same bug appears unfixed in
`mutator_telemetry.c:121` (note: `total_bitmap_size`) — telemetry was the
oracle that produced the M3_0 design data and that file is older. Flag
this in defensibility.

**Failure modes:** if `afl->virgin_bits == NULL` we segfault. AFL guarantees
this is non-null by the time `afl_custom_fuzz` is called for the first
time, but `afl_custom_init` is called earlier — we don't read
`virgin_bits` from `init` so this is fine. **Error handling:** none — a
dereference of a null AFL state is a programmer error.

#### `shm_push_state` (`:189–306`)
The hot path. Single O(MAP_SIZE) pass over `trace_bits`, doing all of:
(a) max-merge into `cumulative_map`, (b) classify into hot/warm/cool/cold,
(c) accumulate `sum`/`sum_sq` for mean and std, (d) build an 8-bin
power-of-two histogram for entropy. Then computes EMA of execution time
(`:259–272`), updates the velocity ring buffer (`:274–286`), writes all 13
features into SHM, and finally bumps `state_seq` with a release-store
(`:289–305`).

**Why one pass:** the alternative — four passes with one stat per pass —
was simpler to write but cache-blew through the 64 KiB cumulative map
four times per mutation. Coverage maps fit in L2 but not always L1, so
four passes had a measurable hot-path cost. The single pass keeps the map
hot in L1 across the histogram and stats.

**Why `double` for accumulators (`:200`):** sum and sum-of-squares can
both overflow `uint32_t` when nonzero edge counts reach the tens of
thousands and individual hits are 8-bit (max 255). At max it's
`65536 * 255 = ~1.7e7` for the sum, and the sum of squares can reach
`65536 * 255^2 = ~4.3e9` which already exceeds u32. Using `double` avoids
any saturation reasoning — accuracy is fine because we only ever divide
to compute mean/variance.

**Order of writes (`:289–305`):** all 13 feature fields are written first;
**then** `state_seq` is bumped with `__ATOMIC_RELEASE`. The release
semantics guarantee that any acquire-load by Python that sees the new
sequence number also sees all 13 prior writes. If we bumped `state_seq`
first, Python could read a sequence-number change and then race to read
half-updated features.

**Failure modes:** none on the hot path — the function is total. If
`m->afl->fsrv.trace_bits` is null we segfault, but again that's a violated
AFL precondition. EMA division-by-zero is avoided by the `step_count <= 1`
branch (`:265`). Velocity is only computed once the ring is full (`:280`),
so we never divide by uninitialised data.

#### `shm_wait_action` (`:308–320`)
Spin-poll loop with `__ATOMIC_ACQUIRE` on `action_seq`, returning when the
sequence number differs from the last-seen value. Sleep is `nanosleep`
with `tv_nsec = SPIN_NS = 100000` (`:82`, 100 µs).

**Why a sentinel u32 not a flag byte:** monotonic counters let a reader
detect not just "new" but "I haven't seen this update specifically." A
flag byte would need ABA handling (the writer would have to clear it after
every read, which itself is racy). With a counter, the reader stores the
last-seen value locally and compares; ABA can only occur on u32 wrap
(2^32 mutations).

**Why `nanosleep` not `sched_yield`:** `sched_yield` returns immediately
under a free CPU; `nanosleep(100µs)` actually de-schedules the process,
giving the Python side a chance to run. A pure spin would burn a full
core and starve Python. **Why 100µs:** matches the OS scheduler's typical
re-dispatch latency; smaller and we just pay the overhead of more wakeups
without seeing fresher data; larger and we add latency to every mutation.

**Tradeoff analysis:** `futex(FUTEX_WAIT)` on the seqno would be more
efficient (zero CPU when waiting, kernel-mediated wakeup) but requires
either both processes share a futex address (which mmap'd SHM does
support) or using `eventfd`. Either is a one-syscall-per-mutation cost
versus the current zero-syscall-when-not-waiting cost. At 100µs spin
granularity the wakeup is fast enough that the syscall overhead is not
worth the engineering complexity.

#### `dict_overwrite` (`:324–330`), `dict_insert` (`:332–342`)
Dictionary-token operations.

`dict_overwrite` is straightforward: clip the token to `[pos, len)` and
`memcpy`.

`dict_insert` shifts the tail right by `tlen` bytes and copies the token
in. The arithmetic is fragile — see Defensibility flags below for the
specific bound issue.

#### `pick_user_extra` (`:344–349`), `pick_auto_extra` (`:351–356`)
Pick a random AFL dictionary entry. Both use plain `rand()` (`:346`,
`:353`). AFL has its own `rand_below(afl, n)` that is reproducible across
runs given the same seed; using libc `rand()` means our mutator selection
is not deterministic with `AFL_PRINT_STATS` reproductions, even though
the action chosen by the agent is deterministic (the agent uses Python's
`random` module seeded separately, plus PyTorch). Acceptable for research,
flag for production.

#### `apply_mutation` (`:360–559`)
The 47-arm dispatcher. Every arm is a small in-place edit on `mb`. Macros
`NEED(n)`, `RND(n)`, `RPOS`, `RDELTA` are local convenience for the
"need at least N bytes," "random integer mod N," "random byte position,"
and "random arithmetic delta" idioms (`:371–374`).

**Cases 0–5** are deterministic bit and byte flips (1, 2, 4 bits;
single, double, quad bytes XOR 0xFF).

**Cases 6–15** are deterministic arithmetic (8/16/32-bit; LE+BE for
multi-byte). Note the `*(uint16_t*)(mb+pos)` casts (`:399–418`) — these
are technically strict-aliasing violations and require unaligned access
to be defined (true on x86 with `-O2`; undefined on strict ARM). See
defensibility.

**Cases 16–20** stomp interesting boundary values.

**Cases 21–40** are the havoc single-ops, broken out one per case. They
duplicate the work in the HAVOC stacked path (case 46) but as single ops,
so the agent can pick one targeted havoc primitive without taking the
full stacked-N=2..512 storm.

**Cases 41–44** are dictionary operations. Each has a fallback when the
dictionary is empty: e.g. user-overwrite falls through to a random byte
write at `:471`. The fallbacks are cheap operations — they preserve
"every action does *something*" so the (state, action, reward) signal is
never zero by construction.

**Case 45 — `CUSTOM_MUTATOR`:** 4–8 stacked focused ops drawn from a
narrower 8-arm pool (no deletes, no token splices). Smaller stack and
narrower op pool than HAVOC: this is the "be aggressive but stay
recognisable as the seed" knob.

**Case 46 — `HAVOC`:** 2–512 stacked random ops drawn from a 12-arm pool
that includes deletes, region copies, and seed splicing (the only arm
that uses `add_buf`). `HAVOC_STACK_POW2 = 9` (`:62–63`) — `1 << (1 +
RND(9))` ranges in `[2, 512]`. `HAVOC_STACK_POW2` is `#undef`'d and
re-defined to ensure consistency with the telemetry mutator that
generated the differential analysis data — this is critical because the
M3_0 features were derived from those telemetry traces, and a different
HAVOC stack distribution would invalidate the training data.

**Final clamp** (`:556–558`): a mutation that empties the buffer is
turned back into a 1-byte buffer, and any overshoot is clipped to
`max_size`. This matches AFL's contract: the returned `mut_len` must be
in `[1, max_size]`.

#### `afl_custom_init` (`:563–599`)
Allocate the mutator state, mmap the SHM, optionally seed the bitmap.
Notable: even if SHM init fails (`:580`), we keep going with `m->shm =
NULL`, and the entire `if (m->shm)` block in `afl_custom_fuzz` (`:609`)
is skipped — the mutator falls back to always doing HAVOC. This means a
broken Python server doesn't kill the fuzzer; it gracefully degrades to
random. **Tradeoff:** the user might not notice the agent has died and
think they're getting RL behaviour when they're not. A loud failure
(printf + abort) might be safer for production; the silent degradation is
a research-friendly choice.

**Failure modes:** `calloc` or `malloc` failure returns NULL, AFL handles
it with `FATAL`. SHM open/mmap failure returns the partial state, see
above.

#### `afl_custom_fuzz` (`:601–647`)
The control loop. Branch structure:
- `if (m->shm)` — when SHM is alive:
  - `if (m->step_count == 0)` first-call bootstrap, jump to mutate with
    default action 46.
  - else read coverage, push state, wait for action, validate action in
    `[0, ACTION_SIZE)` (`:633`).
- After the if: copy seed buffer to the mutator's own scratch (`:641`)
  and call `apply_mutation`.

**Why copy the seed:** AFL's contract for `afl_custom_fuzz` says the
input `buf` may be invalidated when the function returns; the caller
gets `*out_buf` and uses it for the actual exec. We own the lifetime of
`m->mutated_buf` (`:570`), so we copy in. **Subtle:** we copy
`buf_size` bytes but mutate up to `max_size` — a HAVOC `dict_insert` can
legitimately grow the buffer, hence the need for our scratch buffer to be
sized to `MAX_MUTATED_SIZE = 1 MiB` (`:58`) regardless of input size.

#### `afl_custom_deinit` (`:649–655`)
Tear down. `munmap`, close fd, free heap. The SHM file at `/tmp/rl_shm_m3_0`
is **not** unlinked — the run_model.sh trap (`scripts/run_model.sh:104`)
removes it. This means a crashed AFL process leaves the SHM file behind;
the next run will reuse the same file (`O_CREAT` at `:575`), which is
benign because the first acquire-load reads the old `action_seq` value
into `last_action_seq` (`:587`) and the loop just naturally waits for
the next bump.

### 2.2 `scripts/models/common.py`

#### `compute_reward` (`:55–58`)
Returns `(coverage_delta + 1000 * log1p_crash_delta, breakdown_dict)`.
Coverage delta is in raw edges (one new edge ≈ +1 reward). Crash delta
is log-scaled then multiplied by 1000 — the log smooths "find 100 crashes
all at once" so it doesn't blow up the loss; the 1000 makes any crash
discovery *dominate* coverage gains in that step. `STEP_COST = 0.0`
(`:48`) is a deliberate choice: there's no per-step penalty, so the
agent isn't incentivised to terminate episodes (it can't, anyway —
the loop is infinite).

#### `PlateauDetector` (`:61–72`)
Rolling window over coverage values, fires when the window's range
(`max - h[0]`) is at most `PLATEAU_MIN_DELTA = 1` edge over
`PLATEAU_WINDOW = 10000` steps **and** epsilon has decayed to ≤
`EPSILON_MIN + 0.01`. The grace period (`:62`, default 70% of train
budget) prevents early-training plateaus from triggering false stops.
**Triggered is sticky** (`:63, :70`): once we've decided to stop, we
stay stopped — keeps `rl_server.py:144` from re-firing the
"plateau" branch every 100 steps.

#### `ReplayBuffer` (`:75–79`)
A `deque(maxlen=cap)`, no priority sampling. Sample is uniform random
without replacement (`random.sample`). For DQN this is fine —
prioritised experience replay would help if reward was sparse, but
coverage gain is dense per step so uniform is OK.

#### `DQN` (`:82–92`)
A `nn.Sequential` of `Linear → ReLU → ... → Linear` from
`state_size → hidden_layers → action_size`. No batch norm, no dropout,
no residual. **Why a plain MLP:** the input is a 13-dim handcrafted
feature vector, not raw bytes — there's nothing to learn at the
representation level that would benefit from depth or normalisation. A
small MLP is the right shape.

#### `DQNAgent` (`:95–168`)
Standard Double-DQN with a target network and ε-greedy exploration.
Notable engineering choices:
- **`_infer_buf` pre-allocated** (`:111–112`): a 1×state_size CPU/GPU
  tensor that gets overwritten per `select_action`. Avoids a per-step
  malloc on the hot path, where "hot" is once per AFL exec (~1k–5k Hz).
  This is the only real perf-critical micro-opt on the Python side.
- **Double DQN** (`:135–138`): use the *online* net to pick the best
  next-state action and the *target* net to evaluate it. Reduces the
  positive bias of vanilla DQN in noisy reward landscapes — appropriate
  here because coverage delta is noisy when the corpus is queue-saturated.
- **Entropy regulariser** (`:142–144`): the Q-distribution is softmaxed
  and its entropy added to the loss with weight `ENTROPY_COEF = 0.01`
  (`:43`). This is non-standard for DQN — normally exploration is
  controlled by ε alone — but having the network itself prefer
  high-entropy distributions helps when ε is low and the network has
  hardened on a dominant action. Defensible as a research choice.
- **Linear ε decay** (`:120–122`) with a configurable `decay_steps`,
  defaulting to 60% of the train budget (`rl_server.py:80`). At training
  start ε=1.0, drops linearly to `EPSILON_MIN = 0.05` then plateaus.
- **Target sync every 1000 train steps** (`:148–150`).
- **Save/load** (`:153–168`) round-trip optimizer state, ε, train and
  total step counters. Eval mode (`:165`) ignores the saved ε and stays
  at 0 (pure greedy).

**Failure modes:** `train_step` early-returns when the buffer is below
batch size or in eval mode. The `cuda` device picks itself; if PyTorch
sees no GPU it falls back to CPU and the training loop continues, just
slower. `load()` on a missing file logs and returns — fresh init.

#### `BanditNet` (`:171–186`)
A trunk-then-two-heads architecture: shared MLP feature extractor, then
one linear head producing per-action mean and another producing per-action
log-variance. Outputs (47, 47) per state.

#### `ContextualBanditAgent` (`:189–259`)
Same `select_action / remember / train_step / save / load` interface as
`DQNAgent` so `rl_server.py:78` can pick either via flag. Differences:
- **No replay buffer** (`:208`): the bandit does an immediate online
  update on the most recent (state, action, reward) tuple. The bandit
  assumption is that consecutive states are independent — there's no
  credit assignment to share across steps, so replay buys nothing.
- **Thompson sampling** (`:215–218`): instead of ε-greedy, draw one
  sample from `N(mean, exp(logvar))` per action and argmax. Exploration
  comes from posterior uncertainty, not random injection. This is a
  better fit when most actions are "close" in expected reward — the bandit
  will preferentially try actions whose posteriors are wide.
- **NLL loss** (`:236–237`): `0.5 * (logvar + (r - mu)^2 / var)` is the
  Gaussian negative log-likelihood with the constant dropped. Fits both
  heads jointly.

The shared interface in `common.py` is real polymorphism — see Section 7
for whether `rl_server.py` ever has to break it.

#### `create_shm`, `shm_write_action` (`:262–268`)
Open the file, zero-fill, mmap. Action write packs an `=i` (signed 32-bit
little-endian) at the action offset, then `=I` (unsigned little-endian)
at the action_seq offset. **Note the order:** action data first, sequence
counter second. This is the Python-side mirror of the C-side release
ordering — on x86 it's effectively free (TSO), on ARM it's *technically*
relying on the absence of store-store reordering between the two
`mmap.mmap.write` calls. Python's `mmap` module ultimately performs
plain stores; there's no acquire/release atomic primitive on the Python
side. Practically safe on supported hardware, but flag this when asked.

### 2.3 `scripts/models/m3_0.py`

#### `shm_read` (`:44–62`)
Reads the entire 128-byte SHM in one `read()` call, then `struct.unpack_from`
unpacks each field at its compile-time offset. **Why a single read:**
two reads could observe a state torn across an in-progress write from
the C side. The whole-SHM read followed by per-field unpack means we're
guaranteed an atomic snapshot relative to the *next* C write, but we
might observe the *current* C write half-done. The safety here comes
from `rl_server.py:105–110`: the loop only calls `shm_read` after seeing
a fresh `state_seq`, and the C side bumps the sequence *after* the
features (release-store), so a fresh seqno guarantees the features are
all from the same state.

The `coverage` and `total_edges` keys are aliases for the same
field at offset 4 (`:48–49`). Defensive: makes the dict usable by
both legacy code paths and by `compute_reward`.

#### `build_state` (`:65–81`)
Constructs the 13-dim float32 ndarray. Three classes of normalisation:
- **Pre-normed in C** (entropy, hit_mean, hit_std, exec_time, velocity):
  the C side already mapped these into [0, 1]. Python passes through.
- **Re-normed in Python with a magic constant** (corpus_size, crashes,
  new_edges): `log1p(x) / log1p(max)` for log-scaled features and a
  hard-clip + divide for `new_edges`. The `max` constants (10000 for
  corpus, 1000 for crashes, 100 for new_edges) are duplicated between
  Python and C/`docs/m3_0_feature_derivation.md` — see the consistency
  discussion in Section 6.
- **Ratio-normed in Python** (hot/warm/cool): `feature / total_edges`
  — these are written as raw counts on the C side and turned into a
  ratio in Python. `te = max(total_edges, 1)` prevents divide-by-zero
  on the very first state.

`cold_edges / MAP_SIZE` (`:69`) uses the constant `MAP_SIZE = 65536.0`
defined locally (`:38`). This is a *third* place the value 65536 is
asserted to be true (the others are `mutator_m3_0.c:57` `MAP_SIZE` and
the implicit assumption in `count_coverage` that the bitmap is at most
that size). Discussed in defensibility.

#### `zero_state_data`, `csv_extra_fields`, `log_extra`, `exit_summary`
Boilerplate that satisfies the model-module interface enforced by
`rl_server.py:88` (`CSV_EXTRA_HEADER`), `:125` (`csv_extra_fields`),
`:131` (`log_extra`), `:156` (`exit_summary`). `zero_state_data` is used
once at startup (`:86`) to give the loop a synthetic "state s_{-1}"
before the first real read — without it, the very first `compute_reward`
call would crash on a None dict.

### 2.4 `scripts/rl_server.py`

#### `_format_milestone` (`:22–28`)
Formats step counts as "500k" / "1m" tags for checkpoint filenames.
Pure helper.

#### `main` (`:31–157`)
The entire control loop in one function. ~120 lines of orchestration.

**Flag parsing** (`:32–52`): every knob is documented inline. Note
`--algorithm` (`:49`) wires the DQN-vs-bandit switch into one variable,
and `--train-freq N` (`:43–45`) decouples training cadence from action
selection — the `_skip` model variants (e.g., `m3_0_skip`) train every
4 steps but still pick an action every step. `run_model.sh:60–63` sets
this when `--model-id ends with _skip`.

**Setup phase** (`:54–95`):
1. Dynamic import of the model module (`:59`) — `--model-id m3_0`
   resolves to `models.m3_0`.
2. SHM creation, agent construction, optional load from checkpoint.
3. Pre-write `(action=46 / HAVOC, seq=1)` (`:84`) so the C side, which
   blocks waiting for action_seq != 0, gets immediate liftoff. This is
   the only deliberate handshake between the two sides — a fairly
   minimal protocol.
4. Open the metrics CSV and write the header (`:87–90`).

**Hot loop** (`:101–147`):
- Termination check on step count.
- **Inner spin** (`:105–109`): poll `state_seq`, `time.sleep(0.0001)` =
  100 µs per iteration. Mirrors the C-side spin granularity.
- Read state, compute reward against the previous step.
- `agent.remember(s', a', r, s)` then `agent.train_step()` — both gated
  on `step % args.train_freq == 0` (`:118`). For `train_freq = 1` (the
  default), every step trains.
- `agent.select_action(s)` and write back `(action, ++aseq)`.
- Every 100 steps, append to CSV and emit a one-line console log.
- Every 1000 steps, save the checkpoint (`:137–138`).
- Optional milestone copies (`:139–143`).
- Plateau check (`:144–145`).

**Failure modes:**
- `KeyboardInterrupt` is caught at `:149`, the `finally` block flushes
  the CSV and writes a final checkpoint. AFL-side process is the parent's
  responsibility (`run_model.sh:98–105`).
- A torn read of `state_seq` is impossible because the loop only proceeds
  on a *changed* value, and on x86/ARM aligned 32-bit stores are atomic.
- A skipped state (Python is too slow, C bumps `state_seq` twice while
  Python is still computing) is silently OK: Python sees the latest
  state, computes a reward against its own previous state (which lags),
  and the (s, a, r, s') tuple is from *consecutive Python observations*,
  not consecutive C observations. The training data is biased toward
  states where C made it back to the wait — i.e., when C was actually
  blocked waiting for Python. This is a soft bias, not a correctness bug,
  but worth flagging.

### 2.5 Defensibility flags

Issues I'd not surprise myself with mid-interview. Each has a
recommended posture: **own**, **explain as adapted**, or **skip**.

1. **`apply_mutation` — strict aliasing / unaligned access via
   `*(uint16_t*)(mb+pos)` casts** (`mutator_m3_0.c:399–453`).
   Modern x86 + clang `-O2 -ffast-math` (the build flags in
   `run_model.sh:131`) handle these fine but it is undefined behaviour
   per the C standard. AFL++'s own havoc code uses the same idiom, so
   it's defensible as "matches AFL++ conventions."
   **Posture: explain as adapted from AFL++ havoc.** If pushed, say
   you'd rewrite with `memcpy` for portability (and that the codebase
   has `memcpy` used correctly in `count_coverage:169`).

2. **`dict_insert` arithmetic clip is upper-bounded by `nlen` not by the
   destination buffer's true capacity** (`mutator_m3_0.c:332–342`).
   When `len + tlen > max`, the function caps `nlen = max` and then
   `tail = nlen - pos`, then `memmove(buf + pos + tlen, buf + pos, tail)`
   — that writes through index `pos + tlen + tail - 1 = nlen + tlen - 1
   = max + tlen - 1`, which can exceed `max`. In practice we're safe
   because the caller passes `max = max_size <= MAX_MUTATED_SIZE = 1 MiB`
   and the actual scratch buffer is allocated at exactly
   `MAX_MUTATED_SIZE` (`:570`), giving `tlen` slack — but the function
   *as written* relies on that external invariant. **Posture: own.**
   If asked, say "the bound is enforced by the caller, not the function;
   fixing this is a one-line change and is on my list." A clean fix is
   `if (nlen + tlen > max) tail = max - pos - tlen;`.

3. **HAVOC and CUSTOM_MUTATOR cases (45, 46) are visibly modeled on
   AFL++'s own havoc loop** (`mutator_m3_0.c:486–548`). The structure
   — switch over an arm count, weighted toward common operations,
   stack of N=2..512 — is canonical AFL. The arm pool and the stack
   power are tuned, but the skeleton is recognisable.
   **Posture: explain as adapted.** The decision to make HAVOC one
   action among 47 (rather than the only action, as in vanilla AFL)
   is the interesting research choice; the implementation of the
   primitive itself is necessarily AFL-compatible because the M3_0
   features were derived from a telemetry mutator that ran exactly
   this primitive set.

4. **`count_coverage`'s 8-byte chunk skip** (`mutator_m3_0.c:167–173`).
   Pattern is well-known but easy for an interviewer to ask "did you
   write this?" The answer is yes — it's a textbook bitmap-scan
   optimisation, but it's *the right one* for this workload (as the
   comment at `:160–163` shows, the previous bug was much subtler). Be
   ready to explain why 8 bytes (matches a 64-bit register, single
   load, single compare to `0xFFFFFFFFFFFFFFFFULL`). **Posture: own.**

5. **`rand()` (libc) used everywhere, not AFL's `rand_below`**
   (`mutator_m3_0.c` throughout `apply_mutation` plus `:346`, `:353`).
   `rand()` is global, non-thread-safe, and not deterministic with
   AFL's own seed. Acceptable for research. **Posture: own** — call out
   as a known gap if asked about reproducibility.

6. **Volatile is used to keep loads honest, but it does not synchronise
   between processes** (`mutator_m3_0.c:146–154`). The actual memory
   ordering comes from `__ATOMIC_RELEASE` / `__ATOMIC_ACQUIRE` on
   `state_seq`/`action_seq`. **Be precise about this if asked.**

7. **Python side does not use atomic primitives** (`common.py:266–268`).
   Two ordinary `mmap.write` calls — action then seq. On x86 (TSO)
   and on ARM with C/Python's de-facto store ordering, this is
   correct in practice but not by the standard. **Posture: own** —
   if asked, say you'd port the Python writer to a C extension that
   does proper acquire/release for an apples-to-apples guarantee. (A
   pure-Python alternative: write the action and seqno into a single
   `struct.pack("=Iq", ...)` so they ride one mmap write, but the
   CPython mmap implementation is not guaranteed to issue a single
   store either.)

8. **`mutator_telemetry.c:121` still uses `total_bitmap_size`** —
   the bug fixed in `mutator_m3_0.c:160–163` was not back-ported.
   **Posture: own as known divergence**, explain that telemetry was
   "frozen" once the differential analysis was complete.

9. **`first-call bootstrap` skips reward attribution for step 0**
   (`mutator_m3_0.c:610–624`). The agent loses one (s, a, r, s')
   tuple at the start of every campaign. Negligible in 500k-step
   training; flag if asked.

---

## 3. Lock-Free Synchronisation Deep Dive

This is the section the interview is most likely to drill into.

### 3.1 The protocol in one paragraph

C writes 13 features then bumps `state_seq` with a release-store. Python
acquire-loads `state_seq`, observes the new value, then reads features.
Python writes one action then bumps `action_seq` with an
ordinary mmap write. C acquire-loads `action_seq`, observes a new value,
then reads action. Two sentinels = two unidirectional handshakes. No
locks, no syscalls when both sides keep up.

### 3.2 Field ownership and direction

| Offset | Field         | Writer | Reader | Ordering on writer            | Ordering on reader            |
|--------|---------------|--------|--------|-------------------------------|-------------------------------|
| 0      | state_seq     | C      | Python | RELEASE (mutator_m3_0.c:305)  | acquire (rl_server.py:107, plain mmap read after seqno-changed test) |
| 4..55  | 13 features   | C      | Python | plain volatile stores         | plain reads                    |
| 56..63 | (padding)     | —      | —      | —                             | —                              |
| 64     | action_seq    | Python | C      | plain mmap write (common.py:268) | ACQUIRE (mutator_m3_0.c:313)  |
| 68     | action        | Python | C      | plain mmap write              | plain `*i32_at` read           |

### 3.3 Why release-store + acquire-load is the right pairing

The happens-before chain we need:
> (C writes features) HB (C bumps state_seq) HB (Python reads state_seq new value) HB (Python reads features)

The first HB edge is intra-thread program order on the writer. The third
is intra-thread on the reader. The crucial one is the middle: the
release-store on the writer "publishes" all prior memory operations,
and the acquire-load on the reader "subscribes" — every store that
preceded the release in program order on the writer is visible to every
load that follows the acquire on the reader.

This is the C11 memory-model definition of release/acquire pairing. It
gives us mutual exclusion-like behaviour for the *features* without a
lock — features are immutable from the moment the seqno is bumped until
the next bump.

### 3.4 Why this is correct on x86 (TSO)

x86's memory model is Total Store Order: stores from one core are seen
in issue order by other cores, and loads do not reorder past loads or
stores. The only reordering allowed is StoreLoad (a later load can
complete before an earlier store retires). Acquire and release in C11
require: release prevents any *prior* loads or stores from being
reordered after it; acquire prevents any *later* loads or stores from
being reordered before it.

On x86, ordinary `mov` already respects every reordering rule we need
*except* StoreLoad, and we don't have a StoreLoad pair across these
sentinels — the C side stores (features then seq) are all stores, and
the Python side does loads (seq then features) that are all loads.
Acquire/release on x86 emit exactly `mov`s, no fences. The compiler
fence (`__ATOMIC_RELEASE` prevents the compiler from reordering stores
across the atomic) is the only thing the intrinsic does on x86.

### 3.5 Why this is correct on ARM64

ARM64's memory model is much weaker — both reads and writes can be
freely reordered. The C11 release-store on ARM64 lowers to `STLR`
(store-release), and the acquire-load to `LDAR` (load-acquire). At the
microarchitectural level:
- `STLR` ensures all prior loads and stores in program order have
  retired (or at least become visible to coherence) before the store
  itself is observed by other agents.
- `LDAR` ensures the load completes before any subsequent loads or
  stores in program order issue.

These are the architecture's exact implementation of release/acquire,
and together they implement the same happens-before chain as on x86 —
just at higher hardware cost (ARM64 needs a real fence-like operation
where x86 just needs a `mov`).

### 3.6 Why monotonic counters not flag bytes

A flag byte protocol would look like: writer sets flag = 1, reader
checks flag, processes, sets flag = 0 (acks). This has three problems:
- **ABA**: writer sets flag=1, reader processes, before reader can ack
  the writer publishes again with flag=1. Reader can't tell it's a new
  message.
- **Two-way coupling**: the writer needs the reader to ack. If the
  reader is dead the writer blocks. Our protocol is one-shot
  publication — writer doesn't care about reader state.
- **Reader synchronisation across replays**: if Python misses a state
  (it was busy training), it has no way to know — flag was already
  zero when it next checked. The counter tells Python exactly which
  states it skipped (by gap in the sequence, not that the current
  code uses this — but it could).

Monotonic counters give us a free "have I seen this update?" check via
local-variable comparison, no acks needed.

### 3.7 Why u32 not u64

- **Atomicity**: x86 guarantees aligned 32-bit and 64-bit stores are
  atomic. ARM64 also guarantees both. So this is a pure size choice.
- **Wrap interval**: at 5000 mutations/sec (the bandwidth of this
  pipeline), u32 wraps in `2^32 / 5000 ≈ 9.5 days`. At 50 000
  mutations/sec (closer to vanilla AFL throughput) it wraps in ~24
  hours. **Is wrap handled?** No — the comparison is `cur != m->last_action_seq`
  (`mutator_m3_0.c:314`) which works after wrap by accident
  (`UINT32_MAX + 1 == 0 != last`), but if the writer wraps *exactly* back
  to the reader's last-seen value the reader would miss the update
  forever. With sequential bumps, that requires the writer to advance
  by exactly `2^32` between two reader polls — impossible in one tick
  but possible if Python pauses for hours. **Mitigation:** none in code.
  **Acceptable?** Yes for a research codebase running at most a few
  hours per campaign. For production, switch to u64 (8 bytes is cheap;
  wrap interval is `~1.2e9` years at the same rate).

### 3.8 The spin interval — chosen by what?

`SPIN_NS = 100000` (`mutator_m3_0.c:82`) and Python's
`time.sleep(0.0001)` (`rl_server.py:109`) both sleep 100 µs between
polls. Why 100 µs?

- **Lower bound — OS scheduling**: Linux's default scheduler
  granularity (`CONFIG_HZ=1000` → 1 ms tick, but `nanosleep` uses
  hrtimer) lets us sleep for arbitrarily small intervals, but the
  cost of a wakeup is on the order of 5–20 µs (cache miss + context
  switch). Below ~50 µs the sleep cost approaches the work cost.
- **Upper bound — pipeline tail latency**: at 100 µs polling the
  worst-case extra latency per mutation round-trip is ~200 µs
  (one C-side wakeup + one Python-side wakeup). At 1 ms polling the
  worst-case is ~2 ms — that would cap throughput at 500 Hz.

100 µs is the sweet spot. Tested empirically in
`scripts/benchmark_latency.py`.

### 3.9 Concrete alternatives

| Mechanism                                  | Per-event cost                | Pros                                        | Cons                                                  |
|--------------------------------------------|-------------------------------|---------------------------------------------|-------------------------------------------------------|
| `pthread_mutex_t` + `pthread_cond_t`, `PTHREAD_PROCESS_SHARED` in SHM | ~1–5 µs uncontended futex syscall on signal | Familiar primitive, kernel-mediated wakeup, no spin | Two-process pthread sharing is fragile across implementations; PTHREAD_PROCESS_SHARED on glibc requires careful init; if either side crashes mid-critical-section the mutex is dead. |
| `eventfd` (Linux only)                      | One `write` + one `read` syscall per direction (~1 µs) | Simple kernel object, integrates with `epoll` | Linux only; one syscall per mutation across both sides ≈ 4 µs/round-trip overhead. |
| POSIX `sem_t` with `pshared=1`              | One syscall per `sem_post`/`sem_wait` (~1 µs) | Counting; multiple states queued without races | Same syscall cost; needs SHM-resident `sem_t`. |
| SPSC ring buffer (let C run ahead)          | Zero syscalls; just atomic head/tail ops | Decouples C throughput from Python latency  | Breaks the (s,a,r,s') tuple — the agent's "next state" wouldn't be the result of "this action" anymore. The DQN can't learn from this. **Only viable for the bandit** (immediate reward, no temporal credit assignment). |
| Current: seqno + 100µs spin                 | Zero syscalls when keeping up; `nanosleep` on slow path | Trivial, debuggable, portable, no kernel dependency | 100 µs polling latency added in the worst case; both sides burn an idle thread. |

The SPSC alternative is the most interesting trade-off because it is
conditioned on the agent type. **The current architecture is correct
because it is sound for both the DQN and the bandit, and is correct
*by design* rather than relying on a particular agent assumption.**

### 3.10 Hostile-questioning playbook

> **"How do you know there's no torn read of the features?"**
> Because Python only reads features after observing a fresh
> `state_seq` via acquire-load, and C only bumps `state_seq` after
> every feature store completes — the release-store guarantees the
> features are visible to any acquire-load that sees the new seqno.
> Within the read itself, the entire 128-byte SHM is `read()` into a
> bytes buffer in one call, then unpacked locally — there's no
> additional reordering risk after that. The next C update could begin
> mid-read, but it can't *complete* the seqno bump before all features
> are written, so even if Python observed an in-flight seqno-bump
> mid-read, the seqno would not match — and the inner loop would skip
> back and reread.

> **"What if the compiler reorders the feature stores after the seqno
> bump?"**
> The `__ATOMIC_RELEASE` intrinsic is a compiler barrier — the compiler
> is not allowed to reorder any prior memory operation past the release.
> That's a C11 guarantee. Independently, the `volatile` qualifier on
> the SHM pointers prevents the compiler from caching writes in
> registers across function calls.

> **"What if the CPU reorders?"**
> On x86, TSO disallows StoreStore reordering — the writes happen in
> issue order. The release intrinsic on x86 emits a plain `mov`
> because nothing stronger is needed. On ARM64, the release
> intrinsic emits `STLR`, which by the architecture's definition
> orders all prior memory operations before the store-release.

> **"What about ABA?"**
> The seqno is monotonic. ABA could only fire on `2^32`-step wrap; at
> our throughput that's days of continuous run. Acceptable risk for a
> research tool; production should switch to u64.

> **"What if Python crashes mid-write?"**
> The action and action_seq are two separate writes (action first, then
> seq). If Python dies after writing action but before writing seq, C
> never sees a fresh seqno and continues to spin in `shm_wait_action`.
> AFL has a watchdog (`AFL_NO_AFFINITY`, `AFL_AUTORESUME`) but won't
> kill the spin — `run_model.sh:165` waits on the Python PID, then on
> exit kills AFL. So a Python crash terminates the campaign cleanly.

> **"What if AFL crashes mid-write?"**
> Symmetric: Python keeps spinning on `state_seq` and the bash trap at
> `run_model.sh:98–105` reaps both processes.

> **"Could you have used C11 atomics directly instead of GCC builtins?"**
> Yes — `<stdatomic.h>` `atomic_store_explicit(&seq, v, memory_order_release)`
> is the portable spelling. The GCC builtins predate C11 and cover all
> compilers in our build matrix (clang-18, gcc 11+). Functionally
> equivalent.

---

## 4. Memory Safety and Lifetime Analysis

### 4.1 Ownership table

| Allocation                                    | Allocator                       | Lifetime                                   | Freer                              |
|-----------------------------------------------|---------------------------------|--------------------------------------------|------------------------------------|
| `my_mutator_t` struct                          | `calloc` at `mutator_m3_0.c:566` | AFL session (init → deinit)                | `free(m)` at `:654`                |
| `m->mutated_buf` (1 MiB scratch)              | `malloc` at `:570`              | AFL session                                 | `free(m->mutated_buf)` at `:653`   |
| `m->cumulative_map` (64 KiB inline)           | inline in struct, zeroed by calloc | AFL session                                 | with the struct                    |
| `m->edge_ring` (1000 × u32 inline)            | inline in struct                 | AFL session                                 | with the struct                    |
| SHM file `/tmp/rl_shm_m3_0`                   | `open(O_CREAT)` at `:575`       | persists across sessions                    | `unlink` in `run_model.sh:104`     |
| SHM mapping                                   | `mmap` at `:579`                 | AFL session                                 | `munmap` at `:651`                 |
| `m->shm_fd`                                   | `open` at `:575`                 | AFL session                                 | `close` at `:652`                  |
| Python `mmap`                                 | `create_shm` (`common.py:262`)  | Python process                              | implicit on process exit (no explicit close in `rl_server.py`) |
| Python `DQNAgent` and tensors                  | PyTorch tensors                  | Python process                              | GC                                 |
| AFL's own `trace_bits`, `virgin_bits`         | AFL                              | AFL session, owned by `afl_state_t`         | AFL                                |
| AFL seed buffer `buf` in `afl_custom_fuzz`    | AFL                              | one mutator call (the comment in afl-fuzz.h:1048 says we should write into `*out_buf` we own) | AFL — we MUST NOT free   |

### 4.2 SHM lifecycle and stale-SHM

The SHM file is created with `open(O_RDWR | O_CREAT, 0600)` and
`ftruncate`'d to 128 bytes. On a normal exit, the bash trap unlinks it.
On crash, the file persists. Next run: `O_CREAT` is idempotent;
`ftruncate` to the same size is a no-op; `mmap` over the same region
gives the same byte-pattern as the old run.

The hand-shake at startup is permissive enough to recover: the C side
reads the current `action_seq` into `last_action_seq` (`:587`) before
entering the loop, so it will only return on a *new* bump from Python.
Python, in turn, pre-writes `(action=46, seq=1)` (`rl_server.py:84`)
which is guaranteed different from the old persisted value (in the
worst case, where the old value was exactly `1`, Python will then read
the C side's first state — which it published with `state_seq=1` — but
because C reads the seqno *into* `last_action_seq` at init, it observed
the persisted `aseq=N` and now waits for `N+1`; the next Python tick
writes `N+1+1 = N+2`, which differs).

There's a subtle edge case: if AFL crashes immediately after `init` but
before Python has written its first action, the SHM file at next start
contains `state_seq=0, action_seq=0`. Both sides interpret this as
"nothing has happened yet" and the protocol is back to a clean start.
Verified behaviour, not formally proved.

### 4.3 AFL's seed-buffer contract

`afl-fuzz.h:1048` says `afl_custom_fuzz` receives `buf` as input and
must produce a mutated buffer in `*out_buf` whose memory the mutator
manages. This codebase respects that contract — `mutator_m3_0.c:641`
copies `buf` into our own `m->mutated_buf`, and `*out_buf =
m->mutated_buf` at `:642` points AFL into our buffer for execution. We
never free `buf`. AFL is free to invalidate `buf` after the call
returns; we will not have retained a pointer.

### 4.4 Defensive checks present

- Null check on `m` from calloc (`:567`) and on `m->mutated_buf` from
  malloc (`:571`).
- SHM fd check (`:576`) — but does not abort, just sets `m->shm = NULL`
  and continues degraded.
- `MAP_FAILED` check on mmap (`:580`) — same degraded fallback.
- `mut_len` clamps in `apply_mutation` (`:556–558`).
- Action-ID validation: `if (action < 0 || action >= ACTION_SIZE) action = 46`
  (`:633`) — Python could write a garbage value; C falls back to HAVOC.
- Map-size clamp: `if (sz > MAP_SIZE) sz = MAP_SIZE` (`:195`, `:615`).

### 4.5 Defensive checks missing

- No bounds check on `pos` in many `apply_mutation` arms — relies on the
  `RND` macro returning a value in range, which it does *if* `mut_len`
  is small enough that `(int)mut_len * 8` doesn't overflow. For
  `mut_len > INT_MAX / 8 ≈ 256 MiB`, the multiplication overflows. Our
  scratch is 1 MiB so we're safe by construction, but the function
  itself doesn't check.
- `dict_insert` bounds bug (Section 2.5).
- No timeout on `shm_wait_action` — if Python is dead, we spin forever
  (mitigated by AFL's parent handling).

### 4.6 ASan / UBSan prediction

Things ASan/UBSan would likely catch on a stress run:
- UBSan: misaligned 16/32-bit accesses in `apply_mutation` for any seed
  buffer that lands at an odd offset — but `m->mutated_buf` from
  `malloc` is at least 16-byte aligned on x86-64, so static `pos+1`
  shifts are the only sources, and most arms test `pos = RND(mut_len-1)`
  or similar, so unaligned `*(uint16_t*)` is reachable.
- ASan: would not flag the `dict_insert` bound issue unless we ran a
  workload that triggered the case `pos + tlen + tail > max`. We have
  scratch slack, so this might not fire.
- TSan: would flag the SHM region as a data race (release/acquire on
  the seqno is fine; the *features* are not atomic — TSan doesn't know
  about the seqno ordering). Suppressing TSan on SHM regions is normal.

---

## 5. The 47-Action Design

### 5.1 Where the 47 are defined

- **C side**: as the literal `case 0` through `case 46` in
  `apply_mutation` (`mutator_m3_0.c:376–549`). `ACTION_SIZE` is also
  declared at `:59`.
- **Python side**: `ACTION_COLUMNS` list in `common.py:9–31`, with
  `ACTION_SIZE = len(ACTION_COLUMNS)` and an `assert ACTION_SIZE == 47`
  at `:32–33` to catch divergence at import time.
- **README**: enumerates the families at `README.md:75–82`.

The mapping is 1:1 with AFL++'s internal mutator IDs by design — the
agent isn't choosing between AFL primitives and synthetic alternatives,
it's choosing between AFL primitives. This makes the action-effectiveness
data from `mutator_telemetry.c` directly comparable to any per-action
metric AFL would emit for its own mutators.

### 5.2 Discrete + granular vs. abstract action groups

**The argument for discrete + granular (current choice):**
- Direct agency. The agent picks `DET_ARITH_ADD_TWO_LE` (action 8) and
  knows exactly what byte-level edit will happen. Reward is attributable
  to a primitive, not to a strategy.
- Compatible with AFL's existing analysis. The mutation-attribution CSV
  in `mutator_telemetry.c:506–517` records per-action effectiveness;
  this only works if the agent and AFL share a common vocabulary.
- Differential analysis (`docs/m3_0_feature_derivation.md:142–168`)
  showed that *specific* mutations were disproportionately effective on
  buggy versions — XML005 favoured arithmetic (action 33), XML017
  favoured dictionary inserts (action 42). An action-group abstraction
  (e.g., "do an arithmetic mutation") would lose this signal.

**The argument for abstract groups:**
- Smaller action space → faster DQN convergence (Q output layer is
  47-wide today, would be ~6-wide with group abstraction).
- Each action becomes a "macro" the C side could implement adaptively
  (e.g., "arithmetic" picks the byte-width based on local state). This
  pushes intelligence into C and reduces what the agent has to learn.
- Generalization: an agent that learned "arithmetic family is good for
  numeric bugs" might transfer better than one that memorised "action
  33 is good."

**Why granular won here:** the project goal is to compare against
vanilla AFL's selection logic. Vanilla AFL effectively picks among the
same 47 primitives stochastically; replacing that picker with a learned
one is a clean A/B test. An action-group picker would be a different
experiment, conflating the design (47 vs 6 actions) with the algorithm
(random vs DQN).

### 5.3 Action space size and DQN convergence

The DQN output is `state_size × ... × ACTION_SIZE` (`common.py:90`).
For M3_0 that's `13 → 128 → 128 → 64 → 47` ≈ 38 K parameters. The
output layer is 64×47 + 47 ≈ 3 K parameters. Convergence cost from the
larger action head is small in absolute terms; the bigger issue is
*sample efficiency* — with 47 actions, ε=1.0 for the first ~30% of
training (`rl_server.py:80`) gives each action about `0.3 × N / 47`
exploratory samples. At N=500 000 that's ~3000 random pulls per action
before greedy kicks in. Adequate.

### 5.4 Action space for the contextual bandit

The bandit's `BanditNet` (`common.py:171–186`) has a per-action mean
*and* per-action log-variance head. With 47 actions and the same
hidden-layer size that's `47 + 47 = 94` outputs at the head — each
action gets its own posterior. Thompson sampling draws one sample per
action and argmaxes (`:215–218`), so effectively each step does 47
independent reads from 47 posteriors. The width is a non-issue at this
scale.

### 5.5 Are all 47 used?

Yes. Every case 0–46 has a body that does work. The fallbacks in
dictionary actions (`:471, :474, :478, :481`) make those actions
non-no-op even when no dictionary is loaded. The agent will still
pick them via ε-exploration and observe a degenerate reward; but they
are never literally inert.

A subtle detail: if the seed file is a single byte, several actions
hit their `NEED(n)` guards (`:371`) and fall through to a no-op — the
buffer is unchanged. AFL will then execute the same input and observe
no new coverage; reward to that action is zero. This is fine: the
agent learns "for very short inputs, multi-byte mutations are
unproductive." It's a feature, not a bug.

---

## 6. State Representation Engineering

### 6.1 The 13 features

Listed in source order (the order they appear in the SHM and in
`build_state` at `m3_0.py:67–80`):

| # | Feature              | Computed in C at                | Normalised in     | Range  |
|---|----------------------|--------------------------------|-------------------|--------|
| 0 | total_edges          | `mutator_m3_0.c:626` via count_coverage | Python: `/ MAP_SIZE` | [0, 1] |
| 1 | cold_edges           | `mutator_m3_0.c:231`             | Python: `/ MAP_SIZE` | [0, 1] |
| 2 | hot_edges (ratio)    | `mutator_m3_0.c:216` (count)     | Python: `/ total_edges` | [0, 1] |
| 3 | warm_edges (ratio)   | `mutator_m3_0.c:217` (count)     | Python: `/ total_edges` | [0, 1] |
| 4 | cool_edges (ratio)   | `mutator_m3_0.c:218` (count)     | Python: `/ total_edges` | [0, 1] |
| 5 | edge_entropy         | `mutator_m3_0.c:236–242`         | C: `/ 3.0` (max log2(8)) | [0, 1] |
| 6 | edge_hit_mean        | `mutator_m3_0.c:248, 251`        | C: `/ 255.0` | [0, 1] |
| 7 | edge_hit_std         | `mutator_m3_0.c:249, 252`        | C: `sqrt(var)/255` | [0, 1] |
| 8 | corpus_size          | `mutator_m3_0.c:256` (raw)       | Python: `log1p(x)/log1p(10000)` | [0, ~1] |
| 9 | crashes              | `mutator_m3_0.c:627` (raw)       | Python: `log1p(x)/log1p(1000)` | [0, ~1] |
| 10| new_edges            | `mutator_m3_0.c:628` (raw)       | Python: `min(x,100)/100` | [0, 1] |
| 11| avg_exec_time        | `mutator_m3_0.c:271` (already normed) | C: `log1p(us)/log1p(100000)` | [0, 1] |
| 12| coverage_velocity    | `mutator_m3_0.c:282–285` (already normed) | C: `velocity / 0.1` clamped | [0, 1] |

### 6.2 C-side vs Python-side normalisation

**Why split:**
- Features that depend on a constant the C side already knows
  (MAX_HIT_COUNT=255, MAX_ENTROPY=log2(8)=3, the 100ms/0.1 clamp on
  velocity) are normalised in C — fewer bytes over the wire (1 float
  not 2 ints) and no need to re-derive the constant in Python.
- Features that depend on a *training-budget*-dependent constant
  (corpus_size cap of 10000, crash cap of 1000) are normalised in
  Python — the C side doesn't and shouldn't know what training budget
  this run is using.
- Ratio features (hot/warm/cool over total) require both numerator and
  denominator. They could be normalised in C (one float instead of one
  int + a separate denominator), but keeping the raw counts in SHM
  makes the wire format more debuggable (`xxd` shows the raw counts).

### 6.3 Are the constants kept in sync?

Imperfectly. The risks are:
- `MAP_SIZE = 65536` appears at `mutator_m3_0.c:57` and `m3_0.py:38`.
  Two authoritative declarations. Since the AFL trace bitmap size is
  set by the AFL build, both must equal that build's value or
  features are wrong by a constant factor. **No shared header.**
- The hit-count cap `255 = UINT8_MAX` is hard-coded at three places
  in `mutator_m3_0.c` and once implicitly in the entropy bin pattern.
  Changing one without the others would silently corrupt the feature
  vector.
- The entropy max value `3.0 = log2(8 bins)` at `mutator_m3_0.c:242`
  is paired with the 8-element `bins[]` array at `:201`. If you
  changed bin count you'd have to update both.

**What I'd do to fix:** put a `mutator_m3_0_constants.h` between
the C and Python sides with `#define`s and a Python-side import that
parses the same header. The `__atomic` choices and the SHM offsets
are already paired by hand (`mutator_m3_0.c:65–80` ↔ `m3_0.py:21–36`)
— same pattern, no schema validation. **Defensible:** the offsets are
short enough that hand-pairing is checkable in a code review; if the
struct grew, schema tooling would be worth it.

### 6.4 Why these 13 and not more or fewer

Documented in `docs/m3_0_feature_derivation.md`. The selection
methodology was:
1. Run the telemetry mutator (uniform-random over the same 47 actions)
   on 12 libxml2 targets — 6 buggy/fixed pairs covering 2 CVEs.
2. Compute Vargha-Delaney A12 effect size for each candidate feature
   between buggy and fixed versions, at 5 timepoints per run.
3. Rank features by `|A12 - 0.5|` (distance from no-effect).
4. Drop features with zero discriminative power: `edge_hit_max` (max
   value collapses to 255 quickly on every run) and
   `edge_discovery_rate` (per-interval rate is identical across
   buggy/fixed, only cumulative counts differ).
5. Two non-discriminative features (`new_edges` and
   `coverage_velocity`) are *kept anyway* because they are the
   per-step reward signal and the temporal-context signal — without
   them, the agent has no instantaneous learning signal at all.

This is a real selection process, defensible in interview. The honest
caveat: with only 3 seeds per condition, Mann-Whitney U has no
statistical power; the ranking is by effect size only, not significance
(noted at `docs/m3_0_feature_derivation.md:178–180`).

### 6.5 Encoding choices

- **Scalar [0,1] for everything.** No one-hot encoding, no embeddings.
  All features are continuous. ReLU MLPs handle scalars in [0,1]
  naturally; gradients are well-conditioned.
- **Log-scale for counts that grow without bound** (corpus, crashes,
  exec_time): `log1p(x) / log1p(max)`. log1p prevents `log(0)` and
  smooths the early-training high-derivative region.
- **Hard clip for new_edges**: `min(x, 100) / 100`. New-edge deltas can
  spike dramatically when a new code region opens; clipping prevents a
  single outlier from saturating the gradient.
- **Ratio for hot/warm/cool**: division by `total_edges` makes them
  comparable across targets with very different coverage levels.
- **Pre-normalised sum/std**: divided by 255 (the max byte value) so
  they live in [0,1] without further work.

No feature is one-hot because no feature is categorical. No feature is
embedded because the cardinality is tiny.

---

## 7. DQN-vs-Bandit Dual Implementation

### 7.1 The agent interface

Both `DQNAgent` and `ContextualBanditAgent` expose:
- `__init__(state_size, hidden_layers, label, eval_mode, decay_steps)`
- `select_action(state) -> int`
- `remember(s, a, r, ns) -> None`
- `train_step() -> float` (loss)
- `save(path) -> None`
- `load(path) -> None`
- `epsilon` attribute (float, used for logging)

`rl_server.py:77` selects the class via flag and uses it polymorphically
through these six methods. There is no `isinstance` branching in
`rl_server.py`.

### 7.2 LOC and complexity cost

- `DQNAgent`: 73 lines (`common.py:95–168`)
- `BanditNet` + `ContextualBanditAgent`: 89 lines (`:171–259`)
- Branch in `rl_server.py`: 2 lines (`:77–78`)
- CLI flag: 3 lines (`:49–51`)

Total dual-agent overhead: ~165 lines on top of the single-agent
DQN-only baseline. That's about 60% of `common.py`. **Worth it?** The
two agents capture fundamentally different assumptions about the
environment:

- **DQN** assumes the reward at step t depends on actions taken at
  steps t-k for some k; the discounted return `r_t + γ r_{t+1} + ...`
  is meaningful, with γ = 0.99 (`common.py:37`).
- **Bandit** assumes the reward at step t depends only on (state_t,
  action_t); no temporal credit assignment; immediate online update.

For fuzzing, the bandit assumption is *almost* true: a single mutation
either produces a coverage-gaining input or it doesn't, and the
coverage gain is observed on the next exec. Multi-step credit
assignment would matter if a mutation set up state for a *later*
mutation to exploit — possible in principle (e.g., insert a
dictionary token, then arithmetic on the result), but the C-side
HAVOC stack already absorbs that pattern within a single action.

So both are defensible models of the problem. Carrying both lets us
ablate the "does temporal credit assignment matter?" question
empirically. That ablation is the entire point of the dual
implementation.

### 7.3 Is the abstraction leaky?

Two minor leaks:
- `agent.epsilon` is read directly by `rl_server.py:128, :134, :155`
  for CSV/log output. The bandit overloads `epsilon` to mean "minimum
  exploration rate" (`common.py:203`); semantically misleading but
  doesn't break logging.
- The `_decay` method exists on DQN (`common.py:120–122`) but not on
  bandit. `remember` calls it on DQN but not bandit. This is internal
  to each class — not a leak. `rl_server.py` doesn't touch it.

No `isinstance` checks. The polymorphism is real.

---

## 8. Testing and Validation

### 8.1 What tests exist

**None as a test suite.** No `pytest` directory, no `tests/` folder,
no `Makefile` test target. Validation has been entirely empirical:
multi-target campaign runs (`scripts/experiment*.sh`,
`docs/experiment_*_*.md`) compare M3_0 against vanilla AFL and against
earlier model variants on real benchmarks (libxml2, jsoncpp, freetype2,
harfbuzz, libpng, re2 — `benchmarks/*/build_recipe.sh`).

### 8.2 What tests should exist

In rough priority order:

1. **State round-trip**: write known feature values into the SHM from
   Python, read them in C, verify bit-exact equality. Covers offset
   drift, endianness, and packing assumptions.
2. **Synchronization fuzz**: a multi-process test that hammers the
   SHM with rapid state/action exchanges and asserts no torn read,
   no missed sequence number. Run under TSan.
3. **`apply_mutation` arm coverage**: for each of the 47 actions,
   apply it to a known seed and assert the output differs from input
   in the expected way (e.g., bit-flip changes exactly 1 bit;
   arithmetic changes exactly 1, 2, or 4 contiguous bytes).
4. **Bound checks under stress**: feed `apply_mutation` adversarial
   inputs (length 0, length 1, length max_size, dictionary-empty,
   dictionary-full) and run under ASan.
5. **`compute_reward` regression**: 10–20 fixed (cov, pcov, cr, pcr)
   tuples with hand-computed expected rewards.
6. **`PlateauDetector` state machine**: verify (a) doesn't fire before
   grace, (b) doesn't fire while ε > min, (c) fires once and stays
   triggered.
7. **`ContextualBanditAgent` train step**: a synthetic state where one
   action has higher mean than others; verify Thompson sampling
   converges to picking it within N steps.
8. **`DQNAgent` learns a trivial environment**: 2-state 2-action MDP
   with known optimal policy.

### 8.3 Property-based tests for the protocol

If I were upstreaming the SHM protocol I'd write:
- **Invariant**: after any sequence of (write_state, write_action),
  every state ever seen by Python via `shm_read` corresponds to some
  state that was written by C, and the features within that read are
  all from the *same* C write (no inter-write tearing).
- **Schema**: the field offsets in C and Python agree. Code-generate
  both from a single TOML file, fail the build on drift.
- **Crash injection**: kill the C process at random points in
  `shm_push_state`; verify Python either sees the old state or the
  new state, never a mix. (This is hard to implement rigorously
  without single-stepping; a poor man's version is to use `gdb`
  scripting to break in the middle of the writes.)

### 8.4 Cross-language invariant maintenance

Today: hand-paired offsets and constants, separately defined in
`mutator_m3_0.c:65–80` and `m3_0.py:21–36`. The pairing is short
enough to eyeball — about 20 lines on each side — but not
machine-checked.

Better: a shared schema (TOML or JSON) listing fields, types, and
offsets; a code-gen step that produces both a C header and a Python
module. This is overkill for a 5-author research codebase but
mandatory for a multi-team production service.

---

## 9. Code-Quality Self-Audit

### 9.1 Magic numbers

Numbers that should be named constants but aren't:
- The hit-count thresholds `128`, `8`, `1` in
  `mutator_m3_0.c:216–228` (heat classification). These also appear in
  the documenting comment at `:14–17`. Should be `HOT_THRESHOLD`,
  `WARM_THRESHOLD`, `COOL_THRESHOLD`.
- The 8 entropy bins (`mutator_m3_0.c:201` and the 8 elsif chain at
  `:221–228`). Should be `ENTROPY_BINS`.
- The 100000 in `log1p(100000)` (`mutator_m3_0.c:272`) — max expected
  exec time in microseconds. Should be `MAX_EXEC_TIME_US`.
- The `0.1f` velocity normaliser (`mutator_m3_0.c:283`). Should be
  `VELOCITY_NORM`.
- Python: `10000`, `1000`, `100` in `m3_0.py:76–78`. Same as above.
- `MAX_COVERAGE`, `MAX_NEW_EDGES`, `MAX_CRASHES` in `common.py:45–47`
  *are* named — those are the ones that got the treatment. The
  others didn't because they're model-specific not shared.

### 9.2 Error-handling inconsistency

- `afl_custom_init` returns NULL on calloc/malloc fail (AFL aborts) but
  *returns the partial mutator* on SHM fail (degraded mode).
- `count_coverage` has no error path at all.
- Dictionary helpers (`pick_user_extra`, `pick_auto_extra`) return 0
  on empty dictionary; callers check.
- Python: `agent.load()` on missing file is silently OK (logs only),
  matching the bash convention of `--no-checkpoint-required`.
- `compute_reward` cannot fail.

The mix of error-code returns and silent degradation is consistent
*within* each subsystem (init returns NULL, hot path can't fail) but
not consistent across the codebase. Defensible for research; uniform
error codes would be required for shipping.

### 9.3 Research vs production flags

| Item | Research stance | What would change for prod |
|------|-----------------|----------------------------|
| `rand()` use | OK — non-deterministic but seeded once | Use AFL's `rand_below(afl, n)` for reproducibility |
| u32 sequence wrap | OK — wraps in days | u64; wraps in eons |
| `volatile` for SHM cells | OK — relies on compiler honesty | C11 `_Atomic` + `memcpy` for cross-process types |
| Single-process Python server | OK — one campaign at a time | Multi-tenant: one Python server per AFL fuzzer with per-target SHM names |
| Hard-coded `/tmp/rl_shm_*` paths | OK — local dev | Configurable, with a deterministic naming scheme that won't collide with concurrent runs |
| No tests | OK — empirical validation | Tier-1 unit tests + property tests + crash injection |
| `printf` logging from C | OK — appears in AFL stdout | Structured logging (libafl-style) so harvesters can ingest |
| Implicit AFL header dep | OK — built against one AFL version | Pin AFL version in build system, check `AFL_VERSION` macro at compile |
| Two normalisation sites (C + Py) | OK — small & code-reviewed | Code-gen from a schema |

### 9.4 What I'd change to upstream

If this were a PR to AFLplusplus/AFLplusplus:
- **API**: factor out the SHM protocol into a header, document it as
  the canonical IPC for "agent-driven mutators."
- **Configuration**: env vars rather than constants
  (`AFL_RL_SHM_PATH`, `AFL_RL_SHM_SIZE`, `AFL_RL_SPIN_NS`).
- **Dependencies**: drop the libm dep where possible (we use it in
  `count_coverage` only via `log2`; could be replaced with `__builtin_clz`).
- **Build**: provide a CMake target and a `Makefile.example`.
- **Docs**: a single README with the wire format and a Python
  reference implementation in <50 lines.
- **Tests**: bare minimum, the round-trip test in 8.2.1 above plus a
  short integration test that runs against `samples/` for 100 steps.

---

## 10. Anticipated Probe Questions

Pick of 12 regions an interviewer is likely to chase. Each has a
30–60-second answer in plain spoken English — what to actually say
out loud.

### 10.1 `count_coverage` — the chunk skip (`mutator_m3_0.c:158–177`)

**One-line:** counts seen edges by scanning AFL's virgin bitmap, with
a 64-bit-chunk fast-skip for unseen regions.

**Q: Why memcpy into a local instead of a direct cast?**
Because `v` is a `uint8_t*` and casting to `uint64_t*` would be a
strict-aliasing violation. memcpy at this size compiles to a single
load on every modern compiler — no overhead, full UB-free.

**Q: Why 8 bytes?**
Matches a 64-bit register and a single compare-immediate against
`0xFFFFFFFFFFFFFFFFULL`. Larger chunks would need SIMD; smaller
chunks would do more work per byte.

**Q: What's the worst case?**
A bitmap with no all-`0xFF` chunks falls through to the byte loop
inside the chunk and pays both the chunk compare and the per-byte test.
On our targets, observed coverage is ~3% of the bitmap, so the
fast-skip hits ~95%+ of the time.

**Q: What was the bug this fixed?**
The earlier version used `afl->total_bitmap_size`, which is an
accumulating *sum* across corpus entries — it grows unboundedly past
the actual `virgin_bits` allocation. We were doing a multi-MB OOB read
into adjacent AFL state. The fix is `afl->fsrv.map_size`, which is
literally the trace bitmap allocation size. The comment at `:160–163`
records the bug for future readers.

### 10.2 `shm_push_state` single-pass loop (`mutator_m3_0.c:189–306`)

**One-line:** computes all 13 features in a single O(MAP_SIZE) pass to
keep the bitmap hot in L1.

**Q: Why one pass over four?**
Cache. The cumulative bitmap is 64 KiB, which fits in most L2 caches
but not all L1. Four passes pay the L1-or-L2 round trip four times.
One pass keeps the bitmap loaded once. Benchmarked, materially faster
on our targets.

**Q: Why double for the accumulators?**
Sum-of-squares of `uint8_t` values across 65 K edges can exceed `2^32`.
I could have used `uint64_t` instead but `double` is the same width,
the same cycle count, and the divisions for mean/var fall out
naturally without a cast.

**Q: What if the trace bits change mid-pass?**
They don't — `trace_bits` is updated by AFL between target executions,
not during. We're called inside `afl_custom_fuzz` which runs after the
target has already finished its prior exec.

**Q: Race with the Python reader?**
The release-store on `state_seq` (`:305`) is the only thing Python
synchronises on. Python doesn't read features until it sees a fresh
seqno. The features are written *before* the seqno bump; release-store
guarantees they're visible to any acquire-load that sees the new seqno.

### 10.3 The release/acquire pairing on `state_seq` (`mutator_m3_0.c:305` and `rl_server.py:107`)

**One-line:** monotonic 32-bit sequence number with C-side release-store
and Python-side acquire-load, no locks.

**Q: Why not a mutex?**
Two reasons: a process-shared pthread mutex requires careful init in
SHM-resident memory, and if either process crashes mid-critical-section
the mutex can become unrecoverable. The seqno protocol is forward-progress
under crash — both sides just spin or exit cleanly.

**Q: What's the order of the writes?**
Features first, all 13 of them as plain volatile stores. Then the
release-store bumps `state_seq`. The release ensures all 13 prior
stores are visible before the seqno bump is observed.

**Q: How does Python avoid a torn read?**
It only proceeds past the inner spin loop after observing a *changed*
`state_seq`. By the release/acquire pairing, that means all 13
features written before that bump are visible. The whole-SHM `read()`
into a Python bytes object is one syscall — no further reordering risk.

**Q: What if you missed a state — Python is too slow?**
Not a correctness issue. The reward is computed against Python's
*previous observation*, not C's previous observation. The agent learns
from a stream of (s, a, r, s') tuples that are consistent in their own
right; the rate is just lower than C's update rate.

### 10.4 `shm_wait_action` spin (`mutator_m3_0.c:308–320`)

**One-line:** acquire-load on `action_seq` with 100µs `nanosleep`
between polls; returns the new action.

**Q: Why nanosleep and not a busy spin?**
A busy spin would pin a full core and starve the Python side, which
is on the same machine. `nanosleep(100µs)` actually de-schedules so
Python gets the CPU.

**Q: Why not futex?**
`futex(FUTEX_WAIT)` would give us zero CPU when waiting and a
kernel-mediated wakeup, at the cost of one syscall per wakeup. At our
mutation rate that's ~5K syscalls/sec — not free, ~5% CPU on the
syscall path alone. The spin protocol is zero-syscall when both sides
keep up; only slow when something's wrong.

**Q: Why 100µs and not 10µs or 1ms?**
Below ~50µs, the cost of waking up and re-polling approaches the cost
of the work itself. Above ~500µs we're adding visible latency to every
mutation. 100µs sits at the sweet spot — measured in
`benchmark_latency.py`.

**Q: What if Python never writes?**
We spin forever. The watchdog is at the bash level
(`run_model.sh:165`): if the Python process dies, bash kills AFL and
the trap cleans up the SHM file. No timeout in C itself.

### 10.5 `apply_mutation` 47-arm switch (`mutator_m3_0.c:360–559`)

**One-line:** dispatches the agent's chosen action ID to one of 47
in-place buffer mutations, mirroring AFL's own mutator vocabulary.

**Q: Why 47?**
It matches the count of distinct primitive operations AFL itself
exposes in its havoc + det stages. The point of the agent is to pick
*among the same primitives AFL would have picked among*; we wanted a
clean A/B vs vanilla, not a different action surface.

**Q: Why discrete actions instead of grouped families?**
Differential analysis on the libxml2 CVEs showed that *specific*
mutations were disproportionately effective on buggy versions —
arithmetic for the integer overflow, dictionary inserts for the
parser overread. Grouping would lose that resolution.

**Q: What about the strict-aliasing casts?**
You're right — `*(uint16_t*)(mb+pos)` is a strict-aliasing violation
per the C standard. AFL's own havoc code uses the same idiom; clang at
`-O2` accepts it. For portability I'd rewrite with memcpy into a
local. It's on my list.

**Q: What happens if the agent picks an invalid action?**
The check at `:633` clips out-of-range to action 46 (HAVOC). Cheap,
fail-safe.

### 10.6 First-call bootstrap (`mutator_m3_0.c:610–624`)

**One-line:** on the very first mutation, seed the cumulative bitmap
from AFL's first trace and skip the state-publish/action-wait round.

**Q: Why skip the round?**
There's no prior (s, a, r, s') tuple to publish. The agent has nothing
to react to. So we just do a default HAVOC and let the second call be
the first real exchange.

**Q: Doesn't this lose a training tuple?**
One tuple per campaign of 500K. Negligible. Could be rectified with a
synthetic "zero state" but the value isn't worth the code path.

**Q: Why seed cumulative_map from the trace?**
Otherwise the first real `shm_push_state` would compute features over
an entirely empty bitmap, and the early heat distribution would be
garbage. Seeding gives the agent a meaningful first state.

### 10.7 Reward function (`common.py:55–58`)

**One-line:** `reward = coverage_delta + 1000 * log1p(crash_delta)`.

**Q: Why no step cost?**
We don't want the agent to terminate episodes early (it can't anyway
— the loop runs until the budget is exhausted). A negative step cost
would push the value function down by a constant, which a Q-network
absorbs without effect on argmax.

**Q: Why log1p on crashes but not coverage?**
Coverage delta is bounded by the bitmap size (64 K) and is typically
0 or 1; no log needed. Crashes can spike (one input opens 100
indistinct crashes); without log, a single bad-input attribution
swamps the gradient.

**Q: Why the 1000 multiplier?**
Crashes are the actual goal — coverage is the proxy. 1000 makes any
crash discovery dominate any coverage gain in that step. Hand-tuned;
not from a sweep.

**Q: Is this reward sparse?**
Coverage is dense (~10% of steps yield a non-zero delta in early
training); crashes are extremely sparse (often zero across an entire
campaign on hardened targets). The bandit gets per-step learning
signal from coverage; the DQN exploits the temporal credit assignment
when crash deltas eventually appear.

### 10.8 `DQNAgent.train_step` Double-DQN (`common.py:128–151`)

**One-line:** standard Double DQN with an entropy regulariser.

**Q: Why Double DQN over vanilla?**
Vanilla DQN's `argmax(target(s'))` overestimates Q because the same
network selects and evaluates the action. Decoupling — argmax with
online, evaluate with target — reduces this bias. Especially relevant
in noisy reward landscapes like fuzzing where coverage deltas are
stochastic.

**Q: Why the entropy term?**
Once ε decays to 0.05, the network alone drives exploration. If the
softmax over Q has collapsed to one action, the agent stops trying
others. Adding `−λ·H(softmax(Q))` to the loss penalises that collapse.
Coefficient 0.01 — small enough not to hurt convergence on the strong
actions, large enough to keep the distribution from going to a delta.

**Q: Sample efficiency?**
Replay buffer 100K, batch 128, target sync every 1000 train steps,
γ=0.99. At 500K training steps with `train_freq=1` we do ~500K updates.
Convergence is empirical — coverage curves in
`docs/experiment_2_multi_benchmark_10m.md` show plateau around
200–300K steps on most targets.

### 10.9 `ContextualBanditAgent` Thompson sampling (`common.py:210–218`)

**One-line:** per-action posterior over reward; pick action with
max-sampled value.

**Q: Why bandit alongside DQN?**
The bandit assumes immediate reward (no temporal credit assignment).
For fuzzing, that's *almost* true — most actions either produce a
coverage gain on the next exec or don't. Carrying both lets us
ablate empirically: does the DQN actually use γ=0.99, or would γ=0
work just as well?

**Q: Why Thompson over UCB?**
Thompson is straightforward to implement when the network already
outputs a mean and variance per action. UCB would need a
confidence-interval calculator. They are theoretically similar for
many action distributions; we picked Thompson for cleanliness.

**Q: Why no replay buffer?**
A bandit's update is independent across steps. Replay buys nothing.

**Q: Are the heads tied or independent?**
They share the trunk (`BanditNet:trunk`) and split into mean and
logvar heads at the last linear layer. Trunk learns a state
representation; heads parameterise the per-action posterior over it.

### 10.10 `_skip` model variants and `train_freq`
(`run_model.sh:60–63`, `rl_server.py:118–119`)

**One-line:** `m3_0_skip` is the same agent as `m3_0` but trains
only every 4th step.

**Q: Why decouple action selection from training?**
Training is the expensive op (gradient + Adam + backward). Action
selection is one forward pass. By training every 4th step we get 4x
inference per train step at the same wall-clock budget — useful for
ablating sample efficiency vs. compute budget.

**Q: Do the `_skip` variants share C mutators with their parents?**
Yes. `run_model.sh:60` strips the `_skip` suffix when picking the
mutator `.so` and the SHM path. So `m3_0` and `m3_0_skip` share
`/tmp/rl_shm_m3_0` and `mutator_m3_0.so` — they cannot run
concurrently. That's enforced by the experimental harness, not by
code; flag.

### 10.11 `PlateauDetector` early stop (`common.py:61–72`)

**One-line:** declare convergence when coverage hasn't grown by
more than 1 edge in 10K steps and ε has decayed below 0.06.

**Q: Why both conditions?**
Without ε, an agent in early-training random exploration could appear
to plateau just because it's not yet learning. Without the coverage
window, an agent in a long stretch of unproductive but non-zero
exploration would never trigger.

**Q: Why grace period?**
Default 70% of train budget (`rl_server.py:82`). Coverage often
plateaus locally early in training when the agent first finds a
basin and then climbs out — premature stop would miss late-stage
gains.

**Q: What if coverage genuinely is plateauing because the target is
exhausted?**
That's the use case. Early-stop saves wall-clock time we'd otherwise
spend on diminishing returns. Hooked up at `rl_server.py:144`; sets
`stop = "coverage plateau"` and breaks the loop cleanly.

### 10.12 SHM file persistence and cleanup
(`mutator_m3_0.c:575`, `run_model.sh:104`)

**One-line:** SHM is a regular file at `/tmp/rl_shm_m3_0`, mmap'd
shared, and cleaned up by the bash trap.

**Q: Why a file instead of POSIX shm_open?**
Named SHM via `shm_open` lives in `/dev/shm` and must be unlinked
explicitly. A plain file in `/tmp` is identically `mmap`-able with
`MAP_SHARED`, gives us `xxd` inspectability, and survives a crash
without leaking into kernel-managed namespaces.

**Q: What if AFL crashes and leaves it behind?**
Next run reuses it. Both sides re-init their local sequence counters
from the persisted values, so the protocol re-syncs naturally on the
first new bump.

**Q: Race condition if two AFL processes share the path?**
Yes — undefined behaviour. The harness enforces single-tenant per SHM
path; for multi-tenant we'd need a path scheme that includes PID or
campaign ID.

---

*End of document.*
