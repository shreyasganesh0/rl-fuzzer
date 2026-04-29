# Experiments Runbook

How to build benchmark targets and run the three experiment scripts.
Covers the recipe contract used by `scripts/build_benchmark.sh`, every
flag accepted by `scripts/experiment{1,2,3}.sh`, and a smoke test for
each so a one-time setup can be validated end-to-end in a few minutes.

---

## 1. Building a target

`scripts/build_benchmark.sh <name>` is a generic driver. It does not
know how to build any specific target — it sources a recipe at
`benchmarks/<name>/build_recipe.sh` that supplies the project-specific
bits (where to clone, how to invoke cmake/autotools/meson, how to link
the final fuzzer). The driver handles everything that is the same
across benchmarks: cloning at a pinned commit, wiring AFL++'s compiler
wrappers, copying seeds and dictionary into the rl-fuzzer tree, and
smoke-testing the binary.

```
scripts/build_benchmark.sh jsoncpp
        │
        ├─ checks: AFL_ROOT, FUZZBENCH, recipe exists
        ├─ sources benchmarks/jsoncpp/build_recipe.sh
        │      ↳ sets FUZZBENCH_NAME, GIT_URL, BUILD_STEPS(),
        │        LINK_STEPS(), optionally SEEDS_DIR / DICT_PATH
        ├─ reads pinned commit from
        │       $FUZZBENCH/benchmarks/$FUZZBENCH_NAME/benchmark.yaml
        ├─ git clone $GIT_URL → ~/targets/<project>/src ; checkout commit
        ├─ runs BUILD_STEPS()    (recipe-defined)
        ├─ runs LINK_STEPS()     (recipe-defined) → bin/target
        ├─ installs dictionary    → dictionaries/target.dict
        ├─ installs seeds         → inputs/
        └─ smoke-tests the binary
```

### 1.1 Prerequisites checked at startup

These are checked in `scripts/build_benchmark.sh:82–108`:

| Path | Default | Purpose | If missing |
|---|---|---|---|
| `$AFL_ROOT/afl-clang-fast++` | `~/packages/AFLplusplus` | AFL-instrumented C++ compiler used as `CXX` in recipes. | Build AFL++ from source. |
| `$AFL_ROOT/libAFLDriver.a`  | same | libFuzzer-compatible driver: provides `main()` that calls `LLVMFuzzerTestOneInput`. Linked in `LINK_STEPS()` as `$AFLDRIVER`. | `cd $AFL_ROOT && make libAFLDriver.a`. |
| `$FUZZBENCH`                | `~/fuzzbench` | Source of pinned commit (`benchmark.yaml`) and harness file (`benchmarks/<name>/target.cc`). | `git clone --depth=1 https://github.com/google/fuzzbench.git ~/fuzzbench`. |
| `$FUZZBENCH/benchmarks/<FUZZBENCH_NAME>/benchmark.yaml` | inside the FuzzBench clone | Source of `commit:` line read at `build_benchmark.sh:169`. | Use a different benchmark name or remove the pinning. |
| `~/targets/<project>/src/` | (created lazily) | Where the project source is cloned. The driver only re-clones if `.git` is missing (`:178–182`). | Created automatically from `GIT_URL`. |

Two optional locations matter for system-level dependencies:

- `packages/local/` under the repo. If present, headers/libs are added
  to `CFLAGS`/`LDFLAGS`/`PKG_CONFIG_PATH` (`:67–75`). Use this when you
  cannot install distro packages and need `libpng`/`liblzma`/etc.
- `.venv/bin/meson` — added to `PATH` if it exists (`:78–80`). Required
  for benchmarks that build with meson (freetype2, harfbuzz).

### 1.2 The recipe contract

A recipe is a sourced shell file. The contract
(`scripts/build_benchmark.sh:151–155`) requires four things:

| Element | Meaning |
|---|---|
| `FUZZBENCH_NAME=...` | Lookup key under `$FUZZBENCH/benchmarks/`. Drives commit pinning, harness path, and (if present) seed corpus. For libxml2 it's `libxml2_xml`; for jsoncpp it's `jsoncpp_jsoncpp_fuzzer`. |
| `GIT_URL=...` | Where to clone the project. Cloned to `~/targets/<basename of url>/src`. |
| `BUILD_STEPS()` | Function that produces a static library at a known location. The driver doesn't link inside this — it just expects something usable to exist when the function returns. |
| `LINK_STEPS()` | Function that links the static lib + the harness + `$AFLDRIVER` into `$RL_FUZZER/bin/target`. |

Optional vars:

| Var | Purpose |
|---|---|
| `DICT_PATH` | Set inside `BUILD_STEPS()` (after clone) to point at a dictionary the project ships. jsoncpp uses `$SRC_DIR/src/test_lib_json/fuzz.dict` (`benchmarks/jsoncpp/build_recipe.sh:38`). |
| `SEEDS_DIR` | Path to a seeds directory the recipe wants to install. If unset and FuzzBench has `seeds/`, those are used; if neither, a synthetic `FUZZ` seed is written. |

### 1.3 Variables passed in to the recipe

Set right before `source "$RECIPE"` (`build_benchmark.sh:135–145`):

```
CC        = $AFL_ROOT/afl-clang-fast      # AFL-instrumented C compiler
CXX       = $AFL_ROOT/afl-clang-fast++    # AFL-instrumented C++ compiler
CFLAGS    = "-g -O2"  + (-I$LOCAL_PREFIX/include if local pkgs present)
CXXFLAGS  = "-g -O2"  + same
LDFLAGS   = ""        + (-L$LOCAL_PREFIX/lib...) if local pkgs present
AFLDRIVER = $AFL_ROOT/libAFLDriver.a
SRC_DIR   = $TARGET_DIR/src               # = ~/targets/<project>/src
RL_FUZZER = repo root
FUZZBENCH = ~/fuzzbench
```

The recipe reads these and is expected to do the build.

### 1.4 How the dictionary is wired

Three sources, in priority order
(`build_benchmark.sh:202–222`):

1. **Recipe-set `DICT_PATH`** (e.g., jsoncpp's `fuzz.dict` from inside
   the source tree).
2. **A `.dict` file under `$FUZZBENCH/benchmarks/<FUZZBENCH_NAME>/`** or
   anywhere in the cloned source. The first match is copied.
3. **A synthetic 1-line fallback** `kw1="FUZZ"` so the dict argument
   never points at a missing file.

Destination is always `dictionaries/target.dict`.
`run_model.sh:125–126` picks it up if present and adds
`-x dictionaries/target.dict` to the `afl-fuzz` command. Missing or
empty dictionary just means `-x` is not added — fuzzing still runs.

### 1.5 How seeds are wired

Same three-source pattern (`build_benchmark.sh:225–251`):

1. **Recipe-set `SEEDS_DIR`** copied into `inputs/` if `inputs/` is
   empty.
2. **`$FUZZBENCH/benchmarks/<FUZZBENCH_NAME>/seeds/`** if it exists.
3. **Synthetic `inputs/seed_default`** containing the literal bytes
   `FUZZ`.

In all three cases, **`inputs/` is left alone if it already has files**
(`-z "$(ls -A ...)"` check). Deliberate — lets you keep custom seeds
across rebuilds — but means switching benchmarks requires
`rm -rf inputs/*` first.

The jsoncpp recipe is a special case worth flagging: FuzzBench has no
seeds for it, so the recipe defines a `jsoncpp_install_seeds` function
that writes 4 hand-crafted JSON files
(`benchmarks/jsoncpp/build_recipe.sh:55–72`). The driver never actually
calls that function — it's dead code. The 4 named JSON files in the
current `inputs/` directory are there because someone manually copied
them once. Worth fixing if redistributing.

### 1.6 Per-benchmark quirks

| Benchmark | Build system | Special needs |
|---|---|---|
| `jsoncpp` | cmake | None — pure C++ static lib. |
| `libxml2` | autotools (`./autogen.sh`) | Needs `zlib1g-dev`, `liblzma-dev` system packages. Links `-lz -llzma`. Falls back to in-tree `fuzz/xml.c` if the FuzzBench `target.cc` is missing (`benchmarks/libxml2/build_recipe.sh:35–38`). |
| `freetype2`, `harfbuzz` | meson | Need meson (`pip install meson` into `.venv`) — the driver auto-adds `.venv/bin` to PATH if present. May need extra system libs. |
| `libpng`, `re2` | varies | Check the recipe directly. |

### 1.7 What you actually need to start fresh

```bash
# One-time setup
git clone https://github.com/AFLplusplus/AFLplusplus ~/packages/AFLplusplus
cd ~/packages/AFLplusplus && make -j$(nproc) && make libAFLDriver.a

git clone --depth=1 https://github.com/google/fuzzbench.git ~/fuzzbench

# Per-benchmark
cd ~/projects/rl-fuzzer
bash scripts/build_benchmark.sh jsoncpp        # or libxml2, freetype2, etc.
```

After a successful run you have:

```
bin/target                       — instrumented binary, takes a file as @@
dictionaries/target.dict         — dictionary (or 1-line fallback)
inputs/                          — at least one seed
outputs/, outputs_eval/, plots/  — empty, ready for run_model.sh
```

`scripts/build_benchmark.sh --help` lists every available recipe.

---

## 2. Experiment 1 — `scripts/experiment1.sh`

One target (whatever `bin/target` points at — typically jsoncpp), 5 RL
models trained once, then evaluated N times against the same
checkpoint. Optional vanilla AFL++ baseline runs in two flavours
(same-steps and same-time).

**Default invocation (full):**
```bash
bash scripts/experiment1.sh
```
Builds & trains all 5 models on `bin/target`, runs 5 eval rounds, and
produces a same-steps comparison report at
`comparison_results/same_steps/`. ~55 min on a typical desktop CPU.

### 2.1 Flags (`scripts/experiment1.sh:49–63`)

| Flag | Default | Effect |
|---|---|---|
| `--skip-train` | off | Reuse `bin/rl_<model>.pt` checkpoints; skips the ~10-min/model training phase. Verifies all checkpoints exist before starting (`:91–97`). |
| `--eval-runs N` | `5` | Total eval rounds. The first eval is harvested from the end of training (`run_1`); rounds 2..N are pure `--eval-only` (`:174–177`). |
| `--train-steps N` | `500000` | Per-model training step cap. |
| `--eval-steps N` | `500000` | Per-eval-round step cap. |
| `--models CSV` | `m0_0,m1_0,m1_1,m1_2,m2` | Subset of models. Add `_skip` variants here (e.g. `m1_0,m1_0_skip`). |
| `--run-baseline` | off | Adds vanilla AFL++ runs in two flavours: same-steps (`baseline/`) and same-time (`baseline_time/`) at the median RL eval wall-clock (`:240–254`, `:256–284`). |
| `--no-plateau` | off | Forwarded to `rl_server.py` to disable plateau early-stopping. |
| `--compare-mode` | `steps` | Affects CSV output dir naming; the script always runs both same-steps and (if baseline enabled) same-time comparisons regardless. |
| `--out DIR` | `comparison_results` | Where the final reports land. |

### 2.2 Phases

1. **Phase 0** (`:99–125`): clear `outputs_eval/` and `--out` dir. If
   not `--skip-train`, also clear `outputs/`, `plots/`, and old
   `bin/mutator_*.so`.
2. **Phase 1** (`:127–178`): training. Calls `build_and_compare.sh`
   without `--eval-only` so it does train + first eval; that first
   eval becomes `run_1`.
3. **Phase 2** (`:180–238`): N-1 (or N if `--skip-train`) more eval
   rounds, each preceded by `rm -rf outputs_eval/` to defeat
   `AFL_AUTORESUME` queue contamination (rationale at `:13–17`).
4. **Phase 2b** (`:256–284`, only if `--run-baseline`): time-based
   baseline at median RL eval wall-clock.
5. **Phase 3** (`:286–308`): verification — counts CSV files per model.
6. **Phase 4a/4b** (`:310–341`): `compare_metrics.py` produces
   `comparison_report.txt` plus PNG plots.

### 2.3 Smoke test (~3 min)

```bash
bash scripts/build_benchmark.sh jsoncpp           # one-time target build
bash scripts/experiment1.sh \
    --models m0_0 \
    --train-steps 2000 \
    --eval-steps 1000 \
    --eval-runs 2 \
    --no-plateau \
    --out comparison_results/smoke1
```
Validates: build → train → checkpoint → eval → CSV → compare path on
the smallest model. Confirm
`comparison_results/smoke1/same_steps/comparison_report.txt` exists
and is non-empty. Add `--run-baseline` to also smoke the baseline +
same-time path.

---

## 3. Experiment 2 — `scripts/experiment2.sh`

Multi-benchmark, multi-model, milestone-snapshotted. Per-benchmark
loop, per-model nested loop. Eval runs once per model and is sliced
post-hoc at each milestone via `slice_milestones.py`. Designed to be
resumable — partial runs skip work that's already complete.

**Default invocation:**
```bash
bash scripts/experiment2.sh
```
Iterates `jsoncpp,freetype2,libxml2,re2,harfbuzz,libpng` ×
`m1_0,m1_1,m1_2`. For each benchmark: builds it, trains every model
once with milestone checkpoints at 500K/1M/2M/10M, runs same-steps and
same-time baselines, then slices and summarizes. Hours-to-days of
runtime depending on benchmark count and milestone budget.

### 3.1 Flags (`scripts/experiment2.sh:40–55`)

| Flag | Default | Effect |
|---|---|---|
| `--benchmarks CSV` | all six | Restrict to a subset. Each name must have `benchmarks/<name>/build_recipe.sh`. |
| `--models CSV` | `m1_0,m1_1,m1_2` | Models to train per benchmark. |
| `--milestones CSV` | `500000,1000000,2000000,10000000` | Step counts at which to snapshot the training checkpoint (`run_model.sh:139–143` calls `cp` per milestone). Max milestone also drives the `--train-steps`/`--eval-steps` defaults (`:62–67`). |
| `--eval-runs N` | `1` | Eval repetitions per model. |
| `--skip-train` | off | Reuse `<exp_dir>/bin/rl_<model>.pt`; runs eval-only. Recovery: skips per-model eval if its CSV already covers ≥95% of `--eval-steps` (`:262–266`). |
| `--skip-build` | off | Reuse existing `bin/target`. Asserts it exists (`:209`). |
| `--no-plateau` | off | Forwarded to `run_model.sh`. |
| `--exp-root DIR` | `experiments/` | Per-benchmark trees go under `<exp-root>/<benchmark>/`. |
| `--train-steps N` | max milestone | Override default. |
| `--eval-steps N` | max milestone | Override default. |

### 3.2 Phases (per benchmark)

1. **Phase 0** (`:198–210`): build the benchmark via
   `build_benchmark.sh <name>` unless `--skip-build`.
2. **Phase 1** (`:212–284`): train each model in sequence; save
   milestone checkpoints. With `--skip-train`, runs eval-only and
   resumes via `csv_has_enough_steps` (`:169–178`).
3. **Phase 2** (`:286–296`): same-steps baseline via local
   `run_baseline()` helper (`:103–166`) using `afl-fuzz -E`.
4. **Phase 3** (`:298–320`): same-time baseline. RL median wall-clock
   computed by `slice_milestones.py --query-time`.
5. **Phase 4** (`:322–336`): `slice_milestones.py` cuts each eval CSV
   at every milestone and produces per-milestone reports under
   `<exp-dir>/milestones/`.
6. **Cross-benchmark summary** (`:341–351`): `summarize_benchmarks.py`
   aggregates across all benchmarks into `<exp-root>/summary/`.

### 3.3 Smoke test (~5 min)

```bash
bash scripts/experiment2.sh \
    --benchmarks jsoncpp \
    --models m1_0 \
    --milestones 1000,2000 \
    --train-steps 2000 \
    --eval-steps 1000 \
    --eval-runs 1 \
    --no-plateau \
    --exp-root experiments/smoke2
```
Validates: per-benchmark build → train → milestone checkpoints
(`bin/rl_m1_0.pt.1k`, `.2k`) → eval → baseline (steps + time) →
`slice_milestones.py` → `summarize_benchmarks.py`. Check
`experiments/smoke2/jsoncpp/milestones/` exists and
`experiments/smoke2/summary/` has output. To skip the lengthy
benchmark build, add `--skip-build` and reuse whatever `bin/target` is
already there.

---

## 4. Experiment 3 — `scripts/experiment3.sh`

Differential-informed RL fuzzing. Trains M3_0 (DQN + bandit variants)
and M1_0 on `xml005_buggy`, evaluates each plus an AFL++ baseline on
`xml005_buggy` (in-distribution) and `xml017_buggy` (transfer).
Fewer flags than the other two; many choices are hard-coded.

**Default invocation:**
```bash
bash scripts/experiment3.sh
```

### 4.1 Flags (`scripts/experiment3.sh:26–33`)

| Flag | Default | Effect |
|---|---|---|
| `--train-steps N` | `500000` | Per-model training cap. Same value passed to all three trains. |
| `--eval-steps N` | `500000` | Per-eval cap. Forwarded to `run_model.sh` and to `afl-fuzz -E` for the baseline. |
| `--eval-runs N` | `5` | Eval repetitions per (target × variant). |

### 4.2 Hard-coded choices (edit the script to vary)

| Setting | Value | Where |
|---|---|---|
| Train target | `xml005_buggy` | `:24` |
| Eval targets | `xml005_buggy`, `xml017_buggy` | `:103` |
| Variants | `m3_0_dqn`, `m3_0_bandit`, `m1_0_compare` | `:116` |
| Plateau early-stop | always disabled | `:77` (`--no-plateau` always forwarded) |
| Baseline timeout | `EVAL_STEPS / 50 + 30` seconds | `:155` |

### 4.3 Required pre-staged artifacts

This script does **not** call `build_benchmark.sh`. It expects:

| Path | Contents |
|---|---|
| `experiments/differential/targets/xml005_buggy/target` | Pre-built libxml2 v2.9.4 with CVE-2017-5130 (integer overflow). |
| `experiments/differential/targets/xml005_fixed/target` | Same library, version with the fix. (Used by docs / analysis, not by experiment3.sh itself.) |
| `experiments/differential/targets/xml017_buggy/target` | Pre-built libxml2 v2.9.3 with CVE-2016-1762 (heap overread). |
| `experiments/differential/targets/xml017_fixed/target` | Fixed v2.9.4. (Used by docs / analysis.) |
| `experiments/differential/dictionaries/libxml2.dict` | 89 entries from libxml2's `fuzz/xml.dict`. Already checked in. |
| `experiments/differential/seeds/` | 38 XML files from libxml2's `test/` corpus. Already checked in. |

If `xml017_buggy/target` is missing, the script logs and continues
(`:105`); the in-distribution eval (`xml005_buggy`) still runs. The
exact build commands for the libxml2 differential targets are
documented in `docs/differential_fuzzing_experiment_plan.md`.

### 4.4 Phases

1. **Phase 1–3** (`:82–100`): three separate trainings — M3_0 DQN,
   M3_0 bandit, M1_0 — each on `xml005_buggy`. Each training writes
   to its own `<results>/<variant>/bin/rl_<model>.pt`.
2. **Phase 4** (`:102–161`): for each eval target, for each run,
   for each variant: copy the trained checkpoint into the per-run
   directory, run eval-only via `run_model.sh`. Then run an AFL++
   baseline using `afl-fuzz -E $EVAL_STEPS`.

### 4.5 Smoke test (~4 min)

```bash
# Prereq: differential targets must be pre-built. If you don't have
# them, build at least xml005_buggy following
# docs/differential_fuzzing_experiment_plan.md.
bash scripts/experiment3.sh \
    --train-steps 2000 \
    --eval-steps 1000 \
    --eval-runs 1
```
Validates: 3 trainings (DQN, bandit, M1_0) → 1 eval round × 2 targets
× 4 variants. Check
`experiments/differential/results/{m3_0_dqn,m3_0_bandit,m1_0_compare}/bin/rl_*.pt`
exist and
`experiments/differential/results/eval_xml005_buggy/{m3_0_dqn,m3_0_bandit,m1_0_compare,baseline}/run_1/`
are populated.

---

## 5. Cross-cutting tips

- **Build the target first.** Experiments 1 and 2 assume `bin/target`
  is ready. Run `bash scripts/build_benchmark.sh jsoncpp` (or another
  recipe name).
- **`--no-plateau` for short smokes.** Plateau early-stop fires after
  `0.7 × train_steps`; with very small `--train-steps` it stops
  training instantly otherwise.
- **Watch `outputs_eval/`.** Experiment 1 deletes it before every eval
  round (`:194`) to avoid `AFL_AUTORESUME` reusing the previous
  queue. If you write your own runner, mirror that.
- **Logs.** Experiment 1: `comparison_results/experiment.log`.
  Experiment 2: `experiments/full_experiment.log`. Experiment 3: only
  stdout — pipe to `tee` if you want one.
- **Recovery.** Experiment 2 has the most resumable design —
  `csv_has_enough_steps` (`:169–178`) lets you re-run after a crash and
  only the missing pieces redo. Experiments 1 and 3 don't have this;
  partial runs need to be restarted.

---

## 6. Pass-through: `run_model.sh` flags

All three experiment scripts ultimately invoke `scripts/run_model.sh`.
Its flags are documented at the top of that script. The most
load-bearing ones:

| Flag | Effect |
|---|---|
| `--model-id ID` | Required. Selects mutator `.so` and Python module. `_skip` suffix → train every 4 steps. |
| `--train-steps N` / `--eval-steps N` | Caps. |
| `--target PATH` / `--seeds DIR` / `--dict PATH` | Override defaults of `bin/target`, `inputs/`, `dictionaries/target.dict`. |
| `--exp-dir DIR` | Redirects checkpoint, AFL outputs, and plots into `<exp-dir>/...` instead of repo-root subdirs. |
| `--milestones CSV` | Forwarded to `rl_server.py --milestones`; produces `bin/rl_<id>.pt.<tag>` snapshots. |
| `--algorithm dqn\|bandit` | Selects DQN (default) or contextual bandit agent. |
| `--no-build` | Skip mutator recompile. |
| `--eval-only` | Skip train phase. |
| `--no-plateau` | Disable plateau early-stop. |
