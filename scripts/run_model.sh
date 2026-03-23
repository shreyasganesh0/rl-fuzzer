#!/usr/bin/env bash
# scripts/run_model.sh — Unified train + eval pipeline for any RL model.
# Run from repo root:  bash scripts/run_model.sh --model-id m0_0 [OPTIONS]
#
# Options:
#   --model-id    ID  model identifier: m0_0, m1_0, m1_1, m2  (required)
#   --train-steps N   step limit (default: 500000)
#   --eval-steps  N   eval steps (default: 50000)
#   --afl-dir     DIR AFL++ root (default: $AFL_ROOT or ~/packages/AFLplusplus)
#   --target      PATH            (default: bin/target)
#   --seeds       DIR             (default: inputs/)
#   --no-build        skip recompile
#   --eval-only       skip training
#   --no-plateau      disable plateau early-stopping
#   --exp-dir     DIR experiment root (default: REPO_ROOT — fully backward compatible)
#   --milestones  LIST comma-separated step counts passed through to rl_server.py

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON=python3

MODEL_ID=""
TRAIN_STEPS=500000
EVAL_STEPS=50000
AFL_DIR="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
TARGET="${REPO_ROOT}/bin/target"
SEEDS="${REPO_ROOT}/inputs"
NO_BUILD=0
EVAL_ONLY=0
NO_PLATEAU=0
EXP_DIR=""
MILESTONES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)    MODEL_ID="$2";    shift 2 ;;
        --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
        --eval-steps)  EVAL_STEPS="$2";  shift 2 ;;
        --afl-dir)     AFL_DIR="$2";     shift 2 ;;
        --target)      TARGET="$2";      shift 2 ;;
        --seeds)       SEEDS="$2";       shift 2 ;;
        --no-build)    NO_BUILD=1;       shift   ;;
        --eval-only)   EVAL_ONLY=1;      shift   ;;
        --no-plateau)  NO_PLATEAU=1;     shift   ;;
        --exp-dir)     EXP_DIR="$2";     shift 2 ;;
        --milestones)  MILESTONES="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -n "$MODEL_ID" ]] || { echo "[-] --model-id is required"; exit 1; }

# _skip models reuse the parent model's C mutator and SHM path
BASE_MODEL_ID="${MODEL_ID%_skip}"   # strip _skip suffix (no-op if absent)
TRAIN_FREQ=1
if [[ "$MODEL_ID" != "$BASE_MODEL_ID" ]]; then
    TRAIN_FREQ=4
fi

# Derive all paths — mutator/SHM use BASE_MODEL_ID, everything else uses MODEL_ID
SHM_PATH="/tmp/rl_shm_${BASE_MODEL_ID}"
AFL_FUZZ="${AFL_DIR}/afl-fuzz"
AFL_INC="${AFL_DIR}/include"
DICT="${REPO_ROOT}/dictionaries/target.dict"
MUTATOR_SO="${REPO_ROOT}/bin/mutator_${BASE_MODEL_ID}.so"

# When --exp-dir is set, redirect checkpoint/output/plots into that tree
if [[ -n "$EXP_DIR" ]]; then
    MODEL_PT="${EXP_DIR}/bin/rl_${MODEL_ID}.pt"
    AFL_TRAIN_DIR="${EXP_DIR}/outputs/${MODEL_ID}"
    AFL_EVAL_DIR="${EXP_DIR}/outputs_eval/${MODEL_ID}"
    PLOTS_DIR="${EXP_DIR}/plots/${MODEL_ID}"
    mkdir -p "${EXP_DIR}/bin" "$PLOTS_DIR" \
             "${EXP_DIR}/outputs" "${EXP_DIR}/outputs_eval"
else
    MODEL_PT="${REPO_ROOT}/bin/rl_${MODEL_ID}.pt"
    AFL_TRAIN_DIR="${REPO_ROOT}/outputs/${MODEL_ID}"
    AFL_EVAL_DIR="${REPO_ROOT}/outputs_eval/${MODEL_ID}"
    PLOTS_DIR="${REPO_ROOT}/plots/${MODEL_ID}"
    mkdir -p "${REPO_ROOT}/bin" "$PLOTS_DIR" \
             "${REPO_ROOT}/outputs" "${REPO_ROOT}/outputs_eval"
fi

LABEL="${MODEL_ID^^}"   # uppercase for log tags (m0_0 → M0_0)

# Safe initial values — guards in cleanup prevent signalling PID 0
AFL_PID=0
RL_PID=0

log() { echo "[$(date +%H:%M:%S)] [${LABEL}] $*"; }
die() { echo "[-] $*" >&2; exit 1; }

cleanup() {
    # Disarm immediately to prevent re-entry on repeated Ctrl+C
    trap - EXIT SIGINT SIGTERM
    log "cleanup: killing background jobs"
    [[ $RL_PID  -gt 0 ]] && { kill -9 "$RL_PID"  2>/dev/null || true; wait "$RL_PID"  2>/dev/null || true; }
    [[ $AFL_PID -gt 0 ]] && { kill -9 "$AFL_PID" 2>/dev/null || true; wait "$AFL_PID" 2>/dev/null || true; }
    rm -f "$SHM_PATH"
}

trap cleanup EXIT SIGINT SIGTERM

log "======================================================"
log "  RL Fuzzer — Model ${LABEL}"
log "  repo     : $REPO_ROOT"
log "  afl_dir  : $AFL_DIR"
log "  target   : $TARGET"
log "  seeds    : $SEEDS"
log "  train_steps=$TRAIN_STEPS  eval_steps=$EVAL_STEPS"
log "======================================================"

[[ -x "$AFL_FUZZ" ]] || die "afl-fuzz not found at $AFL_FUZZ  (set AFL_ROOT or --afl-dir)"
[[ -x "$TARGET" ]]   || die "target not found: $TARGET  (run build_benchmark.sh first)"
[[ -d "$SEEDS" ]]    || die "seeds not found: $SEEDS"
command -v "$PYTHON" >/dev/null || die "python3 not found"
"$PYTHON" -c "import torch" 2>/dev/null || die "PyTorch not installed"

# ── Dict flag ─────────────────────────────────────────────────────────────────
DICT_FLAG=""; [[ -f "$DICT" ]] && DICT_FLAG="-x $DICT"

# ── Build mutator ─────────────────────────────────────────────────────────────
if [[ $NO_BUILD -eq 0 ]]; then
    log "Compiling src/mutator_${BASE_MODEL_ID}.c → $MUTATOR_SO"
    clang -O2 -march=native -ffast-math -shared -fPIC -I"${AFL_INC}" -lm \
        -o "$MUTATOR_SO" "${REPO_ROOT}/src/mutator_${BASE_MODEL_ID}.c"
    log "Mutator compiled OK"
else
    [[ -f "$MUTATOR_SO" ]] || die "--no-build but $MUTATOR_SO not found"
    log "Skipping build (--no-build)"
fi

# ── Train ─────────────────────────────────────────────────────────────────────
if [[ $EVAL_ONLY -eq 0 ]]; then
    log "--- TRAIN PHASE ---"
    rm -f "$MODEL_PT"   # always start fresh for a clean comparison run
    rm -rf "$AFL_TRAIN_DIR"; rm -f "$SHM_PATH"

    "$PYTHON" "${REPO_ROOT}/scripts/rl_server.py" \
        --model-id "$MODEL_ID" \
        --mode train --model "$MODEL_PT" \
        --train-steps "$TRAIN_STEPS" --results-dir "$PLOTS_DIR" \
        --train-freq "$TRAIN_FREQ" \
        $( [[ $NO_PLATEAU -eq 1 ]] && echo "--no-plateau" ) \
        $( [[ -n "$MILESTONES" ]] && echo "--milestones $MILESTONES" ) &
    RL_PID=$!
    log "RL server PID $RL_PID"
    sleep 2

    # shellcheck disable=SC2086
    AFL_AUTORESUME=1 AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
    AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
    AFL_CUSTOM_MUTATOR_LIBRARY="$MUTATOR_SO" AFL_CUSTOM_MUTATOR_ONLY=1 \
        "$AFL_FUZZ" -i "$SEEDS" -o "$AFL_TRAIN_DIR" $DICT_FLAG -- "$TARGET" @@ &
    AFL_PID=$!
    log "AFL++ PID $AFL_PID"

    wait "$RL_PID" || true
    log "RL server exited"
    kill -9 "$AFL_PID" 2>/dev/null || true
    wait "$AFL_PID" 2>/dev/null || true
    AFL_PID=0; RL_PID=0
    log "AFL++ stopped"

    [[ -f "${AFL_TRAIN_DIR}/default/fuzzer_stats" ]] && \
        cp "${AFL_TRAIN_DIR}/default/fuzzer_stats" "${PLOTS_DIR}/fuzzer_stats_train.txt"
    log "--- TRAIN COMPLETE ---"
fi

# ── Eval ──────────────────────────────────────────────────────────────────────
log "--- EVAL PHASE ---"
[[ -f "$MODEL_PT" ]] || die "No checkpoint at $MODEL_PT — train first"
rm -rf "$AFL_EVAL_DIR"; rm -f "$SHM_PATH"

EVAL_SEEDS="$SEEDS"
log "Seeding eval from original corpus: $EVAL_SEEDS"

"$PYTHON" "${REPO_ROOT}/scripts/rl_server.py" \
    --model-id "$MODEL_ID" \
    --mode eval --model "$MODEL_PT" \
    --eval-steps "$EVAL_STEPS" --train-steps "$TRAIN_STEPS" --results-dir "$PLOTS_DIR" \
    --train-freq "$TRAIN_FREQ" &
RL_PID=$!
log "RL eval server PID $RL_PID"
sleep 2

# shellcheck disable=SC2086
AFL_AUTORESUME=1 AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
AFL_CUSTOM_MUTATOR_LIBRARY="$MUTATOR_SO" AFL_CUSTOM_MUTATOR_ONLY=1 \
    "$AFL_FUZZ" -i "$EVAL_SEEDS" -o "$AFL_EVAL_DIR" $DICT_FLAG -- "$TARGET" @@ &
AFL_PID=$!
log "AFL++ eval PID $AFL_PID"

wait "$RL_PID" || true
log "RL eval exited"
kill -9 "$AFL_PID" 2>/dev/null || true
wait "$AFL_PID" 2>/dev/null || true
AFL_PID=0; RL_PID=0
log "AFL++ eval stopped"

[[ -f "${AFL_EVAL_DIR}/default/fuzzer_stats" ]] && \
    cp "${AFL_EVAL_DIR}/default/fuzzer_stats" "${PLOTS_DIR}/fuzzer_stats_eval.txt"

# Clear trap before clean exit so EXIT handler doesn't double-fire
trap - EXIT SIGINT SIGTERM

log "======================================================"
log "  ${LABEL} done."
log "    checkpoint : $MODEL_PT"
log "    metrics    : $PLOTS_DIR"
log "    afl train  : $AFL_TRAIN_DIR"
log "    afl eval   : $AFL_EVAL_DIR"
log "======================================================"
