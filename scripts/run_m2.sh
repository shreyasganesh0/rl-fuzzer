#!/usr/bin/env bash
# scripts/run_m2.sh  —  train + eval pipeline for Model M2
# Run from repo root:  bash scripts/run_m2.sh [OPTIONS]
#
# Options:
#   --train-steps N   step limit (default: 50000)
#   --eval-steps  N   eval steps (default: 20000)
#   --afl-dir     DIR AFL++ root (default: $AFL_ROOT or ~/packages/AFLplusplus)
#   --target      PATH            (default: bin/target)
#   --seeds       DIR             (default: inputs/)
#   --no-build        skip recompile
#   --eval-only       skip training

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRAIN_STEPS=500000
EVAL_STEPS=50000
AFL_DIR="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
TARGET="${REPO_ROOT}/bin/target"
SEEDS="${REPO_ROOT}/inputs"
NO_BUILD=0
EVAL_ONLY=0
SHM_PATH="/tmp/rl_shm_m2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
        --eval-steps)  EVAL_STEPS="$2";  shift 2 ;;
        --afl-dir)     AFL_DIR="$2";     shift 2 ;;
        --target)      TARGET="$2";      shift 2 ;;
        --seeds)       SEEDS="$2";       shift 2 ;;
        --no-build)    NO_BUILD=1;       shift   ;;
        --eval-only)   EVAL_ONLY=1;      shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

AFL_FUZZ="${AFL_DIR}/afl-fuzz"
AFL_INC="${AFL_DIR}/include"
DICT="${REPO_ROOT}/dictionaries/target.dict"
MUTATOR_SO="${REPO_ROOT}/bin/mutator_m2.so"
MODEL_PT="${REPO_ROOT}/bin/rl_m2.pt"
AFL_TRAIN_DIR="${REPO_ROOT}/outputs/m2"
AFL_EVAL_DIR="${REPO_ROOT}/outputs_eval/m2"
PLOTS_DIR="${REPO_ROOT}/plots/m2"

mkdir -p "${REPO_ROOT}/bin" "$PLOTS_DIR" \
         "${REPO_ROOT}/outputs" "${REPO_ROOT}/outputs_eval"

# Safe initial values — guards in cleanup prevent signalling PID 0
AFL_PID=0
RL_PID=0

log() { echo "[$(date +%H:%M:%S)] [M2] $*"; }
die() { echo "[-] $*" >&2; exit 1; }

cleanup() {
    # Disarm immediately to prevent re-entry on repeated Ctrl+C
    trap - EXIT SIGINT SIGTERM
    log "cleanup: killing background jobs"
    [[ $RL_PID  -gt 0 ]] && { kill -9 "$RL_PID"  2>/dev/null || true; wait "$RL_PID"  2>/dev/null || true; }
    [[ $AFL_PID -gt 0 ]] && { kill -9 "$AFL_PID" 2>/dev/null || true; wait "$AFL_PID" 2>/dev/null || true; }
    rm -f "$SHM_PATH"
}

# Trap EXIT, SIGINT (Ctrl+C), and SIGTERM so cleanup fires regardless of
# whether this script is run directly or as a subprocess of build_and_compare.sh
trap cleanup EXIT SIGINT SIGTERM

log "======================================================"
log "  RL Fuzzer — Model M2"
log "  repo     : $REPO_ROOT"
log "  afl_dir  : $AFL_DIR"
log "  target   : $TARGET"
log "  seeds    : $SEEDS"
log "  train_steps=$TRAIN_STEPS  eval_steps=$EVAL_STEPS"
log "======================================================"

[[ -x "$AFL_FUZZ" ]] || die "afl-fuzz not found at $AFL_FUZZ  (set AFL_ROOT or --afl-dir)"
[[ -x "$TARGET" ]]   || die "target not found: $TARGET  (run build_jsoncpp.sh first)"
[[ -d "$SEEDS" ]]    || die "seeds not found: $SEEDS"
command -v python3 >/dev/null || die "python3 not found"
python3 -c "import torch" 2>/dev/null || die "PyTorch not installed"

# ── Dict flag ─────────────────────────────────────────────────────────────────
DICT_FLAG=""; [[ -f "$DICT" ]] && DICT_FLAG="-x $DICT"

# ── Build mutator ─────────────────────────────────────────────────────────────
if [[ $NO_BUILD -eq 0 ]]; then
    log "Compiling src/mutator_m2.c → $MUTATOR_SO"
    clang -O2 -shared -fPIC -I"${AFL_INC}" \
        -o "$MUTATOR_SO" "${REPO_ROOT}/src/mutator_m2.c"
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

    python3 "${REPO_ROOT}/scripts/rl_server_m2.py" \
        --mode train --model "$MODEL_PT" \
        --train-steps "$TRAIN_STEPS" --results-dir "$PLOTS_DIR" &
    RL_PID=$!
    log "RL server PID $RL_PID"
    sleep 2

    # shellcheck disable=SC2086
    AFL_AUTORESUME=1 AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
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

# Always evaluate on the original seed corpus so all models start from the
# same initial conditions. Using the training queue as eval seeds confounds
# model quality with queue richness accumulated during training.
EVAL_SEEDS="$SEEDS"
log "Seeding eval from original corpus: $EVAL_SEEDS"

python3 "${REPO_ROOT}/scripts/rl_server_m2.py" \
    --mode eval --model "$MODEL_PT" \
    --eval-steps "$EVAL_STEPS" --train-steps "$TRAIN_STEPS" --results-dir "$PLOTS_DIR" &
RL_PID=$!
log "RL eval server PID $RL_PID"
sleep 2

# shellcheck disable=SC2086
AFL_AUTORESUME=1 AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
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
log "  M2 done."
log "    checkpoint : $MODEL_PT"
log "    metrics    : $PLOTS_DIR"
log "    afl train  : $AFL_TRAIN_DIR"
log "    afl eval   : $AFL_EVAL_DIR"
log "======================================================"
