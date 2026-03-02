#!/usr/bin/env bash
# scripts/build_and_compare.sh  —  Orchestrator for all 4 RL fuzzer models
#
# Run from repo root:  bash scripts/build_and_compare.sh [OPTIONS]
#
# Modes:
#   (a) Run all 4 model pipelines sequentially, then compare  [default]
#   (b) --compare-only  — skip model runs, aggregate existing outputs
#
# Example — full run:
#   bash scripts/build_and_compare.sh --train-steps 50000
#
# Example — compare only (models already ran separately):
#   bash scripts/build_and_compare.sh --compare-only
#
# Options:
#   --train-steps N     per-model step limit    (default: 50000)
#   --eval-steps  N     per-model eval steps    (default: 20000)
#   --afl-dir     DIR   AFL++ root              (default: $AFL_ROOT or ~/packages/AFLplusplus)
#   --target      PATH  fuzzer binary           (default: bin/target)
#   --seeds       DIR   seed corpus             (default: inputs/)
#   --models      LIST  comma-separated subset  (default: m0_0,m1_0,m1_1,m2)
#   --compare-only      skip all model runs
#   --no-build          pass --no-build to each run script
#   --eval-only         pass --eval-only to each run script
#   --phase       train|eval|both               (default: both)
#   --smooth      N     plot rolling window     (default: 200)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRAIN_STEPS=500000
EVAL_STEPS=50000
AFL_DIR="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
TARGET="${REPO_ROOT}/bin/target"
SEEDS="${REPO_ROOT}/inputs"
MODELS_TO_RUN="m0_0,m1_0,m1_1,m2"
COMPARE_ONLY=0
NO_BUILD=0
EVAL_ONLY=0
PHASE="both"
SMOOTH=500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-steps)  TRAIN_STEPS="$2";     shift 2 ;;
        --eval-steps)   EVAL_STEPS="$2";      shift 2 ;;
        --afl-dir)      AFL_DIR="$2";         shift 2 ;;
        --target)       TARGET="$2";          shift 2 ;;
        --seeds)        SEEDS="$2";           shift 2 ;;
        --models)       MODELS_TO_RUN="$2";   shift 2 ;;
        --compare-only) COMPARE_ONLY=1;       shift   ;;
        --no-build)     NO_BUILD=1;           shift   ;;
        --eval-only)    EVAL_ONLY=1;          shift   ;;
        --phase)        PHASE="$2";           shift 2 ;;
        --smooth)       SMOOTH="$2";          shift 2 ;;
        --help|-h)
            head -40 "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

COMPARE_DIR="${REPO_ROOT}/comparison_results"
mkdir -p "$COMPARE_DIR" "${REPO_ROOT}/bin"
MASTER_LOG="${REPO_ROOT}/comparison_results/build_and_compare.log"

# log() writes to file only — each run script prints its own output live to the
# terminal. Piping run scripts through tee breaks signal propagation and
# prevents the AFL++ live UI from rendering correctly.
log()     { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }
log_sep() { log "$(printf '─%.0s' {1..58})"; }
die()     { echo "[-] $*" >&2; exit 1; }

EXTRA_FLAGS=""
[[ $NO_BUILD  -eq 1 ]] && EXTRA_FLAGS="$EXTRA_FLAGS --no-build"
[[ $EVAL_ONLY -eq 1 ]] && EXTRA_FLAGS="$EXTRA_FLAGS --eval-only"

log_sep
log "  RL Fuzzer — build_and_compare"
log "  repo_root     : $REPO_ROOT"
log "  train_steps   : $TRAIN_STEPS"
log "  eval_steps    : $EVAL_STEPS"
log "  models to run : $MODELS_TO_RUN"
log "  compare_only  : $COMPARE_ONLY"
log "  phase         : $PHASE"
log_sep

if [[ $COMPARE_ONLY -eq 0 ]]; then
    [[ -x "${AFL_DIR}/afl-fuzz" ]] || die "afl-fuzz not found at ${AFL_DIR}/afl-fuzz"
    [[ -x "$TARGET" ]]             || die "target not found: $TARGET (run build_jsoncpp.sh first)"
    [[ -d "$SEEDS" ]]              || die "seeds not found: $SEEDS"
fi
command -v python3 >/dev/null || die "python3 not found"
python3 -c "import torch" 2>/dev/null || die "PyTorch not installed"

# ── Run models ────────────────────────────────────────────────────────────────
run_model() {
    local model="$1"
    local script="${REPO_ROOT}/scripts/run_${model}.sh"
    [[ -f "$script" ]] || die "$script not found"
    chmod +x "$script"

    log_sep; log "  Starting model: $model"; log_sep
    local t0; t0=$(date +%s)

    # Run directly — no pipe, no subshell. This ensures:
    #   1. SIGINT/SIGTERM reach the run script's trap handlers
    #   2. AFL++ stderr (live UI) renders correctly on the terminal
    #   3. AFL++ background process is properly killed when RL server exits
    # shellcheck disable=SC2086
    bash "$script" \
        --train-steps "$TRAIN_STEPS" \
        --eval-steps  "$EVAL_STEPS" \
        --afl-dir     "$AFL_DIR" \
        --target      "$TARGET" \
        --seeds       "$SEEDS" \
        $EXTRA_FLAGS

    log "  $model complete in $(( $(date +%s) - t0 ))s"
}

if [[ $COMPARE_ONLY -eq 0 ]]; then
    IFS=',' read -ra MODEL_LIST <<< "$MODELS_TO_RUN"
    for model in "${MODEL_LIST[@]}"; do
        model="${model// /}"
        case "$model" in
            m0_0|m1_0|m1_1|m2) run_model "$model" ;;
            *) log "  [warn] Unknown model '$model' — skipping" ;;
        esac
    done
    log_sep; log "  All model runs complete."; log_sep
else
    log "  --compare-only: skipping model runs"
fi

# ── Comparison ────────────────────────────────────────────────────────────────
log_sep; log "  Running comparison..."; log_sep

COMPARE_ARGS=""
for model in m0_0 m1_0 m1_1 m2; do
    plots_dir="${REPO_ROOT}/plots/${model}"
    if [[ -d "$plots_dir" ]]; then
        # Convert m0_0 → --m0-0 etc.
        flag="--${model//_/-}"
        COMPARE_ARGS="$COMPARE_ARGS $flag $plots_dir"
    fi
done

if [[ -z "$COMPARE_ARGS" ]]; then
    log "  [warn] No plots/ subdirs found — nothing to compare yet."
    exit 0
fi

# shellcheck disable=SC2086
python3 "${REPO_ROOT}/scripts/compare_metrics.py" \
    $COMPARE_ARGS \
    --out           "$COMPARE_DIR" \
    --phase         "$PHASE" \
    --smooth-window "$SMOOTH"

log_sep
log "  build_and_compare done."
log ""
log "  Per-model metrics : ${REPO_ROOT}/plots/<model>/"
log "  Model checkpoints : ${REPO_ROOT}/bin/rl_m*.pt"
log "  AFL++ train output: ${REPO_ROOT}/outputs/<model>/"
log "  AFL++ eval output : ${REPO_ROOT}/outputs_eval/<model>/"
log "  Comparison output : $COMPARE_DIR"
log "    comparison_report.txt"
log "    comparison_summary.json"
log "    plot_*.png"
log ""
log "  Master log: $MASTER_LOG"
log_sep
