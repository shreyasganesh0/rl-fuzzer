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
#   --smooth      N     plot rolling window     (default: 500)
#   --no-plateau        disable plateau early-stopping in all models
#   --run-baseline      run a plain AFL++ baseline (no RL) for eval_steps and
#                       include it in the comparison plots

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON=python3   # fallback to system python3

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
NO_PLATEAU=0
RUN_BASELINE=0
BASELINE_TIME_SECONDS=0
BASELINE_TAG="baseline"
BASELINE_ONLY=0

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
        --no-plateau)   NO_PLATEAU=1;         shift   ;;
        --run-baseline) RUN_BASELINE=1;       shift   ;;
        --baseline-time-seconds) BASELINE_TIME_SECONDS="$2"; shift 2 ;;
        --baseline-tag)          BASELINE_TAG="$2";           shift 2 ;;
        --baseline-only)         BASELINE_ONLY=1;             shift   ;;
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
[[ $NO_BUILD   -eq 1 ]] && EXTRA_FLAGS="$EXTRA_FLAGS --no-build"
[[ $EVAL_ONLY  -eq 1 ]] && EXTRA_FLAGS="$EXTRA_FLAGS --eval-only"
[[ $NO_PLATEAU -eq 1 ]] && EXTRA_FLAGS="$EXTRA_FLAGS --no-plateau"

log_sep
log "  RL Fuzzer — build_and_compare"
log "  repo_root     : $REPO_ROOT"
log "  train_steps   : $TRAIN_STEPS"
log "  eval_steps    : $EVAL_STEPS"
log "  models to run : $MODELS_TO_RUN"
log "  compare_only  : $COMPARE_ONLY"
log "  no_plateau    : $NO_PLATEAU"
log "  run_baseline  : $RUN_BASELINE"
log "  baseline_tag  : $BASELINE_TAG"
log "  baseline_time : $BASELINE_TIME_SECONDS"
log "  baseline_only : $BASELINE_ONLY"
log "  phase         : $PHASE"
log_sep

if [[ $COMPARE_ONLY -eq 0 ]]; then
    [[ -x "${AFL_DIR}/afl-fuzz" ]] || die "afl-fuzz not found at ${AFL_DIR}/afl-fuzz"
    [[ -x "$TARGET" ]]             || die "target not found: $TARGET (run build_jsoncpp.sh first)"
    [[ -d "$SEEDS" ]]              || die "seeds not found: $SEEDS"
fi
command -v "$PYTHON" >/dev/null || die "python3 not found"
"$PYTHON" -c "import torch" 2>/dev/null || die "PyTorch not installed"

# ── Run models ────────────────────────────────────────────────────────────────
run_model() {
    local model="$1"
    local script="${REPO_ROOT}/scripts/run_model.sh"
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
        --model-id    "$model" \
        --train-steps "$TRAIN_STEPS" \
        --eval-steps  "$EVAL_STEPS" \
        --afl-dir     "$AFL_DIR" \
        --target      "$TARGET" \
        --seeds       "$SEEDS" \
        $EXTRA_FLAGS

    log "  $model complete in $(( $(date +%s) - t0 ))s"
}

if [[ $COMPARE_ONLY -eq 0 && $BASELINE_ONLY -eq 0 ]]; then
    IFS=',' read -ra MODEL_LIST <<< "$MODELS_TO_RUN"
    for model in "${MODEL_LIST[@]}"; do
        model="${model// /}"
        case "$model" in
            m0_0|m1_0|m1_1|m2|m0_0_skip|m1_0_skip|m1_1_skip|m2_skip) run_model "$model" ;;
            *) log "  [warn] Unknown model '$model' — skipping" ;;
        esac
    done
    log_sep; log "  All model runs complete."; log_sep
else
    log "  --compare-only: skipping model runs"
fi

# ── Baseline (plain AFL++ without RL) ─────────────────────────────────────────
# Run vanilla AFL++ for EVAL_STEPS using the same seeds and target as the RL
# models, with no custom mutator.  We poll fuzzer_stats every 5 seconds and
# emit a CSV whose 'coverage' column matches the RL eval CSVs so compare_metrics
# can overlay it on the eval coverage plot.
run_baseline() {
    local AFL_FUZZ="${AFL_DIR}/afl-fuzz"
    [[ -x "$AFL_FUZZ" ]] || { log "  [error] afl-fuzz not found at $AFL_FUZZ"; return 1; }
    local BASE_DIR="${REPO_ROOT}/outputs_eval/${BASELINE_TAG}"
    local PLOTS_DIR="${REPO_ROOT}/plots/${BASELINE_TAG}"
    local CSV="${PLOTS_DIR}/rl_metrics_${BASELINE_TAG}_eval.csv"
    mkdir -p "$BASE_DIR" "$PLOTS_DIR"
    rm -rf "$BASE_DIR"

    log_sep; log "  Running baseline AFL++ (tag=${BASELINE_TAG}, time=${BASELINE_TIME_SECONDS}s, steps=${EVAL_STEPS})"; log_sep
    local t0; t0=$(date +%s)

    # shellcheck disable=SC2086
    if [[ "$BASELINE_TIME_SECONDS" -gt 0 ]]; then
        # Time-based: AFL++ runs for exactly N seconds via -V flag (flushes stats on clean exit)
        AFL_AUTORESUME=1 AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
            "$AFL_FUZZ" -V "$BASELINE_TIME_SECONDS" -i "$SEEDS" -o "$BASE_DIR" $DICT_FLAG -- "$TARGET" @@ &
    else
        # Steps-based: AFL++ self-terminates at EVAL_STEPS (-E flag)
        AFL_AUTORESUME=1 AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
            "$AFL_FUZZ" -E "$EVAL_STEPS" -i "$SEEDS" -o "$BASE_DIR" $DICT_FLAG -- "$TARGET" @@ &
    fi
    local AFL_BL_PID=$!
    log "  Baseline AFL++ PID $AFL_BL_PID"

    # Write CSV header matching RL eval format so compare_metrics can parse it
    echo "step,reward,coverage_term,crash_term,loss,epsilon,coverage,crashes,action,elapsed_seconds" \
        > "$CSV"

    local step=0
    # We don't have a step counter — instead we poll fuzzer_stats and
    # map wall-clock exec counts to a synthetic step index so the CSV
    # is directly comparable to the RL eval CSVs.
    while kill -0 "$AFL_BL_PID" 2>/dev/null; do
        sleep 1
        local stats="${BASE_DIR}/default/fuzzer_stats"
        [[ -f "$stats" ]] || continue

        local cov; cov=$(grep "^bitmap_cvg" "$stats" 2>/dev/null | awk -F'[: %]+' '{print $2}' || echo 0)
        local execs; execs=$(grep "^execs_done" "$stats" 2>/dev/null | awk -F': *' '{print $2}' || echo 0)
        local crashes; crashes=$(grep "^saved_crashes" "$stats" 2>/dev/null | awk -F': *' '{print $2}' || echo 0)
        # bitmap_cvg is a percentage — convert to edge count using MAP_SIZE=65536
        local edges; edges=$(awk "BEGIN{printf \"%d\", $cov * 65536 / 100}" 2>/dev/null || echo 0)
        step=$execs

        local elapsed; elapsed=$(( $(date +%s) - t0 ))
        echo "${step},0.0,0.0,0.0,0.0,0.0,${edges},${crashes},-1,${elapsed}" >> "$CSV"

        # Stop condition: time-based OR steps-based
        if [[ "$BASELINE_TIME_SECONDS" -gt 0 ]]; then
            if [[ "$elapsed" -ge "$BASELINE_TIME_SECONDS" ]]; then
                log "  Baseline reached ${elapsed}s — stopping"
                break
            fi
        else
            if [[ "$execs" -ge "$EVAL_STEPS" ]]; then
                log "  Baseline reached ${execs} execs — stopping"
                break
            fi
        fi
    done

    kill -9 "$AFL_BL_PID" 2>/dev/null || true
    wait  "$AFL_BL_PID" 2>/dev/null || true

    # Save fuzzer_stats for the report
    [[ -f "${BASE_DIR}/default/fuzzer_stats" ]] && \
        cp "${BASE_DIR}/default/fuzzer_stats" "${PLOTS_DIR}/fuzzer_stats_eval.txt"

    log "  Baseline complete in $(( $(date +%s) - t0 ))s  → $CSV"
}

DICT_FLAG=""
[[ -f "${REPO_ROOT}/dictionaries/target.dict" ]] && \
    DICT_FLAG="-x ${REPO_ROOT}/dictionaries/target.dict"

if [[ $BASELINE_ONLY -eq 1 || ( $COMPARE_ONLY -eq 0 && $RUN_BASELINE -eq 1 ) ]]; then
    run_baseline
fi

# --baseline-only: skip models and comparison — just run baseline then exit
if [[ $BASELINE_ONLY -eq 1 ]]; then
    exit 0
fi

# ── Comparison ────────────────────────────────────────────────────────────────
log_sep; log "  Running comparison..."; log_sep

COMPARE_ARGS=""
for model in m0_0 m1_0 m1_1 m2 m0_0_skip m1_0_skip m1_1_skip m2_skip; do
    plots_dir="${REPO_ROOT}/plots/${model}"
    if [[ -d "$plots_dir" ]]; then
        # Convert m0_0 → --m0-0, m0_0_skip → --m0-0-skip etc.
        flag="--${model//_/-}"
        COMPARE_ARGS="$COMPARE_ARGS $flag $plots_dir"
    fi
done

# Include baseline if it was run or if its plots dir already exists
BASELINE_PLOTS="${REPO_ROOT}/plots/${BASELINE_TAG}"
if [[ -d "$BASELINE_PLOTS" ]]; then
    COMPARE_ARGS="$COMPARE_ARGS --baseline $BASELINE_PLOTS"
fi

if [[ -z "$COMPARE_ARGS" ]]; then
    log "  [warn] No plots/ subdirs found — nothing to compare yet."
    exit 0
fi

# shellcheck disable=SC2086
"$PYTHON" "${REPO_ROOT}/scripts/compare_metrics.py" \
    $COMPARE_ARGS \
    --out           "$COMPARE_DIR" \
    --phase         "$PHASE" \
    --smooth-window "$SMOOTH"

log_sep
log "  build_and_compare done."
log ""
log "  Per-model metrics : ${REPO_ROOT}/plots/<model>/"
log "  Baseline metrics  : ${REPO_ROOT}/plots/${BASELINE_TAG}/  (if --run-baseline used)"
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
