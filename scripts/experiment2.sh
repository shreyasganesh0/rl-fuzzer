#!/usr/bin/env bash
# scripts/run_full_experiment.sh — Multi-benchmark, multi-milestone experiment
#
# Runs all model×benchmark combinations with incremental eval: one continuous
# 10M-step eval run, snapshotted at milestones via post-hoc CSV slicing.
#
# Usage:
#   bash scripts/run_full_experiment.sh [OPTIONS]
#
# Options:
#   --benchmarks LIST   comma-separated (default: jsoncpp,freetype2,libxml2,re2,harfbuzz,libpng)
#   --models LIST       comma-separated (default: m1_0,m1_1,m1_2)
#   --milestones LIST   comma-separated step counts (default: 500000,1000000,2000000,10000000)
#   --eval-runs N       number of eval repetitions per model (default: 1)
#   --skip-train        reuse existing checkpoints
#   --skip-build        reuse existing bin/target
#   --no-plateau        disable plateau early-stopping
#   --exp-root DIR      base directory (default: experiments/)
#   --train-steps N     training step limit (default: max milestone)
#   --eval-steps N      eval step limit (default: max milestone)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON=python3
SCRIPTS="${REPO_ROOT}/scripts"

# ── Defaults ──────────────────────────────────────────────────────────────────
BENCHMARKS="jsoncpp,freetype2,libxml2,re2,harfbuzz,libpng"
MODELS="m1_0,m1_1,m1_2"
MILESTONES="500000,1000000,2000000,10000000"
EVAL_RUNS=1
SKIP_TRAIN=0
SKIP_BUILD=0
NO_PLATEAU=0
EXP_ROOT="${REPO_ROOT}/experiments"
TRAIN_STEPS=""
EVAL_STEPS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmarks)  BENCHMARKS="$2";  shift 2 ;;
        --models)      MODELS="$2";      shift 2 ;;
        --milestones)  MILESTONES="$2";  shift 2 ;;
        --eval-runs)   EVAL_RUNS="$2";   shift 2 ;;
        --skip-train)  SKIP_TRAIN=1;     shift   ;;
        --skip-build)  SKIP_BUILD=1;     shift   ;;
        --no-plateau)  NO_PLATEAU=1;     shift   ;;
        --exp-root)    EXP_ROOT="$2";    shift 2 ;;
        --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
        --eval-steps)  EVAL_STEPS="$2";  shift 2 ;;
        --help|-h)     grep '^#' "$0" | sed 's/^# \{0,2\}//'; exit 0 ;;
        *) echo "[!] Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS"
IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra MILESTONE_LIST <<< "$MILESTONES"

# Derive max milestone for default train/eval steps
MAX_MILESTONE=0
for ms in "${MILESTONE_LIST[@]}"; do
    (( ms > MAX_MILESTONE )) && MAX_MILESTONE=$ms
done
[[ -z "$TRAIN_STEPS" ]] && TRAIN_STEPS=$MAX_MILESTONE
[[ -z "$EVAL_STEPS" ]]  && EVAL_STEPS=$MAX_MILESTONE

AFL_DIR="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
AFL_FUZZ="${AFL_DIR}/afl-fuzz"

ts()  { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }
sep() { log "$(printf '─%.0s' {1..60})"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$EXP_ROOT"
MASTER_LOG="${EXP_ROOT}/full_experiment.log"
: > "$MASTER_LOG"

sep
log "  RL Fuzzer — Full Multi-Benchmark Experiment"
log "  benchmarks  : $BENCHMARKS"
log "  models      : $MODELS"
log "  milestones  : $MILESTONES"
log "  train_steps : $TRAIN_STEPS"
log "  eval_steps  : $EVAL_STEPS"
log "  eval_runs   : $EVAL_RUNS"
log "  skip_train  : $SKIP_TRAIN"
log "  skip_build  : $SKIP_BUILD"
log "  no_plateau  : $NO_PLATEAU"
log "  exp_root    : $EXP_ROOT"
sep

# ── Sanity checks ────────────────────────────────────────────────────────────
[[ -x "$AFL_FUZZ" ]] || die "afl-fuzz not found at $AFL_FUZZ (set AFL_ROOT or install AFL++)"
command -v "$PYTHON" >/dev/null || die "python3 not found"
"$PYTHON" -c "import torch" 2>/dev/null || die "PyTorch not installed"
[[ -f "${SCRIPTS}/run_model.sh" ]] || die "run_model.sh not found"
[[ -f "${SCRIPTS}/slice_milestones.py" ]] || die "slice_milestones.py not found"

# ── Helper: run baseline (adapted from build_and_compare.sh) ─────────────────
run_baseline() {
    local exp_dir="$1"
    local baseline_tag="$2"   # "baseline" or "baseline_time"
    local stop_mode="$3"      # "steps" or "time"
    local stop_val="$4"       # step count or seconds

    local base_dir="${exp_dir}/outputs_eval/${baseline_tag}"
    local plots_dir="${exp_dir}/plots/${baseline_tag}"
    local csv="${plots_dir}/rl_metrics_${baseline_tag}_eval.csv"
    mkdir -p "$base_dir" "$plots_dir"
    rm -rf "$base_dir"

    local target="${REPO_ROOT}/bin/target"
    local seeds="${REPO_ROOT}/inputs"
    local dict_flag=""
    [[ -f "${REPO_ROOT}/dictionaries/target.dict" ]] && \
        dict_flag="-x ${REPO_ROOT}/dictionaries/target.dict"

    log "  Running baseline (tag=${baseline_tag}, mode=${stop_mode}, val=${stop_val})"
    local t0; t0=$(date +%s)

    # shellcheck disable=SC2086
    if [[ "$stop_mode" == "time" ]]; then
        AFL_AUTORESUME=1 AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
        AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
            "$AFL_FUZZ" -V "$stop_val" -i "$seeds" -o "$base_dir" $dict_flag -- "$target" @@ &
    else
        AFL_AUTORESUME=1 AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
        AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
            "$AFL_FUZZ" -E "$stop_val" -i "$seeds" -o "$base_dir" $dict_flag -- "$target" @@ &
    fi
    local afl_pid=$!
    log "  Baseline AFL++ PID $afl_pid"

    echo "step,reward,coverage_term,crash_term,loss,epsilon,coverage,crashes,action,elapsed_seconds" > "$csv"

    while kill -0 "$afl_pid" 2>/dev/null; do
        sleep 1
        local stats="${base_dir}/default/fuzzer_stats"
        [[ -f "$stats" ]] || continue

        local cov; cov=$(grep "^bitmap_cvg" "$stats" 2>/dev/null | awk -F'[: %]+' '{print $2}' || echo 0)
        local execs; execs=$(grep "^execs_done" "$stats" 2>/dev/null | awk -F': *' '{print $2}' || echo 0)
        local crashes; crashes=$(grep "^saved_crashes" "$stats" 2>/dev/null | awk -F': *' '{print $2}' || echo 0)
        local edges; edges=$(awk "BEGIN{printf \"%d\", $cov * 65536 / 100}" 2>/dev/null || echo 0)

        local elapsed; elapsed=$(( $(date +%s) - t0 ))
        echo "${execs},0.0,0.0,0.0,0.0,0.0,${edges},${crashes},-1,${elapsed}" >> "$csv"

        if [[ "$stop_mode" == "time" ]]; then
            [[ "$elapsed" -ge "$stop_val" ]] && break
        else
            [[ "$execs" -ge "$stop_val" ]] && break
        fi
    done

    kill -9 "$afl_pid" 2>/dev/null || true
    wait "$afl_pid" 2>/dev/null || true

    [[ -f "${base_dir}/default/fuzzer_stats" ]] && \
        cp "${base_dir}/default/fuzzer_stats" "${plots_dir}/fuzzer_stats_eval.txt"

    log "  Baseline ${baseline_tag} done in $(( $(date +%s) - t0 ))s → $csv"
}

# ── Helper: check if eval CSV has enough steps ───────────────────────────────
csv_has_enough_steps() {
    local csv_path="$1"
    local required_steps="$2"
    [[ -f "$csv_path" ]] || return 1
    "$PYTHON" -c "
import pandas as pd, sys
df = pd.read_csv('$csv_path')
sys.exit(0 if len(df) > 0 and df['step'].max() >= $required_steps * 0.95 else 1)
" 2>/dev/null
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP: iterate over benchmarks
# ══════════════════════════════════════════════════════════════════════════════

TOTAL_BENCHMARKS=${#BENCHMARK_LIST[@]}
BENCH_IDX=0

for benchmark in "${BENCHMARK_LIST[@]}"; do
    benchmark="${benchmark// /}"
    (( BENCH_IDX++ )) || true
    EXP_DIR="${EXP_ROOT}/${benchmark}"
    mkdir -p "$EXP_DIR"

    sep
    log "  BENCHMARK ${BENCH_IDX}/${TOTAL_BENCHMARKS}: ${benchmark}"
    log "  exp_dir: ${EXP_DIR}"
    sep

    # ── Phase 0: Build benchmark target ──────────────────────────────────────
    if [[ $SKIP_BUILD -eq 0 ]]; then
        log "  Phase 0: Building ${benchmark}..."
        if [[ -f "${SCRIPTS}/build_benchmark.sh" ]]; then
            bash "${SCRIPTS}/build_benchmark.sh" "$benchmark" 2>&1 | tee -a "$MASTER_LOG"
        else
            die "build_benchmark.sh not found"
        fi
        log "  Build complete."
    else
        log "  Phase 0: Skipped (--skip-build)"
        [[ -x "${REPO_ROOT}/bin/target" ]] || die "bin/target not found (run without --skip-build)"
    fi

    # ── Phase 1: Train all RL models ─────────────────────────────────────────
    if [[ $SKIP_TRAIN -eq 0 ]]; then
        sep
        log "  Phase 1: Training models for ${benchmark}"
        sep

        for model in "${MODEL_LIST[@]}"; do
            model="${model// /}"
            model_pt="${EXP_DIR}/bin/rl_${model}.pt"

            # Recovery: skip if checkpoint already exists with enough training
            if [[ -f "$model_pt" ]]; then
                log "  [skip] ${model}: checkpoint already exists at ${model_pt}"
                continue
            fi

            log "  Training ${model}..."
            T0=$(date +%s)

            EXTRA_FLAGS=""
            [[ $NO_PLATEAU -eq 1 ]] && EXTRA_FLAGS="$EXTRA_FLAGS --no-plateau"

            bash "${SCRIPTS}/run_model.sh" \
                --model-id    "$model" \
                --exp-dir     "$EXP_DIR" \
                --train-steps "$TRAIN_STEPS" \
                --eval-steps  "$EVAL_STEPS" \
                --milestones  "$MILESTONES" \
                $EXTRA_FLAGS \
                2>&1 | tee -a "$MASTER_LOG"

            log "  ${model} done in $(( $(date +%s) - T0 ))s"
        done
    else
        sep
        log "  Phase 1: Skipped (--skip-train)"
        for model in "${MODEL_LIST[@]}"; do
            model_pt="${EXP_DIR}/bin/rl_${model}.pt"
            [[ -f "$model_pt" ]] || die "Checkpoint missing: ${model_pt}"
        done
        log "  All checkpoints verified."

        # Run eval-only for each model
        sep
        log "  Phase 1b: Running eval-only for all models"
        sep
        for model in "${MODEL_LIST[@]}"; do
            model="${model// /}"
            eval_csv="${EXP_DIR}/plots/${model}/rl_metrics_${model}_eval.csv"

            # Recovery: skip if eval CSV already has enough steps
            if csv_has_enough_steps "$eval_csv" "$EVAL_STEPS"; then
                log "  [skip] ${model}: eval CSV already complete"
                continue
            fi

            log "  Eval ${model}..."
            T0=$(date +%s)

            EXTRA_FLAGS="--eval-only"
            [[ $NO_PLATEAU -eq 1 ]] && EXTRA_FLAGS="$EXTRA_FLAGS --no-plateau"

            bash "${SCRIPTS}/run_model.sh" \
                --model-id    "$model" \
                --exp-dir     "$EXP_DIR" \
                --train-steps "$TRAIN_STEPS" \
                --eval-steps  "$EVAL_STEPS" \
                $EXTRA_FLAGS \
                2>&1 | tee -a "$MASTER_LOG"

            log "  ${model} eval done in $(( $(date +%s) - T0 ))s"
        done
    fi

    # ── Phase 2: Baseline same-steps ─────────────────────────────────────────
    sep
    log "  Phase 2: Baseline (same-steps) for ${benchmark}"
    sep

    bl_csv="${EXP_DIR}/plots/baseline/rl_metrics_baseline_eval.csv"
    if csv_has_enough_steps "$bl_csv" "$EVAL_STEPS"; then
        log "  [skip] baseline: eval CSV already complete"
    else
        run_baseline "$EXP_DIR" "baseline" "steps" "$EVAL_STEPS"
    fi

    # ── Phase 3: Baseline same-time ──────────────────────────────────────────
    sep
    log "  Phase 3: Baseline (same-time) for ${benchmark}"
    sep

    # Find median RL eval time at max milestone
    T_RL=$("$PYTHON" "${SCRIPTS}/slice_milestones.py" \
        --query-time \
        --exp-dir "$EXP_DIR" \
        --models "$MODELS" \
        --milestone "$MAX_MILESTONE" 2>/dev/null || echo 60)
    log "  Median RL eval time at ${MAX_MILESTONE} steps: ${T_RL}s"

    bt_csv="${EXP_DIR}/plots/baseline_time/rl_metrics_baseline_time_eval.csv"
    if [[ -f "$bt_csv" ]]; then
        log "  [skip] baseline_time: eval CSV already exists"
    else
        if [[ "$T_RL" -gt 0 ]] 2>/dev/null; then
            run_baseline "$EXP_DIR" "baseline_time" "time" "$T_RL"
        else
            log "  [warn] Could not determine RL eval time — skipping time baseline"
        fi
    fi

    # ── Phase 4: Slice + Compare ─────────────────────────────────────────────
    sep
    log "  Phase 4: Slicing milestones and running comparisons"
    sep

    "$PYTHON" "${SCRIPTS}/slice_milestones.py" \
        --exp-dir    "$EXP_DIR" \
        --milestones "$MILESTONES" \
        --models     "$MODELS" \
        --eval-runs  "$EVAL_RUNS" \
        2>&1 | tee -a "$MASTER_LOG"

    sep
    log "  ${benchmark} complete. Results in: ${EXP_DIR}/milestones/"
    sep

done

# ── Cross-benchmark summary ──────────────────────────────────────────────────
sep
log "  Generating cross-benchmark summary..."
sep

"$PYTHON" "${SCRIPTS}/summarize_benchmarks.py" \
    --exp-root   "$EXP_ROOT" \
    --benchmarks "$BENCHMARKS" \
    --models     "$MODELS" \
    --milestones "$MILESTONES" \
    2>&1 | tee -a "$MASTER_LOG"

sep
log "  ALL BENCHMARKS COMPLETE"
log ""
log "  Results:"
for benchmark in "${BENCHMARK_LIST[@]}"; do
    benchmark="${benchmark// /}"
    log "    ${benchmark}: ${EXP_ROOT}/${benchmark}/milestones/"
done
log "  Cross-benchmark: ${EXP_ROOT}/summary/"
log ""
log "  Log: ${MASTER_LOG}"
sep
