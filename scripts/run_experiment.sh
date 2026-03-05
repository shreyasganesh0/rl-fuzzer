#!/usr/bin/env bash
# scripts/run_experiment.sh
#
# DESIGN: Train once, eval 5 times, compare averages.
#
# WHY train once:
#   Training = learning a policy. The checkpoint is the result.
#   Running eval N times measures how consistently that policy
#   performs given AFL++'s inherent randomness (queue ordering,
#   seed selection). That is the variance we want to characterize.
#
# WHY clear outputs_eval/ before every eval run:
#   AFL_AUTORESUME resumes from the previous run's queue if the
#   dir exists. This means run_2 starts with run_1's discovered
#   inputs already in the queue -- coverage "gained" shows as 0
#   and throughput looks artificially high. Always delete it first.
#
# USAGE:
#   # Full experiment (~55 min on CPU):
#   bash scripts/run_experiment.sh
#
#   # Skip training, use existing checkpoints (~13 min):
#   bash scripts/run_experiment.sh --skip-train
#
#   # With AFL++ baseline comparison:
#   bash scripts/run_experiment.sh --run-baseline
#
#   # Faster (3 eval runs, less statistical power):
#   bash scripts/run_experiment.sh --skip-train --eval-runs 3

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="${REPO}/scripts"
PYTHON="${REPO}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON=python3   # fallback to system python3

# ── Defaults ──────────────────────────────────────────────────────────────────
SKIP_TRAIN=0
EVAL_RUNS=5
TRAIN_STEPS=500000
EVAL_STEPS=500000
MODELS_CSV="m0_0,m1_0,m1_1,m2"   # override with --models to add _skip variants
RUN_BASELINE=0
NO_PLATEAU=""
COMPARE_MODE="steps"
OUT_DIR="${REPO}/comparison_results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-train)    SKIP_TRAIN=1;          shift   ;;
        --eval-runs)     EVAL_RUNS="$2";         shift 2 ;;
        --train-steps)   TRAIN_STEPS="$2";       shift 2 ;;
        --eval-steps)    EVAL_STEPS="$2";        shift 2 ;;
        --models)        MODELS_CSV="$2";        shift 2 ;;
        --run-baseline)  RUN_BASELINE=1;         shift   ;;
        --no-plateau)    NO_PLATEAU="--no-plateau"; shift ;;
        --compare-mode)  COMPARE_MODE="$2";      shift 2 ;;
        --out)           OUT_DIR="$2";           shift 2 ;;
        --help|-h)       grep '^#' "$0" | sed 's/^# \{0,2\}//'; exit 0 ;;
        *) echo "[!] Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra MODEL_LIST <<< "$MODELS_CSV"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/experiment.log"
: > "$LOG"

ts()  { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }
sep() { log "$(printf '─%.0s' {1..60})"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

sep
log "  RL Fuzzer — Full Experiment (train-once, eval-${EVAL_RUNS}-times)"
log "  skip_train  : $SKIP_TRAIN"
log "  eval_runs   : $EVAL_RUNS"
log "  train_steps : $TRAIN_STEPS"
log "  eval_steps  : $EVAL_STEPS"
log "  models      : $MODELS_CSV"
log "  run_baseline: $RUN_BASELINE"
sep

# ── Sanity checks ──────────────────────────────────────────────────────────────
[[ -f "${SCRIPTS}/build_and_compare.sh" ]] || die "build_and_compare.sh not found"
[[ -f "${SCRIPTS}/compare_metrics.py"  ]] || die "compare_metrics.py not found"
command -v "$PYTHON" >/dev/null             || die "python3 not found"
"$PYTHON" -c "import torch" 2>/dev/null     || die "PyTorch not installed"

if [[ $SKIP_TRAIN -eq 1 ]]; then
    for model in "${MODEL_LIST[@]}"; do
        [[ -f "${REPO}/bin/rl_${model}.pt" ]] || \
            die "Checkpoint missing: bin/rl_${model}.pt  (run without --skip-train first)"
    done
    log "  Checkpoints verified. Training will be skipped."
fi

# ── PHASE 0: Clear all generated files ────────────────────────────────────────
sep
log "  PHASE 0: Clearing generated directories"
sep

# Always clear these regardless
rm -rf "${REPO}/outputs_eval"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
log "  Cleared: outputs_eval/, ${OUT_DIR}"

if [[ $SKIP_TRAIN -eq 0 ]]; then
    # Fresh training run — wipe everything including old checkpoints and CSVs
    rm -rf "${REPO}/outputs"
    rm -rf "${REPO}/plots"
    rm -f  "${REPO}"/bin/mutator_*.so
    # Keep bin/rl_m*.pt only if they exist (build_and_compare.sh will overwrite)
    log "  Cleared: outputs/, plots/, bin/*.so"
else
    # Just wipe old per-run eval subdirs, keep training CSVs intact
    for model in "${MODEL_LIST[@]}"; do
        rm -rf "${REPO}/plots/${model}/run_"* 2>/dev/null || true
    done
    rm -rf "${REPO}/plots/baseline/run_"*           2>/dev/null || true
    rm -rf "${REPO}/plots/baseline_time/run_"*      2>/dev/null || true
    log "  Cleared: plots/*/run_N/ eval subdirs (training CSVs preserved)"
fi

# ── PHASE 1: Training ─────────────────────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
    sep
    log "  PHASE 1: Training all models once (~10 min/model)"
    log "  This is the ONLY training run. Output: bin/rl_m*.pt"
    sep

    T0=$(date +%s)
    BL_FLAG=""
    [[ $RUN_BASELINE -eq 1 ]] && BL_FLAG="--run-baseline"
    # build_and_compare.sh without --eval-only runs BOTH train and eval.
    # The eval it produces here becomes run_1 (we save it below).
    bash "${SCRIPTS}/build_and_compare.sh" \
        --train-steps "$TRAIN_STEPS" \
        --eval-steps  "$EVAL_STEPS" \
        --models      "$MODELS_CSV" \
        $NO_PLATEAU \
        $BL_FLAG \
        2>&1 | tee -a "$LOG"

    log "  Training done in $(( $(date +%s) - T0 ))s"

    # The eval that ran at end of training = run_1
    log "  Saving run_1 (from end-of-training eval)..."
    for model in "${MODEL_LIST[@]}"; do
        dst="${REPO}/plots/${model}/run_1"
        mkdir -p "$dst"
        [[ -f "${REPO}/plots/${model}/rl_metrics_${model}_eval.csv" ]] && \
            cp "${REPO}/plots/${model}/rl_metrics_${model}_eval.csv" \
               "${dst}/rl_metrics_${model}_eval.csv"
        [[ -f "${REPO}/outputs_eval/${model}/default/fuzzer_stats" ]] && \
            cp "${REPO}/outputs_eval/${model}/default/fuzzer_stats" \
               "${dst}/fuzzer_stats_eval.txt"
    done

    FIRST_EVAL_RUN=2   # runs 2..N are pure eval-only
else
    log "  PHASE 1: Skipped (--skip-train)"
    FIRST_EVAL_RUN=1   # all N runs are eval-only
fi

# ── PHASE 2: Eval N times (or N-1 if training already ran eval once) ──────────
sep
log "  PHASE 2: Running $(( EVAL_RUNS - FIRST_EVAL_RUN + 1 )) more eval rounds"
log "  (Each clears outputs_eval/ first to prevent AFL_AUTORESUME bias)"
sep

for run_idx in $(seq "$FIRST_EVAL_RUN" "$EVAL_RUNS"); do
    sep
    log "  ── Eval run ${run_idx} / ${EVAL_RUNS} ──"

    # CRITICAL: delete AFL++ output dirs.
    # If outputs_eval/<model>/default/ exists, AFL_AUTORESUME will resume
    # from the prior run's queue. coverage "gained" will show as 0 and
    # the baseline result will be inflated. Always delete before each run.
    rm -rf "${REPO}/outputs_eval"
    log "  Cleared outputs_eval/ (fresh queue for this run)"

    T0=$(date +%s)
    BL_FLAG=""
    [[ $RUN_BASELINE -eq 1 ]] && BL_FLAG="--run-baseline"

    bash "${SCRIPTS}/build_and_compare.sh" \
        --eval-only \
        --eval-steps  "$EVAL_STEPS" \
        --models      "$MODELS_CSV" \
        $NO_PLATEAU \
        $BL_FLAG \
        2>&1 | tee -a "$LOG"

    log "  Run ${run_idx} done in $(( $(date +%s) - T0 ))s"

    # Save this run's CSV and fuzzer_stats
    for model in "${MODEL_LIST[@]}"; do
        dst="${REPO}/plots/${model}/run_${run_idx}"
        mkdir -p "$dst"

        csv_src="${REPO}/plots/${model}/rl_metrics_${model}_eval.csv"
        if [[ -f "$csv_src" ]]; then
            cp "$csv_src" "${dst}/rl_metrics_${model}_eval.csv"
            log "  Saved ${model}/run_${run_idx} CSV"
        else
            log "  [WARN] ${model}/run_${run_idx}: CSV not found at $csv_src"
        fi

        fs_src="${REPO}/outputs_eval/${model}/default/fuzzer_stats"
        [[ -f "$fs_src" ]] && cp "$fs_src" "${dst}/fuzzer_stats_eval.txt"
    done

    if [[ $RUN_BASELINE -eq 1 ]]; then
        dst="${REPO}/plots/baseline/run_${run_idx}"
        mkdir -p "$dst"
        [[ -f "${REPO}/plots/baseline/rl_metrics_baseline_eval.csv" ]] && \
            cp "${REPO}/plots/baseline/rl_metrics_baseline_eval.csv" \
               "${dst}/rl_metrics_baseline_eval.csv"
        [[ -f "${REPO}/outputs_eval/baseline/default/fuzzer_stats" ]] && \
            cp "${REPO}/outputs_eval/baseline/default/fuzzer_stats" \
               "${dst}/fuzzer_stats_eval.txt"
    fi
done

# ── Compute median RL eval elapsed time for same-time baseline ────────────────
T_RL=60  # fallback
T_RL=$("$PYTHON" -c "
import pandas as pd, glob, numpy as np
files = glob.glob('${REPO}/plots/m*/run_*/rl_metrics_*_eval.csv')
times = []
for f in files:
    try:
        df = pd.read_csv(f)
        if 'elapsed_seconds' in df.columns and len(df) > 0:
            times.append(float(df['elapsed_seconds'].iloc[-1]))
    except: pass
print(int(np.median(times)) if times else 60)
" 2>/dev/null || echo 60)
log "  Median RL eval time: ${T_RL}s  (will use for same-time baseline)"

# ── PHASE 2b: Time-based baseline (same wall-clock as RL models) ──────────────
if [[ $RUN_BASELINE -eq 1 ]]; then
    sep
    log "  PHASE 2b: Running time-based baseline (${T_RL}s per run × ${EVAL_RUNS} runs)"
    sep

    for run_idx in $(seq 1 "$EVAL_RUNS"); do
        sep
        log "  ── Time-baseline run ${run_idx} / ${EVAL_RUNS} ──"
        rm -rf "${REPO}/outputs_eval/baseline_time"

        bash "${SCRIPTS}/build_and_compare.sh" \
            --baseline-only \
            --baseline-time-seconds "$T_RL" \
            --baseline-tag "baseline_time" \
            --eval-steps "$EVAL_STEPS" \
            2>&1 | tee -a "$LOG"

        dst="${REPO}/plots/baseline_time/run_${run_idx}"
        mkdir -p "$dst"
        [[ -f "${REPO}/plots/baseline_time/rl_metrics_baseline_time_eval.csv" ]] && \
            cp "${REPO}/plots/baseline_time/rl_metrics_baseline_time_eval.csv" \
               "${dst}/rl_metrics_baseline_time_eval.csv"
        [[ -f "${REPO}/outputs_eval/baseline_time/default/fuzzer_stats" ]] && \
            cp "${REPO}/outputs_eval/baseline_time/default/fuzzer_stats" \
               "${dst}/fuzzer_stats_eval.txt"
        log "  Saved baseline_time/run_${run_idx}"
    done
fi

# ── PHASE 3: Verify run count ─────────────────────────────────────────────────
sep
log "  PHASE 3: Verifying output files"
sep
for model in "${MODEL_LIST[@]}"; do
    n=0
    for run_idx in $(seq 1 "$EVAL_RUNS"); do
        [[ -f "${REPO}/plots/${model}/run_${run_idx}/rl_metrics_${model}_eval.csv" ]] && (( n++ )) || true
    done
    log "  ${model}: ${n}/${EVAL_RUNS} eval CSVs found"
done
if [[ $RUN_BASELINE -eq 1 ]]; then
    n=0
    for run_idx in $(seq 1 "$EVAL_RUNS"); do
        [[ -f "${REPO}/plots/baseline/run_${run_idx}/rl_metrics_baseline_eval.csv" ]] && (( n++ )) || true
    done
    log "  baseline: ${n}/${EVAL_RUNS} eval CSVs found"
    n=0
    for run_idx in $(seq 1 "$EVAL_RUNS"); do
        [[ -f "${REPO}/plots/baseline_time/run_${run_idx}/rl_metrics_baseline_time_eval.csv" ]] && (( n++ )) || true
    done
    log "  baseline_time: ${n}/${EVAL_RUNS} eval CSVs found"
fi

# ── PHASE 4a: Same-steps comparison ───────────────────────────────────────────
sep
log "  PHASE 4a: Same-steps comparison (RL 50K steps vs baseline 50K execs)"
sep
OUT_STEPS="${OUT_DIR}/same_steps"
mkdir -p "$OUT_STEPS"

COMP_ARGS_STEPS=(--phase eval --multi-run --compare-mode steps --out "$OUT_STEPS")
for model in "${MODEL_LIST[@]}"; do
    flag="--${model//_/-}"
    [[ -d "${REPO}/plots/${model}" ]] && COMP_ARGS_STEPS+=("$flag" "${REPO}/plots/${model}")
done
[[ $RUN_BASELINE -eq 1 && -d "${REPO}/plots/baseline" ]] && \
    COMP_ARGS_STEPS+=(--baseline "${REPO}/plots/baseline")
"$PYTHON" "${SCRIPTS}/compare_metrics.py" "${COMP_ARGS_STEPS[@]}" 2>&1 | tee -a "$LOG"

# ── PHASE 4b: Same-time comparison ────────────────────────────────────────────
if [[ $RUN_BASELINE -eq 1 && -d "${REPO}/plots/baseline_time" ]]; then
    sep
    log "  PHASE 4b: Same-time comparison (all models run for ~${T_RL}s)"
    sep
    OUT_TIME="${OUT_DIR}/same_time"
    mkdir -p "$OUT_TIME"

    COMP_ARGS_TIME=(--phase eval --multi-run --compare-mode time --out "$OUT_TIME")
    for model in "${MODEL_LIST[@]}"; do
        flag="--${model//_/-}"
        [[ -d "${REPO}/plots/${model}" ]] && COMP_ARGS_TIME+=("$flag" "${REPO}/plots/${model}")
    done
    COMP_ARGS_TIME+=(--baseline "${REPO}/plots/baseline_time")
    "$PYTHON" "${SCRIPTS}/compare_metrics.py" "${COMP_ARGS_TIME[@]}" 2>&1 | tee -a "$LOG"
fi

sep
log "  Done."
log "  Per-run CSVs      : plots/<model>/run_1/ .. run_${EVAL_RUNS}/"
log "  Training CSVs     : plots/<model>/rl_metrics_<model>_train.csv"
log "  Same-steps report : ${OUT_DIR}/same_steps/comparison_report.txt"
log "  Same-time report  : ${OUT_DIR}/same_time/comparison_report.txt"
log "  Log               : ${LOG}"
sep
