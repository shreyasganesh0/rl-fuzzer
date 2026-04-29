#!/usr/bin/env bash
# scripts/experiment3.sh  —  Differential-informed RL fuzzing (M3_0 vs M1_0 vs baseline)
#
# 1. Train M3_0 on xml005_buggy for 500K steps (both DQN and bandit variants)
# 2. Evaluate on xml005_buggy (in-distribution) and xml017_buggy (transfer)
# 3. Compare against vanilla AFL++ baseline and M1_0
# 4. 5 eval runs per configuration for statistical significance
#
# Usage:
#   bash scripts/experiment3.sh [--train-steps N] [--eval-steps N] [--eval-runs N]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON=python3
AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
EXP_DIR="$REPO_ROOT/experiments/differential"
RESULTS_DIR="$EXP_DIR/results"

TRAIN_STEPS=500000
EVAL_STEPS=500000
EVAL_RUNS=5
TRAIN_TARGET="xml005_buggy"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
        --eval-steps)  EVAL_STEPS="$2";  shift 2 ;;
        --eval-runs)   EVAL_RUNS="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

TRAIN_BIN="$EXP_DIR/targets/${TRAIN_TARGET}/target"
SEEDS="$EXP_DIR/seeds"
DICT="$EXP_DIR/dictionaries/libxml2.dict"

[[ -f "$TRAIN_BIN" ]] || { echo "[-] Training target not found: $TRAIN_BIN"; exit 1; }
[[ -f "$DICT" ]]      || { echo "[-] Dictionary not found: $DICT"; exit 1; }

echo "============================================"
echo "  M* Differential Fuzzing Experiment"
echo "============================================"
echo "Train target:  $TRAIN_TARGET"
echo "Train steps:   $TRAIN_STEPS"
echo "Eval steps:    $EVAL_STEPS"
echo "Eval runs:     $EVAL_RUNS"
echo "Algorithms:    dqn, bandit"
echo "Eval targets:  xml005_buggy (in-dist), xml017_buggy (transfer)"
echo ""

# ── Helper: run a single train+eval cycle ─────────────────────────────────
run_model() {
    local MODEL_ID="$1"
    local ALGO="$2"
    local TARGET_BIN="$3"
    local RUN_DIR="$4"
    local PHASE="$5"  # "train" or "eval-only"

    local ALGO_FLAG=""
    [[ -n "$ALGO" ]] && ALGO_FLAG="--algorithm $ALGO"

    local EVAL_ONLY_FLAG=""
    [[ "$PHASE" == "eval-only" ]] && EVAL_ONLY_FLAG="--eval-only"

    bash "$REPO_ROOT/scripts/run_model.sh" \
        --model-id "$MODEL_ID" \
        --train-steps "$TRAIN_STEPS" \
        --eval-steps "$EVAL_STEPS" \
        --target "$TARGET_BIN" \
        --seeds "$SEEDS" \
        --dict "$DICT" \
        --exp-dir "$RUN_DIR" \
        --no-plateau \
        $ALGO_FLAG \
        $EVAL_ONLY_FLAG
}

# ── Phase 1: Train M* (DQN variant) on xml005_buggy ──────────────────────
echo "=== Training M* (DQN) on ${TRAIN_TARGET} ==="
MSTAR_DQN_DIR="$RESULTS_DIR/m3_0_dqn"
mkdir -p "$MSTAR_DQN_DIR"
run_model "m3_0" "dqn" "$TRAIN_BIN" "$MSTAR_DQN_DIR" "train"

# ── Phase 2: Train M* (Bandit variant) on xml005_buggy ────────────────────
echo ""
echo "=== Training M* (Bandit) on ${TRAIN_TARGET} ==="
MSTAR_BAN_DIR="$RESULTS_DIR/m3_0_bandit"
mkdir -p "$MSTAR_BAN_DIR"
run_model "m3_0" "bandit" "$TRAIN_BIN" "$MSTAR_BAN_DIR" "train"

# ── Phase 3: Train M1_0 (comparison baseline) on xml005_buggy ────────────
echo ""
echo "=== Training M1_0 (comparison) on ${TRAIN_TARGET} ==="
M10_DIR="$RESULTS_DIR/m1_0_compare"
mkdir -p "$M10_DIR"
run_model "m1_0" "" "$TRAIN_BIN" "$M10_DIR" "train"

# ── Phase 4: Multi-run evaluation on both targets ─────────────────────────
for EVAL_TARGET in xml005_buggy xml017_buggy; do
    EVAL_BIN="$EXP_DIR/targets/${EVAL_TARGET}/target"
    [[ -f "$EVAL_BIN" ]] || { echo "[-] Eval target not found: $EVAL_BIN"; continue; }

    echo ""
    echo "============================================"
    echo "  Evaluating on: $EVAL_TARGET"
    echo "============================================"

    for RUN in $(seq 1 "$EVAL_RUNS"); do
        echo ""
        echo "--- Eval run $RUN/$EVAL_RUNS on $EVAL_TARGET ---"

        for VARIANT in m3_0_dqn m3_0_bandit m1_0_compare; do
            SRC_DIR="$RESULTS_DIR/$VARIANT"
            EVAL_DIR="$RESULTS_DIR/eval_${EVAL_TARGET}/${VARIANT}/run_${RUN}"
            mkdir -p "$EVAL_DIR/bin" "$EVAL_DIR/plots/$(echo "$VARIANT" | sed 's/_compare//' | sed 's/_dqn//' | sed 's/_bandit//')"

            # Determine model-id and algorithm
            local_model_id="m3_0"
            local_algo=""
            if [[ "$VARIANT" == "m3_0_dqn" ]]; then
                local_algo="dqn"
            elif [[ "$VARIANT" == "m3_0_bandit" ]]; then
                local_algo="bandit"
            else
                local_model_id="m1_0"
            fi

            # Copy checkpoint from training
            SRC_PT="$SRC_DIR/bin/rl_${local_model_id}.pt"
            if [[ -f "$SRC_PT" ]]; then
                cp "$SRC_PT" "$EVAL_DIR/bin/rl_${local_model_id}.pt"
            else
                echo "  [-] No checkpoint for $VARIANT, skipping"
                continue
            fi

            echo "  Evaluating $VARIANT (run $RUN)..."
            run_model "$local_model_id" "$local_algo" "$EVAL_BIN" "$EVAL_DIR" "eval-only" || true
        done

        # Vanilla AFL++ baseline
        echo "  Evaluating baseline (run $RUN)..."
        BL_DIR="$RESULTS_DIR/eval_${EVAL_TARGET}/baseline/run_${RUN}"
        BL_AFL_OUT="$BL_DIR/afl_out"
        BL_CSV="$BL_DIR/baseline_eval.csv"
        mkdir -p "$BL_DIR"
        rm -rf "$BL_AFL_OUT"

        AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 \
        AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
            timeout $((EVAL_STEPS / 50 + 30)) "$AFL_ROOT/afl-fuzz" \
            -i "$SEEDS" -o "$BL_AFL_OUT" -x "$DICT" \
            -E "$EVAL_STEPS" \
            -- "$EVAL_BIN" @@ \
            > "$BL_DIR/afl.log" 2>&1 || true
    done
done

echo ""
echo "============================================"
echo "  M* Experiment Complete"
echo "============================================"
echo ""
echo "Results: $RESULTS_DIR/"
echo ""
echo "Structure:"
echo "  m3_0_dqn/     — M* trained with DQN"
echo "  m3_0_bandit/  — M* trained with contextual bandit"
echo "  m1_0_compare/   — M1_0 comparison model"
echo "  eval_xml005_buggy/{variant}/run_N/  — in-distribution eval"
echo "  eval_xml017_buggy/{variant}/run_N/  — transfer eval"
