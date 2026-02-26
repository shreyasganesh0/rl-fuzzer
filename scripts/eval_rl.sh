#!/bin/bash
# [sysrel]
#
# Loads the trained checkpoint (rl_dqn.pt) and runs with ε=0 (pure
# greedy — no random exploration, no weight updates).  AFL++ findings go
# to outputs_eval/ so training results are not overwritten.
#
# The RL server runs for --eval-steps RL steps (default 20 000) then
# exits.  The EXIT trap kills AFL++ automatically.
#
# Prerequisites:
#   bin/target, bin/rl_mutator.so, dictionaries/, inputs/ — from build_jsoncpp.sh
#   rl_dqn.pt                                        — from train_rl.sh
#
# Usage:
#   cd ~/projects/rl-fuzzer
#   bash scripts/eval_rl.sh [--eval-steps N]
#
# Outputs:
#   rl_metrics_eval.csv   — per-100-step eval metrics (coverage, action, reward)
#   outputs_eval/         — AFL++ findings during eval
#
# Comparing to a vanilla AFL++ baseline:
#   Run AFL++ without the custom mutator for the same wall-clock time:
#     ~/packages/AFLplusplus/afl-fuzz \
#         -i inputs -o outputs_baseline \
#         -x dictionaries/target.dict \
#         -- bin/target -
#   Then compare outputs_eval/default/fuzzer_stats edges_found
#   against outputs_baseline/default/fuzzer_stats edges_found.

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
EVAL_STEPS=20000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-steps) EVAL_STEPS="$2"; shift 2 ;;
        *) echo "[-] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[!] Stopping eval processes..."
    [ -n "${SERVER_PID:-}" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
    pkill -f "rl_server.py" 2>/dev/null || true
    pkill -f "afl-fuzz"     2>/dev/null || true
    rm -f /tmp/rl_shm
    echo "[+] Cleanup done."
}
trap cleanup EXIT SIGINT SIGTERM

# ── AFL++ detection ───────────────────────────────────────────────────────────
if [ -z "${AFL_ROOT:-}" ]; then
    for candidate in \
        "$HOME/packages/AFLplusplus" \
        "$HOME/AFLplusplus" \
        "/usr/local/bin"
    do
        [ -d "$candidate" ] && { export AFL_ROOT="$candidate"; break; }
    done
fi
[ -z "${AFL_ROOT:-}" ] && { echo "[-] Set AFL_ROOT=/path/to/AFLplusplus"; exit 1; }
export PATH="$AFL_ROOT:$PATH"
export AFL_PATH="$AFL_ROOT"
echo "[+] AFL_ROOT=$AFL_ROOT"

# ── Verify prerequisites ──────────────────────────────────────────────────────
[ -x bin/target ]          || { echo "[-] bin/target not found. Run build_jsoncpp.sh."; exit 1; }
[ -f bin/rl_mutator.so ]   || { echo "[-] bin/rl_mutator.so not found. Run build_jsoncpp.sh."; exit 1; }
[ -d inputs ] && [ -n "$(ls -A inputs/)" ] || { echo "[-] inputs/ empty. Run build_jsoncpp.sh."; exit 1; }

MODEL_PATH="${RL_MODEL_PATH:-rl_dqn.pt}"
[ -f "$MODEL_PATH" ] || {
    echo "[-] Checkpoint not found at $MODEL_PATH"
    echo "    Run train_rl.sh first to produce a trained model."
    exit 1
}
echo "[+] Build outputs OK."
echo "[+] Model checkpoint: $MODEL_PATH"
echo "[+] Eval steps: $EVAL_STEPS"

# ── Locate disable CSV ────────────────────────────────────────────────────────
if [ -z "${RL_DISABLE_CSV:-}" ]; then
    for candidate in \
        "./edgeDisablingMutator.csv" \
        "./edge_disable_prob.csv" \
        "${EDGE_CSV_DIR:-./data}/edgeDisablingMutator.csv"
    do
        [ -f "$candidate" ] && { export RL_DISABLE_CSV="$candidate"; break; }
    done
fi

if [ -n "${RL_DISABLE_CSV:-}" ] && [ -f "$RL_DISABLE_CSV" ]; then
    echo "[+] Disable CSV: $RL_DISABLE_CSV"
    CSV_ARG="--disable-csv $RL_DISABLE_CSV"
else
    echo "[!] edgeDisablingMutator.csv not found — disable features will be zero."
    CSV_ARG=""
fi

# ── Python venv ───────────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[+] Activated .venv"
fi

# ── Determine input corpus for eval ──────────────────────────────────────────
# Prefer the corpus built during training (outputs/default/queue) so that
# eval continues from a warm start rather than cold seeds.
# Fall back to inputs/ if no training corpus exists yet.
TRAIN_QUEUE="outputs/default/queue"
if [ -d "$TRAIN_QUEUE" ] && [ -n "$(ls -A "$TRAIN_QUEUE" 2>/dev/null)" ]; then
    EVAL_INPUT="$TRAIN_QUEUE"
    echo "[+] Using training corpus as eval input: $TRAIN_QUEUE"
    echo "[+] Corpus size: $(ls "$TRAIN_QUEUE" | wc -l) items"
else
    EVAL_INPUT="inputs"
    echo "[!] No training corpus found at $TRAIN_QUEUE — using raw seeds."
    echo "    Run train_rl.sh first for a meaningful eval."
fi

# ── Prepare eval output directory ─────────────────────────────────────────────
echo "[+] Clearing outputs_eval/ for fresh eval run..."
mkdir -p outputs_eval plots
rm -rf outputs_eval/*
rm -f  /tmp/rl_shm rl_metrics_eval.csv

# ── Start RL brain in eval mode ───────────────────────────────────────────────
echo "[+] Starting RL brain (eval mode, ε=0)..."

python3 scripts/rl_server.py \
    --mode eval \
    --model "$MODEL_PATH" \
    --eval-steps "$EVAL_STEPS" \
    $CSV_ARG &
SERVER_PID=$!

# Wait for SHM
echo "[+] Waiting for RL brain to initialise..."
for i in $(seq 1 20); do
    [ -f /tmp/rl_shm ] && { echo "[+] SHM ready."; break; }
    sleep 0.5
done
[ -f /tmp/rl_shm ] || {
    echo "[-] rl_server.py did not create /tmp/rl_shm after 10 s."
    exit 1
}

# ── Start AFL++ ───────────────────────────────────────────────────────────────
echo "[+] Starting AFL++ (eval run → outputs_eval/)..."
export AFL_CUSTOM_MUTATOR_LIBRARY="bin/rl_mutator.so"
export AFL_CUSTOM_MUTATOR_ONLY=1
export AFL_SKIP_CPUFREQ=1

"$AFL_ROOT/afl-fuzz" \
    -i "$EVAL_INPUT" \
    -o outputs_eval \
    -x dictionaries/target.dict \
    -- bin/target - &
AFL_PID=$!

# ── Wait for eval to finish ───────────────────────────────────────────────────
wait "$SERVER_PID"
EXIT_CODE=$?

echo ""
echo "[+] Eval complete (exit code $EXIT_CODE)."
echo "[+] Eval metrics : rl_metrics_eval.csv"
echo "[+] AFL++ corpus : outputs_eval/"
echo ""

# Print final coverage from fuzzer_stats if available
STATS="outputs_eval/default/fuzzer_stats"
if [ -f "$STATS" ]; then
    EDGES=$(grep "^edges_found" "$STATS" | awk '{print $3}')
    EXECS=$(grep "^execs_done"  "$STATS" | awk '{print $3}')
    echo "[+] AFL++ edges_found : $EDGES"
    echo "[+] AFL++ execs_done  : $EXECS"
fi

echo ""
echo "     To compare against vanilla AFL++:"
echo "       \$AFL_ROOT/afl-fuzz -i inputs -o outputs_baseline \\"
echo "           -x dictionaries/target.dict -- bin/target -"
echo "       # run for same wall-clock time, then compare:"
echo "       # grep edges_found outputs_eval/default/fuzzer_stats"
echo "       # grep edges_found outputs_baseline/default/fuzzer_stats"
