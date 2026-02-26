#!/bin/bash
# [sysrel]
#
# Starts the RL brain (rl_server.py --mode train) and AFL++ together.
# The RL brain runs until it detects a coverage plateau, then saves the
# model checkpoint and exits cleanly.  The EXIT trap here kills AFL++
# automatically when the server exits, so you don't need to do anything.
#
# Prerequisites (run build_jsoncpp.sh first):
#   bin/target           — instrumented fuzzer binary
#   bin/rl_mutator.so    — compiled custom mutator
#   dictionaries/        — dictionary
#   inputs/              — seed corpus
#
# Usage:
#   cd ~/projects/rl-fuzzer
#   bash scripts/train_rl.sh
#
# Outputs:
#   rl_dqn.pt       — trained model checkpoint (saved on plateau/exit)
#   rl_metrics.csv       — per-100-step training metrics
#   outputs/             — AFL++ findings during training

set -euo pipefail

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[!] Stopping training processes..."
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

# ── Verify build outputs exist ────────────────────────────────────────────────
[ -x bin/target ]          || { echo "[-] bin/target not found. Run build_jsoncpp.sh first."; exit 1; }
[ -f bin/rl_mutator.so ]   || { echo "[-] bin/rl_mutator.so not found. Run build_jsoncpp.sh first."; exit 1; }
[ -d inputs ] && [ -n "$(ls -A inputs/)" ] || { echo "[-] inputs/ is empty. Run build_jsoncpp.sh first."; exit 1; }
echo "[+] Build outputs OK."

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

# ── Prepare training output directories ──────────────────────────────────────
# Training always writes to outputs/.  Clear it for a fresh run so AFL++
# doesn't try to resume a previous session.
echo "[+] Clearing outputs/ for fresh training run..."
mkdir -p outputs plots
rm -rf outputs/*
rm -f  /tmp/rl_shm rl_metrics.csv

# ── Start RL brain in train mode ──────────────────────────────────────────────
echo "[+] Starting RL brain (train mode)..."
MODEL_ARG=""
[ -n "${RL_MODEL_PATH:-}" ] && MODEL_ARG="--model $RL_MODEL_PATH"

python3 scripts/rl_server.py \
    --mode train \
    $CSV_ARG \
    $MODEL_ARG &
SERVER_PID=$!

# Wait for SHM file — means the server is ready
echo "[+] Waiting for RL brain to initialise..."
for i in $(seq 1 20); do
    [ -f /tmp/rl_shm ] && { echo "[+] SHM ready."; break; }
    sleep 0.5
done
[ -f /tmp/rl_shm ] || {
    echo "[-] rl_server.py did not create /tmp/rl_shm after 10 s."
    echo "    Check rl_server.py output above for Python errors."
    exit 1
}

# ── Start AFL++ ───────────────────────────────────────────────────────────────
echo "[+] Starting AFL++ (training run)..."
export AFL_CUSTOM_MUTATOR_LIBRARY="bin/rl_mutator.so"
export AFL_CUSTOM_MUTATOR_ONLY=1
export AFL_SKIP_CPUFREQ=1

"$AFL_ROOT/afl-fuzz" \
    -i inputs \
    -o outputs \
    -x dictionaries/target.dict \
    -- bin/target - &
AFL_PID=$!

# ── Wait for the RL server to finish (plateau or Ctrl-C) ──────────────────────
# When the server detects a plateau it exits cleanly.  The EXIT trap above then
# kills AFL++.  If you Ctrl-C, the same trap fires.
wait "$SERVER_PID"
EXIT_CODE=$?

echo ""
echo "[+] RL server exited (code $EXIT_CODE)."
echo "[+] Model saved to: ${RL_MODEL_PATH:-rl_dqn.pt}"
echo "[+] Training metrics: rl_metrics.csv"
echo "[+] AFL++ corpus: outputs/"
echo ""
echo "     To evaluate the trained model:"
echo "       bash scripts/eval_rl.sh"
