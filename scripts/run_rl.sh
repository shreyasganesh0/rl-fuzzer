#!/bin/bash
# [sysrel]
# run_muofuzz.sh — compile mutator, start RL brain, launch AFL++
#
# Expects to be run from the rl-fuzzer project root, i.e.:
#   cd ~/rl-fuzzer && bash scripts/run_muofuzz.sh
#
# Pre-requisites (handled by build_and_run_jsoncpp.sh):
#   bin/target           — instrumented fuzzer binary
#   src/mutator.c        — custom mutator source
#   scripts/rl_server.py — RL brain

set -euo pipefail

# ── 1. Cleanup handler ────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[!] Stopping processes..."
    [ -n "${SERVER_PID:-}" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
    pkill -f "scripts/rl_server.py" 2>/dev/null || true
    pkill -f afl-fuzz 2>/dev/null || true
    rm -f /tmp/muofuzz_shm
}
trap cleanup EXIT SIGINT SIGTERM

# ── 2. AFL++ detection ────────────────────────────────────────────────────────
if [ -z "${AFL_ROOT:-}" ]; then
    for candidate in \
        "$HOME/packages/AFLplusplus" \
        "$HOME/AFLplusplus" \
        "$HOME/Packages/AFLplusplus" \
        "/usr/local/bin"
    do
        [ -d "$candidate" ] && { export AFL_ROOT="$candidate"; break; }
    done
fi
[ -z "${AFL_ROOT:-}" ] && { echo "[-] Set AFL_ROOT=/path/to/AFLplusplus"; exit 1; }
export PATH="$AFL_ROOT:$PATH"
export AFL_PATH="$AFL_ROOT"
echo "[+] AFL_ROOT=$AFL_ROOT"

# ── 3. LLVM detection ─────────────────────────────────────────────────────────
# Prefer the same clang AFL++ was built against.
# On Ubuntu/Debian: /usr/bin/clang or /usr/bin/clang-16 etc.
LLVM_BIN=""
for llvm_candidate in \
    "/opt/homebrew/opt/llvm/bin" \
    "/usr/local/opt/llvm/bin" \
    "/usr/lib/llvm-16/bin" \
    "/usr/lib/llvm-15/bin" \
    "/usr/lib/llvm-14/bin" \
    "/usr/bin"
do
    [ -d "$llvm_candidate" ] && { LLVM_BIN="$llvm_candidate"; break; }
done
echo "[+] LLVM_BIN=$LLVM_BIN"

# ── 4. Prepare directories ────────────────────────────────────────────────────
echo "[+] Preparing directories..."
mkdir -p bin inputs outputs dictionaries plots
rm -f bin/rl_mutator.so rl_metrics.csv /tmp/muofuzz_shm
rm -rf outputs/*

# Only create a fallback seed if inputs/ is empty (build_and_run_jsoncpp.sh
# may have already placed a real seed corpus here — don't overwrite it).
if [ -z "$(ls -A inputs/)" ]; then
    echo '{"key": "value", "number": 42, "array": [1, 2, 3]}' > inputs/seed.json
    echo "[+] Created default JSON seed."
fi

# Dictionary: AFL++ format — one token per line, quoted strings.
# build_and_run_jsoncpp.sh copies the FuzzBench fuzz.dict here already.
# Only write a minimal fallback if no dict exists.
if [ ! -s dictionaries/target.dict ]; then
    cat > dictionaries/target.dict << 'DICTEOF'
# Minimal JSON token dictionary (AFL++ format)
kw1="{}"
kw2="[]"
kw3="null"
kw4="true"
kw5="false"
kw6=":"
kw7=","
DICTEOF
    echo "[+] Created minimal JSON dictionary."
fi

# ── 5. Locate disable probability CSV ─────────────────────────────────────────
# New rl_server.py uses ONLY the disable CSV (enable CSV was dropped).
# Canonical filename expected by rl_server.py: edgeDisablingMutator.csv
# If yours lives elsewhere, set MUOFUZZ_DISABLE_CSV before calling this script.
if [ -z "${MUOFUZZ_DISABLE_CSV:-}" ]; then
    # Search in common places
    for csv_candidate in \
        "./edgeDisablingMutator.csv" \
        "./edge_disable_prob.csv" \
        "${EDGE_CSV_DIR:-./data}/edgeDisablingMutator.csv"
    do
        if [ -f "$csv_candidate" ]; then
            export MUOFUZZ_DISABLE_CSV="$csv_candidate"
            break
        fi
    done
fi

if [ -n "${MUOFUZZ_DISABLE_CSV:-}" ] && [ -f "$MUOFUZZ_DISABLE_CSV" ]; then
    echo "[+] Using disable CSV: $MUOFUZZ_DISABLE_CSV"
else
    echo "[!] Warning: disable CSV not found — edge disable features will be zero."
    echo "    Place edgeDisablingMutator.csv in the project root, or:"
    echo "    export MUOFUZZ_DISABLE_CSV=/path/to/edgeDisablingMutator.csv"
    unset MUOFUZZ_DISABLE_CSV
fi

# ── 6. Verify target binary ───────────────────────────────────────────────────
[ -x bin/target ] || {
    echo "[-] bin/target not found or not executable."
    echo "    Run build_and_run_jsoncpp.sh first."
    exit 1
}
echo "[+] bin/target OK."

# ── 7. Compile custom mutator ─────────────────────────────────────────────────
echo "[+] Compiling custom mutator..."

[ -f src/mutator.c ] || { echo "[-] src/mutator.c not found."; exit 1; }

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_FLAGS="-shared"
else
    LIB_FLAGS="-dynamiclib -undefined dynamic_lookup"
fi

# Use clang++ for C++ compatibility of afl-fuzz.h (it pulls in some C++ headers)
CLANG_BIN="$LLVM_BIN/clang"
[ -x "$CLANG_BIN" ] || CLANG_BIN="clang"   # fallback to PATH
echo "[+] Compiler: $CLANG_BIN"

"$CLANG_BIN" $LIB_FLAGS -O2 -fPIC \
    -I "$AFL_ROOT/include" \
    src/mutator.c \
    -o bin/rl_mutator.so 2>&1

[ -f bin/rl_mutator.so ] || { echo "[-] Mutator compilation failed."; exit 1; }
echo "[+] bin/rl_mutator.so built."

# ── 8. Start RL brain ─────────────────────────────────────────────────────────
echo "[+] Starting RL brain..."

# Pass the disable CSV path and model path explicitly.
# rl_server.py also respects MUOFUZZ_DISABLE_CSV and MUOFUZZ_MODEL_PATH env vars,
# but explicit flags are clearer.
RL_SERVER_ARGS=""
[ -n "${MUOFUZZ_DISABLE_CSV:-}" ] && \
    RL_SERVER_ARGS="$RL_SERVER_ARGS --disable-csv $MUOFUZZ_DISABLE_CSV"
[ -n "${MUOFUZZ_MODEL_PATH:-}" ] && \
    RL_SERVER_ARGS="$RL_SERVER_ARGS --model $MUOFUZZ_MODEL_PATH"

python3 scripts/rl_server.py $RL_SERVER_ARGS &
SERVER_PID=$!

# Wait for rl_server to create /tmp/muofuzz_shm before AFL++ starts.
echo "[+] Waiting for RL brain to initialise..."
for i in $(seq 1 20); do
    [ -f /tmp/muofuzz_shm ] && { echo "[+] SHM file ready."; break; }
    sleep 0.5
done
[ -f /tmp/muofuzz_shm ] || {
    echo "[-] rl_server.py did not create /tmp/muofuzz_shm after 10 s."
    echo "    Check scripts/rl_server.py output above for errors."
    exit 1
}

# ── 9. Start AFL++ ────────────────────────────────────────────────────────────
echo "[+] Starting AFL++ with MuoFuzz mutator (ACTION_SIZE=47)..."

export AFL_CUSTOM_MUTATOR_LIBRARY="bin/rl_mutator.so"
export AFL_CUSTOM_MUTATOR_ONLY=1
export AFL_SKIP_CPUFREQ=1          # remove if you ran sysctl tuning

"$AFL_ROOT/afl-fuzz" \
    -i inputs \
    -o outputs \
    -x dictionaries/target.dict \
    -- bin/target -

wait "$SERVER_PID"
