#!/bin/bash
# [sysrel]

# ── 1. Cleanup handler ────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[!] Stopping processes..."
    [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null
    pkill -f "scripts/rl_server.py" 2>/dev/null
    pkill -f afl-fuzz 2>/dev/null
    rm -f /tmp/muofuzz_shm
}
trap cleanup EXIT SIGINT SIGTERM

# ── 2. AFL++ detection ────────────────────────────────────────────────────────
if [ -z "$AFL_ROOT" ]; then
    for candidate in "$HOME/AFLplusplus" "$HOME/Packages/AFLplusplus" "/usr/local/bin"; do
        [ -d "$candidate" ] && { export AFL_ROOT="$candidate"; break; }
    done
fi
[ -z "$AFL_ROOT" ] && { echo "[-] Set AFL_ROOT=/path/to/AFLplusplus"; exit 1; }
export PATH="$AFL_ROOT:$PATH"
export AFL_PATH="$AFL_ROOT"

# ── 3. LLVM detection ─────────────────────────────────────────────────────────
for llvm_candidate in "/opt/homebrew/opt/llvm/bin" "/usr/local/opt/llvm/bin" "/usr/bin"; do
    [ -d "$llvm_candidate" ] && { export LLVM_BIN="$llvm_candidate"; break; }
done

# ── 4. Prepare directories & seed ────────────────────────────────────────────
echo "[+] Cleaning previous build..."
mkdir -p bin inputs outputs dictionaries plots
rm -f bin/target bin/rl_mutator.dylib rl_metrics.csv /tmp/muofuzz_shm
rm -rf outputs/*

echo "AAAA" > inputs/seed.txt
echo '"BAD!"' > dictionaries/target.dict
echo "[+] Created seed and dictionary."

# ── 5. Copy edge probability CSVs to working directory ────────────────────────
# These files must be present alongside rl_server.py.
# Adjust source paths if your CSVs live elsewhere.
EDGE_CSV_DIR="${EDGE_CSV_DIR:-.}"   # default: same dir as this script
for csv_file in "edge_enable_prob.csv" "edge_disable_prob.csv"; do
    src="$EDGE_CSV_DIR/$csv_file"
    if [ ! -f "$csv_file" ] && [ -f "$src" ]; then
        cp "$src" .
    fi
    [ -f "$csv_file" ] || echo "[!] Warning: $csv_file not found — edge features will be zero."
done

# ── 6. Compile target ─────────────────────────────────────────────────────────
echo "[+] Compiling target..."
"$AFL_ROOT/afl-clang-fast" src/target.c -o bin/target
[ -f bin/target ] || { echo "[-] Target compilation failed."; exit 1; }

# ── 7. Compile custom mutator ─────────────────────────────────────────────────
echo "[+] Compiling custom mutator..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_FLAGS="-shared"
else
    LIB_FLAGS="-dynamiclib -undefined dynamic_lookup"
fi

"$LLVM_BIN/clang" $LIB_FLAGS -O3 -fPIC \
    -I "$AFL_ROOT/include" \
    src/mutator.c -o bin/rl_mutator.dylib
[ -f bin/rl_mutator.dylib ] || { echo "[-] Mutator compilation failed."; exit 1; }

# ── 8. Start RL brain ─────────────────────────────────────────────────────────
echo "[+] Starting RL brain (shared memory, no socket)..."
python3 scripts/rl_server.py &
SERVER_PID=$!
sleep 2   # let Python create and map the shm file first

# ── 9. Start AFL++ ────────────────────────────────────────────────────────────
echo "[+] Starting AFL++ with neuro-symbolic mutator..."

export AFL_CUSTOM_MUTATOR_LIBRARY="bin/rl_mutator.dylib"
export AFL_CUSTOM_MUTATOR_ONLY=1

"$AFL_ROOT/afl-fuzz" \
    -i inputs \
    -o outputs \
    -x dictionaries/target.dict \
    -- bin/target

wait $SERVER_PID
