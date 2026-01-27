#!/bin/bash

# --- 1. SETUP & TRAP ---
cleanup() {
    echo ""
    echo "[!] Stopping processes..."
    if [ ! -z "$SERVER_PID" ]; then kill -9 "$SERVER_PID" 2>/dev/null; fi
    pkill -f rl_server.py 2>/dev/null
    pkill -f afl-fuzz 2>/dev/null
}
trap cleanup EXIT SIGINT SIGTERM

export AFL_ROOT="$HOME/Packages/AFLplusplus"
export AFL_PATH="$AFL_ROOT"
export PATH="$AFL_ROOT:$PATH"

# Force Homebrew LLVM
if [ -d "/opt/homebrew/opt/llvm/bin" ]; then
    export LLVM_BIN="/opt/homebrew/opt/llvm/bin"
elif [ -d "/usr/local/opt/llvm/bin" ]; then
    export LLVM_BIN="/usr/local/opt/llvm/bin"
else
    echo "[-] Error: Could not find Homebrew LLVM."
    exit 1
fi

export AFL_CC="$LLVM_BIN/clang"
export AFL_CXX="$LLVM_BIN/clang++"

if [ ! -f "$AFL_ROOT/afl-clang-fast" ]; then
    echo "[-] Error: afl-clang-fast not found in $AFL_ROOT."
    exit 1
fi

# --- PRE-CLEAN ---
mkdir -p bin inputs outputs
rm -f bin/target bin/rl_mutator.dylib rl_metrics.csv rl_training_plot.png
pkill -f rl_server.py

# --- 2. COMPILE TARGET -> bin/target ---
echo "[+] Compiling Target (bin/target)..."
"$AFL_ROOT/afl-clang-fast" src/target.c -o bin/target

if [ ! -f bin/target ]; then
    echo "[-] Target compilation failed."
    exit 1
fi

# --- 3. COMPILE MUTATOR -> bin/rl_mutator.dylib ---
echo "[+] Compiling Mutator (bin/rl_mutator.dylib)..."
"$LLVM_BIN/clang" -dynamiclib -O3 -fPIC \
    -I "$AFL_ROOT/include" \
    src/mutator.c -o bin/rl_mutator.dylib

if [ ! -f bin/rl_mutator.dylib ]; then
    echo "[-] Mutator compilation failed."
    exit 1
fi

# --- 4. START RL SERVER ---
echo "[+] Starting RL Brain..."
python3 scripts/rl_server.py &
SERVER_PID=$!
sleep 2

# --- 5. RUN FUZZER ---
echo "[+] Starting AFL++..."
echo "AAAA" > inputs/seed.txt

export AFL_CUSTOM_MUTATOR_LIBRARY=$(pwd)/bin/rl_mutator.dylib

"$AFL_ROOT/afl-fuzz" -i inputs -o outputs -V 60 -- ./bin/target

# --- 6. SAVE PLOTS ---
echo ""
echo "[+] Fuzzing finished. Generating plots..."
python3 scripts/plot_metrics.py

echo "[+] Done. Check 'rl_training_plot.png' and 'outputs/'."
