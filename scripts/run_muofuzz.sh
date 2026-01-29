#!/bin/bash

# [sysrel]

# --- 1. SETUP ---
cleanup() {
    echo ""
    echo "[!] Stopping processes..."
    if [ ! -z "$SERVER_PID" ]; then kill -9 "$SERVER_PID" 2>/dev/null; fi
    pkill -f "scripts/rl_server.py" 2>/dev/null
    pkill -f afl-fuzz 2>/dev/null
}
trap cleanup EXIT SIGINT SIGTERM

# --- 2. CONFIGURATION ---
# Attempt to auto-detect AFL++ location
if [ -z "$AFL_ROOT" ]; then
    # Common locations for AFL++
    if [ -d "$HOME/AFLplusplus" ]; then
        export AFL_ROOT="$HOME/AFLplusplus"
    elif [ -d "$HOME/Packages/AFLplusplus" ]; then
        export AFL_ROOT="$HOME/Packages/AFLplusplus"
    elif [ -d "/usr/local/bin/afl-fuzz" ]; then
        export AFL_ROOT="/usr/local/bin"
    else
        echo "[-] Error: AFL_ROOT is not set. Please export AFL_ROOT=/path/to/AFLplusplus"
        exit 1
    fi
fi

export PATH="$AFL_ROOT:$PATH"
export AFL_PATH="$AFL_ROOT"

# Force LLVM (Mac/Linux compatible)
if [ -d "/opt/homebrew/opt/llvm/bin" ]; then
    export LLVM_BIN="/opt/homebrew/opt/llvm/bin"
elif [ -d "/usr/local/opt/llvm/bin" ]; then
    export LLVM_BIN="/usr/local/opt/llvm/bin"
else
    export LLVM_BIN="/usr/bin"
fi

# --- 3. CLEAN & PREPARE ---
echo "[+] Cleaning previous build..."
mkdir -p bin inputs outputs dictionaries plots
rm -f bin/target bin/rl_mutator.dylib rl_metrics.csv constraints.json
rm -rf outputs/*

# Create initial seed
echo "AAAA" > inputs/seed.txt

# --- CRITICAL: Create Dictionary ---
# The RL Mutator checks if a dictionary is loaded. 
echo '"BAD!"' > dictionaries/target.dict
echo "[+] Created dictionary with 'BAD!' token."

# --- 4. RUN MOCK STATIC ANALYSIS ---
echo "[+] Running Mock Static Analysis..."
if [ -f "scripts/mock_analysis.py" ]; then
    python3 scripts/mock_analysis.py
else
    echo "[-] Error: scripts/mock_analysis.py not found! Run from project root."
    exit 1
fi

# --- 5. COMPILE TARGET ---
echo "[+] Compiling Target..."
"$AFL_ROOT/afl-clang-fast" src/target.c -o bin/target
if [ ! -f bin/target ]; then
    echo "[-] Target compilation failed."
    exit 1
fi

# --- 6. COMPILE MUTATOR ---
echo "[+] Compiling Custom Mutator..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_FLAGS="-shared"
else
    LIB_FLAGS="-dynamiclib -undefined dynamic_lookup"
fi

"$LLVM_BIN/clang" $LIB_FLAGS -O3 -fPIC \
    -I "$AFL_ROOT/include" \
    src/mutator.c -o bin/rl_mutator.dylib

if [ ! -f bin/rl_mutator.dylib ]; then
    echo "[-] Mutator compilation failed."
    exit 1
fi

# --- 7. START RL SERVER ---
echo "[+] Starting RL Brain..."
python3 scripts/rl_server.py &
SERVER_PID=$!
sleep 2 

# --- 8. START AFL++ ---
echo "[+] Starting AFL++ with Neuro-Symbolic Mutator..."

# Env Vars for Custom Mutator
export AFL_CUSTOM_MUTATOR_LIBRARY="bin/rl_mutator.dylib"
#export AFL_CUSTOM_MUTATOR_ONLY=1
#export AFL_NO_UI=1

"$AFL_ROOT/afl-fuzz" \
    -i inputs \
    -o outputs \
    -x dictionaries/target.dict \
    -- bin/target

wait $SERVER_PID
