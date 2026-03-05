#!/bin/bash
# [sysrel]
# build_jsoncpp.sh — one-time build script
#
# Clones jsoncpp at the FuzzBench-pinned commit, builds it with AFL++
# instrumentation, links the fuzzer binary, installs the dictionary,
# and sets up the seed corpus.  Does NOT start any fuzzing.
#
# Run this once before any training or eval session, or whenever you
# want a clean rebuild.
#
# Usage:
#   cd ~/projects/rl-fuzzer
#   bash scripts/build_jsoncpp.sh
#
# After this completes you should have:
#   bin/target                   — instrumented fuzzer binary
#   bin/rl_mutator.so            — compiled custom mutator
#   dictionaries/target.dict     — FuzzBench/jsoncpp dictionary (54 entries)
#   inputs/                      — 4 seed JSON files
#   outputs/                     — empty, ready for AFL++
#   plots/                       — empty, for compare_metrics.py

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
FUZZBENCH="$HOME/fuzzbench"
RL_FUZZER="$HOME/projects/rl-fuzzer"
TARGET_DIR="$HOME/targets/jsoncpp"
BENCHMARK_DIR="$FUZZBENCH/benchmarks/jsoncpp_jsoncpp_fuzzer"

# ── Prerequisites ─────────────────────────────────────────────────────────────
echo "[*] Checking prerequisites..."

[ -x "$AFL_ROOT/afl-clang-fast++" ] || {
    echo "[-] afl-clang-fast++ not found at $AFL_ROOT"
    echo "    Build AFL++ first:"
    echo "      cd ~/packages/AFLplusplus && make -j\$(nproc)"
    echo "      LLVM_CONFIG=llvm-config-16 make -f GNUmakefile.llvm"
    exit 1
}

[ -f "$AFL_ROOT/libAFLDriver.a" ] || {
    echo "[-] libAFLDriver.a not found."
    echo "    cd $AFL_ROOT && make libAFLDriver.a"
    exit 1
}

[ -d "$FUZZBENCH" ] || {
    echo "[-] FuzzBench not found at $FUZZBENCH"
    echo "    git clone --depth=1 https://github.com/google/fuzzbench.git ~/fuzzbench"
    exit 1
}

[ -d "$RL_FUZZER" ] || {
    echo "[-] rl-fuzzer not found at $RL_FUZZER"
    exit 1
}

# ── Detect LLVM ───────────────────────────────────────────────────────────────
LLVM_BIN=""
for candidate in \
    "/opt/homebrew/opt/llvm/bin" \
    "/usr/local/opt/llvm/bin" \
    "/usr/lib/llvm-16/bin" \
    "/usr/lib/llvm-15/bin" \
    "/usr/lib/llvm-14/bin" \
    "/usr/bin"
do
    [ -d "$candidate" ] && { LLVM_BIN="$candidate"; break; }
done
echo "[+] LLVM_BIN=$LLVM_BIN"
echo "[+] AFL_ROOT=$AFL_ROOT"

# ── System tuning ─────────────────────────────────────────────────────────────
echo "[*] Applying system tuning..."
echo core | sudo tee /proc/sys/kernel/core_pattern > /dev/null
(cd /sys/devices/system/cpu && \
    echo performance | sudo tee cpu*/cpufreq/scaling_governor > /dev/null) || \
    echo "[!] Could not set CPU governor — AFL_SKIP_CPUFREQ=1 will be used at runtime."

# ── Read FuzzBench-pinned commit ──────────────────────────────────────────────
COMMIT=$(grep '^commit:' "$BENCHMARK_DIR/benchmark.yaml" | awk '{print $2}')
echo "[*] FuzzBench pins jsoncpp at commit: $COMMIT"

# ── Clone jsoncpp ─────────────────────────────────────────────────────────────
if [ ! -d "$TARGET_DIR/src/.git" ]; then
    echo "[*] Cloning jsoncpp..."
    mkdir -p "$TARGET_DIR"
    git clone https://github.com/open-source-parsers/jsoncpp.git "$TARGET_DIR/src"
fi

echo "[*] Checking out pinned commit $COMMIT..."
cd "$TARGET_DIR/src"
git fetch origin
git checkout "$COMMIT"

# ── Build jsoncpp static library ──────────────────────────────────────────────
# Do NOT add -fsanitize=address to CXXFLAGS — the library objects would contain
# ASAN instrumentation but the final link omits the ASAN runtime, causing
# SIGSEGV before the AFL++ fork server starts.
echo "[*] Building jsoncpp with AFL++ instrumentation..."
rm -rf "$TARGET_DIR/src/build-afl"
mkdir -p "$TARGET_DIR/src/build-afl"
cd "$TARGET_DIR/src/build-afl"

CC="$AFL_ROOT/afl-clang-fast" \
CXX="$AFL_ROOT/afl-clang-fast++" \
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DJSONCPP_WITH_TESTS=OFF \
    -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
    -DCMAKE_C_COMPILER="$AFL_ROOT/afl-clang-fast" \
    -DCMAKE_CXX_COMPILER="$AFL_ROOT/afl-clang-fast++" \
    -DCMAKE_CXX_FLAGS="-g -O2" \
    -DCMAKE_C_FLAGS="-g -O2"

make -j"$(nproc)" 2>&1 | grep -E "Building|Linking|[Ee]rror" || true

[ -f "$TARGET_DIR/src/build-afl/lib/libjsoncpp.a" ] || {
    echo "[-] libjsoncpp.a not built — check cmake output above."
    exit 1
}
echo "[+] libjsoncpp.a built."

# ── Install FuzzBench dictionary ──────────────────────────────────────────────
echo "[*] Installing dictionary..."
mkdir -p "$RL_FUZZER/dictionaries"
DICT_SRC="$TARGET_DIR/src/src/test_lib_json/fuzz.dict"
if [ -f "$DICT_SRC" ]; then
    cp "$DICT_SRC" "$RL_FUZZER/dictionaries/target.dict"
    echo "[+] Copied fuzz.dict ($(wc -l < "$RL_FUZZER/dictionaries/target.dict") entries)"
else
    echo "[!] fuzz.dict not found in jsoncpp source — writing minimal fallback."
    cat > "$RL_FUZZER/dictionaries/target.dict" << 'DICTEOF'
kw1="{}"
kw2="[]"
kw3="null"
kw4="true"
kw5="false"
kw6=":"
kw7=","
DICTEOF
fi

# ── Link fuzzer binary ────────────────────────────────────────────────────────
echo "[*] Linking fuzzer binary..."
mkdir -p "$RL_FUZZER/bin"

"$AFL_ROOT/afl-clang-fast++" \
    -g -O2 \
    -I"$TARGET_DIR/src/include" \
    "$TARGET_DIR/src/src/test_lib_json/fuzz.cpp" \
    "$TARGET_DIR/src/build-afl/lib/libjsoncpp.a" \
    "$AFL_ROOT/libAFLDriver.a" \
    -o "$RL_FUZZER/bin/target"

[ -x "$RL_FUZZER/bin/target" ] || { echo "[-] Failed to build bin/target"; exit 1; }
echo "[+] bin/target built."

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "[*] Smoke testing binary..."
SMOKE_INPUT=$(mktemp /tmp/rl_smoke_XXXX.json)
echo '{"smoke": true}' > "$SMOKE_INPUT"
if "$RL_FUZZER/bin/target" "$SMOKE_INPUT" > /dev/null 2>&1; then
    echo "[+] Smoke test passed."
else
    echo "[!] Smoke test exited non-zero — may be normal outside AFL++. Continuing."
fi
rm -f "$SMOKE_INPUT"

# ── Compile custom mutator ────────────────────────────────────────────────────
echo "[*] Compiling custom mutator..."
cd "$RL_FUZZER"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_FLAGS="-shared"
else
    LIB_FLAGS="-dynamiclib -undefined dynamic_lookup"
fi

CLANG_BIN="$LLVM_BIN/clang"
[ -x "$CLANG_BIN" ] || CLANG_BIN="clang"

# Note: model-specific mutators (mutator_m0_0.so, etc.) are compiled
# by each run_m*.sh script — no generic mutator build needed here.

# ── Set up directory structure ────────────────────────────────────────────────
echo "[*] Setting up directories..."
mkdir -p "$RL_FUZZER"/{inputs,outputs,outputs_eval,plots}

# Clean old AFL++ state so train/eval always start fresh
rm -rf "$RL_FUZZER/outputs"/*
rm -rf "$RL_FUZZER/outputs_eval"/*
rm -f  /tmp/rl_shm

# ── Seed corpus ───────────────────────────────────────────────────────────────
# FuzzBench's jsoncpp benchmark has no seed corpus (zero-seed benchmark).
# We provide 4 synthetic seeds which give the agent more interesting initial
# states without biasing coverage comparison heavily.
# Note: run scripts check if inputs/ is empty before writing — safe to re-run.
echo "[*] Setting up seed corpus..."
if [ -z "$(ls -A "$RL_FUZZER/inputs/" 2>/dev/null)" ]; then
    cat > "$RL_FUZZER/inputs/seed_simple.json"           << 'EOF'
{"key": "value"}
EOF
    cat > "$RL_FUZZER/inputs/seed_complex.json"          << 'EOF'
{"key": "value", "number": 42, "pi": 3.14159, "array": [1, 2, 3], "nested": {"bool": true, "null_val": null}}
EOF
    cat > "$RL_FUZZER/inputs/seed_empty_containers.json" << 'EOF'
{"empty_obj": {}, "empty_arr": [], "str": "hello\nworld\ttab"}
EOF
    cat > "$RL_FUZZER/inputs/seed_edge_cases.json"       << 'EOF'
{"int_max": 2147483647, "int_min": -2147483648, "float": 1.7976931348623157e+308}
EOF
    echo "[+] Created $(ls "$RL_FUZZER/inputs/" | wc -l) seed files."
else
    echo "[+] inputs/ already populated — keeping existing seeds."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "[+] ============================================"
echo "[+]  Build complete."
echo "[+]  bin/target          ready"
echo "[+]  bin/rl_mutator.so   ready"
echo "[+]  dictionaries/       ready ($(wc -l < "$RL_FUZZER/dictionaries/target.dict") dict entries)"
echo "[+]  inputs/             ready ($(ls "$RL_FUZZER/inputs/" | wc -l) seeds)"
echo "[+]  outputs/            clean"
echo "[+]  outputs_eval/       clean"
echo "[+] ============================================"
echo ""
echo "     Next steps:"
echo "       Full experiment : bash scripts/run_experiment.sh"
echo "       Eval only      : bash scripts/run_experiment.sh --skip-train"
echo ""
