#!/bin/bash
# build_and_run_jsoncpp.sh
#
# Builds the FuzzBench jsoncpp target and runs it with rl-fuzzer.
#
# Assumed layout:
#   ~/rl-fuzzer/      — the rl-fuzzer project
#   ~/fuzzbench/      — FuzzBench repo (git clone --depth=1 https://github.com/google/fuzzbench)
#   ~/packages/AFLplusplus/  — AFL++ built from source
#
# Usage:
#   cd ~ && bash ~/rl-fuzzer/scripts/build_and_run_jsoncpp.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
AFL_ROOT="$HOME/packages/AFLplusplus"
FUZZBENCH="$HOME/fuzzbench"
RL_FUZZER="$HOME/rl-fuzzer"
TARGET_DIR="$HOME/targets/jsoncpp"
BENCHMARK_DIR="$FUZZBENCH/benchmarks/jsoncpp_jsoncpp_fuzzer"

# ── Checks ────────────────────────────────────────────────────────────────────
echo "[*] Checking prerequisites..."

[ -x "$AFL_ROOT/afl-clang-fast++" ] || {
    echo "[-] afl-clang-fast++ not found at $AFL_ROOT"
    echo "    Build AFL++ first: cd ~/packages/AFLplusplus && make -j\$(nproc)"
    echo "    Then: LLVM_CONFIG=llvm-config-16 \\"
    echo "          CPPFLAGS=\"-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11\" \\"
    echo "          LDFLAGS=\"-L/usr/lib/gcc/x86_64-linux-gnu/11\" \\"
    echo "          make -f GNUmakefile.llvm"
    exit 1
}

[ -f "$AFL_ROOT/libAFLDriver.a" ] || {
    echo "[-] libAFLDriver.a not found. Build it:"
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

# ── One-time system tuning ────────────────────────────────────────────────────
echo "[*] Applying system tuning..."
echo core | sudo tee /proc/sys/kernel/core_pattern > /dev/null
(cd /sys/devices/system/cpu && echo performance | sudo tee cpu*/cpufreq/scaling_governor > /dev/null) || \
    echo "[!] Could not set CPU governor — set AFL_SKIP_CPUFREQ=1 if needed"

# ── Read pinned commit from FuzzBench ────────────────────────────────────────
COMMIT=$(grep '^commit:' "$BENCHMARK_DIR/benchmark.yaml" | awk '{print $2}')
echo "[*] FuzzBench pins jsoncpp at commit: $COMMIT"

# ── Clone jsoncpp at pinned commit ───────────────────────────────────────────
if [ ! -d "$TARGET_DIR/src/.git" ]; then
    echo "[*] Cloning jsoncpp..."
    mkdir -p "$TARGET_DIR"
    git clone https://github.com/open-source-parsers/jsoncpp.git "$TARGET_DIR/src"
fi

echo "[*] Checking out pinned commit $COMMIT..."
cd "$TARGET_DIR/src"
git checkout "$COMMIT"

# ── Build jsoncpp library (NO -fsanitize=address) ────────────────────────────
# IMPORTANT: Do not add -fsanitize=address here. If the library objects contain
# ASAN instrumentation but the final link does not include the full ASAN runtime,
# you will get dozens of undefined references to __asan_init, __asan_report_*,
# etc., and the binary will SIGSEGV before the AFL++ fork server starts.
echo "[*] Building jsoncpp static library with AFL++ instrumentation..."
rm -rf "$TARGET_DIR/src/build-afl"
mkdir -p "$TARGET_DIR/src/build-afl"
cd "$TARGET_DIR/src/build-afl"

CC="$AFL_ROOT/afl-clang-fast" \
CXX="$AFL_ROOT/afl-clang-fast++" \
CXXFLAGS="-g -O2" \
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DJSONCPP_WITH_TESTS=OFF \
    -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
    -DCMAKE_CXX_COMPILER="$AFL_ROOT/afl-clang-fast++" \
    -DCMAKE_C_COMPILER="$AFL_ROOT/afl-clang-fast" \
    -DCMAKE_CXX_FLAGS="-g -O2"

make -j"$(nproc)" 2>&1 | grep -E "Building|Linking|Error|error" || true

[ -f "$TARGET_DIR/src/build-afl/lib/libjsoncpp.a" ] || {
    echo "[-] libjsoncpp.a not built. Check cmake output above."
    exit 1
}
echo "[+] libjsoncpp.a built successfully."

# ── Copy FuzzBench dictionary ─────────────────────────────────────────────────
echo "[*] Installing FuzzBench dictionary..."
mkdir -p "$RL_FUZZER/dictionaries"
cp "$TARGET_DIR/src/src/test_lib_json/fuzz.dict" "$RL_FUZZER/dictionaries/target.dict" 2>/dev/null || \
    echo "[!] No fuzz.dict found — continuing without dictionary"

# ── Link the fuzzer binary ────────────────────────────────────────────────────
# The harness (fuzz.cpp) is already in the jsoncpp repo — FuzzBench's build.sh
# points to it. We just substitute:
#   $LIB_FUZZING_ENGINE → libAFLDriver.a
#   $CXX               → afl-clang-fast++
echo "[*] Linking fuzzer binary..."
mkdir -p "$RL_FUZZER/bin"

"$AFL_ROOT/afl-clang-fast++" \
    -g -O2 \
    -I"$TARGET_DIR/src/include" \
    "$TARGET_DIR/src/src/test_lib_json/fuzz.cpp" \
    "$TARGET_DIR/src/build-afl/lib/libjsoncpp.a" \
    "$AFL_ROOT/libAFLDriver.a" \
    -o "$RL_FUZZER/bin/target" 2>&1 | grep -v "^afl-cc\|^SanitizerCoverage\|^\[+\] Instrumented"

[ -x "$RL_FUZZER/bin/target" ] || {
    echo "[-] Failed to build bin/target"
    exit 1
}
echo "[+] bin/target built successfully."

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "[*] Smoke testing binary..."
RESULT=$(echo '{"key": "value"}' | "$RL_FUZZER/bin/target" - 2>&1)
echo "$RESULT" | grep -q "Execution successful" || {
    echo "[-] Smoke test failed. Output:"
    echo "$RESULT"
    exit 1
}
echo "[+] Smoke test passed."

# ── Set up seed corpus ────────────────────────────────────────────────────────
echo "[*] Setting up seed corpus..."
mkdir -p "$RL_FUZZER/inputs"
cat > "$RL_FUZZER/inputs/seed.json" << 'EOF'
{"key": "value", "number": 42, "array": [1, 2, 3], "nested": {"bool": true}}
EOF

# ── Run the fuzzer ────────────────────────────────────────────────────────────
echo ""
echo "[+] All done! Launching rl-fuzzer against jsoncpp..."
echo ""
cd "$RL_FUZZER"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "[!] No .venv found — make sure pandas/torch/numpy are installed globally"
fi

export AFL_ROOT="$AFL_ROOT"
bash scripts/run_muofuzz.sh
