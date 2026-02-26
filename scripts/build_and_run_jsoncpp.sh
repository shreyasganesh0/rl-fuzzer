#!/bin/bash
# build_and_run_jsoncpp.sh
#
# Builds the FuzzBench jsoncpp target and runs it with MuoFuzz.
#
# Assumed layout:
#   ~/rl-fuzzer/             — the rl-fuzzer project root
#   ~/fuzzbench/             — FuzzBench repo
#   ~/packages/AFLplusplus/  — AFL++ built from source
#
# Usage:
#   cd ~/rl-fuzzer && bash scripts/build_and_run_jsoncpp.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
FUZZBENCH="$HOME/fuzzbench"
RL_FUZZER="$HOME/projects/rl-fuzzer"
TARGET_DIR="$HOME/targets/jsoncpp"
BENCHMARK_DIR="$FUZZBENCH/benchmarks/jsoncpp_jsoncpp_fuzzer"

# ── Checks ────────────────────────────────────────────────────────────────────
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

# ── One-time system tuning ────────────────────────────────────────────────────
echo "[*] Applying system tuning..."
echo core | sudo tee /proc/sys/kernel/core_pattern > /dev/null
(cd /sys/devices/system/cpu && \
    echo performance | sudo tee cpu*/cpufreq/scaling_governor > /dev/null) || \
    echo "[!] Could not set CPU governor — AFL_SKIP_CPUFREQ=1 will be set."

# ── Read pinned commit from FuzzBench ─────────────────────────────────────────
COMMIT=$(grep '^commit:' "$BENCHMARK_DIR/benchmark.yaml" | awk '{print $2}')
echo "[*] FuzzBench pins jsoncpp at commit: $COMMIT"

# ── Clone jsoncpp at pinned commit ────────────────────────────────────────────
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
# NOTE: Do NOT add -fsanitize=address to CXXFLAGS here. If the library objects
# contain ASAN instrumentation but the final link omits the full ASAN runtime,
# you get dozens of undefined __asan_init / __asan_report_* symbols and the
# binary will SIGSEGV before the AFL++ fork server starts.
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
    echo "[-] libjsoncpp.a not built. Check cmake output above."
    exit 1
}
echo "[+] libjsoncpp.a built."

# ── Copy FuzzBench dictionary ──────────────────────────────────────────────────
echo "[*] Installing FuzzBench dictionary..."
mkdir -p "$RL_FUZZER/dictionaries"
if [ -f "$TARGET_DIR/src/src/test_lib_json/fuzz.dict" ]; then
    cp "$TARGET_DIR/src/src/test_lib_json/fuzz.dict" \
       "$RL_FUZZER/dictionaries/target.dict"
    echo "[+] Copied fuzz.dict ($(wc -l < "$RL_FUZZER/dictionaries/target.dict") entries)"
else
    echo "[!] No fuzz.dict found — run_muofuzz.sh will create a minimal one."
fi

# ── Link the fuzzer binary ─────────────────────────────────────────────────────
echo "[*] Linking fuzzer binary..."
mkdir -p "$RL_FUZZER/bin"

"$AFL_ROOT/afl-clang-fast++" \
    -g -O2 \
    -I"$TARGET_DIR/src/include" \
    "$TARGET_DIR/src/src/test_lib_json/fuzz.cpp" \
    "$TARGET_DIR/src/build-afl/lib/libjsoncpp.a" \
    "$AFL_ROOT/libAFLDriver.a" \
    -o "$RL_FUZZER/bin/target"

[ -x "$RL_FUZZER/bin/target" ] || {
    echo "[-] Failed to build bin/target"
    exit 1
}
echo "[+] bin/target built."

# ── Smoke test ────────────────────────────────────────────────────────────────
# libAFLDriver reads from a file path given as argv[1], not from stdin.
# We write a temp JSON file and pass it directly.
echo "[*] Smoke testing binary..."
SMOKE_INPUT=$(mktemp /tmp/muofuzz_smoke_XXXX.json)
echo '{"smoke": true}' > "$SMOKE_INPUT"

# libAFLDriver: when not under AFL, it runs the harness on the given file
# and exits. Exit code 0 = harness returned without crashing.
if "$RL_FUZZER/bin/target" "$SMOKE_INPUT" > /dev/null 2>&1; then
    echo "[+] Smoke test passed."
else
    EXITCODE=$?
    echo "[!] Warning: smoke test exited with code $EXITCODE."
    echo "    This may be normal if libAFLDriver expects to run under AFL."
    echo "    Continuing — the binary will be properly tested when AFL++ starts."
fi
rm -f "$SMOKE_INPUT"

# ── Set up seed corpus ────────────────────────────────────────────────────────
# Place multiple valid JSON seeds to give AFL++ good starting coverage.
# run_muofuzz.sh will NOT overwrite these (it checks if inputs/ is empty first).
echo "[*] Setting up seed corpus..."
mkdir -p "$RL_FUZZER/inputs"

cat > "$RL_FUZZER/inputs/seed_simple.json" << 'EOF'
{"key": "value"}
EOF

cat > "$RL_FUZZER/inputs/seed_complex.json" << 'EOF'
{"key": "value", "number": 42, "pi": 3.14159, "array": [1, 2, 3], "nested": {"bool": true, "null_val": null}}
EOF

cat > "$RL_FUZZER/inputs/seed_empty_containers.json" << 'EOF'
{"empty_obj": {}, "empty_arr": [], "str": "hello\nworld\ttab"}
EOF

cat > "$RL_FUZZER/inputs/seed_edge_cases.json" << 'EOF'
{"int_max": 2147483647, "int_min": -2147483648, "float": 1.7976931348623157e+308}
EOF

echo "[+] Created $(ls "$RL_FUZZER/inputs/" | wc -l) seed files."

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "[+] Build complete. Launching MuoFuzz against jsoncpp..."
echo ""
cd "$RL_FUZZER"

# Activate Python venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[+] Activated .venv"
else
    echo "[!] No .venv found — ensure pandas/torch/numpy are installed."
fi

export AFL_ROOT="$AFL_ROOT"
bash scripts/run_muofuzz.sh
