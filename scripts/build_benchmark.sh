#!/bin/bash
# scripts/build_benchmark.sh — Generic FuzzBench benchmark build framework
#
# Builds any supported FuzzBench benchmark target with AFL++ instrumentation.
# Per-benchmark build logic is defined in benchmarks/<name>/build_recipe.sh.
#
# Usage:
#   bash scripts/build_benchmark.sh jsoncpp          # build jsoncpp target
#   bash scripts/build_benchmark.sh freetype2         # build freetype2 target
#   bash scripts/build_benchmark.sh jsoncpp --clean   # force clean rebuild
#
# After this completes you should have:
#   bin/target                   — instrumented fuzzer binary
#   dictionaries/target.dict     — dictionary (if available)
#   inputs/                      — seed corpus
#   outputs/                     — empty, ready for AFL++
#   plots/                       — empty, for compare_metrics.py

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
BENCHMARK=""
CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=1; shift ;;
        --help|-h)
            echo "Usage: $0 <benchmark> [--clean]"
            echo ""
            echo "Available benchmarks:"
            for d in "$(dirname "$0")/../benchmarks"/*/; do
                [[ -f "${d}build_recipe.sh" ]] && echo "  $(basename "$d")"
            done
            exit 0
            ;;
        -*)  echo "[-] Unknown flag: $1"; exit 1 ;;
        *)
            if [[ -z "$BENCHMARK" ]]; then
                BENCHMARK="$1"
            else
                echo "[-] Unexpected argument: $1"; exit 1
            fi
            shift
            ;;
    esac
done

[[ -n "$BENCHMARK" ]] || { echo "[-] Usage: $0 <benchmark> [--clean]"; exit 1; }

# ── Paths ─────────────────────────────────────────────────────────────────────
RL_FUZZER="$(cd "$(dirname "$0")/.." && pwd)"
RECIPE="${RL_FUZZER}/benchmarks/${BENCHMARK}/build_recipe.sh"

[[ -f "$RECIPE" ]] || {
    echo "[-] No recipe found at $RECIPE"
    echo "    Available benchmarks:"
    for d in "${RL_FUZZER}/benchmarks"/*/; do
        [[ -f "${d}build_recipe.sh" ]] && echo "      $(basename "$d")"
    done
    exit 1
}

AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
FUZZBENCH="${FUZZBENCH:-$HOME/fuzzbench}"

# ── Local packages (built from source when system packages unavailable) ──────
LOCAL_PREFIX="${RL_FUZZER}/packages/local"
if [[ -d "$LOCAL_PREFIX" ]]; then
    export PKG_CONFIG_PATH="${LOCAL_PREFIX}/lib/pkgconfig:${LOCAL_PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
    export CFLAGS="${CFLAGS:-} -I${LOCAL_PREFIX}/include"
    export CXXFLAGS="${CXXFLAGS:-} -I${LOCAL_PREFIX}/include"
    export LDFLAGS="${LDFLAGS:-} -L${LOCAL_PREFIX}/lib -L${LOCAL_PREFIX}/lib64"
    export LD_LIBRARY_PATH="${LOCAL_PREFIX}/lib:${LOCAL_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
fi

# Ensure meson is available (installed in .venv via pip)
if [[ -x "${RL_FUZZER}/.venv/bin/meson" ]]; then
    export PATH="${RL_FUZZER}/.venv/bin:${PATH}"
fi

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

# ── System tuning (best-effort — AFL_SKIP_CPUFREQ and AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES handle these at runtime)
echo "[*] Applying system tuning..."
echo core | sudo tee /proc/sys/kernel/core_pattern > /dev/null 2>&1 || \
    echo "[!] Could not set core_pattern — AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 will be used at runtime."
(cd /sys/devices/system/cpu && \
    echo performance | sudo tee cpu*/cpufreq/scaling_governor > /dev/null 2>&1) || \
    echo "[!] Could not set CPU governor — AFL_SKIP_CPUFREQ=1 will be used at runtime."

# ── Source recipe ─────────────────────────────────────────────────────────────
# Variables available to recipes:
CC="$AFL_ROOT/afl-clang-fast"
CXX="$AFL_ROOT/afl-clang-fast++"
CFLAGS="-g -O2"
CXXFLAGS="-g -O2"
LDFLAGS=""
if [[ -d "$LOCAL_PREFIX" ]]; then
    CFLAGS="$CFLAGS -I${LOCAL_PREFIX}/include"
    CXXFLAGS="$CXXFLAGS -I${LOCAL_PREFIX}/include"
    LDFLAGS="-L${LOCAL_PREFIX}/lib -L${LOCAL_PREFIX}/lib64"
fi
AFLDRIVER="$AFL_ROOT/libAFLDriver.a"

echo "[*] Loading recipe: $RECIPE"
# shellcheck source=/dev/null
source "$RECIPE"

# Recipe must set FUZZBENCH_NAME and GIT_URL, and define BUILD_STEPS() and LINK_STEPS()
[[ -n "${FUZZBENCH_NAME:-}" ]] || { echo "[-] Recipe must set FUZZBENCH_NAME"; exit 1; }
[[ -n "${GIT_URL:-}" ]]        || { echo "[-] Recipe must set GIT_URL"; exit 1; }
type BUILD_STEPS &>/dev/null    || { echo "[-] Recipe must define BUILD_STEPS()"; exit 1; }
type LINK_STEPS  &>/dev/null    || { echo "[-] Recipe must define LINK_STEPS()"; exit 1; }

BENCHMARK_DIR="$FUZZBENCH/benchmarks/$FUZZBENCH_NAME"
[ -d "$BENCHMARK_DIR" ] || {
    echo "[-] FuzzBench benchmark dir not found: $BENCHMARK_DIR"
    exit 1
}

# Derive project name from GIT_URL
PROJECT="$(basename "$GIT_URL" .git)"
TARGET_DIR="$HOME/targets/${PROJECT}"
SRC_DIR="${TARGET_DIR}/src"

# ── Read FuzzBench-pinned commit ──────────────────────────────────────────────
COMMIT=$(grep '^commit:' "$BENCHMARK_DIR/benchmark.yaml" | awk '{print $2}')
echo "[*] FuzzBench pins $BENCHMARK at commit: $COMMIT"

# ── Clone source ─────────────────────────────────────────────────────────────
if [[ $CLEAN -eq 1 && -d "$SRC_DIR" ]]; then
    echo "[*] --clean: removing $SRC_DIR"
    rm -rf "$SRC_DIR"
fi

if [ ! -d "$SRC_DIR/.git" ]; then
    echo "[*] Cloning $PROJECT..."
    mkdir -p "$TARGET_DIR"
    git clone "$GIT_URL" "$SRC_DIR"
fi

echo "[*] Checking out pinned commit $COMMIT..."
cd "$SRC_DIR"
git fetch origin
git checkout "$COMMIT"

# ── Build ─────────────────────────────────────────────────────────────────────
echo "[*] Building $BENCHMARK with AFL++ instrumentation..."
BUILD_STEPS

# ── Link fuzzer binary ────────────────────────────────────────────────────────
echo "[*] Linking fuzzer binary..."
mkdir -p "$RL_FUZZER/bin"
LINK_STEPS

[ -x "$RL_FUZZER/bin/target" ] || { echo "[-] Failed to build bin/target"; exit 1; }
echo "[+] bin/target built."

# ── Install dictionary ────────────────────────────────────────────────────────
echo "[*] Installing dictionary..."
mkdir -p "$RL_FUZZER/dictionaries"
if [[ -n "${DICT_PATH:-}" && -f "$DICT_PATH" ]]; then
    cp "$DICT_PATH" "$RL_FUZZER/dictionaries/target.dict"
    echo "[+] Copied dictionary from recipe DICT_PATH ($(wc -l < "$RL_FUZZER/dictionaries/target.dict") entries)"
elif [[ -d "$BENCHMARK_DIR" ]]; then
    # Look for dictionary in FuzzBench benchmark dir
    DICT_FOUND=""
    for dpath in "$BENCHMARK_DIR"/*.dict "$SRC_DIR"/**/*.dict; do
        if [[ -f "$dpath" ]]; then
            cp "$dpath" "$RL_FUZZER/dictionaries/target.dict"
            DICT_FOUND="$dpath"
            echo "[+] Found dictionary: $dpath ($(wc -l < "$RL_FUZZER/dictionaries/target.dict") entries)"
            break
        fi
    done
    if [[ -z "$DICT_FOUND" ]]; then
        echo "[!] No dictionary found — writing minimal fallback."
        echo 'kw1="FUZZ"' > "$RL_FUZZER/dictionaries/target.dict"
    fi
fi

# ── Install seeds ─────────────────────────────────────────────────────────────
echo "[*] Setting up seed corpus..."
mkdir -p "$RL_FUZZER/inputs"
if [[ -n "${SEEDS_DIR:-}" && -d "$SEEDS_DIR" ]]; then
    # Recipe-provided seeds
    if [ -z "$(ls -A "$RL_FUZZER/inputs/" 2>/dev/null)" ]; then
        cp "$SEEDS_DIR"/* "$RL_FUZZER/inputs/" 2>/dev/null || true
        echo "[+] Copied seeds from recipe SEEDS_DIR"
    else
        echo "[+] inputs/ already populated — keeping existing seeds."
    fi
elif [[ -d "$FUZZBENCH/benchmarks/$FUZZBENCH_NAME/seeds" ]]; then
    # FuzzBench seeds
    if [ -z "$(ls -A "$RL_FUZZER/inputs/" 2>/dev/null)" ]; then
        cp "$FUZZBENCH/benchmarks/$FUZZBENCH_NAME/seeds"/* "$RL_FUZZER/inputs/" 2>/dev/null || true
        echo "[+] Copied FuzzBench seeds"
    else
        echo "[+] inputs/ already populated — keeping existing seeds."
    fi
else
    # Synthetic fallback seed
    if [ -z "$(ls -A "$RL_FUZZER/inputs/" 2>/dev/null)" ]; then
        echo "FUZZ" > "$RL_FUZZER/inputs/seed_default"
        echo "[+] Created synthetic fallback seed."
    else
        echo "[+] inputs/ already populated — keeping existing seeds."
    fi
fi

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "[*] Smoke testing binary..."
SMOKE_INPUT=$(mktemp /tmp/rl_smoke_XXXX)
echo "FUZZ" > "$SMOKE_INPUT"
if "$RL_FUZZER/bin/target" "$SMOKE_INPUT" > /dev/null 2>&1; then
    echo "[+] Smoke test passed."
else
    echo "[!] Smoke test exited non-zero — may be normal outside AFL++. Continuing."
fi
rm -f "$SMOKE_INPUT"

# ── Set up directory structure ────────────────────────────────────────────────
echo "[*] Setting up directories..."
mkdir -p "$RL_FUZZER"/{inputs,outputs,outputs_eval,plots}

# Clean old AFL++ state so train/eval always start fresh
rm -rf "$RL_FUZZER/outputs"/*
rm -rf "$RL_FUZZER/outputs_eval"/*
rm -f  /tmp/rl_shm

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "[+] ============================================"
echo "[+]  Build complete: $BENCHMARK"
echo "[+]  bin/target          ready"
echo "[+]  dictionaries/       ready"
echo "[+]  inputs/             ready ($(ls "$RL_FUZZER/inputs/" 2>/dev/null | wc -l) seeds)"
echo "[+]  outputs/            clean"
echo "[+]  outputs_eval/       clean"
echo "[+] ============================================"
echo ""
echo "     Next steps:"
echo "       Full experiment : bash scripts/run_experiment.sh"
echo "       Eval only      : bash scripts/run_experiment.sh --skip-train"
echo ""
