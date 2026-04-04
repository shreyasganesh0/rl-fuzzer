#!/usr/bin/env bash
# build_libxml2_targets.sh — Build 4 libxml2 targets for differential fuzzing
#
# Builds buggy/fixed pairs for 2 CVEs:
#   xml005_buggy: CVE-2017-5130, libxml2 v2.9.4
#   xml005_fixed: CVE-2017-5130, libxml2 v2.9.5
#   xml017_buggy: CVE-2016-1762, libxml2 v2.9.3
#   xml017_fixed: CVE-2016-1762, libxml2 v2.9.4
#
# Note: v2.9.4 is both xml005_buggy AND xml017_fixed (same source, different bugs).
#
# Usage: bash experiments/differential/build/build_libxml2_targets.sh [--clean]

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"

AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
FUZZBENCH="${FUZZBENCH:-$HOME/fuzzbench}"
BENCHMARK_DIR="$FUZZBENCH/benchmarks/libxml2_xml"
TARGET_BASE="$HOME/targets/libxml2_differential"

TARGETS_DIR="$EXP_DIR/targets"
SEEDS_DIR="$EXP_DIR/seeds"
DICT_DIR="$EXP_DIR/dictionaries"

CC="$AFL_ROOT/afl-clang-fast"
CXX="$AFL_ROOT/afl-clang-fast++"
AFLDRIVER="$AFL_ROOT/libAFLDriver.a"

# Detect GCC C++ stdlib paths (clang may not find them automatically)
GCC_VER=$(ls /usr/include/c++/ 2>/dev/null | sort -V | tail -1)
if [[ -n "$GCC_VER" ]]; then
    CXX_INCLUDE="-I/usr/include/c++/$GCC_VER -I/usr/include/x86_64-linux-gnu/c++/$GCC_VER"
    CXX_LIBDIR="-L/usr/lib/gcc/x86_64-linux-gnu/$GCC_VER"
else
    CXX_INCLUDE=""
    CXX_LIBDIR=""
fi

CLEAN=0
[[ "${1:-}" == "--clean" ]] && CLEAN=1

# ── Prerequisite checks ───────────────────────────────────────────────────
echo "=== Phase 1: Build Differential libxml2 Targets ==="
echo ""
echo "Checking prerequisites..."

for f in "$CC" "$CXX" "$AFLDRIVER"; do
    [[ -f "$f" ]] || { echo "[-] MISSING: $f"; exit 1; }
done
echo "[+] AFL++ found: $AFL_ROOT"

[[ -d "$BENCHMARK_DIR" ]] || { echo "[-] MISSING: $BENCHMARK_DIR"; exit 1; }
echo "[+] FuzzBench libxml2_xml benchmark found: $BENCHMARK_DIR"

pkg-config --libs liblzma >/dev/null 2>&1 || { echo "[-] liblzma not found (install liblzma-dev)"; exit 1; }
pkg-config --libs zlib    >/dev/null 2>&1 || { echo "[-] zlib not found (install zlib1g-dev)"; exit 1; }
echo "[+] System libraries: liblzma OK, zlib OK"

[[ -f "$BENCHMARK_DIR/target.cc" ]] || { echo "[-] MISSING: $BENCHMARK_DIR/target.cc (FuzzBench harness)"; exit 1; }
echo "[+] Harness: $BENCHMARK_DIR/target.cc"

# ── Copy harness, seeds, dictionary ───────────────────────────────────────
echo ""
echo "=== Copying FuzzBench assets ==="

# Harness: byte-identical copy from FuzzBench (NEVER modify this file).
# Portability fixes (missing uint8_t/size_t) are handled via -include flags
# at compile time, not by patching the source.
cp "$BENCHMARK_DIR/target.cc" "$EXP_DIR/build/harness.cc"
echo "[+] Harness: byte-identical copy of FuzzBench libxml2_xml/target.cc"
echo "    SHA256 (FuzzBench): $(sha256sum "$BENCHMARK_DIR/target.cc" | awk '{print $1}')"
echo "    SHA256 (local):     $(sha256sum "$EXP_DIR/build/harness.cc" | awk '{print $1}')"

# Seeds: canonical test/*.xml from the libxml2 source tree (FuzzBench-pinned commit).
# These are already checked into experiments/differential/seeds/ — verify they exist.
SEED_COUNT=$(ls "$SEEDS_DIR"/*.xml 2>/dev/null | wc -l)
if [[ $SEED_COUNT -eq 0 ]]; then
    echo "[-] STOP: No seed files in $SEEDS_DIR/"
    echo "    Seeds must come from libxml2/test/*.xml — see provenance audit."
    exit 1
fi
echo "[+] Seeds: $SEEDS_DIR/ ($SEED_COUNT files from libxml2/test/*.xml)"
echo "    Total size: $(du -sh "$SEEDS_DIR" | awk '{print $1}')"

# Dictionary: canonical fuzz/xml.dict from the libxml2 source tree.
# Already checked into experiments/differential/dictionaries/ — verify it exists.
if [[ ! -f "$DICT_DIR/libxml2.dict" ]]; then
    echo "[-] STOP: Dictionary missing: $DICT_DIR/libxml2.dict"
    echo "    Must come from libxml2/fuzz/xml.dict — see provenance audit."
    exit 1
fi
echo "[+] Dictionary: $DICT_DIR/libxml2.dict ($(wc -l < "$DICT_DIR/libxml2.dict") entries from libxml2/fuzz/xml.dict)"

# ── CVE definitions ───────────────────────────────────────────────────────
# Format: NAME TAG CVE_ID DESCRIPTION
declare -A BUILD_TAGS=(
    [xml005_buggy]="v2.9.4"
    [xml005_fixed]="v2.9.5"
    [xml017_buggy]="v2.9.3"
    [xml017_fixed]="v2.9.4"
)

declare -A BUILD_CVES=(
    [xml005_buggy]="CVE-2017-5130 (integer overflow in xmlMemoryStrdup, xmlmemory.c)"
    [xml005_fixed]="CVE-2017-5130 fix (commit 897dffbae322b46b83f99a607d527058a72c51ed)"
    [xml017_buggy]="CVE-2016-1762 (heap overread in xmlNextChar, parserInternals.c)"
    [xml017_fixed]="CVE-2016-1762 fix (commit a7a94612aa3b16779e2c74e1fa353b5d9786c602)"
)

# ── Clone libxml2 ─────────────────────────────────────────────────────────
echo ""
echo "=== Cloning libxml2 ==="

LIBXML2_REPO="$TARGET_BASE/libxml2.git"
if [[ ! -d "$LIBXML2_REPO" ]] || [[ $CLEAN -eq 1 ]]; then
    rm -rf "$LIBXML2_REPO"
    mkdir -p "$TARGET_BASE"
    git clone --bare https://gitlab.gnome.org/GNOME/libxml2.git "$LIBXML2_REPO"
    echo "[+] Bare clone: $LIBXML2_REPO"
else
    echo "[+] Using existing clone: $LIBXML2_REPO"
fi

# ── Build function ────────────────────────────────────────────────────────
build_target() {
    local NAME="$1"
    local TAG="${BUILD_TAGS[$NAME]}"
    local CVE="${BUILD_CVES[$NAME]}"
    local BUILD_DIR="$TARGET_BASE/$NAME"
    local OUT_DIR="$TARGETS_DIR/$NAME"

    echo ""
    echo "=== Building: $NAME (tag: $TAG) ==="
    echo "    CVE: $CVE"

    # Skip if already built and not --clean
    if [[ -f "$OUT_DIR/target" ]] && [[ $CLEAN -eq 0 ]]; then
        echo "[+] Already built: $OUT_DIR/target ($(stat -c%s "$OUT_DIR/target") bytes)"
        return 0
    fi

    # Checkout source at tag
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR/src"
    git clone --shared "$LIBXML2_REPO" "$BUILD_DIR/src"
    cd "$BUILD_DIR/src"
    git checkout "$TAG" 2>/dev/null || {
        echo "[-] STOP: Tag '$TAG' not found in libxml2 repo."
        echo "    Available tags matching v2.9.*:"
        git tag -l 'v2.9.*'
        exit 1
    }
    local COMMIT_HASH
    COMMIT_HASH=$(git rev-parse HEAD)
    echo "    Commit: $COMMIT_HASH"

    # Build libxml2 with AFL++ instrumentation + ASAN
    # Configure flags from FuzzBench build.sh + existing benchmarks/libxml2/build_recipe.sh
    export CC="$AFL_ROOT/afl-clang-fast"
    export CXX="$AFL_ROOT/afl-clang-fast++"
    export AFL_USE_ASAN=1
    export CFLAGS="-g -O2"
    export CXXFLAGS="-g -O2 $CXX_INCLUDE"
    export LDFLAGS="$CXX_LIBDIR"

    [[ -f Makefile ]] && make distclean 2>/dev/null || true

    ./autogen.sh \
        --prefix="$BUILD_DIR/install" \
        --disable-shared \
        --without-debug \
        --without-ftp \
        --without-http \
        --without-legacy \
        --without-python \
        CC="$CC" CXX="$CXX" \
        CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" LDFLAGS="$LDFLAGS" 2>&1 | tail -3

    make -j"$(nproc)" 2>&1 | tail -3
    make install 2>&1 | tail -1

    # Verify libxml2.a was built
    [[ -f "$BUILD_DIR/install/lib/libxml2.a" ]] || {
        echo "[-] STOP: libxml2.a not produced for $NAME"
        exit 1
    }
    echo "    libxml2.a: $(stat -c%s "$BUILD_DIR/install/lib/libxml2.a") bytes"

    # Link harness against this version's libxml2
    # -include flags provide missing standard types (uint8_t, size_t) without
    # modifying the FuzzBench harness source file — it stays byte-identical.
    mkdir -p "$OUT_DIR"
    AFL_USE_ASAN=1 "$CXX" $CXXFLAGS \
        -include cstdint -include cstddef \
        -I"$BUILD_DIR/install/include/libxml2" \
        "$EXP_DIR/build/harness.cc" \
        "$BUILD_DIR/install/lib/libxml2.a" \
        "$AFLDRIVER" \
        $CXX_LIBDIR -lz -llzma -lstdc++ \
        -o "$OUT_DIR/target" 2>&1 || {
        echo "[-] STOP: Harness failed to compile against $NAME ($TAG)"
        echo "    Compiler: $CXX"
        echo "    Include: $BUILD_DIR/install/include/libxml2"
        exit 1
    }

    echo "    target: $(stat -c%s "$OUT_DIR/target") bytes"
    echo "    SHA256: $(sha256sum "$OUT_DIR/target" | awk '{print $1}')"

    # Save build metadata
    cat > "$OUT_DIR/build_info.json" << METAEOF
{
    "name": "$NAME",
    "git_tag": "$TAG",
    "git_commit": "$COMMIT_HASH",
    "cve": "$CVE",
    "compiler": "$(${CC} --version 2>&1 | head -1)",
    "cflags": "$CFLAGS",
    "asan": true,
    "configure_flags": "--disable-shared --without-debug --without-ftp --without-http --without-legacy --without-python",
    "harness": "FuzzBench libxml2_xml/target.cc",
    "binary_sha256": "$(sha256sum "$OUT_DIR/target" | awk '{print $1}')",
    "binary_size": $(stat -c%s "$OUT_DIR/target"),
    "built_at": "$(date -Iseconds)"
}
METAEOF

    # Smoke test
    echo '<a/>' | timeout 5 "$OUT_DIR/target" 2>/dev/null && echo "    Smoke test: PASS" || {
        echo "[-] WARNING: Smoke test failed for $NAME (may be expected with ASAN)"
    }

    # Cleanup env
    unset AFL_USE_ASAN

    cd "$REPO_ROOT"
}

# ── Build all 4 targets ──────────────────────────────────────────────────
for NAME in xml005_buggy xml005_fixed xml017_buggy xml017_fixed; do
    build_target "$NAME"
done

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "=== Phase 1 Build Summary ==="
echo "============================================"
echo ""
echo "Targets built:"
for NAME in xml005_buggy xml005_fixed xml017_buggy xml017_fixed; do
    TAG="${BUILD_TAGS[$NAME]}"
    if [[ -f "$TARGETS_DIR/$NAME/target" ]]; then
        SIZE=$(stat -c%s "$TARGETS_DIR/$NAME/target")
        echo "  [OK] $NAME (tag: $TAG, $SIZE bytes)"
    else
        echo "  [FAIL] $NAME (tag: $TAG)"
    fi
done
echo ""
echo "FuzzBench assets:"
echo "  Harness: $EXP_DIR/build/harness.cc ($(sha256sum "$EXP_DIR/build/harness.cc" | awk '{print $1}'))"
echo "  Seeds:   $SEEDS_DIR/ ($(ls "$SEEDS_DIR"/*.xml 2>/dev/null | wc -l) files)"
echo "  Dict:    $DICT_DIR/libxml2.dict ($(wc -l < "$DICT_DIR/libxml2.dict") entries)"
echo ""
echo "Build tags used:"
for NAME in xml005_buggy xml005_fixed xml017_buggy xml017_fixed; do
    echo "  $NAME: ${BUILD_TAGS[$NAME]}"
done
echo ""
echo "Configure flags (all builds): --disable-shared --without-debug --without-ftp --without-http --without-legacy --without-python"
echo "ASAN: AFL_USE_ASAN=1 (AFL++ handles injection)"
echo "Compiler: $CC"
echo ""
echo "Next: Run experiments/differential/build/verify_bugs.sh to verify bug reachability"
