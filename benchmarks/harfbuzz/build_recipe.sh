# benchmarks/harfbuzz/build_recipe.sh — HarfBuzz FuzzBench benchmark recipe
#
# Sourced by scripts/build_benchmark.sh.
# Requires: meson, ninja-build, pkg-config

FUZZBENCH_NAME="harfbuzz_hb-shape-fuzzer"
GIT_URL="https://github.com/harfbuzz/harfbuzz.git"

BUILD_STEPS() {
    cd "$SRC_DIR"

    # Clean old build
    rm -rf builddir

    CC="$CC" CXX="$CXX" \
    CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" \
    meson setup builddir \
        --default-library=static \
        -Dtests=disabled \
        -Ddocs=disabled \
        -Dbenchmark=disabled \
        -Dicu=disabled \
        -Dcairo=disabled \
        -Dfreetype=disabled \
        -Dglib=disabled

    ninja -C builddir

    [ -f "$SRC_DIR/builddir/src/libharfbuzz.a" ] || {
        echo "[-] libharfbuzz.a not built"
        exit 1
    }
    echo "[+] libharfbuzz.a built."
}

LINK_STEPS() {
    HARNESS="$FUZZBENCH/benchmarks/$FUZZBENCH_NAME/target.cc"
    if [[ ! -f "$HARNESS" ]]; then
        HARNESS="$SRC_DIR/test/fuzzing/hb-shape-fuzzer.cc"
    fi

    "$CXX" \
        $CXXFLAGS \
        -I"$SRC_DIR/src" \
        "$HARNESS" \
        "$SRC_DIR/builddir/src/libharfbuzz.a" \
        "$AFLDRIVER" \
        -lpthread \
        -o "$RL_FUZZER/bin/target"
}
