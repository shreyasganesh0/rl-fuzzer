# benchmarks/libpng/build_recipe.sh — libpng FuzzBench benchmark recipe
#
# Sourced by scripts/build_benchmark.sh.
# Requires: zlib1g-dev

FUZZBENCH_NAME="libpng_libpng_read_fuzzer"
GIT_URL="https://github.com/glennrp/libpng.git"

BUILD_STEPS() {
    cd "$SRC_DIR"

    [[ -f Makefile ]] && make distclean 2>/dev/null || true

    autoreconf -fi 2>/dev/null || true

    CC="$CC" CXX="$CXX" \
    CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" \
    ./configure \
        --prefix="$SRC_DIR/install" \
        --disable-shared

    make -j"$(nproc)"
    make install

    [ -f "$SRC_DIR/install/lib/libpng16.a" ] || \
    [ -f "$SRC_DIR/install/lib/libpng.a" ] || {
        echo "[-] libpng.a not built"
        exit 1
    }
    echo "[+] libpng built."
}

LINK_STEPS() {
    HARNESS="$FUZZBENCH/benchmarks/$FUZZBENCH_NAME/target.cc"
    if [[ ! -f "$HARNESS" ]]; then
        HARNESS="$SRC_DIR/contrib/libtests/pngfuzz.c"
    fi

    # Find the actual .a name
    PNG_LIB=""
    for lib in "$SRC_DIR/install/lib/libpng16.a" "$SRC_DIR/install/lib/libpng.a"; do
        [[ -f "$lib" ]] && { PNG_LIB="$lib"; break; }
    done

    "$CXX" \
        $CXXFLAGS \
        -I"$SRC_DIR/install/include" \
        "$HARNESS" \
        "$PNG_LIB" \
        "$AFLDRIVER" \
        -lz \
        -o "$RL_FUZZER/bin/target"
}
