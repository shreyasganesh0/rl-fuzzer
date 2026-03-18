# benchmarks/freetype2/build_recipe.sh — FreeType2 FuzzBench benchmark recipe
#
# Sourced by scripts/build_benchmark.sh.
# Requires: libarchive-dev (for bzip2/zlib)

FUZZBENCH_NAME="freetype2_ftfuzzer"
GIT_URL="https://gitlab.freedesktop.org/freetype/freetype.git"

BUILD_STEPS() {
    cd "$SRC_DIR"

    # Clean previous build artifacts
    [[ -f Makefile ]] && make distclean 2>/dev/null || true

    ./autogen.sh 2>/dev/null || true  # may not exist in all versions

    CC="$CC" CXX="$CXX" \
    CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" \
    ./configure \
        --prefix="$SRC_DIR/install" \
        --disable-shared \
        --with-harfbuzz=no \
        --with-bzip2=no \
        --with-png=no

    make -j"$(nproc)"
    make install

    [ -f "$SRC_DIR/install/lib/libfreetype.a" ] || {
        echo "[-] libfreetype.a not built"
        exit 1
    }
    echo "[+] libfreetype.a built."
}

LINK_STEPS() {
    # FuzzBench ftfuzzer harness
    HARNESS="$FUZZBENCH/benchmarks/$FUZZBENCH_NAME/ftfuzzer.cc"
    if [[ ! -f "$HARNESS" ]]; then
        # Fallback: use the one from freetype source
        HARNESS="$SRC_DIR/src/tools/ftfuzzer/ftfuzzer.cc"
    fi

    "$CXX" \
        $CXXFLAGS \
        -I"$SRC_DIR/install/include/freetype2" \
        "$HARNESS" \
        "$SRC_DIR/install/lib/libfreetype.a" \
        "$AFLDRIVER" \
        $LDFLAGS -lz -larchive \
        -o "$RL_FUZZER/bin/target"
}
