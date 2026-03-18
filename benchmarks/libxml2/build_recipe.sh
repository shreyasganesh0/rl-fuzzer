# benchmarks/libxml2/build_recipe.sh — libxml2 FuzzBench benchmark recipe
#
# Sourced by scripts/build_benchmark.sh.
# Requires: zlib1g-dev, liblzma-dev

FUZZBENCH_NAME="libxml2_xml"
GIT_URL="https://gitlab.gnome.org/GNOME/libxml2.git"

BUILD_STEPS() {
    cd "$SRC_DIR"

    [[ -f Makefile ]] && make distclean 2>/dev/null || true

    ./autogen.sh \
        --prefix="$SRC_DIR/install" \
        --disable-shared \
        --without-python \
        --without-debug \
        --without-ftp \
        --without-http \
        CC="$CC" CXX="$CXX" \
        CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" LDFLAGS="$LDFLAGS"

    make -j"$(nproc)"
    make install

    [ -f "$SRC_DIR/install/lib/libxml2.a" ] || {
        echo "[-] libxml2.a not built"
        exit 1
    }
    echo "[+] libxml2.a built."
}

LINK_STEPS() {
    HARNESS="$FUZZBENCH/benchmarks/$FUZZBENCH_NAME/target.cc"
    if [[ ! -f "$HARNESS" ]]; then
        HARNESS="$SRC_DIR/fuzz/xml.c"
    fi

    "$CXX" \
        $CXXFLAGS \
        -I"$SRC_DIR/install/include/libxml2" \
        "$HARNESS" \
        "$SRC_DIR/install/lib/libxml2.a" \
        "$AFLDRIVER" \
        $LDFLAGS -lz -llzma \
        -o "$RL_FUZZER/bin/target"
}
