# benchmarks/re2/build_recipe.sh — RE2 FuzzBench benchmark recipe
#
# Sourced by scripts/build_benchmark.sh.

FUZZBENCH_NAME="re2_fuzzer"
GIT_URL="https://github.com/google/re2.git"

BUILD_STEPS() {
    cd "$SRC_DIR"

    make clean 2>/dev/null || true

    CC="$CC" CXX="$CXX" \
    CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" \
    make -j"$(nproc)" obj/libre2.a

    [ -f "$SRC_DIR/obj/libre2.a" ] || {
        echo "[-] libre2.a not built"
        exit 1
    }
    echo "[+] libre2.a built."
}

LINK_STEPS() {
    HARNESS="$FUZZBENCH/benchmarks/$FUZZBENCH_NAME/target.cc"
    if [[ ! -f "$HARNESS" ]]; then
        HARNESS="$SRC_DIR/re2/fuzzing/re2_fuzzer.cc"
    fi

    "$CXX" \
        $CXXFLAGS \
        -I"$SRC_DIR" \
        "$HARNESS" \
        "$SRC_DIR/obj/libre2.a" \
        "$AFLDRIVER" \
        -lpthread \
        -o "$RL_FUZZER/bin/target"
}
