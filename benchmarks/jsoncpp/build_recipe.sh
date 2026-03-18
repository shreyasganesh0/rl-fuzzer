# benchmarks/jsoncpp/build_recipe.sh — jsoncpp FuzzBench benchmark recipe
#
# Sourced by scripts/build_benchmark.sh. Sets recipe variables and defines
# BUILD_STEPS() and LINK_STEPS() functions.

FUZZBENCH_NAME="jsoncpp_jsoncpp_fuzzer"
GIT_URL="https://github.com/open-source-parsers/jsoncpp.git"

# Dictionary lives inside the jsoncpp source tree
# (set after clone, see BUILD_STEPS)

BUILD_STEPS() {
    rm -rf "$SRC_DIR/build-afl"
    mkdir -p "$SRC_DIR/build-afl"
    cd "$SRC_DIR/build-afl"

    CC="$CC" CXX="$CXX" \
    cmake "$SRC_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_STATIC_LIBS=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DJSONCPP_WITH_TESTS=OFF \
        -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
        -DCMAKE_C_FLAGS="$CFLAGS"

    make -j"$(nproc)" 2>&1 | grep -E "Building|Linking|[Ee]rror" || true

    [ -f "$SRC_DIR/build-afl/lib/libjsoncpp.a" ] || {
        echo "[-] libjsoncpp.a not built — check cmake output above."
        exit 1
    }
    echo "[+] libjsoncpp.a built."

    # Set dictionary path (inside source tree)
    DICT_PATH="$SRC_DIR/src/test_lib_json/fuzz.dict"
}

LINK_STEPS() {
    "$CXX" \
        $CXXFLAGS \
        -I"$SRC_DIR/include" \
        "$SRC_DIR/src/test_lib_json/fuzz.cpp" \
        "$SRC_DIR/build-afl/lib/libjsoncpp.a" \
        "$AFLDRIVER" \
        -o "$RL_FUZZER/bin/target"
}

# jsoncpp is a zero-seed benchmark in FuzzBench — we provide synthetic seeds
SEEDS_DIR=""  # will trigger synthetic seed generation in build_benchmark.sh

# Override default seed setup: provide jsoncpp-specific JSON seeds
_orig_seed_setup=1  # marker for build_benchmark.sh to detect recipe seeds
jsoncpp_install_seeds() {
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
        echo "[+] Created $(ls "$RL_FUZZER/inputs/" | wc -l) jsoncpp seed files."
    fi
}
