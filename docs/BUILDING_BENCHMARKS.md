# Building FuzzBench Benchmarks

## Overview

The rl-fuzzer project uses a generic build framework to compile any supported
FuzzBench benchmark target with AFL++ instrumentation. Each benchmark has a
**recipe** (`benchmarks/<name>/build_recipe.sh`) that defines how to build and
link its source code.

## Quick Start

```bash
# Build jsoncpp (default benchmark)
bash scripts/build_benchmark.sh jsoncpp

# Build with clean rebuild
bash scripts/build_benchmark.sh jsoncpp --clean

# List available benchmarks
bash scripts/build_benchmark.sh --help
```

## Available Benchmarks

| Benchmark  | FuzzBench Name                  | Build System | Dependencies       |
|------------|--------------------------------|-------------|-------------------|
| jsoncpp    | jsoncpp_jsoncpp_fuzzer         | cmake       | —                 |
| freetype2  | freetype2_ftfuzzer             | autotools   | zlib              |
| libxml2    | libxml2_xml                    | autotools   | zlib, liblzma     |
| re2        | re2_fuzzer                     | make        | —                 |
| harfbuzz   | harfbuzz_hb-shape-fuzzer       | meson       | meson, ninja      |
| libpng     | libpng_libpng_read_fuzzer      | autotools   | zlib              |

## Writing a New Recipe

Create `benchmarks/<name>/build_recipe.sh` with the following structure:

```bash
# Required variables
FUZZBENCH_NAME="<fuzzbench_benchmark_name>"
GIT_URL="<git_clone_url>"

# Optional variables
DICT_PATH=""    # Path to dictionary file (auto-detected if empty)
SEEDS_DIR=""    # Path to seed corpus directory

# Required functions
BUILD_STEPS() {
    # Build the library from $SRC_DIR
    # Available variables: $CC, $CXX, $CFLAGS, $CXXFLAGS, $SRC_DIR
    cd "$SRC_DIR"
    # ... build commands ...
}

LINK_STEPS() {
    # Link the fuzzer binary to $RL_FUZZER/bin/target
    # Available variables: $CXX, $CXXFLAGS, $AFLDRIVER, $RL_FUZZER, $SRC_DIR
    "$CXX" $CXXFLAGS \
        -I"$SRC_DIR/include" \
        "<harness_source>" \
        "<library.a>" \
        "$AFLDRIVER" \
        -o "$RL_FUZZER/bin/target"
}
```

### Variables Available to Recipes

| Variable    | Description                                |
|------------|--------------------------------------------|
| `AFL_ROOT` | AFL++ installation directory               |
| `CC`       | `afl-clang-fast` path                      |
| `CXX`      | `afl-clang-fast++` path                    |
| `CFLAGS`   | C compiler flags (`-g -O2`)                |
| `CXXFLAGS` | C++ compiler flags (`-g -O2`)              |
| `AFLDRIVER` | Path to `libAFLDriver.a`                  |
| `TARGET_DIR` | `~/targets/<project>/`                   |
| `SRC_DIR`  | `~/targets/<project>/src/`                 |
| `RL_FUZZER` | rl-fuzzer project root                    |
| `FUZZBENCH` | FuzzBench clone root (`~/fuzzbench`)      |
| `LLVM_BIN` | Detected LLVM binary directory             |

## Directory Layout

```
~/fuzzbench/                      FuzzBench clone (benchmarks + seeds)
~/packages/AFLplusplus/           AFL++ installation
~/targets/<project>/src/          Cloned benchmark source (at pinned commit)
~/projects/rl-fuzzer/
  benchmarks/<name>/build_recipe.sh   Per-benchmark build recipes
  scripts/build_benchmark.sh          Generic build framework
  scripts/build_jsoncpp.sh            Backward-compat wrapper → build_benchmark.sh
  bin/target                          Compiled fuzzer binary
  dictionaries/target.dict            Fuzzing dictionary
  inputs/                             Seed corpus
```

## Troubleshooting

### ASAN pitfalls
Do NOT add `-fsanitize=address` to CFLAGS/CXXFLAGS. The library objects would
contain ASAN instrumentation but the final link omits the ASAN runtime, causing
SIGSEGV before the AFL++ fork server starts.

### Missing dependencies
Each recipe may require system packages. Check the recipe header comments for
required dependencies. Install with:
```bash
sudo apt-get install zlib1g-dev liblzma-dev meson ninja-build
```

### Stale builds
Use `--clean` to force a fresh clone and rebuild:
```bash
bash scripts/build_benchmark.sh freetype2 --clean
```

### LLVM detection
The framework searches for LLVM in standard paths. If your LLVM is elsewhere,
ensure it's on your PATH or set `LLVM_BIN` before running.
