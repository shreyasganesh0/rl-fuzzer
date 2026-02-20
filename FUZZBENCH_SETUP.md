# rl-fuzzer × FuzzBench Target Setup Guide

This guide covers everything needed to clone, build, and run a FuzzBench target
against the rl-fuzzer system. It documents all the non-obvious fixes discovered
during initial setup.

---

## Assumed Directory Layout

```
~/
├── rl-fuzzer/          # this project
├── fuzzbench/          # FuzzBench benchmarks repo
├── packages/
│   └── AFLplusplus/    # AFL++ built from source
└── targets/
    └── jsoncpp/        # built target (created by build script)
```

---

## 1. Prerequisites

### System packages

```bash
sudo apt update
sudo apt install -y \
    build-essential git cmake ninja-build \
    clang-16 llvm-16 llvm-16-dev llvm-16-tools \
    python3 python3-pip python3-venv \
    libstdc++-11-dev
```

> **Why clang-16 specifically?**  
> AFL++ LLVM mode requires a consistent LLVM version. clang-14's libc++ causes
> ABI incompatibilities. clang-16 from `apt.llvm.org` uses libstdc++ and avoids
> this entirely. Install from the official LLVM apt repo if not available:
>
> ```bash
> wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
> echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main" | \
>     sudo tee /etc/apt/sources.list.d/llvm-16.list
> sudo apt update && sudo apt install -y clang-16 llvm-16 llvm-16-dev
> ```

---

## 2. Clone FuzzBench

```bash
# Shallow clone — we only need the benchmark metadata and harness paths
git clone --depth=1 https://github.com/google/fuzzbench.git ~/fuzzbench
```

FuzzBench benchmark directories live at `~/fuzzbench/benchmarks/<name>/`.
Each one contains:
- `benchmark.yaml` — pinned commit, project name, fuzz target name
- `build.sh` — exact build instructions (the harness source path is here)
- `Dockerfile` — for reference

---

## 3. Build AFL++ from source (LLVM mode)

```bash
git clone https://github.com/AFLplusplus/AFLplusplus.git ~/packages/AFLplusplus
cd ~/packages/AFLplusplus
make -j$(nproc)

# Build LLVM instrumentation mode with explicit stdlib paths
# (required because llvm-config-16 does not expose C++ stdlib paths)
LLVM_CONFIG=llvm-config-16 \
CPPFLAGS="-I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11" \
LDFLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/11" \
make -f GNUmakefile.llvm

# Verify
echo 'int main(){return 0;}' > /tmp/t.c
~/packages/AFLplusplus/afl-clang-fast /tmp/t.c -o /tmp/t
# Should print: mode: LLVM-PCGUARD
```

> **Do NOT export `AFL_ROOT` or `AFL_LLVM_CONFIG` to your shell environment.**
> AFL++ treats any unrecognised `AFL_*` variable as an error. The run script
> sets `AFL_ROOT` internally only for PATH purposes.

---

## 4. Python environment

```bash
cd ~/rl-fuzzer
python3 -m venv .venv
source .venv/bin/activate
pip install pandas torch numpy
```

> **torch is large (~2 GB with CUDA deps).** If you don't have a GPU and want a
> smaller install:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 5. One-time system tuning

These are required every boot (or add to `/etc/rc.local`):

```bash
# Allow AFL++ to see crashes (default sends them to apport/systemd)
echo core | sudo tee /proc/sys/kernel/core_pattern

# Reduce CPU frequency scaling jitter
cd /sys/devices/system/cpu
echo performance | sudo tee cpu*/cpufreq/scaling_governor
cd ~
```

---

## 6. How to read a FuzzBench `build.sh` to find the harness

Every `build.sh` follows the same pattern. For jsoncpp:

```bash
# build.sh key lines:
$CXX $CXXFLAGS -I../include $LIB_FUZZING_ENGINE \
    ../src/test_lib_json/fuzz.cpp -o $OUT/jsoncpp_fuzzer \
    lib/libjsoncpp.a
```

To adapt for AFL++, substitute:
- `$CXX` → `afl-clang-fast++`
- `$CXXFLAGS` → `-g -O2` (no ASAN — see note below)
- `$LIB_FUZZING_ENGINE` → `~/packages/AFLplusplus/libAFLDriver.a`
- `$OUT/jsoncpp_fuzzer` → `~/rl-fuzzer/bin/target`

> **Critical: do NOT compile with `-fsanitize=address`.**  
> If the library (`libjsoncpp.a`) is compiled with ASAN, every object file
> contains `asan.module_ctor` symbols that require the full ASAN runtime at
> link time. Without `-fsanitize=address` on the final link, you get dozens of
> undefined reference errors for `__asan_init`, `__asan_report_*`, etc., and
> the binary crashes with SIGSEGV before the fork server even starts.  
> AFL++'s coverage instrumentation catches memory bugs without ASAN.

---

## 7. Building a target: general pattern

```bash
BENCHMARK="jsoncpp_jsoncpp_fuzzer"   # from ~/fuzzbench/benchmarks/
COMMIT=$(grep '^commit:' ~/fuzzbench/benchmarks/$BENCHMARK/benchmark.yaml | awk '{print $2}')
PROJECT=$(grep '^project:' ~/fuzzbench/benchmarks/$BENCHMARK/benchmark.yaml | awk '{print $2}')

# 1. Clone at the pinned commit
mkdir -p ~/targets/$PROJECT && cd ~/targets/$PROJECT
git clone https://github.com/open-source-parsers/jsoncpp.git src
cd src && git checkout $COMMIT

# 2. Build library (NO -fsanitize=address)
mkdir -p build-afl && cd build-afl
CC=$HOME/packages/AFLplusplus/afl-clang-fast \
CXX=$HOME/packages/AFLplusplus/afl-clang-fast++ \
CXXFLAGS="-g -O2" \
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF \
         -DJSONCPP_WITH_TESTS=OFF -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF
make -j$(nproc)

# 3. Link harness
AFL=$HOME/packages/AFLplusplus
SRC=$HOME/targets/$PROJECT/src
$AFL/afl-clang-fast++ -g -O2 \
    -I$SRC/include \
    $SRC/src/test_lib_json/fuzz.cpp \
    $SRC/build-afl/lib/libjsoncpp.a \
    $AFL/libAFLDriver.a \
    -o ~/rl-fuzzer/bin/target

# 4. Smoke test
echo '{}' | ~/rl-fuzzer/bin/target -
# Expected: "Reading N bytes from -\nExecution successful."
```

---

## 8. Running the fuzzer

```bash
cd ~/rl-fuzzer
source .venv/bin/activate
export AFL_ROOT=$HOME/packages/AFLplusplus
bash scripts/run_muofuzz.sh
```

The AFL++ TUI will appear. Key things to watch:
- `corpus count` — grows as new inputs are found
- `map density` — coverage percentage of the target
- RL brain logs print to terminal alongside the TUI (step, coverage, loss, ε)

---

## 9. Troubleshooting reference

| Symptom | Cause | Fix |
|---|---|---|
| `fatal error: 'list' file not found` during AFL++ build | llvm-config doesn't expose C++ stdlib paths | Add `CPPFLAGS` / `LDFLAGS` pointing to GCC-11 stdlib (see §3) |
| `[!] WARNING: Mistyped AFL environment variable: AFL_ROOT` | `AFL_ROOT` exported in shell | Remove from `.bashrc`; only set inside run script |
| `undefined reference to '__asan_init'` | Library compiled with ASAN, linker command missing `-fsanitize=address` | Rebuild library **without** ASAN flags (see §6) |
| `Fork server crashed with signal 11` | Binary has ASAN module ctors but ASAN runtime not linked | Same as above — rebuild without ASAN |
| `ModuleNotFoundError: No module named 'pandas'` | Python deps missing | `pip install pandas torch numpy` in venv |
| `PROGRAM ABORT: Suboptimal CPU scaling governor` | CPU in powersave mode | `echo performance \| sudo tee cpu*/cpufreq/scaling_governor` |
| `Pipe at the beginning of 'core_pattern'` | System sends crashes to apport | `echo core \| sudo tee /proc/sys/kernel/core_pattern` |
| `Program 'bin/target' not found` | run script deleted it on startup | Fixed in current script (only `rl_mutator.dylib` is deleted) |
| `Creating a tensor from a list of numpy.ndarrays is extremely slow` | PyTorch warning in `Agent.train()` | Fixed in current `rl_server.py` (uses `np.array()` before `FloatTensor`) |

---

## 10. Adding a new FuzzBench target

1. Find the benchmark: `ls ~/fuzzbench/benchmarks/`
2. Read its `build.sh` to find: the git repo URL, the harness `.cpp` file, and the library to build
3. Read `benchmark.yaml` for the pinned commit
4. Clone at that commit, build the library with `afl-clang-fast++` and **no ASAN**
5. Link: harness + library + `libAFLDriver.a` → `~/rl-fuzzer/bin/target`
6. Smoke test with `echo 'seed' | ~/rl-fuzzer/bin/target -`
7. Run `bash scripts/run_muofuzz.sh`

No harness writing needed — FuzzBench provides all harnesses already.
