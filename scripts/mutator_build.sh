#!/bin/bash
clang -dynamiclib -I ~/Packages/AFLplusplus/include -o bin/rl_mutator.dylib src/mutator.c

export AFL_CUSTOM_MUTATOR_LIBRARY=./bin/rl_mutator.dylib
afl-fuzz -i inputs -o outputs -- bin/target_bin
