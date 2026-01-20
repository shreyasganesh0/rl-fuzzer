#!/bin/bash

afl-cc -o bin/target_bin src/target.c

afl-fuzz -i inputs -o outputs bin/target_bin
