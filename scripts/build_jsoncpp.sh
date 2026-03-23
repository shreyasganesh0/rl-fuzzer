#!/bin/bash
# scripts/build_jsoncpp.sh — Backward-compatible wrapper
#
# Delegates to the generic build framework. See scripts/build_benchmark.sh.

exec "$(dirname "$0")/build_benchmark.sh" jsoncpp "$@"
