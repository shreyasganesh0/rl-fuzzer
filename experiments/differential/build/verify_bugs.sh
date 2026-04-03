#!/usr/bin/env bash
# verify_bugs.sh — Verify bug reachability for differential fuzzing targets
#
# Tests:
# 1. CVE-2016-1762 (xml017): Known PoC from Bugzilla #759671
# 2. CVE-2017-5130 (xml005): Quick 2-minute AFL++ run to check for crashes
#
# Usage: bash experiments/differential/build/verify_bugs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
TARGETS_DIR="$EXP_DIR/targets"
LOG="$EXP_DIR/build/verification.log"

echo "=== Bug Reachability Verification ===" | tee "$LOG"
echo "Date: $(date -Iseconds)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── CVE-2016-1762 (xml017): Craft a triggering input ─────────────────────
echo "--- CVE-2016-1762 (xml017): Heap overread in xmlNextChar ---" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# The bug is triggered by malformed internal subset with specific byte sequences
# that cause xmlNextChar to read past the buffer boundary.
# Create a minimal trigger: a document with a malformed DTD internal subset
# containing bytes that trigger the multi-byte UTF-8 parsing path.
POC_FILE="$EXP_DIR/build/poc_cve_2016_1762.xml"

# This crafts a minimal input that exercises the xmlNextChar overread path:
# An incomplete entity definition in an internal DTD subset with trailing
# multi-byte chars near the buffer boundary
printf '<?xml version="1.0"?>\n<!DOCTYPE a [\n<!ENTITY x "\xff\xfe">\n]>\n<a/>' > "$POC_FILE"

echo "PoC file: $POC_FILE ($(stat -c%s "$POC_FILE") bytes)" | tee -a "$LOG"

# Test on buggy version (v2.9.3) — should crash or show ASAN error
echo -n "xml017_buggy (v2.9.3): " | tee -a "$LOG"
ASAN_OPTIONS=detect_leaks=0 timeout 10 "$TARGETS_DIR/xml017_buggy/target" "$POC_FILE" > /tmp/verify_buggy.out 2>&1
BUGGY_EXIT=$?
if [[ $BUGGY_EXIT -ne 0 ]]; then
    echo "EXIT CODE $BUGGY_EXIT (crash/error detected)" | tee -a "$LOG"
    grep -i "ERROR\|ASAN\|overflow\|heap\|stack\|segfault" /tmp/verify_buggy.out | head -3 >> "$LOG" 2>/dev/null || true
else
    echo "EXIT CODE 0 (no crash — PoC may not trigger this specific path)" | tee -a "$LOG"
fi

# Test on fixed version (v2.9.4) — should NOT crash
echo -n "xml017_fixed (v2.9.4): " | tee -a "$LOG"
ASAN_OPTIONS=detect_leaks=0 timeout 10 "$TARGETS_DIR/xml017_fixed/target" "$POC_FILE" > /tmp/verify_fixed.out 2>&1
FIXED_EXIT=$?
if [[ $FIXED_EXIT -eq 0 ]]; then
    echo "EXIT CODE 0 (clean — as expected)" | tee -a "$LOG"
else
    echo "EXIT CODE $FIXED_EXIT (unexpected crash on fixed version!)" | tee -a "$LOG"
    grep -i "ERROR\|ASAN" /tmp/verify_fixed.out | head -3 >> "$LOG" 2>/dev/null || true
fi

echo "" | tee -a "$LOG"

# ── CVE-2017-5130 (xml005): Quick AFL++ run ──────────────────────────────
echo "--- CVE-2017-5130 (xml005): Integer overflow in xmlMemoryStrdup ---" | tee -a "$LOG"
echo "No public PoC available. Running a quick 2-minute AFL++ campaign." | tee -a "$LOG"
echo "" | tee -a "$LOG"

VERIFY_OUT="$EXP_DIR/build/verify_afl_out"
rm -rf "$VERIFY_OUT"

# Check kernel core_pattern
CORE_PAT=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo "unknown")
if [[ "$CORE_PAT" != "core" ]]; then
    echo "WARNING: core_pattern is '$CORE_PAT' (should be 'core' for AFL++)" | tee -a "$LOG"
    echo "Run: echo core | sudo tee /proc/sys/kernel/core_pattern" | tee -a "$LOG"
    echo "Skipping AFL++ verification run." | tee -a "$LOG"
else
    echo "core_pattern: OK ($CORE_PAT)" | tee -a "$LOG"

    # Run AFL++ for 2 minutes on buggy target
    echo "Running: afl-fuzz -V 120 on xml005_buggy..." | tee -a "$LOG"
    AFL_SKIP_CPUFREQ=1 AFL_NO_AFFINITY=1 AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
        timeout 130 "$AFL_ROOT/afl-fuzz" \
        -i "$EXP_DIR/seeds" \
        -o "$VERIFY_OUT" \
        -x "$EXP_DIR/dictionaries/libxml2.dict" \
        -V 120 \
        -- "$TARGETS_DIR/xml005_buggy/target" @@ \
        > /tmp/verify_afl.out 2>&1 || true

    # Check for crashes
    CRASH_COUNT=$(ls "$VERIFY_OUT/default/crashes/" 2>/dev/null | grep -v README | wc -l)
    echo "Crashes found: $CRASH_COUNT" | tee -a "$LOG"

    if [[ $CRASH_COUNT -gt 0 ]]; then
        echo "Crash files:" | tee -a "$LOG"
        ls -la "$VERIFY_OUT/default/crashes/" 2>/dev/null | head -5 | tee -a "$LOG"
    fi

    # Check coverage
    if [[ -f "$VERIFY_OUT/default/fuzzer_stats" ]]; then
        EDGES=$(grep "bitmap_cvg" "$VERIFY_OUT/default/fuzzer_stats" | awk '{print $3}')
        EXECS=$(grep "execs_done" "$VERIFY_OUT/default/fuzzer_stats" | awk '{print $3}')
        echo "Coverage: $EDGES, Execs: $EXECS" | tee -a "$LOG"
    fi
fi

echo "" | tee -a "$LOG"

# ── Summary ───────────────────────────────────────────────────────────────
echo "============================================" | tee -a "$LOG"
echo "=== Verification Summary ===" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "CVE-2016-1762 (xml017):" | tee -a "$LOG"
echo "  Buggy (v2.9.3): exit=$BUGGY_EXIT" | tee -a "$LOG"
echo "  Fixed (v2.9.4): exit=$FIXED_EXIT" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "CVE-2017-5130 (xml005):" | tee -a "$LOG"
if [[ "$CORE_PAT" == "core" ]]; then
    echo "  AFL++ 2-min run: $CRASH_COUNT crashes" | tee -a "$LOG"
else
    echo "  Skipped (core_pattern not set)" | tee -a "$LOG"
fi
echo "" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"

rm -f /tmp/verify_buggy.out /tmp/verify_fixed.out /tmp/verify_afl.out
