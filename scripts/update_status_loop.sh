#!/usr/bin/env bash
# update_status_loop.sh — Continuously updates STATUS.md with campaign progress
# Runs in background, updates every 60 seconds.
set -euo pipefail

TEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/experiments/differential/telemetry"
STATUS_FILE="$TEL_DIR/STATUS.md"
START_TIME=$(date +%s)

while true; do
    AFL_COUNT=$(pgrep -c "afl-fuzz" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_TIME ))
    ELAPSED_H=$(echo "scale=1; $ELAPSED/3600" | bc)

    {
        echo "# Differential Fuzzing Campaign Status"
        echo ""
        echo "**Updated**: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "**Elapsed**: ${ELAPSED_H}h / 24.0h"
        echo "**Active AFL++ processes**: $AFL_COUNT"
        echo ""

        # Telemetry runs
        echo "## Telemetry Runs (12)"
        echo ""
        echo "| Target | Seed | CSV Rows | Snapshots | AFL Execs | Crashes | Status |"
        echo "|--------|------|----------|-----------|-----------|---------|--------|"

        TEL_DONE=0
        TEL_TOTAL=0
        for TARGET in xml005_buggy xml005_fixed xml017_buggy xml017_fixed; do
            for SEED in 1 2 3; do
                TEL_TOTAL=$((TEL_TOTAL + 1))
                RUN_ID="${TARGET}_seed${SEED}"
                COV_CSV="$TEL_DIR/coverage_dynamics_${RUN_ID}.csv"
                SNAP_DIR="$TEL_DIR/snapshots_${RUN_ID}"
                STATS="$TEL_DIR/afl_out_${RUN_ID}/default/fuzzer_stats"

                ROWS=0; [[ -f "$COV_CSV" ]] && ROWS=$(( $(wc -l < "$COV_CSV") - 1 ))
                SNAPS=0; [[ -d "$SNAP_DIR" ]] && SNAPS=$(ls "$SNAP_DIR"/*.bin 2>/dev/null | wc -l)
                EXECS="-"; CRASHES="-"; ST="waiting"
                if [[ -f "$STATS" ]]; then
                    EXECS=$(grep "execs_done" "$STATS" 2>/dev/null | awk '{print $3}')
                    CRASHES=$(grep "saved_crashes" "$STATS" 2>/dev/null | awk '{print $3}')
                    ST="running"
                    # Check if AFL process is still alive for this output dir
                    if ! pgrep -f "afl_out_${RUN_ID}" >/dev/null 2>&1; then
                        ST="done"
                        TEL_DONE=$((TEL_DONE + 1))
                    fi
                fi

                echo "| $TARGET | $SEED | $ROWS | $SNAPS | $EXECS | $CRASHES | $ST |"
            done
        done

        echo ""

        # Baseline runs
        echo "## Baseline Runs (12)"
        echo ""
        echo "| Target | Seed | AFL Execs | Crashes | Status |"
        echo "|--------|------|-----------|---------|--------|"

        BL_DONE=0
        BL_TOTAL=0
        for TARGET in xml005_buggy xml005_fixed xml017_buggy xml017_fixed; do
            for SEED in 1 2 3; do
                BL_TOTAL=$((BL_TOTAL + 1))
                RUN_ID="baseline_${TARGET}_seed${SEED}"
                STATS="$TEL_DIR/afl_out_${RUN_ID}/default/fuzzer_stats"

                EXECS="-"; CRASHES="-"; ST="waiting"
                if [[ -f "$STATS" ]]; then
                    EXECS=$(grep "execs_done" "$STATS" 2>/dev/null | awk '{print $3}')
                    CRASHES=$(grep "saved_crashes" "$STATS" 2>/dev/null | awk '{print $3}')
                    ST="running"
                    if ! pgrep -f "afl_out_${RUN_ID}" >/dev/null 2>&1; then
                        ST="done"
                        BL_DONE=$((BL_DONE + 1))
                    fi
                fi

                echo "| $TARGET | $SEED | $EXECS | $CRASHES | $ST |"
            done
        done

        echo ""
        echo "## Summary"
        echo ""
        echo "- Telemetry: $TEL_DONE/$TEL_TOTAL complete"
        echo "- Baseline: $BL_DONE/$BL_TOTAL complete"
        echo "- Disk: $(du -sh "$TEL_DIR" 2>/dev/null | awk '{print $1}')"

        if [[ $AFL_COUNT -eq 0 ]] && [[ $ELAPSED -gt 300 ]]; then
            echo ""
            echo "**ALL CAMPAIGNS FINISHED**"
        fi

    } > "$STATUS_FILE"

    # Exit if no AFL++ processes and we've been running > 5 min
    if [[ $AFL_COUNT -eq 0 ]] && [[ $ELAPSED -gt 300 ]]; then
        break
    fi

    sleep 60
done
