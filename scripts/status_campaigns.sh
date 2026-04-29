#!/usr/bin/env bash
# status_differential.sh — Check progress of running differential campaigns
set -euo pipefail

TEL_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/experiments/differential/telemetry}"

echo "=== Differential Campaign Status ($(date +%H:%M:%S)) ==="
echo ""

# Running AFL++ instances
AFL_COUNT=$(pgrep -c "afl-fuzz" 2>/dev/null || echo 0)
echo "Running AFL++ instances: $AFL_COUNT"
echo ""

# Telemetry runs
echo "--- Telemetry Runs ---"
printf "%-35s %8s %8s %8s %8s\n" "Run" "CSVRows" "Snaps" "AFLExecs" "Crashes"
for TARGET in xml005_buggy xml005_fixed xml017_buggy xml017_fixed; do
    for SEED in 1 2 3; do
        RUN_ID="${TARGET}_seed${SEED}"
        COV_CSV="$TEL_DIR/coverage_dynamics_${RUN_ID}.csv"
        SNAP_DIR="$TEL_DIR/snapshots_${RUN_ID}"
        STATS="$TEL_DIR/afl_out_${RUN_ID}/default/fuzzer_stats"

        ROWS="-"
        [[ -f "$COV_CSV" ]] && ROWS=$(( $(wc -l < "$COV_CSV") - 1 ))

        SNAPS="-"
        [[ -d "$SNAP_DIR" ]] && SNAPS=$(ls "$SNAP_DIR"/*.bin 2>/dev/null | wc -l)

        EXECS="-"; CRASHES="-"
        if [[ -f "$STATS" ]]; then
            EXECS=$(grep "execs_done" "$STATS" 2>/dev/null | awk '{print $3}')
            CRASHES=$(grep "saved_crashes" "$STATS" 2>/dev/null | awk '{print $3}')
        fi

        printf "%-35s %8s %8s %8s %8s\n" "$RUN_ID" "$ROWS" "$SNAPS" "$EXECS" "$CRASHES"
    done
done

echo ""
echo "--- Baseline Runs ---"
printf "%-35s %10s %8s\n" "Run" "Execs" "Crashes"
for TARGET in xml005_buggy xml005_fixed xml017_buggy xml017_fixed; do
    for SEED in 1 2 3; do
        RUN_ID="baseline_${TARGET}_seed${SEED}"
        STATS="$TEL_DIR/afl_out_${RUN_ID}/default/fuzzer_stats"

        EXECS="-"; CRASHES="-"
        if [[ -f "$STATS" ]]; then
            EXECS=$(grep "execs_done" "$STATS" 2>/dev/null | awk '{print $3}')
            CRASHES=$(grep "saved_crashes" "$STATS" 2>/dev/null | awk '{print $3}')
        fi

        printf "%-35s %10s %8s\n" "$RUN_ID" "$EXECS" "$CRASHES"
    done
done

echo ""
# Disk usage
echo "Telemetry disk usage: $(du -sh "$TEL_DIR" 2>/dev/null | awk '{print $1}')"
