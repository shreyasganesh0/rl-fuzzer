#!/usr/bin/env bash
# run_baseline_vanilla.sh — Run vanilla AFL++ (no custom mutator) on differential targets
#
# Collects fuzzer_stats at 1-second intervals for baseline comparison.
# Same 4 targets x 3 seeds = 12 runs.
#
# Usage:
#   bash scripts/run_baseline_vanilla.sh [--duration SECONDS] [--targets LIST] [--seeds LIST]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
EXP_DIR="$REPO_ROOT/experiments/differential"
TARGETS_DIR="$EXP_DIR/targets"
TEL_DIR="$EXP_DIR/telemetry"
SEEDS_DIR="$EXP_DIR/seeds"
DICT_FILE="$EXP_DIR/dictionaries/libxml2.dict"

# Defaults
DURATION=3600
TARGET_LIST="xml005_buggy,xml005_fixed,xml017_buggy,xml017_fixed"
SEED_LIST="1,2,3"
SKIP_EXISTING=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)      DURATION="$2"; shift 2 ;;
        --targets)       TARGET_LIST="$2"; shift 2 ;;
        --seeds)         SEED_LIST="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra TARGETS <<< "$TARGET_LIST"
IFS=',' read -ra SEEDS <<< "$SEED_LIST"

echo "=== Vanilla AFL++ Baseline Campaign ==="
echo "Duration per run: ${DURATION}s"
echo "Targets: ${TARGETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $((${#TARGETS[@]} * ${#SEEDS[@]}))"
echo ""

[[ -f "$AFL_ROOT/afl-fuzz" ]] || { echo "[-] afl-fuzz not found"; exit 1; }

mkdir -p "$TEL_DIR"

# ── Polling function ──────────────────────────────────────────────────────
poll_stats() {
    local STATS_FILE="$1"
    local CSV_FILE="$2"
    local START=$3

    echo "step,reward,coverage_term,crash_term,loss,epsilon,coverage,crashes,action,elapsed_seconds" > "$CSV_FILE"

    while [[ -f "$STATS_FILE" ]] || sleep 1; do
        [[ -f "$STATS_FILE" ]] || continue

        local EXECS BITMAP_CVG CRASHES EDGES NOW ELAPSED
        EXECS=$(grep "execs_done" "$STATS_FILE" 2>/dev/null | awk '{print $3}' || echo 0)
        BITMAP_CVG=$(grep "bitmap_cvg" "$STATS_FILE" 2>/dev/null | awk '{print $3}' | tr -d '%' || echo 0)
        CRASHES=$(grep "saved_crashes" "$STATS_FILE" 2>/dev/null | awk '{print $3}' || echo 0)
        NOW=$(date +%s)
        ELAPSED=$((NOW - START))

        # Convert bitmap_cvg% to edge count (map size = 65536)
        EDGES=$(echo "$BITMAP_CVG * 65536 / 100" | bc -l 2>/dev/null | cut -d. -f1 || echo 0)
        [[ -z "$EDGES" ]] && EDGES=0

        echo "$EXECS,0,0,0,0,0,$EDGES,$CRASHES,-1,$ELAPSED" >> "$CSV_FILE"
        sleep 1
    done
}

# ── Run campaigns ─────────────────────────────────────────────────────────
RUN_COUNT=0
TOTAL_RUNS=$((${#TARGETS[@]} * ${#SEEDS[@]}))

for TARGET_NAME in "${TARGETS[@]}"; do
    TARGET_BIN="$TARGETS_DIR/$TARGET_NAME/target"
    [[ -f "$TARGET_BIN" ]] || { echo "[-] Target missing: $TARGET_BIN"; exit 1; }

    for SEED in "${SEEDS[@]}"; do
        RUN_ID="baseline_${TARGET_NAME}_seed${SEED}"
        RUN_COUNT=$((RUN_COUNT + 1))

        echo ""
        echo "=== Run $RUN_COUNT/$TOTAL_RUNS: $RUN_ID ==="

        CSV_FILE="$TEL_DIR/baseline_${TARGET_NAME}_seed${SEED}.csv"
        if [[ $SKIP_EXISTING -eq 1 ]] && [[ -f "$CSV_FILE" ]]; then
            echo "[+] Skipping (existing: $CSV_FILE)"
            continue
        fi

        AFL_OUT="$TEL_DIR/afl_out_${RUN_ID}"
        rm -rf "$AFL_OUT"

        echo "  Target:   $TARGET_BIN"
        echo "  Duration: ${DURATION}s"

        START_TIME=$(date +%s)

        # Start AFL++ (vanilla, no custom mutator)
        AFL_SKIP_CPUFREQ=1 \
        AFL_NO_AFFINITY=1 \
        AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
        timeout $((DURATION + 30)) "$AFL_ROOT/afl-fuzz" \
            -i "$SEEDS_DIR" \
            -o "$AFL_OUT" \
            -x "$DICT_FILE" \
            -t 1000 \
            -V "$DURATION" \
            -s "$SEED" \
            -- "$TARGET_BIN" @@ \
            > "$TEL_DIR/afl_log_${RUN_ID}.txt" 2>&1 &

        AFL_PID=$!

        # Wait for fuzzer_stats to appear
        STATS_FILE="$AFL_OUT/default/fuzzer_stats"
        for i in $(seq 1 30); do
            [[ -f "$STATS_FILE" ]] && break
            sleep 1
        done

        # Poll stats in background
        poll_stats "$STATS_FILE" "$CSV_FILE" "$START_TIME" &
        POLL_PID=$!

        # Wait for AFL++ to finish
        wait $AFL_PID 2>/dev/null || true

        # Give poller time to capture final stats, then kill it
        sleep 2
        kill $POLL_PID 2>/dev/null || true
        wait $POLL_PID 2>/dev/null || true

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        echo "  Elapsed: ${ELAPSED}s"
        if [[ -f "$CSV_FILE" ]]; then
            ROWS=$(wc -l < "$CSV_FILE")
            echo "  CSV rows: $((ROWS - 1))"
        fi

        if [[ -f "$STATS_FILE" ]]; then
            EXECS=$(grep "execs_done" "$STATS_FILE" | awk '{print $3}')
            CRASHES=$(grep "saved_crashes" "$STATS_FILE" | awk '{print $3}')
            echo "  Final: execs=$EXECS crashes=$CRASHES"
        fi
    done
done

echo ""
echo "============================================"
echo "=== Baseline Campaign Complete ==="
echo "============================================"
echo ""
echo "Baseline CSVs: $(ls "$TEL_DIR"/baseline_*.csv 2>/dev/null | wc -l)"
