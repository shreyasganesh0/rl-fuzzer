#!/usr/bin/env bash
# run_telemetry_campaign.sh — Run instrumented telemetry collection campaigns
#
# Runs AFL++ with the telemetry mutator (random mutation selection + logging)
# on all 4 differential targets, 3 seeds each = 12 runs.
#
# Usage:
#   bash scripts/run_telemetry_campaign.sh [--duration SECONDS] [--targets LIST] [--seeds LIST]
#
# Options:
#   --duration N     Fuzzing time per run in seconds (default: 3600 = 1 hour)
#   --targets LIST   Comma-separated target names (default: xml005_buggy,xml005_fixed,xml017_buggy,xml017_fixed)
#   --seeds LIST     Comma-separated random seeds (default: 1,2,3)
#   --log-interval N Steps between CSV writes (default: 1000)
#   --skip-existing  Skip runs that already have telemetry CSVs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
EXP_DIR="$REPO_ROOT/experiments/differential"
TARGETS_DIR="$EXP_DIR/targets"
TEL_DIR="$EXP_DIR/telemetry"
SEEDS_DIR="$EXP_DIR/seeds"
DICT_FILE="$EXP_DIR/dictionaries/libxml2.dict"
MUTATOR_SO="$REPO_ROOT/bin/mutator_telemetry.so"

# Defaults
DURATION=3600
TARGET_LIST="xml005_buggy,xml005_fixed,xml017_buggy,xml017_fixed"
SEED_LIST="1,2,3"
LOG_INTERVAL=1000
SKIP_EXISTING=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)     DURATION="$2"; shift 2 ;;
        --targets)      TARGET_LIST="$2"; shift 2 ;;
        --seeds)        SEED_LIST="$2"; shift 2 ;;
        --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra TARGETS <<< "$TARGET_LIST"
IFS=',' read -ra SEEDS <<< "$SEED_LIST"

# ── Prerequisites ─────────────────────────────────────────────────────────
echo "=== Telemetry Campaign ==="
echo "Duration per run: ${DURATION}s"
echo "Targets: ${TARGETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $((${#TARGETS[@]} * ${#SEEDS[@]}))"
echo ""

[[ -f "$AFL_ROOT/afl-fuzz" ]] || { echo "[-] afl-fuzz not found at $AFL_ROOT"; exit 1; }

# Compile mutator if needed
if [[ ! -f "$MUTATOR_SO" ]] || [[ "$REPO_ROOT/src/mutator_telemetry.c" -nt "$MUTATOR_SO" ]]; then
    echo "[*] Compiling mutator_telemetry.c..."
    clang -O2 -shared -fPIC -g \
        -I "$AFL_ROOT/include" \
        -o "$MUTATOR_SO" \
        "$REPO_ROOT/src/mutator_telemetry.c" \
        -lm
    echo "[+] Compiled: $MUTATOR_SO"
fi

[[ -d "$SEEDS_DIR" ]] || { echo "[-] Seeds dir missing: $SEEDS_DIR"; exit 1; }
[[ -f "$DICT_FILE" ]] || { echo "[-] Dictionary missing: $DICT_FILE"; exit 1; }

mkdir -p "$TEL_DIR"

# ── Run campaigns ─────────────────────────────────────────────────────────
RUN_COUNT=0
TOTAL_RUNS=$((${#TARGETS[@]} * ${#SEEDS[@]}))

for TARGET_NAME in "${TARGETS[@]}"; do
    TARGET_BIN="$TARGETS_DIR/$TARGET_NAME/target"
    [[ -f "$TARGET_BIN" ]] || { echo "[-] Target binary missing: $TARGET_BIN"; exit 1; }

    for SEED in "${SEEDS[@]}"; do
        RUN_ID="${TARGET_NAME}_seed${SEED}"
        RUN_COUNT=$((RUN_COUNT + 1))

        echo ""
        echo "=== Run $RUN_COUNT/$TOTAL_RUNS: $RUN_ID ==="

        # Check if already done
        COV_CSV="$TEL_DIR/coverage_dynamics_${RUN_ID}.csv"
        if [[ $SKIP_EXISTING -eq 1 ]] && [[ -f "$COV_CSV" ]]; then
            echo "[+] Skipping (existing CSV: $COV_CSV)"
            continue
        fi

        # Create snapshot directory
        SNAP_DIR="$TEL_DIR/snapshots_${RUN_ID}"
        mkdir -p "$SNAP_DIR"

        # AFL++ output directory
        AFL_OUT="$TEL_DIR/afl_out_${RUN_ID}"
        rm -rf "$AFL_OUT"

        echo "  Target:   $TARGET_BIN"
        echo "  Seed:     $SEED"
        echo "  Duration: ${DURATION}s"
        echo "  CSV dir:  $TEL_DIR"
        echo "  Snap dir: $SNAP_DIR"

        START_TIME=$(date +%s)

        # Run AFL++ with telemetry mutator
        TELEMETRY_CSV_DIR="$TEL_DIR" \
        TELEMETRY_VERSION="$RUN_ID" \
        TELEMETRY_LOG_INTERVAL="$LOG_INTERVAL" \
        TELEMETRY_SNAPSHOT_INTERVAL=10000 \
        AFL_CUSTOM_MUTATOR_LIBRARY="$MUTATOR_SO" \
        AFL_CUSTOM_MUTATOR_ONLY=1 \
        AFL_SKIP_CPUFREQ=1 \
        AFL_NO_AFFINITY=1 \
        AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
        AFL_AUTORESUME=1 \
        timeout $((DURATION + 30)) "$AFL_ROOT/afl-fuzz" \
            -i "$SEEDS_DIR" \
            -o "$AFL_OUT" \
            -x "$DICT_FILE" \
            -t 1000 \
            -V "$DURATION" \
            -s "$SEED" \
            -- "$TARGET_BIN" @@ \
            > "$TEL_DIR/afl_log_${RUN_ID}.txt" 2>&1 || true

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        # Report results
        echo "  Elapsed:  ${ELAPSED}s"
        if [[ -f "$COV_CSV" ]]; then
            ROWS=$(wc -l < "$COV_CSV")
            echo "  CSV rows: $((ROWS - 1)) (coverage dynamics)"
        else
            echo "  WARNING: No coverage CSV produced"
        fi

        MUT_CSV="$TEL_DIR/mutation_attribution_${RUN_ID}.csv"
        if [[ -f "$MUT_CSV" ]]; then
            ROWS=$(wc -l < "$MUT_CSV")
            echo "  CSV rows: $((ROWS - 1)) (mutation attribution)"
        fi

        SNAP_COUNT=$(ls "$SNAP_DIR"/*.bin 2>/dev/null | wc -l)
        echo "  Snapshots: $SNAP_COUNT"

        # Extract final stats from AFL++
        STATS_FILE="$AFL_OUT/default/fuzzer_stats"
        if [[ -f "$STATS_FILE" ]]; then
            EXECS=$(grep "execs_done" "$STATS_FILE" | awk '{print $3}')
            CRASHES=$(grep "saved_crashes" "$STATS_FILE" | awk '{print $3}')
            echo "  AFL++ execs: $EXECS, crashes: $CRASHES"
        fi
    done
done

echo ""
echo "============================================"
echo "=== Campaign Complete ==="
echo "============================================"
echo ""
echo "Output: $TEL_DIR/"
echo "Coverage CSVs: $(ls "$TEL_DIR"/coverage_dynamics_*.csv 2>/dev/null | wc -l)"
echo "Mutation CSVs: $(ls "$TEL_DIR"/mutation_attribution_*.csv 2>/dev/null | wc -l)"
echo ""
echo "Next steps:"
echo "  1. Run scripts/run_baseline_vanilla.sh for baseline comparison"
echo "  2. Run scripts/analysis/differential_analysis.py for analysis"
