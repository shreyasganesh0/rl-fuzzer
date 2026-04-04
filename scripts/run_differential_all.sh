#!/usr/bin/env bash
# run_differential_all.sh — Run all differential fuzzing campaigns in parallel
#
# Runs 24 AFL++ instances (12 telemetry + 12 baseline) across available cores.
# Each instance runs for --duration seconds (default: 86400 = 24 hours).
#
# Usage:
#   bash scripts/run_differential_all.sh [--duration SECONDS] [--jobs N]
#
# Output:
#   experiments/differential/telemetry/
#     coverage_dynamics_<target>_seed<N>.csv        (12 files)
#     mutation_attribution_<target>_seed<N>.csv     (12 files)
#     snapshots_<target>_seed<N>/                   (12 dirs)
#     baseline_<target>_seed<N>.csv                 (12 files)
#     afl_out_<target>_seed<N>/                     (12 dirs, telemetry AFL state)
#     afl_out_baseline_<target>_seed<N>/            (12 dirs, baseline AFL state)
#     campaign_manifest.json                        (experiment metadata)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AFL_ROOT="${AFL_ROOT:-$HOME/packages/AFLplusplus}"
EXP_DIR="$REPO_ROOT/experiments/differential"
TARGETS_DIR="$EXP_DIR/targets"
TEL_DIR="$EXP_DIR/telemetry"
SEEDS_DIR="$EXP_DIR/seeds"
DICT_FILE="$EXP_DIR/dictionaries/libxml2.dict"
MUTATOR_SRC="$REPO_ROOT/src/mutator_telemetry.c"
MUTATOR_SO="$REPO_ROOT/bin/mutator_telemetry.so"

DURATION=86400  # 24 hours
MAX_JOBS=$(( $(nproc) - 1 ))  # leave 1 core for system
[[ $MAX_JOBS -lt 1 ]] && MAX_JOBS=1

TARGETS=("xml005_buggy" "xml005_fixed" "xml017_buggy" "xml017_fixed")
SEEDS=(1 2 3)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        --jobs)     MAX_JOBS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Preflight ─────────────────────────────────────────────────────────────
echo "============================================"
echo "  Differential Fuzzing — Full Campaign"
echo "============================================"
echo ""
echo "Duration per run: ${DURATION}s ($(echo "scale=1; $DURATION/3600" | bc)h)"
echo "Parallel jobs:    $MAX_JOBS"
echo "Targets:          ${TARGETS[*]}"
echo "Seeds:            ${SEEDS[*]}"
echo "Total runs:       $(( ${#TARGETS[@]} * ${#SEEDS[@]} * 2 ))  (12 telemetry + 12 baseline)"
echo ""

# Check core_pattern
CORE_PAT=$(cat /proc/sys/kernel/core_pattern)
if [[ "$CORE_PAT" != "core" ]]; then
    echo "[-] FATAL: core_pattern is '$CORE_PAT'"
    echo "    AFL++ requires: echo core | sudo tee /proc/sys/kernel/core_pattern"
    exit 1
fi
echo "[+] core_pattern: OK"

# Check AFL++
[[ -f "$AFL_ROOT/afl-fuzz" ]] || { echo "[-] afl-fuzz not found at $AFL_ROOT"; exit 1; }
echo "[+] AFL++: $AFL_ROOT"

# Check targets
for T in "${TARGETS[@]}"; do
    [[ -f "$TARGETS_DIR/$T/target" ]] || { echo "[-] Target missing: $TARGETS_DIR/$T/target"; exit 1; }
done
echo "[+] All 4 targets found"

# Compile telemetry mutator
echo "[*] Compiling mutator_telemetry.c..."
"$AFL_ROOT/afl-clang-fast" \
    -O2 -shared -fPIC -g \
    -I "$AFL_ROOT/include" \
    -o "$MUTATOR_SO" "$MUTATOR_SRC" -lm 2>/dev/null
echo "[+] Compiled: $MUTATOR_SO"

mkdir -p "$TEL_DIR"

# ── Job launcher ──────────────────────────────────────────────────────────
PIDS=()
JOBS=()
LOG_DIR="$TEL_DIR/logs"
mkdir -p "$LOG_DIR"

launch_job() {
    local JOB_NAME="$1"
    local CMD="$2"

    # Wait if at max capacity
    while [[ ${#PIDS[@]} -ge $MAX_JOBS ]]; do
        # Check for finished jobs
        NEW_PIDS=()
        NEW_JOBS=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                NEW_PIDS+=("${PIDS[$i]}")
                NEW_JOBS+=("${JOBS[$i]}")
            else
                wait "${PIDS[$i]}" 2>/dev/null || true
                echo "  [done] ${JOBS[$i]} (PID ${PIDS[$i]})"
            fi
        done
        PIDS=("${NEW_PIDS[@]+"${NEW_PIDS[@]}"}")
        JOBS=("${NEW_JOBS[@]+"${NEW_JOBS[@]}"}")
        [[ ${#PIDS[@]} -ge $MAX_JOBS ]] && sleep 5
    done

    eval "$CMD" > "$LOG_DIR/${JOB_NAME}.log" 2>&1 &
    local PID=$!
    PIDS+=("$PID")
    JOBS+=("$JOB_NAME")
    echo "  [launched] $JOB_NAME (PID $PID)"
}

# ── Launch telemetry campaigns ────────────────────────────────────────────
echo ""
echo "=== Launching telemetry campaigns (12 runs) ==="

for TARGET_NAME in "${TARGETS[@]}"; do
    TARGET_BIN="$TARGETS_DIR/$TARGET_NAME/target"

    for SEED in "${SEEDS[@]}"; do
        RUN_ID="${TARGET_NAME}_seed${SEED}"
        AFL_OUT="$TEL_DIR/afl_out_${RUN_ID}"
        SNAP_DIR="$TEL_DIR/snapshots_${RUN_ID}"
        mkdir -p "$SNAP_DIR"
        rm -rf "$AFL_OUT"

        CMD="TELEMETRY_CSV_DIR='$TEL_DIR' \
             TELEMETRY_VERSION='$RUN_ID' \
             TELEMETRY_LOG_INTERVAL=1000 \
             TELEMETRY_SNAPSHOT_INTERVAL=10000 \
             AFL_CUSTOM_MUTATOR_LIBRARY='$MUTATOR_SO' \
             AFL_CUSTOM_MUTATOR_ONLY=1 \
             AFL_SKIP_CPUFREQ=1 \
             AFL_NO_AFFINITY=1 \
             AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
             timeout $((DURATION + 60)) '$AFL_ROOT/afl-fuzz' \
                 -i '$SEEDS_DIR' \
                 -o '$AFL_OUT' \
                 -x '$DICT_FILE' \
                 -t 2000 \
                 -V $DURATION \
                 -s $SEED \
                 -- '$TARGET_BIN' @@"

        launch_job "tel_${RUN_ID}" "$CMD"
    done
done

# ── Launch baseline campaigns ─────────────────────────────────────────────
echo ""
echo "=== Launching baseline campaigns (12 runs) ==="

for TARGET_NAME in "${TARGETS[@]}"; do
    TARGET_BIN="$TARGETS_DIR/$TARGET_NAME/target"

    for SEED in "${SEEDS[@]}"; do
        RUN_ID="baseline_${TARGET_NAME}_seed${SEED}"
        AFL_OUT="$TEL_DIR/afl_out_${RUN_ID}"
        rm -rf "$AFL_OUT"

        CMD="AFL_SKIP_CPUFREQ=1 \
             AFL_NO_AFFINITY=1 \
             AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
             timeout $((DURATION + 60)) '$AFL_ROOT/afl-fuzz' \
                 -i '$SEEDS_DIR' \
                 -o '$AFL_OUT' \
                 -x '$DICT_FILE' \
                 -t 2000 \
                 -V $DURATION \
                 -s $SEED \
                 -- '$TARGET_BIN' @@"

        launch_job "$RUN_ID" "$CMD"
    done
done

TOTAL_LAUNCHED=${#PIDS[@]}
echo ""
echo "============================================"
echo "  All $TOTAL_LAUNCHED jobs launched"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/*.log"
echo "  bash scripts/status_differential.sh"
echo ""

# ── Write campaign manifest ───────────────────────────────────────────────
AFL_VERSION=$("$AFL_ROOT/afl-fuzz" 2>&1 | head -1 || echo "unknown")
cat > "$TEL_DIR/campaign_manifest.json" << MANIFESTEOF
{
  "experiment": "differential_fuzzing_libxml2",
  "started": "$(date -Iseconds)",
  "duration_seconds": $DURATION,
  "max_parallel_jobs": $MAX_JOBS,
  "afl_version": "$AFL_VERSION",
  "targets": {
    "xml005_buggy": {"tag": "v2.9.4", "cve": "CVE-2017-5130", "binary": "$TARGETS_DIR/xml005_buggy/target"},
    "xml005_fixed": {"tag": "v2.9.5", "cve": "CVE-2017-5130 (fixed)", "binary": "$TARGETS_DIR/xml005_fixed/target"},
    "xml017_buggy": {"tag": "v2.9.3", "cve": "CVE-2016-1762", "binary": "$TARGETS_DIR/xml017_buggy/target"},
    "xml017_fixed": {"tag": "v2.9.4", "cve": "CVE-2016-1762 (fixed)", "binary": "$TARGETS_DIR/xml017_fixed/target"}
  },
  "seeds_dir": "$SEEDS_DIR",
  "dictionary": "$DICT_FILE",
  "random_seeds": [1, 2, 3],
  "telemetry_runs": $(( ${#TARGETS[@]} * ${#SEEDS[@]} )),
  "baseline_runs": $(( ${#TARGETS[@]} * ${#SEEDS[@]} )),
  "total_runs": $(( ${#TARGETS[@]} * ${#SEEDS[@]} * 2 )),
  "output_dir": "$TEL_DIR"
}
MANIFESTEOF
echo "[+] Manifest: $TEL_DIR/campaign_manifest.json"

# ── Wait for all jobs ─────────────────────────────────────────────────────
echo ""
echo "Waiting for all $TOTAL_LAUNCHED jobs to complete..."
echo "(Expected: ~${DURATION}s = $(echo "scale=1; $DURATION/3600" | bc)h)"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null || true
    # Check if outputs exist
    JOB="${JOBS[$i]}"
    echo "  [finished] $JOB"
done

# ── Post-campaign: collect baseline stats from fuzzer_stats ───────────────
echo ""
echo "=== Collecting baseline stats ==="

for TARGET_NAME in "${TARGETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_ID="baseline_${TARGET_NAME}_seed${SEED}"
        AFL_OUT="$TEL_DIR/afl_out_${RUN_ID}"
        STATS_FILE="$AFL_OUT/default/fuzzer_stats"
        CSV_FILE="$TEL_DIR/${RUN_ID}.csv"

        if [[ -f "$STATS_FILE" ]]; then
            # Extract key metrics
            EXECS=$(grep "execs_done" "$STATS_FILE" | awk '{print $3}')
            BITMAP=$(grep "bitmap_cvg" "$STATS_FILE" | awk '{print $3}' | tr -d '%')
            CRASHES=$(grep "saved_crashes" "$STATS_FILE" | awk '{print $3}')
            EDGES=$(echo "$BITMAP * 65536 / 100" | bc -l 2>/dev/null | cut -d. -f1)
            [[ -z "$EDGES" ]] && EDGES=0

            echo "  $RUN_ID: execs=$EXECS edges=$EDGES crashes=$CRASHES"

            # Write summary CSV from plot_data if available
            PLOT_DATA="$AFL_OUT/default/plot_data"
            if [[ -f "$PLOT_DATA" ]]; then
                echo "step,coverage,crashes,elapsed_seconds" > "$CSV_FILE"
                # plot_data format: unix_time, cycles_done, cur_item, paths_total,
                #                   pending_total, pending_favs, map_size, saved_crashes,
                #                   saved_hangs, max_depth, execs_per_sec, total_execs, edges_found
                tail -n+2 "$PLOT_DATA" | while IFS=', ' read -r ts cycles cur paths pend pfav mapsize cr hangs depth eps execs edges rest; do
                    ELAPSED=$((ts - $(head -2 "$PLOT_DATA" | tail -1 | cut -d, -f1 | tr -d ' ')))
                    echo "$execs,$edges,$cr,$ELAPSED"
                done >> "$CSV_FILE"
            fi
        else
            echo "  $RUN_ID: NO fuzzer_stats (may have failed — check $LOG_DIR/${RUN_ID}.log)"
        fi
    done
done

# ── Final summary ─────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Campaign Complete"
echo "============================================"
echo ""
echo "Telemetry CSVs:"
for F in "$TEL_DIR"/coverage_dynamics_*.csv; do
    [[ -f "$F" ]] && echo "  $(basename "$F") ($(wc -l < "$F") rows)"
done
echo ""
echo "Mutation CSVs:"
for F in "$TEL_DIR"/mutation_attribution_*.csv; do
    [[ -f "$F" ]] && echo "  $(basename "$F") ($(wc -l < "$F") rows)"
done
echo ""
echo "Baseline CSVs:"
for F in "$TEL_DIR"/baseline_*.csv; do
    [[ -f "$F" ]] && echo "  $(basename "$F") ($(wc -l < "$F") rows)"
done
echo ""
echo "Snapshots:"
for D in "$TEL_DIR"/snapshots_*/; do
    [[ -d "$D" ]] && echo "  $(basename "$D")/ ($(ls "$D"/*.bin 2>/dev/null | wc -l) files)"
done
echo ""
echo "Logs: $LOG_DIR/"
echo "Manifest: $TEL_DIR/campaign_manifest.json"
echo ""
echo "Completed: $(date -Iseconds)"
echo ""
echo "Next: python3 scripts/analysis/differential_analysis.py \\"
echo "        --telemetry-dir experiments/differential/telemetry \\"
echo "        --output-dir experiments/differential/analysis"
