#!/usr/bin/env bash
# scripts/status.sh — Quick experiment status check
# Usage: bash scripts/status.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO}/.venv/bin/python3"
EXP_ROOT="${REPO}/experiments"

if [[ ! -d "$EXP_ROOT" ]]; then
    echo "No experiments directory found."
    exit 0
fi

# Check if experiment process is running
PID=$(pgrep -f "run_full_experiment" 2>/dev/null | head -1)
if [[ -n "$PID" ]]; then
    echo "  Experiment running (PID $PID)"
else
    echo "  Experiment NOT running"
fi

# Active RL server
ACTIVE=$(ps aux 2>/dev/null | grep "rl_server.py" | grep -v grep | sed -n 's/.*--model-id \([^ ]*\).*--mode \([^ ]*\).*/\1 \2/p' | head -1)
if [[ -n "$ACTIVE" ]]; then
    echo "  Active: $ACTIVE"
fi
echo ""

"$PYTHON" -c "
import pandas as pd, os, glob, sys

models = ['m1_0', 'm1_1', 'm1_2']
benchmarks = ['jsoncpp', 'freetype2', 'libxml2', 're2', 'harfbuzz', 'libpng']
exp_root = '$EXP_ROOT'

total_phases = 0
done_phases = 0

for bench in benchmarks:
    found = False
    lines = []
    for model in models:
        for phase in ['train', 'eval']:
            total_phases += 1
            csv = f'{exp_root}/{bench}/plots/{model}/rl_metrics_{model}_{phase}.csv'
            if os.path.exists(csv):
                try:
                    df = pd.read_csv(csv)
                    if len(df) > 0:
                        step = df['step'].iloc[-1]
                        cov = df['coverage'].iloc[-1]
                        hrs = df['elapsed_seconds'].iloc[-1] / 3600
                        pct = min(step / 10_000_000 * 100, 100)
                        status = 'DONE' if pct >= 99.9 else f'{pct:.0f}%'
                        if pct >= 99.9:
                            done_phases += 1
                        lines.append(f'  {model} {phase:5s}: {step:>10,}/10M  cov={cov:<6}  {hrs:.1f}h  [{status}]')
                        found = True
                except:
                    pass

    # Baselines
    for tag in ['baseline', 'baseline_time']:
        csv = f'{exp_root}/{bench}/plots/{tag}/rl_metrics_{tag}_eval.csv'
        if os.path.exists(csv):
            try:
                df = pd.read_csv(csv)
                if len(df) > 0:
                    step = df['step'].iloc[-1]
                    cov = df['coverage'].iloc[-1]
                    hrs = df['elapsed_seconds'].iloc[-1] / 3600
                    lines.append(f'  {tag:13s}: step={step:>12,}  cov={cov:<6}  {hrs:.1f}h  [DONE]')
                    found = True
            except:
                pass

    # Milestones
    ms_done = []
    for tag in ['500k', '1m', '2m', '10m']:
        ms_dir = f'{exp_root}/{bench}/milestones/{tag}'
        if os.path.isdir(ms_dir) and os.path.exists(f'{ms_dir}/comparison_report.txt'):
            ms_done.append(tag)

    if found:
        print(f'--- {bench} ---')
        for l in lines:
            print(l)
        if ms_done:
            print(f'  milestones: {\" \".join(ms_done)}')
        print()
    else:
        print(f'--- {bench} --- (not started)')
        print()

# Summary
pct = done_phases / total_phases * 100 if total_phases > 0 else 0
print(f'Overall: {done_phases}/{total_phases} phases done ({pct:.0f}%)')

# Check for summary
summary = f'{exp_root}/summary/cross_benchmark_report.txt'
if os.path.exists(summary):
    print(f'Cross-benchmark summary: {summary}')
" 2>/dev/null
