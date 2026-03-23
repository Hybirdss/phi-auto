#!/bin/bash
# phi-auto: Start autonomous experiment loop
# Usage: bash scripts/experiment.sh [--hours 2] [--exp-min 10]

cd "$(dirname "$0")/.."

echo "=== phi-auto Autonomous Experiment ==="
echo "Starting at $(date)"
echo ""

# default: 2 hours total, 10 min per experiment
HOURS=${1:-2}
EXP_MIN=${2:-10}

python3 -u src/agent/experiment.py --hours "$HOURS" --exp-min "$EXP_MIN" 2>&1 | tee logs/experiment_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Finished at $(date)"
