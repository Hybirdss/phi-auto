#!/bin/bash
# phi-auto: Start self-improvement loop
# Usage: bash scripts/self_improve.sh [cycles] [budget_seconds]

cd "$(dirname "$0")/.."

echo "=== phi-auto Self-Improvement ==="
echo "Starting at $(date)"
echo ""

CYCLES=${1:-10}
BUDGET=${2:-3600}

python3 -u src/agent/self_improve.py --cycles "$CYCLES" --budget "$BUDGET" 2>&1 | tee logs/self_improve_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Finished at $(date)"
