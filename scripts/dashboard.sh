#!/bin/bash
# phi-auto: Live training dashboard
# Usage: bash scripts/dashboard.sh

cd "$(dirname "$0")/.."

echo "=== phi-auto Dashboard ==="
echo ""

# System status
python3 -c "
from src.tools.monitor import get_system_snapshot, format_snapshot, get_cpu_usage
snap = get_system_snapshot()
print('System:', format_snapshot(snap))
print(f'CPU load: {get_cpu_usage(0.3):.1f}%')
"

echo ""

# Experiment results
python3 -c "
from src.tools.logger import ExperimentLogger
from src.tools.checkpoint import list_checkpoints

logger = ExperimentLogger()
results = logger.get_all_results()
best_exp, best_bpb = logger.get_best_result()

print(f'Experiments: {len(results)} total')
if best_exp:
    print(f'Best: exp #{best_exp} val_bpb={best_bpb:.4f}')
else:
    print('Best: (no results yet)')

ckpts = list_checkpoints()
print(f'Checkpoints: {len(ckpts)}')
for c in ckpts[:3]:
    print(f'  {c[\"file\"]} ({c[\"size_mb\"]} MB)')

if results:
    print()
    print('Recent experiments:')
    for r in results[-5:]:
        print(f'  #{r.get(\"exp_id\",\"?\")} [{r.get(\"status\",\"?\")}] bpb={r.get(\"val_bpb\",\"?\")} | {r.get(\"notes\",\"\")[:50]}')
"

echo ""
echo "=== $(date) ==="
