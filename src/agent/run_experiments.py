"""
phi-auto Autoresearch Experiment Runner.
Runs a sequence of experiments, compares results, keeps the best.
Inspired by karpathy/autoresearch: modify → train → eval → keep/discard → log.
"""

import os
import sys
import time
import json
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.engine.train import train
from src.tools.logger import update_devlog


# ──────────────────────────────────────────────────────────────
# Experiment Configs
# ──────────────────────────────────────────────────────────────

BASELINE = {
    'name': 'EXP-001: Baseline (Lion, 2000 steps)',
    'vocab_size': 1024, 'n_embd': 128, 'n_head': 4, 'n_layer': 4,
    'seq_len': 128, 'batch_size': 4, 'grad_accum': 2,
    'lr': 3e-4, 'min_lr': 3e-5, 'warmup_steps': 50,
    'max_steps': 2000, 'eval_interval': 200, 'eval_steps': 10,
    'log_interval': 50, 'weight_decay': 0.1, 'grad_clip': 1.0,
    'optimizer': 'lion', 'time_budget': 1800,
}

EXPERIMENTS = [
    {
        'name': 'EXP-002: AdamW, 6000 steps (1.5 epochs)',
        'changes': {
            'optimizer': 'adamw',
            'max_steps': 6000,
            'lr': 6e-4,
            'min_lr': 6e-5,
            'warmup_steps': 100,
            'time_budget': 3600,
            'eval_interval': 300,
        },
        'hypothesis': 'AdamW with proper training length should beat Lion plateau. '
                      '6000 steps = 1.5 epochs = model sees all data.',
    },
    {
        'name': 'EXP-003: AdamW + weight-tied + bigger batch',
        'changes': {
            'optimizer': 'adamw',
            'max_steps': 6000,
            'lr': 1e-3,
            'min_lr': 1e-4,
            'warmup_steps': 200,
            'batch_size': 8,
            'grad_accum': 2,
            'weight_tie': True,
            'time_budget': 3600,
            'eval_interval': 300,
        },
        'hypothesis': 'Larger effective batch (16) + higher lr + weight tying. '
                      'Fewer param + more data per step = faster convergence.',
    },
    {
        'name': 'EXP-004: Scale up (192 dim, 6 layers)',
        'changes': {
            'optimizer': 'adamw',
            'n_embd': 192,
            'n_head': 6,
            'n_layer': 6,
            'max_steps': 4000,
            'lr': 6e-4,
            'min_lr': 6e-5,
            'warmup_steps': 150,
            'batch_size': 4,
            'grad_accum': 2,
            'weight_tie': True,
            'time_budget': 3600,
            'eval_interval': 200,
        },
        'hypothesis': '3x more parameters (3.5M). If EXP-002/003 plateau, '
                      'the bottleneck is model capacity not training.',
    },
]


def make_config(base, changes):
    """Merge experiment changes into base config."""
    cfg = copy.deepcopy(base)
    cfg.update(changes)
    return cfg


def run_experiment(exp_cfg, exp_num):
    """Run a single experiment and return results."""
    name = exp_cfg.pop('name', f'EXP-{exp_num:03d}')
    hypothesis = exp_cfg.pop('hypothesis', '')
    changes = exp_cfg.pop('changes', {})

    config = make_config(BASELINE, changes)
    config.pop('name', None)

    print("\n" + "=" * 70)
    print(f"  {name}")
    print(f"  Hypothesis: {hypothesis}")
    print("=" * 70)

    # key config diffs from baseline
    diffs = []
    for k, v in changes.items():
        if k in BASELINE and BASELINE[k] != v:
            diffs.append(f"  {k}: {BASELINE[k]} → {v}")
        elif k not in BASELINE:
            diffs.append(f"  {k}: (new) {v}")
    if diffs:
        print("Config changes:")
        print("\n".join(diffs))
    print()

    t0 = time.time()
    try:
        model, val_loss, val_bpb = train(config)
        elapsed = time.time() - t0

        result = {
            'name': name,
            'hypothesis': hypothesis,
            'val_loss': val_loss,
            'val_bpb': val_bpb,
            'time': elapsed,
            'status': 'completed',
            'config': {k: v for k, v in config.items()
                       if k in changes or k in ['n_embd', 'n_layer', 'optimizer', 'max_steps']},
        }
        print(f"\n{'='*70}")
        print(f"  RESULT: val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f}  ({elapsed:.0f}s)")
        print(f"{'='*70}")
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED: {e}")
        return {
            'name': name,
            'status': 'failed',
            'error': str(e),
            'time': elapsed,
        }


def run_all(skip_baseline=True):
    """Run all experiments sequentially."""
    results = []

    # results file
    results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'experiment_results.json')

    # load existing results
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    best_bpb = min((r['val_bpb'] for r in results if r.get('status') == 'completed'),
                    default=float('inf'))

    for i, exp in enumerate(EXPERIMENTS):
        exp_copy = copy.deepcopy(exp)
        result = run_experiment(exp_copy, i + 2)
        results.append(result)

        # save incrementally
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # compare with best
        if result.get('status') == 'completed':
            bpb = result['val_bpb']
            if bpb < best_bpb:
                improvement = (best_bpb - bpb) / best_bpb * 100
                print(f"\n  >>> NEW BEST! val_bpb {best_bpb:.4f} → {bpb:.4f} ({improvement:.1f}% better)")
                best_bpb = bpb
                update_devlog(f"{result['name']}: val_bpb={bpb:.4f} — NEW BEST ({improvement:.1f}% improvement)")
            else:
                print(f"\n  >>> No improvement: {bpb:.4f} vs best {best_bpb:.4f}")
                update_devlog(f"{result['name']}: val_bpb={bpb:.4f} — no improvement over {best_bpb:.4f}")

    # summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    for r in results:
        if r.get('status') == 'completed':
            mark = " ★" if r['val_bpb'] == best_bpb else ""
            print(f"  {r['name']}: val_bpb={r['val_bpb']:.4f} ({r['time']:.0f}s){mark}")
        else:
            print(f"  {r['name']}: FAILED — {r.get('error', 'unknown')}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, help='Run specific experiment number (2-4)')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    args = parser.parse_args()

    if args.exp:
        idx = args.exp - 2
        if 0 <= idx < len(EXPERIMENTS):
            run_experiment(copy.deepcopy(EXPERIMENTS[idx]), args.exp)
        else:
            print(f"Experiment {args.exp} not found (valid: 2-{len(EXPERIMENTS)+1})")
    else:
        run_all()
