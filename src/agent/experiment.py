"""
Autonomous experiment loop for phi-auto.
Karpathy-style: modify config -> train -> eval -> keep/discard -> repeat.
"""

import os
import sys
import time
import copy
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.engine.model import GPT, GPTConfig
from src.engine.tokenizer import ByteBPETokenizer
from src.engine.train import SimpleAdamW, collect_param_grads, evaluate, get_lr
from src.data.loader import DataLoader
from src.data.prepare import prepare_all
from src.tools.logger import ExperimentLogger, update_devlog
from src.tools.checkpoint import save_model, load_model, copy_checkpoint
from src.tools.monitor import ResourceGuard, format_snapshot, get_system_snapshot
from src.agent.hypothesis import HypothesisGenerator, estimate_params, estimate_memory_mb


# ---------------------------------------------------------------------------
# Single Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(config, tokenizer, train_path, val_path, time_budget=600):
    """Run a single training experiment. Returns results dict."""
    t_start = time.time()

    # build model
    model_config = GPTConfig(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        seq_len=config['seq_len'],
    )
    model = GPT(model_config)

    # data loaders
    train_loader = DataLoader(train_path, tokenizer, config['batch_size'], config['seq_len'])
    val_loader = DataLoader(val_path, tokenizer, config['batch_size'], config['seq_len'], shuffle=False)

    # optimizer
    optimizer = SimpleAdamW(
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.1),
        grad_clip=config.get('grad_clip', 1.0),
    )

    max_steps = config.get('max_steps', 500)
    grad_accum = config.get('grad_accum', 2)
    warmup_steps = config.get('warmup_steps', 50)
    log_interval = config.get('log_interval', 50)
    eval_interval = config.get('eval_interval', 100)
    eval_steps = config.get('eval_steps', 10)

    best_val_loss = float('inf')
    last_tok_sec = 0

    for step in range(max_steps):
        t0 = time.time()

        # LR schedule
        lr = get_lr(step, warmup_steps, max_steps, config['lr'], config.get('min_lr', 3e-5))
        optimizer.lr = lr

        # gradient accumulation
        total_loss = 0
        for micro_step in range(grad_accum):
            x, y, epoch = train_loader.get_batch()
            _, loss = model.forward(x, y)
            model.backward()

            if micro_step == 0:
                accum_pairs = collect_param_grads(model)
                accum_grads = [g.copy() for _, g in accum_pairs]
            else:
                pairs = collect_param_grads(model)
                for i, (_, g) in enumerate(pairs):
                    accum_grads[i] += g
            total_loss += loss

        for i in range(len(accum_grads)):
            accum_grads[i] /= grad_accum

        final_pairs = [(p, accum_grads[i]) for i, (p, _) in enumerate(accum_pairs)]
        optimizer.step(final_pairs)

        avg_loss = total_loss / grad_accum
        dt = time.time() - t0
        elapsed = time.time() - t_start
        last_tok_sec = config['batch_size'] * grad_accum * config['seq_len'] / dt

        if step % log_interval == 0:
            print(f"  step {step:4d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                  f"{last_tok_sec:.0f} tok/s | {elapsed:.0f}s")

        # eval
        if step > 0 and step % eval_interval == 0:
            val_loss, val_bpb = evaluate(model, val_loader, eval_steps)
            print(f"  >>> val_loss: {val_loss:.4f} | val_bpb: {val_bpb:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # time budget
        if elapsed >= time_budget:
            print(f"  Time budget reached ({time_budget}s)")
            break

    # final eval
    val_loss, val_bpb = evaluate(model, val_loader, eval_steps)
    total_time = time.time() - t_start

    # RAM snapshot
    snap = get_system_snapshot()

    train_loader.close()
    val_loader.close()

    return {
        'model': model,
        'val_loss': val_loss,
        'val_bpb': val_bpb,
        'best_loss': best_val_loss,
        'tok_sec': last_tok_sec,
        'ram_mb': snap['ram_used_mb'],
        'steps': step + 1,
        'time_sec': total_time,
        'n_params': model.n_params,
    }


# ---------------------------------------------------------------------------
# Autonomous Experiment Loop
# ---------------------------------------------------------------------------

def experiment_loop(base_config=None, total_budget_hours=8, experiment_budget_min=10):
    """
    Autonomous experiment loop.
    Runs experiments, keeps improvements, discards failures.
    """
    if base_config is None:
        base_config = {
            'vocab_size': 1024,
            'n_embd': 128,
            'n_head': 4,
            'n_layer': 4,
            'seq_len': 128,
            'batch_size': 4,
            'grad_accum': 2,
            'lr': 3e-4,
            'min_lr': 3e-5,
            'warmup_steps': 50,
            'max_steps': 500,
            'eval_interval': 100,
            'eval_steps': 10,
            'log_interval': 50,
            'weight_decay': 0.1,
            'grad_clip': 1.0,
        }

    print("=" * 60)
    print("phi-auto Autonomous Experiment Loop")
    print("=" * 60)

    # setup
    logger = ExperimentLogger()
    guard = ResourceGuard(max_ram_pct=85, min_battery_pct=15, max_cpu_temp=75)
    hypothesis_gen = HypothesisGenerator()

    # prepare data
    print("\n[1] Preparing data...")
    tokenizer, train_path, val_path = prepare_all(
        vocab_size=base_config['vocab_size'],
        max_stories=20000,
    )

    # check if we have a baseline
    best_exp, best_bpb = logger.get_best_result()
    current_best_config = copy.deepcopy(base_config)
    current_best_bpb = best_bpb if best_bpb < float('inf') else None

    total_budget_sec = total_budget_hours * 3600
    exp_budget_sec = experiment_budget_min * 60
    loop_start = time.time()
    exp_count = 0

    print(f"\n[2] Starting experiment loop")
    print(f"  Total budget: {total_budget_hours}h")
    print(f"  Per-experiment: {experiment_budget_min}min")
    if current_best_bpb:
        print(f"  Current best bpb: {current_best_bpb:.4f}")
    print()

    # --- Experiment 0: Baseline (if no prior results) ---
    if current_best_bpb is None:
        print("=" * 40)
        print("Experiment 0: BASELINE")
        print("=" * 40)

        exp_id = logger.next_id()
        try:
            results = run_experiment(
                base_config, tokenizer, train_path, val_path,
                time_budget=exp_budget_sec
            )
            current_best_bpb = results['val_bpb']
            current_best_config = copy.deepcopy(base_config)
            logger.log_experiment(exp_id, base_config, results,
                                  status="baseline", notes="initial baseline")
            save_model(results['model'], tag="baseline",
                       metadata={'val_bpb': current_best_bpb, 'exp_id': exp_id})
            save_model(results['model'], tag="best",
                       metadata={'val_bpb': current_best_bpb, 'exp_id': exp_id})
            update_devlog(
                f"Baseline established: val_bpb={current_best_bpb:.4f}, "
                f"params={results['n_params']:,}, tok/s={results['tok_sec']:.0f}"
            )
            print(f"\n>>> BASELINE: val_bpb={current_best_bpb:.4f}")
        except Exception as e:
            print(f"Baseline failed: {e}")
            traceback.print_exc()
            logger.log_experiment(exp_id, base_config, {}, status="failed", notes=str(e))
            return

        exp_count += 1

    # --- Main Loop ---
    while True:
        elapsed = time.time() - loop_start
        if elapsed >= total_budget_sec:
            print(f"\n>>> Total time budget reached ({total_budget_hours}h)")
            break

        # resource check
        ok, reason = guard.check(verbose=True)
        if not ok:
            print(f"\n>>> Resource limit: {reason}")
            if 'temp' in reason.lower() or 'hot' in reason.lower():
                guard.wait_for_cooldown()
                continue
            else:
                print("  Pausing 60s...")
                time.sleep(60)
                continue

        # generate hypothesis
        experiments = hypothesis_gen.generate(current_best_config, n=1)
        if not experiments:
            print("  No valid hypothesis generated, trying random...")
            experiments = hypothesis_gen.generate(current_best_config, n=1)
            if not experiments:
                continue

        exp = experiments[0]
        exp_config = exp['config']
        exp_count += 1
        exp_id = logger.next_id()

        remaining = total_budget_sec - elapsed
        this_budget = min(exp_budget_sec, remaining - 60)
        if this_budget < 120:
            print("  Not enough time for another experiment.")
            break

        print("\n" + "=" * 40)
        print(f"Experiment {exp_id}: {exp['strategy'].upper()}")
        print(f"  {exp['hypothesis']}")
        print(f"  est_params: {exp['est_params']:,}  est_mem: {exp['est_memory_mb']}MB")
        print(f"  budget: {this_budget:.0f}s  remaining: {remaining:.0f}s")
        print("=" * 40)

        try:
            results = run_experiment(
                exp_config, tokenizer, train_path, val_path,
                time_budget=this_budget
            )

            new_bpb = results['val_bpb']
            improved = new_bpb < current_best_bpb

            if improved:
                improvement = ((current_best_bpb - new_bpb) / current_best_bpb) * 100
                print(f"\n>>> IMPROVEMENT! val_bpb: {current_best_bpb:.4f} -> {new_bpb:.4f} "
                      f"({improvement:.1f}% better)")
                current_best_bpb = new_bpb
                current_best_config = copy.deepcopy(exp_config)
                save_model(results['model'], tag="best",
                           metadata={'val_bpb': new_bpb, 'exp_id': exp_id})
                status = "improved"
                update_devlog(
                    f"Exp {exp_id} [{exp['strategy']}]: IMPROVED val_bpb "
                    f"{current_best_bpb:.4f} -> {new_bpb:.4f} ({improvement:.1f}%)"
                )
            else:
                print(f"\n>>> No improvement. val_bpb: {new_bpb:.4f} vs best: {current_best_bpb:.4f}")
                status = "no_improvement"

            hypothesis_gen.record(exp_config, new_bpb, exp['strategy'])
            logger.log_experiment(exp_id, exp_config, results,
                                  status=status, notes=exp['hypothesis'])

        except Exception as e:
            print(f"\n>>> Experiment {exp_id} FAILED: {e}")
            traceback.print_exc()
            logger.log_experiment(exp_id, exp_config, {},
                                  status="failed", notes=str(e))
            hypothesis_gen.record(exp_config, float('inf'), exp['strategy'])

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Experiment Loop Complete")
    print("=" * 60)
    print(f"  Experiments run: {exp_count}")
    print(f"  Best val_bpb: {current_best_bpb:.4f}")
    print(f"  Total time: {time.time() - loop_start:.0f}s")

    all_results = logger.get_all_results()
    improvements = [r for r in all_results if r.get('status') == 'improved']
    print(f"  Improvements found: {len(improvements)}")

    update_devlog(
        f"Experiment loop done: {exp_count} experiments, "
        f"best_bpb={current_best_bpb:.4f}, improvements={len(improvements)}"
    )

    return current_best_config, current_best_bpb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=2, help='Total budget in hours')
    parser.add_argument('--exp-min', type=int, default=10, help='Per-experiment budget in minutes')
    args = parser.parse_args()

    experiment_loop(
        total_budget_hours=args.hours,
        experiment_budget_min=args.exp_min,
    )
