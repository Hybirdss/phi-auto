"""
SPIN: Self-Play Fine-Tuning for phi-auto.
Each iteration: model generates responses, then trains to prefer
human/training data over its own generations using DPO loss.

From "Self-Play Fine-Tuning Converts Weak Language Models to Strong
Language Models" (Zhu et al., 2024, UCLA).

Key advantage over STaR: no external reward model needed.
2-3 iterations typically saturate improvements.
"""

import os
import sys
import time
import json
import math
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.engine.model import GPT, GPTConfig, softmax
from src.engine.tokenizer import ByteBPETokenizer
from src.engine.optim import Lion
from src.engine.train import collect_param_grads
from src.tools.checkpoint import save_model, load_model
from src.tools.logger import update_devlog
from src.tools.monitor import ResourceGuard


def get_log_probs(model, tokens, prompt_len):
    """Get log probabilities for tokens after prompt_len."""
    seq_len = model.config.seq_len
    tokens = tokens[:seq_len]

    if len(tokens) < 2:
        return 0.0

    x = np.array([tokens[:-1]], dtype=np.int32)
    y = np.array([tokens[1:]], dtype=np.int32)

    logits, _ = model.forward(x)
    probs = softmax(logits[0], axis=-1)

    # only score the response part (after prompt)
    log_p = 0.0
    count = 0
    for i in range(max(0, prompt_len - 1), len(tokens) - 1):
        p = probs[i, tokens[i + 1]]
        log_p += np.log(max(p, 1e-9))
        count += 1

    return log_p / max(count, 1)


def dpo_loss(log_p_chosen, log_p_rejected, beta=0.1):
    """DPO loss: -log sigmoid(beta * (chosen - rejected))"""
    diff = beta * (log_p_chosen - log_p_rejected)
    return -np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-9)


def generate_response(model, tokenizer, prompt, max_tokens=80, temperature=0.8):
    """Generate response from model."""
    tokens = tokenizer.encode(prompt)
    idx = np.array([tokens], dtype=np.int32)
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    full_text = tokenizer.decode(output[0].tolist())
    response = full_text[len(prompt):]
    return response, output[0].tolist()


def spin_iteration(model, tokenizer, training_texts, n_pairs=50,
                   finetune_steps=100, beta=0.1, lr=1e-5):
    """
    One SPIN iteration:
    1. Sample (prompt, completion) pairs from training data
    2. Generate model's own completions for same prompts
    3. Fine-tune: prefer training data over model's generations (DPO)

    Returns stats dict.
    """
    print(f"  Generating {n_pairs} preference pairs...")

    # 1. Create preference pairs
    pairs = []
    for _ in range(n_pairs):
        text = random.choice(training_texts)
        # split into prompt + completion
        words = text.split()
        if len(words) < 10:
            continue
        split_point = random.randint(3, min(10, len(words) // 2))
        prompt = ' '.join(words[:split_point])
        human_response = ' '.join(words[split_point:])

        # generate model's response
        model_response, _ = generate_response(model, tokenizer, prompt,
                                               max_tokens=len(human_response.split()) + 10,
                                               temperature=0.9)

        prompt_tokens = tokenizer.encode(prompt)
        human_tokens = tokenizer.encode(prompt + ' ' + human_response)
        model_tokens = tokenizer.encode(prompt + ' ' + model_response)

        pairs.append({
            'prompt_len': len(prompt_tokens),
            'chosen': human_tokens,
            'rejected': model_tokens,
        })

    if not pairs:
        return {'loss': 0, 'n_pairs': 0}

    print(f"  Created {len(pairs)} pairs. Fine-tuning with DPO...")

    # 2. DPO fine-tuning
    optimizer = Lion(lr=lr, weight_decay=0.01, grad_clip=1.0)

    total_loss = 0
    n_updates = 0

    for step in range(finetune_steps):
        pair = random.choice(pairs)

        # get log probs for chosen and rejected
        log_p_chosen = get_log_probs(model, pair['chosen'], pair['prompt_len'])
        log_p_rejected = get_log_probs(model, pair['rejected'], pair['prompt_len'])

        loss = dpo_loss(log_p_chosen, log_p_rejected, beta)

        # approximate gradient: just train on chosen (maximize log P(chosen))
        # this is a simplification of full DPO — effective for tiny models
        seq_len = model.config.seq_len
        chosen = pair['chosen'][:seq_len + 1]
        if len(chosen) < 2:
            continue

        x = np.array([chosen[:-1]], dtype=np.int32)
        y = np.array([chosen[1:]], dtype=np.int32)
        _, ce_loss = model.forward(x, y)
        model.backward()

        # weight the gradient by DPO signal
        grad_pairs = collect_param_grads(model)
        # scale gradient by how much we should prefer chosen
        preference_weight = min(2.0, max(0.1, 1.0 + beta * (log_p_rejected - log_p_chosen)))
        weighted_pairs = [(p, g * preference_weight) for p, g in grad_pairs]
        optimizer.step(weighted_pairs)

        total_loss += loss
        n_updates += 1

        if (step + 1) % 25 == 0:
            avg = total_loss / n_updates
            print(f"    step {step+1}/{finetune_steps}: dpo_loss={avg:.4f} pref_w={preference_weight:.2f}")

    avg_loss = total_loss / max(n_updates, 1)
    return {
        'loss': avg_loss,
        'n_pairs': len(pairs),
        'n_updates': n_updates,
    }


def spin_loop(config=None, n_iterations=3, pairs_per_iter=50,
              finetune_steps=100, time_budget=3600):
    """
    Full SPIN loop: 2-3 iterations of self-play fine-tuning.
    """
    if config is None:
        config = {
            'vocab_size': 1024, 'n_embd': 128, 'n_head': 4, 'n_layer': 4,
            'seq_len': 128,
        }

    print("=" * 60)
    print("phi-auto SPIN (Self-Play Fine-Tuning)")
    print("=" * 60)

    # load tokenizer
    from src.data.prepare import TOKENIZER_PATH
    if not os.path.exists(TOKENIZER_PATH):
        print("ERROR: No tokenizer. Run training first.")
        return
    tokenizer = ByteBPETokenizer.load(TOKENIZER_PATH)

    # build and load model
    model_config = GPTConfig(**{k: config[k] for k in
                                ['vocab_size', 'n_embd', 'n_head', 'n_layer', 'seq_len']})
    model = GPT(model_config)
    if not load_model(model, tag="best"):
        if not load_model(model, tag="final"):
            print("ERROR: No checkpoint. Run training first.")
            return
    print("Model loaded.\n")

    # load training texts
    train_path = os.path.expanduser("~/.cache/phi-auto/data/train.jsonl")
    training_texts = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= 2000:
                break
            try:
                training_texts.append(json.loads(line)['text'])
            except:
                continue
    print(f"Loaded {len(training_texts)} training texts.\n")

    guard = ResourceGuard()
    t_start = time.time()

    for iteration in range(n_iterations):
        elapsed = time.time() - t_start
        if elapsed >= time_budget:
            print(f"\nTime budget reached.")
            break

        ok, reason = guard.check(verbose=True)
        if not ok:
            print(f"  Resource limit: {reason}")
            break

        print(f"\n--- SPIN Iteration {iteration + 1}/{n_iterations} ---")
        stats = spin_iteration(
            model, tokenizer, training_texts,
            n_pairs=pairs_per_iter,
            finetune_steps=finetune_steps,
            beta=0.1,
            lr=1e-5,
        )
        print(f"  DPO loss: {stats['loss']:.4f}, pairs: {stats['n_pairs']}, updates: {stats['n_updates']}")

        save_model(model, tag=f"spin_iter{iteration+1}",
                   metadata={'spin_iteration': iteration + 1, **stats})

    save_model(model, tag="spin_final")
    total_time = time.time() - t_start
    print(f"\nSPIN complete. {n_iterations} iterations, {total_time:.0f}s")
    update_devlog(f"SPIN: {n_iterations} iterations completed in {total_time:.0f}s")

    # sample generation
    print("\nSample outputs after SPIN:")
    for prompt in ["Once upon a time", "The little dog", "She was very"]:
        resp, _ = generate_response(model, tokenizer, prompt, max_tokens=50, temperature=0.7)
        print(f"  [{prompt}] {resp[:100]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--pairs', type=int, default=50)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--budget', type=int, default=3600)
    args = parser.parse_args()

    spin_loop(
        n_iterations=args.iterations,
        pairs_per_iter=args.pairs,
        finetune_steps=args.steps,
        time_budget=args.budget,
    )
