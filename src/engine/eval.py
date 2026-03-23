"""
Evaluation utilities for phi-auto.
Computes val_bpb, perplexity, and text quality metrics.
"""

import math
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.engine.model import GPT, GPTConfig, softmax


def evaluate_loss(model, val_loader, n_steps=20):
    """Compute average loss and bits-per-byte on validation set."""
    total_loss = 0
    total_tokens = 0

    for _ in range(n_steps):
        x, y, _ = val_loader.get_batch()
        _, loss = model.forward(x, y)
        B, T = x.shape
        total_loss += loss * B * T
        total_tokens += B * T

    avg_loss = total_loss / max(total_tokens, 1)
    # bits per byte: nats / (avg_bytes_per_token * ln(2))
    # for BPE with ~1.2 bytes/token average
    bpb = avg_loss / (1.2 * math.log(2))
    return avg_loss, bpb


def compute_perplexity_batch(model, tokenizer, texts, max_len=None):
    """Compute perplexity for a batch of texts."""
    if max_len is None:
        max_len = model.config.seq_len

    perplexities = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:max_len + 1]

        x = np.array([tokens[:-1]], dtype=np.int32)
        y = np.array([tokens[1:]], dtype=np.int32)
        _, loss = model.forward(x, y)

        ppl = math.exp(min(loss, 20))
        perplexities.append(ppl)

    return perplexities


def evaluate_generation_quality(model, tokenizer, prompts, max_tokens=80, temperature=0.8):
    """
    Evaluate text generation quality.
    Returns dict with metrics.
    """
    results = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        idx = np.array([tokens], dtype=np.int32)
        output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
        text = tokenizer.decode(output[0].tolist())
        generated = text[len(prompt):]

        # metrics
        words = generated.split()
        n_words = len(words)
        unique_words = len(set(words))
        diversity = unique_words / max(n_words, 1)

        # repetition check
        trigrams = [' '.join(words[i:i+3]) for i in range(max(0, len(words) - 2))]
        from collections import Counter
        trigram_counts = Counter(trigrams)
        max_repeat = trigram_counts.most_common(1)[0][1] if trigram_counts else 0

        # ascii ratio
        ascii_ratio = sum(1 for c in generated if ord(c) < 128) / max(len(generated), 1)

        results.append({
            'prompt': prompt,
            'generated': generated[:200],
            'n_words': n_words,
            'diversity': diversity,
            'max_trigram_repeat': max_repeat,
            'ascii_ratio': ascii_ratio,
            'is_coherent': diversity > 0.3 and max_repeat < 4 and ascii_ratio > 0.8,
        })

    coherent = sum(1 for r in results if r['is_coherent'])
    avg_diversity = sum(r['diversity'] for r in results) / max(len(results), 1)

    return {
        'samples': results,
        'coherence_rate': coherent / max(len(results), 1),
        'avg_diversity': avg_diversity,
        'n_evaluated': len(results),
    }


def full_evaluation(model, tokenizer, val_loader, n_eval_steps=20):
    """Run complete evaluation suite."""
    print("Running full evaluation...")

    # loss & bpb
    val_loss, val_bpb = evaluate_loss(model, val_loader, n_eval_steps)
    print(f"  val_loss: {val_loss:.4f}")
    print(f"  val_bpb:  {val_bpb:.4f}")

    # generation quality
    prompts = [
        "Once upon a time",
        "The little dog",
        "She was very happy",
        "One day, a boy named",
        "In the garden",
    ]
    gen_results = evaluate_generation_quality(model, tokenizer, prompts)
    print(f"  coherence: {gen_results['coherence_rate']:.1%}")
    print(f"  diversity: {gen_results['avg_diversity']:.2f}")

    for s in gen_results['samples'][:2]:
        print(f"  [{s['prompt']}] -> {s['generated'][:80]}...")

    return {
        'val_loss': val_loss,
        'val_bpb': val_bpb,
        'coherence_rate': gen_results['coherence_rate'],
        'diversity': gen_results['avg_diversity'],
    }
