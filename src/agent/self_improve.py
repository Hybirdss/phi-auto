"""
Self-improvement loop for phi-auto.
STaR (Self-Taught Reasoner) + Reflect-Retry approach.
The model generates training data, verifies it, and fine-tunes on correct outputs.
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
from src.engine.train import SimpleAdamW, collect_param_grads, evaluate, get_lr
from src.data.loader import DataLoader
from src.tools.logger import ExperimentLogger, update_devlog
from src.tools.checkpoint import save_model, load_model
from src.tools.monitor import ResourceGuard


# ---------------------------------------------------------------------------
# Task Definitions (for self-improvement)
# ---------------------------------------------------------------------------

STORY_PROMPTS = [
    "Once upon a time, there was a",
    "A little girl named Lily",
    "One sunny day, a boy",
    "In a big garden, there",
    "The friendly dog loved to",
    "A small bird sat on a",
    "There was a happy family",
    "One morning, the cat",
    "The children went to the",
    "A kind old woman had a",
]

COMPLETION_TASKS = [
    # (prompt, expected_pattern) - for verification
    ("Once upon a time", None),  # open-ended, check fluency
    ("The dog", None),
    ("She was very", None),
    ("They played in the", None),
    ("One day, a little", None),
]


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity of text under the model."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float('inf')

    seq_len = model.config.seq_len
    tokens = tokens[:seq_len + 1]

    x = np.array([tokens[:-1]], dtype=np.int32)
    y = np.array([tokens[1:]], dtype=np.int32)

    logits, loss = model.forward(x, y)
    return math.exp(loss) if loss < 20 else float('inf')


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8):
    """Generate text from a prompt."""
    tokens = tokenizer.encode(prompt)
    idx = np.array([tokens], dtype=np.int32)
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    return tokenizer.decode(output[0].tolist())


# ---------------------------------------------------------------------------
# Quality Verification
# ---------------------------------------------------------------------------

def verify_generation(text, min_len=30, max_repeats=3):
    """
    Basic quality checks on generated text.
    Returns (is_good, reason).
    """
    if len(text) < min_len:
        return False, "too_short"

    # check for excessive repetition
    words = text.split()
    if len(words) < 5:
        return False, "too_few_words"

    # check for word-level repetition
    for i in range(len(words) - max_repeats):
        if len(set(words[i:i + max_repeats + 1])) == 1:
            return False, "word_repetition"

    # check for phrase repetition (3-gram)
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    from collections import Counter
    trigram_counts = Counter(trigrams)
    if trigram_counts and trigram_counts.most_common(1)[0][1] > max_repeats:
        return False, "phrase_repetition"

    # check for non-ASCII garbage
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.8:
        return False, "too_much_garbage"

    # check for reasonable word length
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > 15:
        return False, "garbled_words"

    return True, "ok"


def score_generation(model, tokenizer, text):
    """
    Score a generation for training value.
    Higher = better training signal.
    """
    ppl = compute_perplexity(model, tokenizer, text)
    is_good, reason = verify_generation(text)

    if not is_good:
        return 0.0, reason

    # sweet spot: not too predictable (boring), not too surprising (garbage)
    # perplexity between 5-100 is ideal
    if ppl < 2:
        score = 0.3  # too predictable
    elif ppl < 10:
        score = 1.0  # high quality
    elif ppl < 50:
        score = 0.7  # decent
    elif ppl < 200:
        score = 0.4  # borderline
    else:
        score = 0.1  # probably garbage

    return score, f"ppl={ppl:.1f}"


# ---------------------------------------------------------------------------
# STaR: Self-Taught Reasoner Cycle
# ---------------------------------------------------------------------------

def star_cycle(model, tokenizer, n_samples=20, temperature=0.8, min_score=0.5):
    """
    One STaR cycle:
    1. Generate completions from prompts
    2. Verify & score them
    3. Keep high-quality ones as training data
    Returns list of (text, score) pairs that passed verification.
    """
    print(f"  STaR: Generating {n_samples} samples...")
    good_data = []
    total_tried = 0

    prompts = random.choices(STORY_PROMPTS, k=n_samples)

    for prompt in prompts:
        text = generate_text(model, tokenizer, prompt, max_tokens=80, temperature=temperature)
        total_tried += 1

        score, reason = score_generation(model, tokenizer, text)
        if score >= min_score:
            good_data.append((text, score))

    success_rate = len(good_data) / max(total_tried, 1)
    print(f"  STaR: {len(good_data)}/{total_tried} passed (rate={success_rate:.1%})")

    return good_data, success_rate


# ---------------------------------------------------------------------------
# Reflect-Retry: Learn from Failures
# ---------------------------------------------------------------------------

def reflect_retry(model, tokenizer, failed_prompt, n_retries=3, temperatures=None):
    """
    When a generation fails:
    1. Try again with different temperatures
    2. If any retry succeeds, return it
    """
    if temperatures is None:
        temperatures = [0.6, 0.9, 1.1]

    for temp in temperatures[:n_retries]:
        text = generate_text(model, tokenizer, failed_prompt, max_tokens=80, temperature=temp)
        is_good, reason = verify_generation(text)
        if is_good:
            return text, temp
    return None, None


# ---------------------------------------------------------------------------
# Fine-tune on Self-Generated Data
# ---------------------------------------------------------------------------

def finetune_on_data(model, tokenizer, texts, steps=100, lr=1e-4):
    """
    Fine-tune model on self-generated high-quality texts.
    Quick LoRA-free fine-tuning (just update all params with low LR).
    """
    if not texts:
        return 0.0

    optimizer = SimpleAdamW(lr=lr, weight_decay=0.01, grad_clip=1.0)

    total_loss = 0
    n_updates = 0

    for step in range(steps):
        # pick random text
        text = random.choice(texts)
        tokens = tokenizer.encode(text)

        seq_len = model.config.seq_len
        if len(tokens) < 3:
            continue

        tokens = tokens[:seq_len + 1]
        x = np.array([tokens[:-1]], dtype=np.int32)
        y = np.array([tokens[1:]], dtype=np.int32)

        _, loss = model.forward(x, y)
        model.backward()

        pairs = collect_param_grads(model)
        optimizer.step(pairs)

        total_loss += loss
        n_updates += 1

    avg_loss = total_loss / max(n_updates, 1)
    return avg_loss


# ---------------------------------------------------------------------------
# Main Self-Improvement Loop
# ---------------------------------------------------------------------------

def self_improve_loop(config=None, n_cycles=10, samples_per_cycle=20,
                      finetune_steps=50, time_budget=3600):
    """
    Main self-improvement loop.
    Each cycle: generate -> verify -> fine-tune on good data.
    """
    if config is None:
        config = {
            'vocab_size': 1024, 'n_embd': 128, 'n_head': 4, 'n_layer': 4,
            'seq_len': 128, 'batch_size': 4,
        }

    print("=" * 60)
    print("phi-auto Self-Improvement Loop")
    print("=" * 60)

    # load tokenizer
    from src.data.prepare import TOKENIZER_PATH
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = ByteBPETokenizer.load(TOKENIZER_PATH)
    else:
        print("ERROR: No tokenizer found. Run training first.")
        return

    # build model and load best checkpoint
    model_config = GPTConfig(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        seq_len=config['seq_len'],
    )
    model = GPT(model_config)

    if not load_model(model, tag="best"):
        if not load_model(model, tag="final"):
            print("ERROR: No checkpoint found. Run training first.")
            return

    print("Model loaded from checkpoint.\n")

    guard = ResourceGuard()
    logger = ExperimentLogger()
    t_start = time.time()

    # track improvement
    initial_quality = None
    all_good_data = []
    cycle_stats = []

    for cycle in range(n_cycles):
        elapsed = time.time() - t_start
        if elapsed >= time_budget:
            print(f"\nTime budget reached ({time_budget}s)")
            break

        print(f"\n--- Cycle {cycle + 1}/{n_cycles} ---")

        # resource check
        ok, reason = guard.check(verbose=True)
        if not ok:
            print(f"  Resource limit: {reason}")
            if 'temp' in reason.lower():
                guard.wait_for_cooldown()
            else:
                time.sleep(30)
            continue

        # STaR: generate and verify
        good_data, success_rate = star_cycle(
            model, tokenizer,
            n_samples=samples_per_cycle,
            temperature=0.8,
            min_score=0.5,
        )

        # Reflect-Retry on some failures
        if success_rate < 0.5:
            print(f"  Reflect-Retry (low success rate)...")
            for prompt in random.sample(STORY_PROMPTS, min(5, len(STORY_PROMPTS))):
                text, temp = reflect_retry(model, tokenizer, prompt)
                if text:
                    score, _ = score_generation(model, tokenizer, text)
                    if score >= 0.5:
                        good_data.append((text, score))

        if not good_data:
            print(f"  No good data generated this cycle. Skipping fine-tune.")
            cycle_stats.append({'cycle': cycle + 1, 'good': 0, 'loss': 0, 'rate': success_rate})
            continue

        all_good_data.extend(good_data)
        texts = [text for text, _ in good_data]

        # quality before fine-tune
        pre_losses = []
        for text, _ in good_data[:5]:
            ppl = compute_perplexity(model, tokenizer, text)
            pre_losses.append(math.log(ppl) if ppl < float('inf') else 10)

        # fine-tune
        print(f"  Fine-tuning on {len(texts)} examples ({finetune_steps} steps)...")
        ft_loss = finetune_on_data(model, tokenizer, texts,
                                   steps=finetune_steps, lr=1e-4)

        # quality after fine-tune
        post_losses = []
        for text, _ in good_data[:5]:
            ppl = compute_perplexity(model, tokenizer, text)
            post_losses.append(math.log(ppl) if ppl < float('inf') else 10)

        pre_avg = sum(pre_losses) / len(pre_losses) if pre_losses else 0
        post_avg = sum(post_losses) / len(post_losses) if post_losses else 0

        print(f"  FT loss: {ft_loss:.4f} | quality: {pre_avg:.2f} -> {post_avg:.2f}")

        cycle_stats.append({
            'cycle': cycle + 1,
            'good': len(good_data),
            'loss': ft_loss,
            'rate': success_rate,
            'pre_quality': pre_avg,
            'post_quality': post_avg,
        })

        # save checkpoint every 3 cycles
        if (cycle + 1) % 3 == 0:
            save_model(model, tag=f"self_improve_c{cycle+1}",
                       metadata={'cycle': cycle + 1, 'total_good_data': len(all_good_data)})

    # final save
    save_model(model, tag="self_improved",
               metadata={'cycles': len(cycle_stats), 'total_good_data': len(all_good_data)})

    # summary
    total_time = time.time() - t_start
    total_good = sum(s['good'] for s in cycle_stats)
    avg_rate = sum(s['rate'] for s in cycle_stats) / max(len(cycle_stats), 1)

    print("\n" + "=" * 60)
    print("Self-Improvement Complete")
    print("=" * 60)
    print(f"  Cycles: {len(cycle_stats)}")
    print(f"  Total good data: {total_good}")
    print(f"  Avg success rate: {avg_rate:.1%}")
    print(f"  Time: {total_time:.0f}s")

    # sample generation
    print("\nSample outputs:")
    for prompt in STORY_PROMPTS[:3]:
        text = generate_text(model, tokenizer, prompt, max_tokens=60, temperature=0.7)
        print(f"  {text[:120]}")

    update_devlog(
        f"Self-improvement: {len(cycle_stats)} cycles, "
        f"{total_good} good examples, avg_rate={avg_rate:.1%}"
    )

    # save good data for future training
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data_self_gen.jsonl')
    with open(data_path, 'w') as f:
        for text, score in all_good_data:
            f.write(json.dumps({'text': text, 'score': score}) + '\n')
    print(f"\n  Self-generated data saved: {data_path} ({len(all_good_data)} examples)")

    return model, cycle_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=10)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--ft-steps', type=int, default=50)
    parser.add_argument('--budget', type=int, default=3600, help='Time budget in seconds')
    args = parser.parse_args()

    self_improve_loop(
        n_cycles=args.cycles,
        samples_per_cycle=args.samples,
        finetune_steps=args.ft_steps,
        time_budget=args.budget,
    )
