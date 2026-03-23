"""
phi-auto Training Script.
Single-file training loop for on-device GPT.
Usage: python src/engine/train.py
"""

import os
import sys
import time
import math
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.engine.model import GPT, GPTConfig
from src.engine.tokenizer import ByteBPETokenizer
from src.engine.optim import Lion, AdamW as AdamWNew, ScheduleFreeAdamW
from src.data.loader import DataLoader
from src.data.mmap_loader import MMapDataLoader, prepare_mmap_data
from src.data.prepare import prepare_all

# ---------------------------------------------------------------------------
# AdamW Optimizer
# ---------------------------------------------------------------------------

class AdamW:
    """AdamW optimizer with gradient clipping."""
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, grad_clip=1.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.t = 0
        self.states = []
        for p, grad_attr in params:
            self.states.append({
                'param': p,
                'grad_attr': grad_attr,
                'm': np.zeros_like(p),
                'v': np.zeros_like(p),
            })

    def step(self, model):
        """Update parameters."""
        self.t += 1

        # collect all gradients for global clip
        all_grads = []
        for state in self.states:
            grad_attr = state['grad_attr']
            # find the gradient on the object that owns this param
            grad = self._find_grad(model, state['param'], grad_attr)
            if grad is not None:
                all_grads.append(grad)

        # global gradient clipping
        if self.grad_clip > 0 and all_grads:
            total_norm = math.sqrt(sum(np.sum(g ** 2) for g in all_grads))
            clip_coef = self.grad_clip / (total_norm + 1e-8)
            if clip_coef < 1.0:
                for g in all_grads:
                    g *= clip_coef

        # update each parameter
        for i, state in enumerate(self.states):
            if i >= len(all_grads):
                continue
            grad = all_grads[i]
            p = state['param']
            m, v = state['m'], state['v']

            # weight decay
            if self.weight_decay > 0 and p.ndim >= 2:
                p -= self.lr * self.weight_decay * p

            # adam update
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def _find_grad(self, model, param, grad_attr):
        """Find gradient for a parameter by scanning model components."""
        # scan all components
        components = [model.tok_emb, model.ln_f, model.lm_head]
        for block in model.blocks:
            components.extend([
                block.ln1, block.ln2,
                block.attn.qkv, block.attn.proj,
                block.mlp.gate_up, block.mlp.down,
            ])
        for comp in components:
            if hasattr(comp, grad_attr):
                g = getattr(comp, grad_attr)
                if g is not None and hasattr(comp, 'w') and comp.w is param:
                    return g
                if g is not None and hasattr(comp, 'g') and comp.g is param:
                    return g
                if g is not None and hasattr(comp, 'b') and comp.b is param:
                    return g
        return None


class SimpleAdamW:
    """Simpler AdamW that works with (param_array, grad_array) pairs directly."""
    def __init__(self, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, grad_clip=1.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.t = 0
        self.m_dict = {}
        self.v_dict = {}

    def step(self, param_grad_pairs):
        """param_grad_pairs: list of (param_array, grad_array)"""
        self.t += 1

        # global gradient clipping
        if self.grad_clip > 0:
            total_norm = math.sqrt(sum(np.sum(g ** 2) for _, g in param_grad_pairs if g is not None))
            clip_coef = self.grad_clip / (total_norm + 1e-8)
            if clip_coef < 1.0:
                for _, g in param_grad_pairs:
                    if g is not None:
                        g *= clip_coef

        for p, g in param_grad_pairs:
            if g is None:
                continue
            pid = id(p)
            if pid not in self.m_dict:
                self.m_dict[pid] = np.zeros_like(p)
                self.v_dict[pid] = np.zeros_like(p)

            m, v = self.m_dict[pid], self.v_dict[pid]

            # weight decay (only for 2D+ params)
            if self.weight_decay > 0 and p.ndim >= 2:
                p -= self.lr * self.weight_decay * p

            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def collect_param_grads(model):
    """Collect all (param, grad) pairs from model."""
    pairs = []

    def _collect(component):
        for attr in ['w', 'g', 'b']:
            if hasattr(component, attr):
                p = getattr(component, attr)
                if p is None:
                    continue
                grad_attr = 'd' + attr
                grad = getattr(component, grad_attr, None)
                if grad is not None:
                    pairs.append((p, grad))

    _collect(model.tok_emb)
    _collect(model.ln_f)
    _collect(model.lm_head)
    for block in model.blocks:
        _collect(block.ln1)
        _collect(block.ln2)
        _collect(block.attn.qkv)
        _collect(block.attn.proj)
        _collect(block.mlp.gate_up)
        _collect(block.mlp.down)
    return pairs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, val_loader, eval_steps=20):
    """Evaluate model on validation data. Returns average loss and bpb."""
    total_loss = 0
    total_tokens = 0
    total_bytes = 0

    for _ in range(eval_steps):
        x, y, _ = val_loader.get_batch()
        _, loss = model.forward(x, y)
        B, T = x.shape
        total_loss += loss * B * T
        total_tokens += B * T
        # estimate bytes (rough: ~1.2 bytes per token for English BPE)
        total_bytes += B * T * 1.2

    avg_loss = total_loss / total_tokens
    # bits per byte = nats_per_token / (bytes_per_token * ln(2))
    bpb = avg_loss / (1.2 * math.log(2))
    return avg_loss, bpb


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(config=None):
    """Main training function."""
    # defaults
    if config is None:
        config = {
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
            'max_steps': 2000,
            'eval_interval': 100,
            'eval_steps': 10,
            'log_interval': 10,
            'weight_decay': 0.1,
            'grad_clip': 1.0,
            'time_budget': 1800,  # 30 min
        }

    print("=" * 60)
    print("phi-auto Training")
    print("=" * 60)

    # 1. Data preparation
    print("\n[1/5] Preparing data...")
    tokenizer, train_path, val_path = prepare_all(
        vocab_size=config['vocab_size'],
        max_stories=20000
    )

    # 2. Pre-tokenize to mmap (one-time cost, massive speedup after)
    print("\n[2/5] Pre-tokenizing to mmap...")
    train_mmap = prepare_mmap_data(train_path, tokenizer)
    val_mmap = prepare_mmap_data(val_path, tokenizer)

    # 3. Model
    print("\n[3/5] Building model...")
    model_config = GPTConfig(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        seq_len=config['seq_len'],
    )
    model = GPT(model_config, tie_weights=config.get('weight_tie', False))

    # 4. Data loaders (mmap — zero-copy, no runtime tokenization)
    print("\n[4/5] Setting up mmap data loaders...")
    train_loader = MMapDataLoader(train_mmap, config['batch_size'], config['seq_len'])
    val_loader = MMapDataLoader(val_mmap, config['batch_size'], config['seq_len'], shuffle=False)

    # 5. Optimizer
    optim_type = config.get('optimizer', 'lion')
    if optim_type == 'lion':
        optimizer = Lion(
            lr=config.get('lr', 3e-4) * 0.1,
            weight_decay=config.get('weight_decay', 0.1) * 10,
            grad_clip=config['grad_clip'],
        )
        print(f"  Optimizer: Lion (lr={optimizer.lr:.1e})")
    elif optim_type == 'schedule_free':
        optimizer = ScheduleFreeAdamW(
            lr=config.get('lr', 3e-4),
            warmup_steps=config.get('warmup_steps', 50),
            grad_clip=config['grad_clip'],
        )
        print(f"  Optimizer: Schedule-Free AdamW (lr={optimizer.lr:.1e})")
    elif optim_type == 'adamw':
        optimizer = AdamWNew(
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.1),
            grad_clip=config['grad_clip'],
        )
        print(f"  Optimizer: AdamW (lr={optimizer.lr:.1e})")
    else:
        optimizer = SimpleAdamW(
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            grad_clip=config['grad_clip'],
        )
        print(f"  Optimizer: SimpleAdamW (lr={optimizer.lr:.1e})")

    # 6. Training loop
    print(f"\n[5/5] Training for {config['max_steps']} steps...")
    print(f"  Effective batch: {config['batch_size'] * config['grad_accum']}")
    print(f"  Time budget: {config['time_budget']}s")
    print()

    t_start = time.time()
    best_val_loss = float('inf')
    smooth_loss = 0

    for step in range(config['max_steps']):
        t0 = time.time()

        # LR schedule (skip for schedule-free optimizers)
        if not isinstance(optimizer, ScheduleFreeAdamW):
            lr = get_lr(step, config['warmup_steps'], config['max_steps'],
                         config['lr'], config['min_lr'])
            if isinstance(optimizer, Lion):
                lr *= 0.1  # Lion uses ~10x smaller LR
            optimizer.lr = lr
        else:
            lr = optimizer.lr

        # gradient accumulation (pre-allocated buffers, no per-step .copy())
        total_loss = 0

        for micro_step in range(config['grad_accum']):
            x, y, epoch = train_loader.get_batch()
            _, loss = model.forward(x, y)
            model.backward()

            pairs = collect_param_grads(model)

            # one-time allocation after first backward (gradients now exist)
            if step == 0 and micro_step == 0:
                _accum_bufs = [np.zeros_like(g) for _, g in pairs]

            if micro_step == 0:
                for buf in _accum_bufs:
                    buf[:] = 0

            for i, (_, g) in enumerate(pairs):
                _accum_bufs[i] += g

            total_loss += loss

        # average gradients (in-place)
        inv_accum = 1.0 / config['grad_accum']
        for buf in _accum_bufs:
            buf *= inv_accum

        # optimizer step
        final_pairs = [(p, _accum_bufs[i]) for i, (p, _) in enumerate(pairs)]
        optimizer.step(final_pairs)

        avg_loss = total_loss / config['grad_accum']
        dt = time.time() - t0
        elapsed = time.time() - t_start

        # smoothed loss
        beta = 0.9
        smooth_loss = beta * smooth_loss + (1 - beta) * avg_loss if step > 0 else avg_loss
        debiased = smooth_loss / (1 - beta ** (step + 1))

        # logging
        if step % config['log_interval'] == 0:
            tok_per_sec = config['batch_size'] * config['grad_accum'] * config['seq_len'] / dt
            print(f"step {step:5d} | loss {debiased:.4f} | lr {lr:.2e} | "
                  f"{dt*1000:.0f}ms | {tok_per_sec:.0f} tok/s | "
                  f"epoch {epoch} | {elapsed:.0f}s")

        # evaluation
        if step > 0 and step % config['eval_interval'] == 0:
            val_loss, val_bpb = evaluate(model, val_loader, config['eval_steps'])
            print(f"  >>> val_loss: {val_loss:.4f} | val_bpb: {val_bpb:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  >>> New best! Saving checkpoint...")
                save_checkpoint(model, optimizer, step, val_loss, val_bpb)

        # time budget check
        if elapsed >= config['time_budget']:
            print(f"\nTime budget reached ({config['time_budget']}s)")
            break

    # final eval
    print("\nFinal evaluation...")
    val_loss, val_bpb = evaluate(model, val_loader, config['eval_steps'])
    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"val_loss:    {val_loss:.6f}")
    print(f"val_bpb:     {val_bpb:.6f}")
    print(f"best_loss:   {best_val_loss:.6f}")
    print(f"total_time:  {total_time:.1f}s")
    print(f"steps:       {step + 1}")
    print(f"params:      {model.n_params:,}")

    # save final
    save_checkpoint(model, optimizer, step, val_loss, val_bpb, tag="final")

    # generate sample
    print("\nSample generation:")
    prompt = tokenizer.encode("Once upon a time")
    prompt_arr = np.array([prompt], dtype=np.int32)
    generated = model.generate(prompt_arr, max_new_tokens=100, temperature=0.8)
    text = tokenizer.decode(generated[0].tolist())
    print(f"  {text[:200]}")

    train_loader.close()
    val_loader.close()

    return model, val_loss, val_bpb


def save_checkpoint(model, optimizer, step, val_loss, val_bpb, tag="best"):
    """Save model checkpoint."""
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"model_{tag}.npz")

    params = {}
    for i, (p, name) in enumerate(model.all_params()):
        params[f"p_{i}"] = p

    np.savez(path,
             step=step, val_loss=val_loss, val_bpb=val_bpb,
             **params)
    print(f"  Checkpoint saved: {path}")


if __name__ == "__main__":
    train()
