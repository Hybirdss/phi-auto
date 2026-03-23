# phi-auto

**Language model training from scratch on an Android phone.**

GPT architecture, BPE tokenizer, AdamW/Lion optimizers, manual backward pass — all in ~2,500 lines of Python+numpy. Runs on Termux, no PyTorch, no GPU.

Based on the [autoresearch](https://github.com/karpathy/autoresearch) approach: build → measure → improve → repeat.

---

## Setup

```bash
# Termux on Android (or any ARM Linux)
pkg install python numpy
git clone https://github.com/Hybirdss/phi-auto.git && cd phi-auto

python src/engine/train.py              # Train (~30 min)
python src/agent/run_experiments.py     # Automated experiment sweep
```

**Requires:** Python 3.10+, numpy, ~3GB RAM, ~5GB storage

---

## What's In Here

### Model (`src/engine/model.py`)
Standard GPT with RMSNorm, RoPE, SwiGLU, causal attention. Forward and backward pass both implemented manually — no autograd. Weight tying optional.

### Tokenizer (`src/engine/tokenizer.py`)
Byte-pair encoding. Two optimizations made during development:
- Heap-based encoding: O(N log N) instead of O(M×N). 13 min → 37 sec for 19K texts.
- Incremental pair counting during training: O(affected) instead of O(N) per merge. 24 min → 5 min.

### Optimizers (`src/engine/optim.py`)
AdamW, Lion, Schedule-Free AdamW. All from scratch.

### Data (`src/data/`)
TinyStories dataset. Pre-tokenized to `.npy`, loaded via memory-mapped I/O (zero-copy).

### Experiment Runner (`src/agent/`)
Autoresearch-style loop: mutate hyperparameters → train → evaluate → keep or discard. 6 mutation strategies. Logs everything.

Self-improvement modules (STaR, SPIN) are implemented but waiting for a model that can generate coherent text first.

---

## Current Numbers

### Throughput (1.3M params, Snapdragon ARM)
```
Step time       233 ms
Throughput      2,196 tokens/sec
```
After optimization. Started at 326ms / 1,572 tok/s. Main gains from fusing MLP projections (two matmuls → one).

### Time Breakdown
```
Forward     45%    (OpenBLAS matmuls)
Backward    55%    (weight + input gradients per layer)
```

### Experiments Run
| Config | val_loss | Notes |
|--------|----------|-------|
| 128d/4L Lion 2K steps | 6.17 | Plateau at unigram level |
| 128d/4L AdamW 300 steps | 6.37 | AdamW converges ~3x faster than Lion here |
| 128d/4L 5-story overfit | **5.38** | Below unigram — context learning works |
| 192d/6L AdamW lr=6e-4 | diverged | lr too high |

---

## The Unigram Wall

The most useful thing that came out of this so far.

N-gram cross-entropies on TinyStories validation set:

```
Random (uniform)     6.93 nats
Unigram              6.25 nats
Our model            6.17 nats   ← basically unigram
Bigram               3.36 nats
Trigram              1.88 nats
```

The 1.3M model after 2,000 steps learns token frequencies and nothing else. val_loss ≈ unigram CE. No bigram patterns, no word order, no grammar.

But: the 5-story overfit test gets loss to 5.38 (below unigram). So the architecture and gradients are correct. The problem is capacity — 1.3M params can't compress 20K diverse stories beyond marginal statistics.

Next step is scaling to 3.5M+ params to see if the wall breaks.

Full analysis in [DEVLOG.md](DEVLOG.md).

---

## Stuff That Didn't Work

Documenting these because they cost time and the reasons aren't obvious:

- **Custom NEON SIMD matmul in C** — wrote a 4×4 micro-kernel. OpenBLAS was 2.5-4.5× faster. Their ARM kernels are assembly-optimized with cache blocking. Not worth competing.
- **RWKV time-mixing for training** — O(T) complexity but needs a Python for-loop. Attention's O(T²) runs as a single BLAS call, which is faster in practice. (RWKV does win at inference.)
- **Lion optimizer at small scale** — sign-based updates throw away gradient magnitude. At 1.3M params that information matters. AdamW works better here.
- **lr=6e-4 for 3.5M model** — diverges immediately. 3e-4 is the safe zone.

---

## Project Structure

```
src/
├── engine/            # Model, training, tokenizer, optimizers
│   ├── model.py       # GPT forward + backward
│   ├── train.py       # Training loop
│   ├── tokenizer.py   # BPE (heap-optimized)
│   ├── optim.py       # AdamW, Lion, Schedule-Free AdamW
│   └── rwkv_tmix.py   # RWKV (experimental)
├── data/              # Dataset preparation, mmap loader
├── agent/             # Experiment runner, self-improvement
└── tools/             # Monitoring, checkpoints, config
```

---

## Next

1. Scale model to 3.5M+ params, train 2+ epochs
2. Break below bigram-level loss (prove context learning at scale)
3. Run self-improvement loop (STaR/SPIN) once generation works
4. Possibly port to C for 5-10× throughput
