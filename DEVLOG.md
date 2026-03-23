# phi-auto Research Log

> On-device language model training from scratch.
> Snapdragon ARM, Python+numpy, no PyTorch.

---

## Setup

| | |
|-|-|
| Hardware | Snapdragon AArch64, 8 cores, 10GB RAM (3GB usable) |
| Software | Python 3.13, numpy 2.4.3 (OpenBLAS), Termux on Android |
| Dataset | TinyStories — 20K stories, 4.3M train tokens, 224K val tokens |
| Tokenizer | Byte-pair encoding, vocab=1024 |
| Architecture | GPT (RMSNorm, RoPE, SwiGLU, causal attention), manual backward |
| Throughput | 2,196 tok/s at 1.3M params (post-optimization) |

---

## Research Question

**At what model size and data volume does a transformer transition from memorizing token frequencies to learning actual language patterns?**

Small model scaling laws — empirically measuring the critical threshold in the 1K–10M parameter range where a model first breaks past unigram-level performance (token frequency) and begins learning contextual patterns (bigram+).

### Reference Baselines (TinyStories val set)

```
Random (uniform)     6.93 nats
Unigram              6.25 nats
Bigram               3.36 nats     ← below this = contextual learning
Trigram              1.88 nats
```

### Core Hypothesis

The 1.3M model's plateau at unigram level is not a training bug — it's a **capacity-data scaling boundary**. Increasing parameters, reducing data diversity, or adjusting their ratio should break through the wall.

---

## Experiment Table

All experiment results in one place. New experiments appended below.

| ID | Date | Params | Config | Steps | val_loss | vs Unigram | Status | Notes |
|----|------|--------|--------|-------|----------|------------|--------|-------|
| B1 | 03-24 | 1.3M | 128d/4L Lion lr=3e-5 | 2000 | 6.17 | -1.3% | Baseline | Plateau at unigram level |
| μ1 | 03-24 | 1.3M | 128d/4L AdamW lr=3e-4 | 300 | 6.37 | +1.9% | Micro | AdamW converges 3x faster |
| μ2 | 03-24 | 1.3M | 128d/4L Lion lr=3e-5 | 300 | 6.56 | +5.0% | Micro | Lion converges slower |
| E2 | 03-24 | 3.7M | 192d/6L AdamW lr=6e-4 | - | div | - | Failed | lr too high, diverged |
| E3 | 03-24 | 1.3M | 128d/4L AdamW lr=3e-4 clip=5 | 400 | ~6.5 | +4% | Killed | Still above unigram |
| μ3 | 03-24 | 1.3M | 128d/4L 5-story overfit | ~500 | **5.38** | **-13.9%** | Micro | **Below unigram. Architecture works.** |

> **vs Unigram**: position relative to unigram CE (6.25). Negative = better than unigram (learning language patterns).

---

## Log

Reverse chronological (newest first). Each entry should be self-contained.

---

### 2026-03-24: The Unigram Wall

**Observation:** Baseline (1.3M, 2K steps) val_loss=6.17 ≈ unigram CE=6.25. Model learned token frequencies only.

**Controlled experiments:**
- 5-story overfit → loss 5.38 (below unigram). Architecture confirmed correct.
- 100-story overfit → loss 6.98 (random level). Data diversity exceeds capacity.
- Lion vs AdamW both plateau at bpb ~7.5. Not an optimizer issue → capacity limit.

**Diagnosis:**
vocab=1024 → 1M possible bigrams. Model has 1.3M total params. After embedding, attention, and MLP structure, insufficient free capacity to encode bigram statistics over a large dataset. With 5 stories (~200 unique bigrams) it works. With 20K stories, it doesn't.

**Conclusion:** Not a software bug. A capacity-data scaling boundary. Need to increase parameters or reduce data diversity.

---

### 2026-03-24: EXP-002, EXP-003 Failed

**E2 (192d/6L, 3.7M, lr=6e-4):** Immediate divergence. Loss 6.35 → 7.16. lr=6e-4 too high for 3.7M model.

**E3 (128d/4L, 1.3M, AdamW lr=3e-4, clip=5, wd=0.01):** Loss ~6.5 at step 400. Noisy, never dropped below unigram. Same capacity = same result.

**Takeaway:**
- 3.5M+ models need lr ≤ 3e-4
- 1.3M model cannot break unigram regardless of hyperparameter tuning (capacity ceiling)

---

### 2026-03-24: Optimizer Comparison (Lion vs AdamW)

**Setup:** Same model (128d/4L), same data, 300-step micro-experiments.

| | val_loss (300 steps) | Convergence |
|-|-|-|
| AdamW lr=3e-4 | 6.37 | Fast |
| Lion lr=3e-5 | 6.56 | 3x slower |

**Analysis:** Lion's sign-based updates discard gradient magnitude. At large scale this doesn't matter. At 1.3M params, every bit of gradient information counts. AdamW is better at this scale.

---

### 2026-03-24: Baseline Training Complete

**Config:** 128d/4L (1.3M params), Lion, 2000 steps, batch=4, seq=128
**Result:** val_loss=6.17, val_bpb=7.41, 1,350 tok/s
**Generated text:** Complete gibberish
**Time:** ~26 minutes

---

## Optimization Log

Performance improvements. Each entry includes measurements.

| ID | What | Before | After | Gain | Status |
|----|------|--------|-------|------|--------|
| OPT-001 | Fused MLP gate+up | 326ms/step | 233ms/step | +40% tok/s | Kept |
| OPT-002 | Heap-based BPE encode | 13+ min / 19K texts | 37 sec | 21x | Kept |
| OPT-003 | Incremental BPE training | 24+ min | 290 sec | ~50x | Kept |
| OPT-004 | Mmap data loading | ~5ms/batch | ~0ms/batch | eliminated | Kept |
| OPT-005 | RoPE backward (exact) | biased gradients | exact gradients | correctness | Kept |
| OPT-006 | Grad accum buffer fix | crash | works | correctness | Kept |
| OPT-007 | NEON C matmul | OpenBLAS baseline | 2.5-4.5x slower | negative | Abandoned |
| OPT-008 | RWKV time-mixing | attn 6.9ms/block | RWKV 9.8ms/block | 0.7x (slower) | Kept for inference |

<details>
<summary>Details (click to expand)</summary>

**OPT-001: Fused MLP.** Merged gate and up projections into single Linear. Two BLAS dispatches → one. At small matrix sizes, dispatch overhead dominates actual compute.

**OPT-002: Heap BPE.** O(M×N) scan → O(N log N) heap. Doubly-linked list + min-heap for merge candidates. Each merge application is O(log N) amortized.

**OPT-003: Incremental pair count.** Full recount per merge → update only affected neighbors. Python loop iterations: 768M → ~tens of thousands.

**OPT-004: Mmap.** Pre-tokenize entire dataset to uint16 .npy. Load via np.load(mmap_mode='r'), let OS handle paging.

**OPT-005: RoPE backward.** Inverse of rotation matrix = negate sin component. Orthogonal matrices satisfy R⁻¹ = Rᵀ.

**OPT-006: Buffer fix.** Grad accumulation buffers allocated at step 0 before any backward → gradients are None → crash. Fixed with lazy allocation after first backward.

**OPT-007: NEON C.** Wrote 4×4 micro-kernel sgemm. OpenBLAS's assembly-level ARM kernels with cache blocking are 2.5-4.5x faster. Matmul is not the bottleneck — Python overhead is.

**OPT-008: RWKV.** Theoretical O(T) < attention O(T²). In practice, Python for-loop is slower than numpy's single BLAS call for attention. Wins at inference (O(1) per token) but loses at training.

</details>

---

## Profiling (1.3M params, batch=4, seq=128)

```
Forward     45%     Backward    55%
────────────────────────────────────
Embedding    0.1%   lm_head      22% of bwd
Attention   16%     Attention    ~16% of bwd
MLP         20%     MLP          ~20% of bwd
Loss         2%
────────────────────────────────────
Total: 233ms/step, 2,196 tok/s
OpenBLAS: 20-22 GFLOPS on this Snapdragon
```

---

## Confirmed Findings

Only things verified across multiple experiments.

1. **The unigram wall is a capacity limit.** The 1.3M model plateaus near unigram CE (6.25) regardless of optimizer or hyperparameters. Only the 5-story overfit broke through. (B1, μ1, μ2, E3, μ3)

2. **AdamW > Lion at small scale.** Sign-based updates discard gradient magnitude, which hurts when model capacity is tight. (μ1 vs μ2)

3. **BLAS dispatch overhead > FLOPs at small sizes.** Fusing operations beats algorithmic improvements. (OPT-001)

4. **Python overhead is the actual bottleneck.** Custom C SIMD is slower than OpenBLAS. Theoretical O(T) RWKV is slower than O(T²) attention. Same root cause: interpreter overhead between compute kernels. (OPT-007, OPT-008)

---

## Architecture Decisions

Updated with rationale when changed.

| Decision | Choice | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| Framework | numpy only | No PyTorch ARM overhead; educational purpose | PyTorch Mobile, tinygrad |
| Normalization | RMSNorm | 30% cheaper than LayerNorm | LayerNorm |
| Positions | RoPE | Zero learnable params, extrapolation | Learned, ALiBi |
| Activation | SwiGLU (fused) | +1-3% over GELU at same param count | GELU, ReLU |
| Optimizer | AdamW (default) | Better than Lion at this scale | Lion, SF-AdamW |
| Data loading | mmap | Eliminates runtime tokenization | Streaming JSON |
| Attention | Standard causal | Faster than RWKV in numpy training | RWKV (kept for inference) |

---

## Status

**Current phase:** Preparing scaling experiments

**Working:**
- [x] GPT forward + backward (manual gradients, verified)
- [x] Three optimizers (AdamW, Lion, Schedule-Free AdamW)
- [x] BPE tokenizer (heap encode, incremental training)
- [x] Mmap data pipeline
- [x] Autonomous experiment runner
- [x] Self-improvement modules (STaR, SPIN — waiting on model quality)

**Blocked:**
- [ ] Breaking the unigram wall (need 3.5M+ model)
- [ ] Model that generates recognizable English
- [ ] Self-improvement loop (needs generation quality)

**Next:**
- [ ] Model size × data size grid experiment (scaling law measurement)
- [ ] 3.5M model with conservative training (lr=1e-4, 2+ epochs)
- [ ] C port evaluation (Python overhead is the bottleneck)
