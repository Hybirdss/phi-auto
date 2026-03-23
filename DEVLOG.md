# phi-auto Development Log

> On-device self-improving AI for Android/Termux
> Started: 2026-03-23

---

## Phase 0: Project Setup & Research
**Status: COMPLETE**

### 2026-03-23 - Day 1: Project Kickoff

**Research Completed:**
- [x] Analyzed karpathy/autoresearch (train.py, prepare.py, program.md)
- [x] Surveyed 50+ repos for on-device training
- [x] Reviewed 20+ papers on self-improving AI
- [x] Identified viable frameworks (QVAC Fabric, llm.c, tinygrad, MobileFineTuner)
- [x] Confirmed dead ends (llama.cpp training removed, JAX no ARM, Mojo no Android)
- [x] Mapped memory budget: 10GB total, ~5.8GB available for model+training

**Architecture Designed:**
- [x] System overview diagram
- [x] Core experiment loop
- [x] Self-improvement pipeline (STaR + Reflect-Retry)
- [x] Directory structure
- [x] Memory management strategy
- [x] Component specifications

**Environment:**
- CPU: Snapdragon AArch64, 8 cores
- RAM: 10GB (3GB available)
- Storage: 22GB free
- Python: 3.13, numpy 2.4.3
- Tools: clang 21.1.8, cmake 4.2.3, rust 1.93.1

**Key Decisions:**
- Pure Python + numpy for Phase 1 (no PyTorch overhead)
- 1-5M params initial model, scale up after validation
- TinyStories dataset (low entropy, good for small models)
- BPE tokenizer with 1024 vocab
- 256 token context window
- AdamW optimizer (simpler than Muon, works on CPU)

---

## Phase 1: Core Training Engine
**Status: IN PROGRESS (training running)**

### TODO:
- [x] Install numpy (2.4.3 via pkg)
- [x] Implement BPE tokenizer (1024 vocab) - `src/engine/tokenizer.py`
- [x] Download & prepare TinyStories dataset - `src/data/prepare.py`
- [x] Implement streaming data loader - `src/data/loader.py`
- [x] Implement GPT model (RMSNorm, RoPE, causal attention, SwiGLU) - `src/engine/model.py`
- [x] Implement full backward pass (manual gradients, no autograd)
- [x] Implement training loop (AdamW, grad accum, cosine LR) - `src/engine/train.py`
- [x] Implement evaluation module - `src/engine/eval.py`
- [x] Model sanity check: forward + backward pass OK (164K test model)
- [ ] Run baseline training: establish first val_bpb
- [ ] Benchmark: tokens/sec, RAM usage, CPU temp
- [ ] Save first checkpoint

### Milestones:
- [x] Model forward + backward pass works
- [ ] `python src/engine/train.py` completes full run
- [ ] val_bpb < 2.0 (model is learning)
- [ ] val_bpb < 1.5 (model generates recognizable text)

### Bugs Fixed:
- einsum backward pass: numpy 2.4.3 rejected `...i,...j->ij` syntax, fixed with `x_2d.T @ d_2d`
- Python version mismatch: upgraded 3.12 -> 3.13 for numpy compatibility

---

## Phase 2: Autonomous Experiment Loop
**Status: CODE COMPLETE (awaiting Phase 1 baseline)**

### TODO:
- [x] Implement experiment runner - `src/agent/experiment.py`
- [x] Implement results.tsv logging - `src/tools/logger.py`
- [x] Implement keep/discard logic with checkpoint management
- [x] Implement hypothesis generator (6 strategies) - `src/agent/hypothesis.py`
- [x] Implement resource guard (RAM, CPU temp, battery) - `src/tools/monitor.py`
- [x] Implement checkpoint management - `src/tools/checkpoint.py`
- [x] Implement TOML config loader - `src/tools/config.py`
- [ ] Run first automated experiment
- [ ] Run overnight: target 10+ experiments

### Strategies Implemented:
1. **random** - Mutate 1-3 hyperparams randomly
2. **scale_up** - Increase model size (wider/deeper)
3. **scale_down** - Decrease model for speed
4. **lr_search** - Try nearby learning rates
5. **seq_len** - Try different context lengths
6. **efficient** - Optimize throughput (wider+shallower, bigger batch)

### Milestones:
- [ ] First automated experiment completes
- [ ] First improvement discovered autonomously
- [ ] 10 experiments logged in results.tsv
- [ ] val_bpb improved by > 5% from baseline

---

## Phase 3: Self-Improvement Loop
**Status: CODE COMPLETE (awaiting Phase 1 model)**

### TODO:
- [x] Implement STaR cycle (generate -> verify -> score) - `src/agent/self_improve.py`
- [x] Implement Reflect-Retry (retry with different temperatures)
- [x] Implement quality verification (repetition, coherence, ascii checks)
- [x] Implement perplexity-based scoring
- [x] Implement fine-tuning on self-generated data
- [ ] Run first self-improvement cycle
- [ ] Measure improvement from self-generated data

### Milestones:
- [ ] Model successfully generates own training data
- [ ] First self-improvement cycle completes
- [ ] Measurable improvement from self-generated data
- [ ] Stable improvement over 5+ cycles without degradation

---

## Phase 4: Scale Up (Future)
**Status: PLANNING**

### Ideas:
- [ ] Port to llm.c (pure C, NEON optimization)
- [ ] Integrate QVAC Fabric (Vulkan GPU training)
- [ ] Increase model to 30M+ params
- [ ] Add BitNet quantization
- [ ] Multi-task self-improvement
- [ ] Agent that modifies its own training code (full Karpathy Loop)

---

## CLI Commands

```bash
python phi train                    # Train baseline
python phi train --config configs/phone.toml  # Train with config
python phi experiment --hours 4     # Autonomous experiments
python phi improve --cycles 20      # Self-improvement
python phi status                   # System & training status
python phi generate "Once upon"     # Generate text
python phi run                      # Full pipeline
```

---

## Experiment Results

| # | Date | Config | val_bpb | val_loss | tok/s | Status | Notes |
|---|------|--------|---------|----------|-------|--------|-------|
| 1 | 03-24 | 128d/4L Lion 2Kstep | 7.41 | 6.17 | 1,350 | BASELINE | Loss plateau at 6.2, gibberish output |
| μ1 | 03-24 | 128d/4L AdamW 300step | 7.66 | 6.37 | 1,600 | micro | AdamW converges 3x faster than Lion |
| μ2 | 03-24 | 128d/4L Lion 300step | 7.89 | 6.56 | 1,400 | micro | Lion slower convergence |
| 2 | 03-24 | 192d/6L AdamW 6Kstep | - | - | - | RUNNING | 3.5M params, weight-tied, 1.5 epochs |

---

## Notes & Observations

- [2026-03-23] TinyStories downloaded: 2.2GB raw, 20K stories extracted
- [2026-03-23] System: RAM 68%, CPU 50°C, Battery 31% during training
- [2026-03-23] BPE tokenizer training is CPU-intensive (~5min for 768 merges on 5K texts)

---

## Optimization Log (autoresearch-style)

### OPT-001: Fused MLP gate+up projection
**Date:** 2026-03-24
**Problem:** MLP forward takes 16.5ms/block — two separate Linear ops (gate + up) each do matmul independently.
**Hypothesis:** Fusing gate and up into a single `gate_up` Linear (n_embd → 2*hidden) saves one matmul dispatch + memory allocation.
**Result:** Step time 326ms → 233ms (**28.5% faster**). Throughput 1,572 → 2,196 tok/s (**+40%**).
**Insight:** On small matrices, the overhead of dispatching BLAS calls matters as much as the actual FLOPs. Fewer, larger matmuls > many small ones.
**Status:** KEPT ✓

### OPT-002: Cached causal mask
**Date:** 2026-03-24
**Problem:** `np.triu(np.full((T,T), -1e9))` called every forward pass per layer = 4 allocations/step.
**Fix:** Pre-allocate once in `__init__`, slice `[:T,:T]` in forward.
**Result:** Marginal speedup, but eliminates GC pressure over long training runs.
**Status:** KEPT ✓

### OPT-003: Incremental BPE pair counting (tokenizer)
**Date:** 2026-03-24
**Problem:** Original BPE training recounts ALL pairs from scratch each merge = O(merges × total_tokens). 768 merges × 1M tokens = 768M ops in Python. Took >24 minutes on phone.
**Hypothesis:** Maintain running pair counts, only update affected pairs on each merge.
**Result:** 500 texts × 256 merges: 1.3s (was estimated >60s). **~50x speedup**.
**Insight:** The classic BPE bottleneck is pair counting, not the merge itself. Incremental counting turns O(N) per merge into O(affected) per merge.
**Status:** KEPT ✓

### Profiling Baseline (1.3M params, batch=4, seq=128)
```
Component          Time (ms)    % of Total
─────────────────────────────────────────
Embedding          0.2          0.1%
Block×4 (fwd)     124          38%
  Attention        52           16%
  MLP              66           20%
Softmax (loss)     6.7          2%
Backward          178          55%
─────────────────────────────────────────
Total (pre-opt)   326ms        1,572 tok/s
Total (post-opt)  233ms        2,196 tok/s
```

### Key Insights
1. **OpenBLAS ARMV8**: numpy gets 20-22 GFLOPS on this Snapdragon. Decent.
2. **Matmul dispatch overhead dominates** at small sizes — fusing ops helps more than algorithmic tricks.
3. **Python loop overhead** is the #1 enemy. BPE training was 50x slower than it needed to be because of Python `for` loops over token sequences.
4. **Backward > Forward**: backward takes 55% of step time. Must optimize backward pass next.

### OPT-004: Lion optimizer (replace AdamW)
**Date:** 2026-03-24
**Insight from research:** Lion (Google Brain 2023) uses `sign(momentum)` instead of `m/(sqrt(v)+eps)`. Only needs 1 buffer per param instead of 2.
**Expected:** 50% less optimizer memory, simpler computation.
**LR rule:** Use 10x smaller LR than AdamW, 10x larger weight decay.
**Status:** IMPLEMENTED, testing with training run

### OPT-005: Memory-mapped data loader
**Date:** 2026-03-24
**Problem:** JSON-streaming loader tokenizes on-the-fly = slow. Each `get_batch()` does JSON.parse + BPE encode.
**Fix:** Pre-tokenize entire dataset into `.npy` uint16 array. Load via `np.load(mmap_mode='r')` = zero-copy OS paging.
**Expected:** 5-7x faster data loading, near-zero RAM overhead.
**Status:** IMPLEMENTED

### OPT-006: Incremental tokenizer optimization
**Date:** 2026-03-24
**Problem:** Old BPE training took >24 min (killed). New incremental pair counting: 290s (5 min).
**Speedup:** ~5x for tokenizer training.
**Status:** KEPT ✓

### Research Findings (2026-03-24)
From automated research sweep of 2025-2026 papers:

**Architecture:** RWKV-7 "Goose" (March 2025) — O(1) per-token inference vs O(T) for attention. State is (H,D,D) = 1024 elements vs attention's (H,T,T) = 65K elements for T=256. **64x smaller.**
→ TODO: Replace CausalAttention with RWKV time-mixing.

**Optimizer:** Lion > AdamW for mobile (50% less memory). Schedule-Free AdamW eliminates LR scheduling.

**Self-improvement:** SPIN (Self-Play Fine-Tuning) > STaR for tiny models. No external reward model needed. 2-3 iterations saturate.

**Data loading:** mmap > streaming. Pre-tokenize once, load zero-copy.

**muP:** Tune hyperparams on 64-dim model, transfer to 192-dim. Saves days of HP search.

### OPT-007: RWKV Time-Mixing (implemented, benchmarked)
**Date:** 2026-03-24
**Hypothesis:** RWKV replaces O(T²) attention with O(T) linear recurrence. Should be faster.
**Result (training):** RWKV 9.8ms vs Attention 6.9ms at seq=128, dim=64. RWKV **0.7x slower** in training.
**Why:** Python `for t in range(T)` loop in RWKV is slower than numpy's batch matmul for attention. NumPy vectorizes `q @ k.T` into a single BLAS call, while RWKV's recurrence can't be vectorized.
**Key insight:** RWKV wins at **inference** (O(1) per token vs O(T)), not training speed. For generation-heavy workloads (self-improvement loop), RWKV will be crucial.
**Decision:** Keep as optional module. Use attention for training, switch to RWKV for generation/self-improvement.
**Status:** IMPLEMENTED as `src/engine/rwkv_tmix.py`, not default yet

### OPT-008: Heap-based BPE encode (21x faster tokenization)
**Date:** 2026-03-24
**Problem:** BPE `encode()` uses O(merges × text_length) algorithm — scans all pairs to find best merge, applies it, repeat. For 768 merges × ~200 bytes per text = 154K ops per text. 19K texts = 2.9B Python loop iterations. Pre-tokenization took **13+ minutes**.
**Fix:** Replace with heap-based O(N log N) algorithm:
1. Build doubly-linked list of tokens + min-heap of merge candidates
2. Pop lowest-rank merge, apply in O(1) via linked list surgery
3. Add new neighbor pairs to heap
4. Repeat until heap empty
**Result:** 19K texts: 13+ min → **37 seconds** (**21x speedup**). 500 texts/sec.
**Insight:** The BPE encode bottleneck is the O(M) scan per merge to find the best pair. The heap eliminates this entirely — each merge application is O(log N) amortized.
**Status:** KEPT ✓

### OPT-009: Proper RoPE backward pass
**Date:** 2026-03-24
**Problem:** Attention backward skipped RoPE gradients ("approximate" comment). RoPE is a rotation matrix, so gradients were systematically biased by ignoring the rotation inverse.
**Fix:** Implement `apply_rope_backward()` — applies inverse rotation (negate sin component). This is mathematically exact since rotation matrices are orthogonal: R^(-1) = R^T.
**Impact:** More accurate gradients → better convergence, especially for position-dependent patterns.
**Status:** KEPT ✓

### OPT-010: Fix gradient accumulation bug + reduce allocations
**Date:** 2026-03-24
**Problem 1:** Grad accum buffers allocated at step 0 BEFORE any backward pass → all gradients are None → empty buffer list → IndexError.
**Fix:** Allocate after first backward.
**Problem 2:** Embedding backward allocated `np.zeros_like(self.w)` every call (512KB per call × 2000 steps = 1GB GC pressure).
**Fix:** Reuse pre-allocated buffer.
**Status:** KEPT ✓

### OPT-011: Weight tying (lm_head shares embedding)
**Date:** 2026-03-24
**Hypothesis:** Share embedding weights with lm_head output projection. Standard practice in GPT-2/LLaMA/Phi.
**Result:** Saves 131K params (10%). Minimal speed impact at vocab=1024 (matmul sizes unchanged).
**Bug found & fixed:** TiedLinear accumulated grad into `tok_emb.dw`, but `Embedding.backward()` zeroed it. Fixed with `accumulate=True` flag.
**Status:** KEPT ✓

### OPT-012: NEON C extension for sgemm (ABANDONED)
**Date:** 2026-03-24
**Hypothesis:** Custom NEON 4x4 micro-kernel should beat OpenBLAS for our small matrix sizes.
**Result:** OpenBLAS is **2.5-4.5x FASTER** than naive NEON sgemm. OpenBLAS has assembly-level NEON kernels with cache-aware blocking. Years of tuning > naive implementation.
**Key insight:** **Matmul is NOT the bottleneck** on this platform. OpenBLAS gets 20+ GFLOPS on ARM. The bottleneck is Python overhead between operations (array allocation, function calls, GC). To beat this, would need the entire forward+backward in C (= reinventing llm.c).
**Status:** ABANDONED ✗

### INSIGHT-001: Baseline Analysis (2026-03-24)
**Baseline result:** 128d/4L (1.3M), Lion, 2000 steps → val_bpb=7.41
**Problem:** Loss plateaus at 6.2 after ~500 steps. Model generates gibberish.
**Root causes identified via micro-experiments:**
1. **Insufficient training length:** 2000 steps = 0.5 epochs. Model only sees 48% of data ONCE.
2. **Optimizer choice:** AdamW converges 3x faster than Lion for this scale. Lion's sign-based updates lose magnitude info critical for small models.
3. **Model capacity:** 1.3M params saturates at bpb ~7.5. Both optimizers plateau at same level → capacity limit, not optimizer limit.
4. **Random baseline:** CE(random, vocab=1024) = ln(1024) = 6.93. Our 6.17 is only 11% better than random.

**Action:** Scale to 192d/6L (3.5M), use AdamW, train 6000 steps (1.5 epochs).
**Target:** val_bpb < 5.0 (50% better than random). val_bpb < 3.0 for coherent text.
