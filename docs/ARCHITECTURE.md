# phi-auto Architecture

> Self-improving AI system for Android/Termux
> Target: Snapdragon ARM, 10GB RAM, no NVIDIA GPU

## System Overview

```
+===========================================================================+
|                          phi-auto System                                   |
|                                                                           |
|  +---------------------------+    +-----------------------------------+   |
|  |     Agent (Brain)         |    |     Engine (Muscle)               |   |
|  |                           |    |                                   |   |
|  |  +---------------------+ |    |  +-----------------------------+  |   |
|  |  | Experiment Planner  | |    |  | GPT Training Loop           |  |   |
|  |  | - hypothesis gen    | |--->|  | - forward / backward        |  |   |
|  |  | - code modification | |    |  | - AdamW optimizer           |  |   |
|  |  | - result analysis   | |    |  | - gradient accumulation     |  |   |
|  |  +---------------------+ |    |  +-----------------------------+  |   |
|  |                           |    |                                   |   |
|  |  +---------------------+ |    |  +-----------------------------+  |   |
|  |  | Self-Improve Loop   | |    |  | Tokenizer                   |  |   |
|  |  | - STaR cycle        | |    |  | - BPE (1024-2048 vocab)     |  |   |
|  |  | - reflect & retry   | |    |  | - byte-level fallback       |  |   |
|  |  | - keep / discard    | |    |  +-----------------------------+  |   |
|  |  +---------------------+ |    |                                   |   |
|  +---------------------------+    +-----------------------------------+   |
|                                                                           |
|  +---------------------------+    +-----------------------------------+   |
|  |     Data Pipeline         |    |     Evaluation                    |   |
|  |                           |    |                                   |   |
|  |  - TinyStories loader     |    |  - val_bpb (bits per byte)       |   |
|  |  - streaming tokenization |    |  - perplexity                    |   |
|  |  - train/val split        |    |  - text generation samples       |   |
|  |  - memory-mapped I/O      |    |  - experiment comparison         |   |
|  +---------------------------+    +-----------------------------------+   |
|                                                                           |
|  +--------------------------------------------------------------------+  |
|  |                      Devlog & Tracking                              |  |
|  |  - results.tsv (experiment log)                                     |  |
|  |  - DEVLOG.md (development journal)                                  |  |
|  |  - checkpoints/ (model snapshots)                                   |  |
|  +--------------------------------------------------------------------+  |
+===========================================================================+
```

## Core Loop: Autonomous Self-Improvement

```
                    +------------------+
                    |   START          |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Load base model |
                    |  + current best  |
                    +--------+---------+
                             |
              +--------------v--------------+
              |                             |
              |   EXPERIMENT LOOP (forever) |
              |                             |
              |   +---------------------+   |
              |   | 1. Generate         |   |
              |   |    hypothesis       |   |
              |   +----------+----------+   |
              |              |              |
              |   +----------v----------+   |
              |   | 2. Modify           |   |
              |   |    train.py         |   |
              |   +----------+----------+   |
              |              |              |
              |   +----------v----------+   |
              |   | 3. Train            |   |
              |   |    (5-30 min)       |   |
              |   +----------+----------+   |
              |              |              |
              |   +----------v----------+   |
              |   | 4. Evaluate         |   |
              |   |    val_bpb          |   |
              |   +----------+----------+   |
              |              |              |
              |   +----------v----------+   |
              |   | 5. Compare          |   |
              |   |    vs best          |   |
              |   +----+----------+-----+   |
              |        |          |         |
              |   +----v---+ +---v------+  |
              |   | KEEP   | | DISCARD  |  |
              |   | update | | revert   |  |
              |   | best   | | to prev  |  |
              |   +----+---+ +---+------+  |
              |        |         |          |
              |        +----+----+          |
              |             |               |
              |   +---------v-----------+   |
              |   | 6. Log to           |   |
              |   |    results.tsv      |   |
              |   |    + DEVLOG.md      |   |
              |   +---------+-----------+   |
              |             |               |
              +------------>+ (loop back)   |
              |                             |
              +-----------------------------+
```

## Self-Improvement Pipeline (STaR + Reflect-Retry)

```
+-----------------------------------------------------------------------+
|                    Self-Improvement Cycle                               |
|                                                                        |
|   Phase A: Generate (inference only, no GPU needed)                    |
|   +------------------+     +------------------+     +--------------+   |
|   | Sample problems  | --> | Model attempts   | --> | Verify with  |   |
|   | from data        |     | solutions        |     | code/checker |   |
|   +------------------+     +------------------+     +------+-------+   |
|                                                            |           |
|                                              +-------------+-------+   |
|                                              |                     |   |
|                                         +----v----+          +-----v-+ |
|                                         | CORRECT |          | WRONG | |
|                                         +----+----+          +---+---+ |
|                                              |                   |     |
|   Phase B: Learn from correct          Phase C: Reflect & Retry  |     |
|   +------------------+                 +------------------+      |     |
|   | Collect correct  |                 | Generate self-   |      |     |
|   | (question,       |                 | reflection on    |      |     |
|   |  reasoning,      |                 | why it failed    |      |     |
|   |  answer) triples |                 +--------+---------+      |     |
|   +--------+---------+                          |                |     |
|            |                           +--------v---------+      |     |
|   +--------v---------+                 | Retry with       |      |     |
|   | LoRA fine-tune   |                 | reflection in    |      |     |
|   | on correct data  |                 | context          |      |     |
|   | (10min / batch)  |                 +--------+---------+      |     |
|   +--------+---------+                          |                |     |
|            |                           +--------v---------+      |     |
|   +--------v---------+                 | If success:      |      |     |
|   | Updated model    |                 | add to Phase B   |------+     |
|   | (better at task) |                 | If fail: skip    |            |
|   +------------------+                 +------------------+            |
+-----------------------------------------------------------------------+
```

## Directory Structure

```
phi-auto/
+-- docs/
|   +-- RESEARCH.md          # Research findings summary
|   +-- ARCHITECTURE.md      # This file
|   +-- REFERENCES.md        # Papers, repos, links
|
+-- src/
|   +-- engine/
|   |   +-- __init__.py
|   |   +-- model.py          # GPT model (tiny, CPU-optimized)
|   |   +-- train.py          # Training loop
|   |   +-- tokenizer.py      # BPE tokenizer (small vocab)
|   |   +-- eval.py           # Evaluation (val_bpb, perplexity)
|   |
|   +-- agent/
|   |   +-- __init__.py
|   |   +-- experiment.py     # Experiment loop (Karpathy-style)
|   |   +-- hypothesis.py     # Hypothesis generation
|   |   +-- self_improve.py   # STaR + Reflect-Retry loops
|   |
|   +-- data/
|   |   +-- __init__.py
|   |   +-- loader.py         # Streaming data loader
|   |   +-- prepare.py        # Dataset download & prep
|   |
|   +-- tools/
|       +-- __init__.py
|       +-- logger.py         # Experiment logging
|       +-- checkpoint.py     # Model save/load
|       +-- monitor.py        # RAM/CPU/battery monitoring
|
+-- configs/
|   +-- default.toml          # Default hyperparameters
|   +-- tiny.toml             # Ultra-small for testing
|   +-- phone.toml            # Optimized for phone
|
+-- scripts/
|   +-- setup.sh              # One-command setup
|   +-- train.sh              # Quick train launcher
|   +-- experiment.sh         # Start autonomous loop
|
+-- logs/                     # Training logs
+-- models/                   # Saved models
+-- checkpoints/              # Training checkpoints
+-- results.tsv               # Experiment results
+-- DEVLOG.md                 # Development journal
+-- README.md                 # Project overview
```

## Component Details

### 1. Engine: Model (src/engine/model.py)

```
GPT Architecture (Phone-Optimized):
+---------------------------------------------------+
| Config                                             |
| - vocab_size: 1024 (byte-level BPE)               |
| - n_embd: 128-256 (embedding dimension)           |
| - n_head: 4-8 (attention heads)                   |
| - n_layer: 4-8 (transformer blocks)               |
| - seq_len: 256 (context window)                   |
| - Total: 1M - 30M parameters                      |
+---------------------------------------------------+
|                                                    |
| Token Embedding [vocab_size x n_embd]              |
|         |                                          |
|         v                                          |
| +-- Transformer Block x n_layer ---------------+   |
| |   RMSNorm -> Attention -> RMSNorm -> MLP     |   |
| |                                               |   |
| |   Attention:                                  |   |
| |   - Q, K, V projections                      |   |
| |   - RoPE positional encoding                 |   |
| |   - Causal mask (no flash attention)         |   |
| |   - Simple scaled dot-product                |   |
| |                                               |   |
| |   MLP:                                        |   |
| |   - Linear -> SiLU -> Linear                 |   |
| |   - 4x expansion ratio                       |   |
| +-----------------------------------------------+  |
|         |                                          |
|         v                                          |
| RMSNorm -> LM Head [n_embd x vocab_size]          |
|         |                                          |
|         v                                          |
| Softmax -> Loss (Cross Entropy)                    |
+---------------------------------------------------+
```

### 2. Training Phases

```
Phase 0: Validate Setup        (~1 min)
  - microGPT sanity check
  - Verify CPU, RAM, storage

Phase 1: Pretrain Tiny Model   (~1-4 hrs)
  - 1-5M params on TinyStories
  - Establish baseline val_bpb
  - Save checkpoint

Phase 2: Autonomous Research   (overnight, continuous)
  - Karpathy Loop: modify -> train -> eval -> keep/discard
  - Each experiment: 5-30 min
  - ~3-6 experiments/hour

Phase 3: Self-Improvement      (continuous)
  - STaR: generate -> verify -> fine-tune on correct
  - Reflect-Retry on failures
  - Gradually improving model
```

### 3. Memory Management Strategy

```
+-- 10GB Total RAM -------------------------------------------+
|                                                              |
|  [OS + Termux: ~2GB]                                        |
|  [Python Runtime: ~300MB]                                    |
|                                                              |
|  [Model Parameters]     Depends on config:                   |
|    1M  params = ~4MB    (fp32)                               |
|    5M  params = ~20MB   (fp32)                               |
|    30M params = ~120MB  (fp32)                               |
|                                                              |
|  [Gradients]            Same size as parameters              |
|  [Optimizer State]      2x parameters (AdamW: m + v)         |
|  [Activations]          ~batch_size * seq_len * n_embd * 4B  |
|  [Data Buffer]          ~batch_size * seq_len * 4B           |
|                                                              |
|  Example: 5M model, batch=8, seq=256                         |
|    Params:     20MB                                          |
|    Grads:      20MB                                          |
|    Optimizer:  40MB                                          |
|    Activs:     ~50MB                                         |
|    Data:       ~8KB                                          |
|    Total:      ~130MB  (plenty of room!)                     |
|                                                              |
|  Example: 30M model, batch=4, seq=256                        |
|    Params:     120MB                                         |
|    Grads:      120MB                                         |
|    Optimizer:  240MB                                         |
|    Activs:     ~100MB                                        |
|    Total:      ~580MB  (still comfortable)                   |
+--------------------------------------------------------------+
```

### 4. Evaluation Metrics

```
Primary:   val_bpb (validation bits per byte) - lower is better
Secondary: perplexity, generation quality (human-readable text)
Monitor:   RAM usage, CPU temp, battery drain, tokens/second
```

### 5. Technology Stack

```
Layer 0 (Hardware):  ARM CPU (Snapdragon), 10GB RAM
Layer 1 (OS):        Android + Termux
Layer 2 (Runtime):   Python 3.12, numpy
Layer 3 (Engine):    Custom GPT (pure Python/numpy, no PyTorch)
Layer 4 (Agent):     Experiment loop, self-improvement logic
Layer 5 (Interface): CLI scripts, DEVLOG.md, results.tsv
```
