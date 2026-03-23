# Research Summary - On-Device Self-Improving AI

> Compiled: 2026-03-23 | 4 research agents, 142 web searches, 50+ repos/papers

## Key Findings

### On-Device Training Frameworks (Verified Working)

| Framework | Type | Phone Benchmark | Repo |
|-----------|------|-----------------|------|
| QVAC Fabric | Vulkan GPU LoRA | 125M:10min, 1B:78min (S25) | github.com/tetherto/qvac-fabric-llm.cpp |
| MobileFineTuner | C++ Full FT | GPT-2 124M on Pixel 8 Pro | arxiv.org/abs/2512.08211 |
| Alibaba MNN | Train+Infer | 30+ production apps | github.com/alibaba/MNN |
| llm.c | Pure C pretrain | ~10-15s/step (124M, ARM est.) | github.com/karpathy/llm.c |
| tinygrad | CLANG/CPU | Production on Snapdragon 845 | github.com/tinygrad/tinygrad |
| llama.cpp LoRA | CPU LoRA | 3B:4-8hrs (phone est.) | github.com/ggml-org/llama.cpp |

### Self-Improving Techniques (Low Overhead)

| Technique | Overhead | Paper |
|-----------|----------|-------|
| Reflect, Retry, Reward | Inference only | arxiv.org/abs/2505.24726 |
| RLM (Recursive LM) | Inference + Python REPL | arxiv.org/abs/2512.24601 |
| STaR (Self-Taught Reasoner) | LoRA fine-tune per cycle | arxiv.org/abs/2503.04625 |
| SEAL (Self-Adapting LM) | LoRA on Llama-1B | arxiv.org/abs/2506.10943 |
| Absolute Zero Reasoner | RL loop, zero data | arxiv.org/abs/2505.03335 |
| qTTT (Query Test-Time Train) | ~1% params update | arxiv.org/abs/2512.13898 |
| Darwin Godel Machine | Code evolution, no weights | arxiv.org/abs/2505.22954 |

### Dead Ends (Skip These)

- llama.cpp train-text-from-scratch: Removed from repo
- JAX on Termux: No ARM wheels
- Mojo: No Android support
- GGML training: Experimental, incomplete backward pass

### Memory Budget (10GB Phone)

```
Base model (Q4 3B BitNet)     ~375MB
LoRA adapter + optimizer      ~200MB
KV cache                      ~500MB
Training batch buffer         ~500MB
Replay buffer (compressed)    ~100MB
Python runtime + tools        ~500MB
OS/overhead                   ~2GB
Headroom                      ~5.8GB
```

### Optimal Model Sizes for Phone

| Params | RAM (fp32) | RAM (Q4) | Train Time (est.) |
|--------|-----------|----------|-------------------|
| 4K (microGPT) | <1MB | - | ~1 min |
| 1-5M | 4-20MB | - | 5-30 min |
| 10-30M | 40-120MB | - | 1-4 hrs |
| 124M (GPT-2) | 500MB | 125MB | 8-42 hrs |
| 350M+ | >1.4GB | 350MB | Days |
