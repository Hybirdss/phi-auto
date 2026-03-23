# Research Direction: Small Model Scaling Laws

> Where does a language model stop counting tokens and start understanding language?

Last updated: 2026-03-24

---

## The Question

At what combination of model size and data volume does a transformer transition from **memorizing token frequencies** (unigram-level) to **learning contextual patterns** (bigram+ level)?

This is under-studied. Scaling law research (Kaplan 2020, Hoffmann 2022) focuses on models from 70M to 175B parameters. The sub-10M regime — where models operate on the edge of capability — has almost no systematic characterization for language.

### What We Already Know (from phi-auto experiments)

- 1.3M params on 4.3M tokens: plateau at unigram CE (6.25 nats). No contextual learning.
- Same model on 5 stories (~2K tokens): loss drops to 5.38 (below unigram). Context learning works.
- The gap between unigram (6.25) and bigram (3.36) on TinyStories is 46%. That's the space we need to enter.

### What We Want to Find

1. The **minimum model size** that breaks the unigram wall on TinyStories (20K stories).
2. The **capacity-data frontier** — for each model size, how much data can it compress beyond unigram?
3. Whether **phase transitions** exist (sudden jumps from memorization to generalization, a.k.a. grokking).
4. How the **tokens-per-parameter ratio** affects learning in this regime.

---

## Related Work

### Scaling Laws

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) | Loss scales as power law: L(N) ~ N^(-0.076). Tested 768 params to billions. | Baseline formula. Unclear if it holds sub-10M. |
| [Hoffmann et al. 2022 (Chinchilla)](https://arxiv.org/abs/2203.15556) | Compute-optimal ratio ≈ 20 tokens/parameter. Trained 400+ models from 70M–16B. | Our 1.3M model gets 4.3M/1.3M ≈ 3.3 tokens/param — severely undertrained by Chinchilla standards. |
| [Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264) | Up to 4 epochs of repeated data is essentially free (<0.5% loss increase). In data-constrained regime: train smaller models for more epochs. | Directly applicable. Our model trained for <1 epoch. Should push to 4+. |
| [Scaling Laws in the Tiny Regime (2025)](https://arxiv.org/abs/2603.07365) | 22K–19.8M params. Power law exponents are **1.4–2x steeper** than large-model regime. | Scaling behavior is fundamentally different in our parameter range. Cannot directly extrapolate from Chinchilla. |
| [Epoch AI 2024](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt) | Chinchilla replication found discrepancies. Small-model regime may have different optimal ratios. | Supports the hypothesis that sub-10M needs its own scaling law. |

### Phase Transitions

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| [Phase Transitions in Small Transformers (2025)](https://arxiv.org/abs/2511.12768) | 3.6M-param GPT shows phase-transition-like discontinuities during training. Transition from random fragments to coherent words is measurable. | Phase transitions are NOT exclusive to large models. We might see one if we train long enough. |
| [Power et al. 2022 (Grokking)](https://arxiv.org/abs/2201.02177) | Small transformers on modular arithmetic: memorize first, then suddenly generalize at ~step 7000 (step-function jump). | Suggests that our model might need much longer training to "grok" language patterns. |
| [Gromov 2023](https://arxiv.org/abs/2301.02679) | Even 2-layer networks grok on modular arithmetic without regularization. Four phases: comprehension, grokking, memorization, confusion. | Weight decay and regularization can speed up the transition — testable. |

### TinyStories and Small LMs

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| [Eldan & Li 2023 (TinyStories)](https://arxiv.org/abs/2305.07759) | Models <10M params produce fluent stories. Grammar emerges at 1–3M params. 3M model beats GPT-2 125M on story generation. | Our target dataset. Proves our parameter range (1–10M) should be sufficient. |
| [BabyLM Challenge](https://babylm.github.io/) | 10M words (~13M tokens) sufficient for substantial grammatical competence. Architecture and training objective matter more than param count when data is scarce. | Our 4.3M tokens may be enough — if architecture and training are right. |
| [Super Tiny Language Models (2024)](https://arxiv.org/abs/2405.14159) | 10M–100M params with byte-level tokenization, weight tying (90–95% param reduction), self-play training. | Directly relevant. Weight tying and self-play at our scale. |
| [TinyLlama (Zhang et al. 2024)](https://arxiv.org/abs/2401.02385) | 1.1B params trained on 3T tokens (2,700:1 ratio). Beats OPT-1.3B despite "overtraining." | Small models benefit from massive overtraining. Our 3.3:1 ratio is absurdly low. |

### N-gram vs Neural Crossover

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) | Neural LM achieves 24% lower perplexity than trigram on Brown corpus. | The original proof that neural beats n-gram. But on much larger models. |
| Domain-specific studies | In small-data domain-specific settings, trigrams can outperform GPT-2. | Our situation: small data + small model. We're in the regime where n-grams might win. |

### Self-Improvement at Small Scale

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| [STaR (Zelikman et al. 2022)](https://arxiv.org/abs/2203.14465) | Self-taught reasoning via iterative rationale generation. Tested at 6B, not sub-100M. | Method is implemented in our codebase. Unknown if it works at 1–10M. |
| [SPIN (Chen et al. 2024)](https://arxiv.org/abs/2401.01335) | Self-play fine-tuning, no reward model needed. 3 iterations saturate. Tested at 7B. | Also implemented. Open question at our scale. |
| [Super Tiny LMs (2024)](https://arxiv.org/abs/2405.14159) | Self-play training at 10M–100M params. | Only known attempt at self-improvement in our parameter range. |

**Key gap:** STaR, SPIN, and Quiet-STaR are all tested at 3B–7B. Whether self-improvement works when the model lacks capacity to generate useful text is an open question.

### On-Device Training (Prior Art)

| System | What | Scale |
|--------|------|-------|
| [Google Gboard FL (Hard et al. 2019)](https://arxiv.org/abs/1811.03604) | Federated training of 1.4M-param RNN across 1.5M phones | Production deployment |
| [Apple Private FL (2023)](https://machinelearning.apple.com/research/large-vocabulary) | Private federated LM training on devices with DP | Research |
| [FwdLLM (Xu et al. 2024)](https://arxiv.org/abs/2308.13894) | Fine-tuned LLaMA-7B on Pixel 7 in 10 min, 1.5GB peak memory | No backprop needed |
| [PockEngine (MIT, 2023)](https://arxiv.org/abs/2310.17752) | Compile-time sparse backprop. LLaMA2-7B at 550 tok/s on Jetson | 15x speedup over TF |
| [POET (Patil et al. 2022)](https://arxiv.org/abs/2207.07697) | Training on devices with as low as 32KB memory | BERT on Cortex-M |

---

## Experiment Plan

### Phase 1: Break the Unigram Wall

The immediate goal. Can we push val_loss below the bigram baseline (3.36)?

**Grid experiment: model size × training length**

| | 2K steps | 8K steps | 16K steps (4 epochs) |
|-|----------|----------|----------------------|
| 1.3M (128d/4L) | 6.17 (done) | TODO | TODO |
| 3.5M (192d/6L) | TODO | TODO | TODO |
| 8M (256d/8L) | TODO | TODO | TODO |

Config for all: AdamW, lr=3e-4 (scaled down for larger models), batch=8, seq=128, wd=0.1, clip=1.0.

**Rationale:**
- Muennighoff 2023: 4 epochs of repeated data is free. We've only done <1 epoch.
- Chinchilla: our 3.3 tokens/param ratio is ~6x below optimal (20:1). Need more training.
- TinyStories paper: grammar emerges at 1–3M. Our 1.3M should be near the threshold.

### Phase 2: Map the Scaling Frontier

If Phase 1 succeeds (some config breaks unigram), systematically map:

1. **Minimum params to beat unigram** — binary search between 500K and 3.5M
2. **Minimum params to beat bigram** — binary search between 1M and 10M
3. **Optimal tokens-per-parameter ratio** in the 1–10M range
4. **Phase transition detection** — monitor for sudden generalization jumps (grokking-like)

### Phase 3: Self-Improvement

If Phase 2 produces a model that generates coherent text:

1. Test STaR at this scale (already implemented)
2. Test SPIN at this scale (already implemented)
3. Measure: does self-improvement work when model capacity is tight?

This would address the key open gap in the literature — self-improvement has never been rigorously tested below 3B parameters.

---

## Key Hypotheses to Test

| # | Hypothesis | Test | Success Criterion |
|---|-----------|------|-------------------|
| H1 | 1.3M model is undertrained, not undercapacitated | Train B1 config for 16K steps (4 epochs) | val_loss < 6.0 (below unigram) |
| H2 | 3.5M model breaks unigram within 1 epoch | 192d/6L, 4K steps | val_loss < 5.5 |
| H3 | Phase transition exists in training curve | Monitor per-step loss for discontinuities | Sudden loss drop > 10% in <100 steps |
| H4 | 8M model reaches bigram level | 256d/8L, 8K steps | val_loss < 3.5 |
| H5 | Self-improvement works at 3.5M scale | STaR/SPIN on best model | Measurable val_loss decrease from self-generated data |

---

## References

Scaling laws:
- Kaplan et al. (2020). Scaling Laws for Neural Language Models. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Muennighoff et al. (2023). Scaling Data-Constrained Language Models. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)
- Scaling Laws in the Tiny Regime (2025). [arXiv:2603.07365](https://arxiv.org/abs/2603.07365)
- Epoch AI (2024). Chinchilla Scaling: A Replication Attempt. [epoch.ai](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)

Phase transitions:
- Phase Transitions in Small Transformers (2025). [arXiv:2511.12768](https://arxiv.org/abs/2511.12768)
- Power et al. (2022). Grokking: Generalization Beyond Overfitting. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Gromov (2023). Grokking Modular Arithmetic. [arXiv:2301.02679](https://arxiv.org/abs/2301.02679)

Small language models:
- Eldan & Li (2023). TinyStories. [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
- BabyLM Challenge. [babylm.github.io](https://babylm.github.io/)
- Hillier et al. (2024). Super Tiny Language Models. [arXiv:2405.14159](https://arxiv.org/abs/2405.14159)
- Zhang et al. (2024). TinyLlama. [arXiv:2401.02385](https://arxiv.org/abs/2401.02385)

Self-improvement:
- Zelikman et al. (2022). STaR: Self-Taught Reasoner. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Chen et al. (2024). SPIN: Self-Play Fine-Tuning. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335)
- Zelikman et al. (2024). Quiet-STaR. [arXiv:2403.09629](https://arxiv.org/abs/2403.09629)

On-device training:
- Hard et al. (2019). Federated Learning for Mobile Keyboard Prediction. [arXiv:1811.03604](https://arxiv.org/abs/1811.03604)
- Xu et al. (2024). FwdLLM. [arXiv:2308.13894](https://arxiv.org/abs/2308.13894)
- Zhu et al. (2023). PockEngine. [arXiv:2310.17752](https://arxiv.org/abs/2310.17752)
- Patil et al. (2022). POET. [arXiv:2207.07697](https://arxiv.org/abs/2207.07697)

Fundamentals:
- Bengio et al. (2003). A Neural Probabilistic Language Model. [JMLR](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
