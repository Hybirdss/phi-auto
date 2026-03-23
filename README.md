# phi-auto

Self-improving AI that runs entirely on your phone.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted for Android/Termux with no GPU.

## Quick Start

```bash
cd ~/phi-auto
bash scripts/setup.sh        # Install deps & sanity check
python phi train              # Train baseline model (~30 min)
python phi experiment         # Autonomous experiment loop
python phi improve            # Self-improvement loop
python phi run                # Full pipeline (train → experiment → improve)
```

## CLI

```bash
python phi train                              # Default config
python phi train --config configs/phone.toml  # Phone-optimized (5M params)
python phi train --config configs/tiny.toml   # Quick test (~100K params)
python phi train --dim 256 --layers 8         # Custom overrides
python phi experiment --hours 4 --exp-min 15  # 4h budget, 15min/experiment
python phi improve --cycles 20 --budget 7200  # 20 STaR cycles, 2h budget
python phi status                             # RAM, battery, best val_bpb
python phi generate "Once upon a time"        # Generate text from best model
```

## What It Does

1. **Train** - Tiny GPT on TinyStories, pure numpy, no PyTorch
2. **Experiment** - Auto-search hyperparams & architecture (6 strategies)
3. **Self-improve** - STaR loop: generate → verify → fine-tune on good outputs
4. **Monitor** - Tracks RAM, CPU temp, battery; pauses if overheating

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  phi-auto                        │
│                                                  │
│  Agent (Brain)              Engine (Muscle)       │
│  ├─ experiment.py           ├─ model.py (GPT)    │
│  ├─ hypothesis.py           ├─ train.py          │
│  └─ self_improve.py         ├─ tokenizer.py      │
│                             └─ eval.py           │
│                                                  │
│  Data                       Tools                │
│  ├─ prepare.py              ├─ monitor.py        │
│  └─ loader.py               ├─ logger.py         │
│                             ├─ checkpoint.py     │
│                             └─ config.py         │
└─────────────────────────────────────────────────┘
```

## Requirements

- Android + Termux
- Python 3.10+, numpy
- ~3GB free RAM, ~5GB storage

## Status

See [DEVLOG.md](DEVLOG.md) for progress and experiment results.
