"""
Hypothesis generator for phi-auto.
Generates experiment configurations to try next.
Rule-based approach: systematic hyperparameter & architecture search.
"""

import random
import math
import copy


# ---------------------------------------------------------------------------
# Search Space Definition
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    'n_embd':     [64, 96, 128, 160, 192, 256, 320, 384],
    'n_head':     [2, 4, 6, 8],
    'n_layer':    [2, 3, 4, 6, 8, 10, 12],
    'seq_len':    [64, 128, 256, 384, 512],
    'lr':         [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3],
    'min_lr':     [1e-5, 3e-5, 1e-4],
    'batch_size': [2, 4, 8, 16],
    'grad_accum': [1, 2, 4, 8],
    'weight_decay': [0.01, 0.05, 0.1, 0.2],
    'warmup_steps': [20, 50, 100, 200],
    'vocab_size': [512, 1024, 2048, 4096],
}

# Constraints: n_embd must be divisible by n_head
# Memory budget: ~500MB max for model + optimizer + gradients


def estimate_params(config):
    """Estimate total parameters for a config."""
    V = config['vocab_size']
    D = config['n_embd']
    L = config['n_layer']
    H = 4 * D  # MLP hidden size

    emb = V * D
    per_layer = (
        D +                    # ln1 (RMSNorm)
        D * 3 * D +            # QKV projection
        D * D +                # output projection
        D +                    # ln2 (RMSNorm)
        D * H +                # gate
        D * H +                # up
        H * D                  # down
    )
    head = D + D * V           # ln_f + lm_head

    return emb + L * per_layer + head


def estimate_memory_mb(config):
    """Estimate peak memory in MB for training."""
    n_params = estimate_params(config)
    B = config.get('batch_size', 4)
    T = config.get('seq_len', 128)
    D = config['n_embd']
    L = config['n_layer']

    param_mb = n_params * 4 / (1024 * 1024)
    grad_mb = param_mb
    optim_mb = param_mb * 2  # m + v
    # activations: rough estimate
    act_mb = B * T * D * L * 4 * 4 / (1024 * 1024)

    return param_mb + grad_mb + optim_mb + act_mb


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

def strategy_random(base_config, n=1):
    """Generate random valid configs."""
    configs = []
    for _ in range(n):
        cfg = copy.deepcopy(base_config)
        # mutate 1-3 hyperparameters
        keys = random.sample(list(SEARCH_SPACE.keys()), k=random.randint(1, 3))
        for key in keys:
            cfg[key] = random.choice(SEARCH_SPACE[key])

        # fix constraints
        if cfg['n_embd'] % cfg['n_head'] != 0:
            valid_heads = [h for h in SEARCH_SPACE['n_head'] if cfg['n_embd'] % h == 0]
            cfg['n_head'] = random.choice(valid_heads) if valid_heads else 4

        if estimate_memory_mb(cfg) < 500:
            configs.append(cfg)

    return configs


def strategy_scale_model(base_config, direction='up'):
    """Scale model up or down."""
    cfg = copy.deepcopy(base_config)
    if direction == 'up':
        # try bigger model
        options = [
            {'n_embd': min(cfg['n_embd'] + 64, 384)},
            {'n_layer': min(cfg['n_layer'] + 2, 12)},
            {'n_embd': min(cfg['n_embd'] + 32, 384), 'n_layer': min(cfg['n_layer'] + 1, 12)},
        ]
    else:
        # try smaller but faster
        options = [
            {'n_embd': max(cfg['n_embd'] - 64, 64)},
            {'n_layer': max(cfg['n_layer'] - 2, 2)},
            {'batch_size': min(cfg.get('batch_size', 4) * 2, 16)},
        ]

    choice = random.choice(options)
    cfg.update(choice)

    if cfg['n_embd'] % cfg['n_head'] != 0:
        valid_heads = [h for h in SEARCH_SPACE['n_head'] if cfg['n_embd'] % h == 0]
        cfg['n_head'] = random.choice(valid_heads) if valid_heads else 4

    return cfg


def strategy_lr_search(base_config):
    """Try different learning rates."""
    cfg = copy.deepcopy(base_config)
    current_lr = cfg.get('lr', 3e-4)
    # try nearby LRs
    candidates = [current_lr * 0.3, current_lr * 0.5, current_lr * 2.0, current_lr * 3.0]
    candidates = [lr for lr in candidates if 1e-5 <= lr <= 3e-3]
    cfg['lr'] = random.choice(candidates) if candidates else current_lr
    cfg['min_lr'] = cfg['lr'] / 10
    return cfg


def strategy_seq_len_search(base_config):
    """Try different sequence lengths."""
    cfg = copy.deepcopy(base_config)
    current = cfg.get('seq_len', 128)
    candidates = [s for s in SEARCH_SPACE['seq_len'] if s != current]
    cfg['seq_len'] = random.choice(candidates)
    # adjust batch size if needed to fit memory
    if cfg['seq_len'] > current:
        cfg['batch_size'] = max(2, cfg.get('batch_size', 4) // 2)
    return cfg


def strategy_efficient(base_config):
    """Optimize for speed: same compute, better throughput."""
    cfg = copy.deepcopy(base_config)
    options = [
        # wider but shallower
        {'n_embd': min(cfg['n_embd'] + 64, 384), 'n_layer': max(cfg['n_layer'] - 2, 2)},
        # bigger batch
        {'batch_size': min(cfg.get('batch_size', 4) * 2, 16), 'grad_accum': max(1, cfg.get('grad_accum', 2) // 2)},
        # smaller vocab (faster softmax)
        {'vocab_size': max(512, cfg.get('vocab_size', 1024) // 2)},
    ]
    choice = random.choice(options)
    cfg.update(choice)

    if cfg['n_embd'] % cfg['n_head'] != 0:
        valid_heads = [h for h in SEARCH_SPACE['n_head'] if cfg['n_embd'] % h == 0]
        cfg['n_head'] = random.choice(valid_heads) if valid_heads else 4

    return cfg


# ---------------------------------------------------------------------------
# Main Hypothesis Generator
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    ('random', strategy_random, 0.3),
    ('scale_up', lambda c: strategy_scale_model(c, 'up'), 0.2),
    ('scale_down', lambda c: strategy_scale_model(c, 'down'), 0.1),
    ('lr_search', strategy_lr_search, 0.2),
    ('seq_len', strategy_seq_len_search, 0.1),
    ('efficient', strategy_efficient, 0.1),
]


class HypothesisGenerator:
    """Generates experiment hypotheses based on past results."""

    def __init__(self):
        self.history = []  # (config, val_bpb, strategy_name)

    def record(self, config, val_bpb, strategy_name):
        self.history.append((config, val_bpb, strategy_name))

    def generate(self, base_config, n=1):
        """Generate n experiment configs to try."""
        experiments = []

        # weight strategies by past success
        weights = self._compute_weights()

        for _ in range(n):
            # pick strategy
            names, funcs, _ = zip(*ALL_STRATEGIES)
            idx = random.choices(range(len(ALL_STRATEGIES)), weights=weights, k=1)[0]
            strategy_name, strategy_fn, _ = ALL_STRATEGIES[idx]

            if strategy_name == 'random':
                configs = strategy_fn(base_config, n=1)
                cfg = configs[0] if configs else copy.deepcopy(base_config)
            else:
                cfg = strategy_fn(base_config)

            # validate
            mem = estimate_memory_mb(cfg)
            params = estimate_params(cfg)
            if mem > 500:
                continue

            experiments.append({
                'config': cfg,
                'strategy': strategy_name,
                'est_params': params,
                'est_memory_mb': round(mem),
                'hypothesis': self._describe(strategy_name, base_config, cfg),
            })

        return experiments

    def _compute_weights(self):
        """Weight strategies by their past success rate."""
        weights = [w for _, _, w in ALL_STRATEGIES]

        if len(self.history) < 3:
            return weights

        # count improvements per strategy
        strategy_scores = {}
        best_so_far = float('inf')
        for config, bpb, name in self.history:
            if name not in strategy_scores:
                strategy_scores[name] = {'wins': 0, 'total': 0}
            strategy_scores[name]['total'] += 1
            if bpb < best_so_far:
                strategy_scores[name]['wins'] += 1
                best_so_far = bpb

        # boost weights for successful strategies
        for i, (name, _, base_w) in enumerate(ALL_STRATEGIES):
            if name in strategy_scores:
                s = strategy_scores[name]
                if s['total'] > 0:
                    win_rate = s['wins'] / s['total']
                    weights[i] = base_w * (1 + 2 * win_rate)

        return weights

    def _describe(self, strategy, base, new):
        """Human-readable hypothesis description."""
        changes = []
        for key in new:
            if key in base and new[key] != base.get(key):
                changes.append(f"{key}: {base.get(key)} -> {new[key]}")
        change_str = ", ".join(changes[:5]) if changes else "no changes"
        return f"[{strategy}] {change_str}"


if __name__ == "__main__":
    base = {
        'vocab_size': 1024, 'n_embd': 128, 'n_head': 4, 'n_layer': 4,
        'seq_len': 128, 'batch_size': 4, 'grad_accum': 2, 'lr': 3e-4,
        'min_lr': 3e-5, 'warmup_steps': 50, 'weight_decay': 0.1,
    }
    gen = HypothesisGenerator()
    exps = gen.generate(base, n=5)
    for e in exps:
        print(f"  [{e['strategy']}] params={e['est_params']:,} mem={e['est_memory_mb']}MB")
        print(f"    {e['hypothesis']}")
