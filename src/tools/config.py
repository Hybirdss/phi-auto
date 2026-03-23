"""
Configuration loader for phi-auto.
Reads TOML config files and merges with defaults.
"""

import os


def parse_toml_simple(path):
    """Parse a simple TOML file (flat sections, basic types only)."""
    result = {}
    current_section = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip()
                if current_section not in result:
                    result[current_section] = {}
                continue

            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.split('#')[0].strip()  # remove inline comments

                # parse value
                parsed = _parse_value(val)

                if current_section:
                    result[current_section][key] = parsed
                else:
                    result[key] = parsed

    return result


def _parse_value(val):
    """Parse a TOML value string."""
    if val.lower() == 'true':
        return True
    if val.lower() == 'false':
        return False

    # try int
    try:
        return int(val)
    except ValueError:
        pass

    # try float (including scientific notation)
    try:
        return float(val)
    except ValueError:
        pass

    # string
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return val[1:-1]

    return val


def load_config(path):
    """Load config from TOML file and flatten to training config dict."""
    raw = parse_toml_simple(path)

    config = {}
    # flatten sections into a single dict
    for key, val in raw.items():
        if isinstance(val, dict):
            config.update(val)
        else:
            config[key] = val

    return config


def merge_configs(base, override):
    """Merge override config into base config."""
    result = dict(base)
    for key, val in override.items():
        if val is not None:
            result[key] = val
    return result


DEFAULT_CONFIG = {
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
    'time_budget': 1800,
}


def get_config(config_path=None):
    """Get training config, optionally from a TOML file."""
    config = dict(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        overrides = load_config(config_path)
        config = merge_configs(config, overrides)
    return config


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg = get_config(sys.argv[1])
    else:
        cfg = get_config()
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
