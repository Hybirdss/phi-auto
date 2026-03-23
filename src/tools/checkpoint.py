"""
Checkpoint management for phi-auto.
Save/load model states, manage checkpoint history.
"""

import os
import json
import numpy as np
import time

CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')


def save_model(model, path=None, tag="latest", metadata=None):
    """Save model parameters to .npz file."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(CKPT_DIR, f"model_{tag}.npz")

    params = {}
    param_names = []
    for i, (p, name) in enumerate(model.all_params()):
        key = f"p_{i}"
        params[key] = p
        param_names.append(name)

    meta = {
        'n_params': model.n_params,
        'config': {
            'vocab_size': model.config.vocab_size,
            'n_embd': model.config.n_embd,
            'n_head': model.config.n_head,
            'n_layer': model.config.n_layer,
            'seq_len': model.config.seq_len,
        },
        'param_names': param_names,
        'timestamp': time.time(),
    }
    if metadata:
        meta.update(metadata)

    np.savez(path, **params)

    # save metadata alongside
    meta_path = path.replace('.npz', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return path


def load_model(model, path=None, tag="latest"):
    """Load model parameters from .npz file."""
    if path is None:
        path = os.path.join(CKPT_DIR, f"model_{tag}.npz")

    if not os.path.exists(path):
        return False

    data = np.load(path)
    all_params = model.all_params()

    for i, (p, name) in enumerate(all_params):
        key = f"p_{i}"
        if key in data:
            loaded = data[key]
            if loaded.shape == p.shape:
                p[:] = loaded
            else:
                print(f"  Warning: shape mismatch for {key}: "
                      f"{loaded.shape} vs {p.shape}, skipping")

    return True


def load_metadata(path=None, tag="latest"):
    """Load checkpoint metadata."""
    if path is None:
        path = os.path.join(CKPT_DIR, f"model_{tag}_meta.json")
    else:
        path = path.replace('.npz', '_meta.json')

    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)


def list_checkpoints():
    """List all available checkpoints."""
    if not os.path.exists(CKPT_DIR):
        return []

    ckpts = []
    for f in sorted(os.listdir(CKPT_DIR)):
        if f.endswith('.npz'):
            path = os.path.join(CKPT_DIR, f)
            meta = load_metadata(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            ckpts.append({
                'file': f,
                'path': path,
                'size_mb': round(size_mb, 1),
                'metadata': meta,
            })
    return ckpts


def copy_checkpoint(src_tag, dst_tag):
    """Copy a checkpoint (for backup before experiments)."""
    import shutil
    src = os.path.join(CKPT_DIR, f"model_{src_tag}.npz")
    dst = os.path.join(CKPT_DIR, f"model_{dst_tag}.npz")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        src_meta = src.replace('.npz', '_meta.json')
        dst_meta = dst.replace('.npz', '_meta.json')
        if os.path.exists(src_meta):
            shutil.copy2(src_meta, dst_meta)
        return True
    return False


def cleanup_old_checkpoints(keep=5):
    """Remove old experiment checkpoints, keeping the N most recent + best/latest."""
    protected = {'model_best.npz', 'model_latest.npz', 'model_final.npz',
                 'model_baseline.npz'}
    if not os.path.exists(CKPT_DIR):
        return

    exp_ckpts = []
    for f in os.listdir(CKPT_DIR):
        if f.endswith('.npz') and f not in protected:
            path = os.path.join(CKPT_DIR, f)
            exp_ckpts.append((os.path.getmtime(path), path, f))

    exp_ckpts.sort(reverse=True)
    for _, path, f in exp_ckpts[keep:]:
        os.remove(path)
        meta_path = path.replace('.npz', '_meta.json')
        if os.path.exists(meta_path):
            os.remove(meta_path)


if __name__ == "__main__":
    print("=== Checkpoint Manager ===")
    ckpts = list_checkpoints()
    if ckpts:
        for c in ckpts:
            print(f"  {c['file']} ({c['size_mb']} MB)")
    else:
        print("  No checkpoints found.")
