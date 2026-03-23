"""
Memory-mapped data loader for phi-auto.
Pre-tokenizes data into .npy arrays, then loads via mmap for zero-copy access.
~7x faster than JSON-streaming loader.
"""

import os
import json
import numpy as np


CACHE_DIR = os.path.expanduser("~/.cache/phi-auto/data")


def prepare_mmap_data(jsonl_path, tokenizer, output_path=None, max_docs=None):
    """Pre-tokenize entire dataset into a flat numpy array on disk."""
    if output_path is None:
        base = os.path.splitext(os.path.basename(jsonl_path))[0]
        output_path = os.path.join(CACHE_DIR, f"{base}_tokens.npy")

    if os.path.exists(output_path):
        data = np.load(output_path, mmap_mode='r')
        print(f"Mmap data exists: {output_path} ({len(data):,} tokens)")
        return output_path

    print(f"Pre-tokenizing {jsonl_path}...")
    all_tokens = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            try:
                text = json.loads(line.strip())["text"]
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
            except (json.JSONDecodeError, KeyError):
                continue

    arr = np.array(all_tokens, dtype=np.uint16)  # vocab < 65536
    np.save(output_path, arr)
    print(f"Saved {len(arr):,} tokens to {output_path} "
          f"({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
    return output_path


class MMapDataLoader:
    """Zero-copy data loading from memory-mapped numpy array.
    No tokenization at runtime. No JSON parsing. Just raw speed.
    """
    def __init__(self, npy_path, batch_size, seq_len, shuffle=True):
        self.data = np.load(npy_path, mmap_mode='r')
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
        self.shuffle = shuffle
        self._epoch = 0
        self._pos = 0

    def get_batch(self):
        B, T = self.batch_size, self.seq_len

        if self.shuffle:
            # random slices from the dataset — better coverage
            max_start = self.n_tokens - T - 1
            if max_start <= 0:
                max_start = 1
            ix = np.random.randint(0, max_start, size=B)
            x = np.stack([self.data[i:i + T].astype(np.int32) for i in ix])
            y = np.stack([self.data[i + 1:i + T + 1].astype(np.int32) for i in ix])
        else:
            # sequential for validation
            if self._pos + B * (T + 1) > self.n_tokens:
                self._pos = 0
                self._epoch += 1

            x = np.zeros((B, T), dtype=np.int32)
            y = np.zeros((B, T), dtype=np.int32)
            for i in range(B):
                start = self._pos + i * (T + 1)
                chunk = self.data[start:start + T + 1].astype(np.int32)
                if len(chunk) < T + 1:
                    chunk = np.pad(chunk, (0, T + 1 - len(chunk)))
                x[i] = chunk[:T]
                y[i] = chunk[1:T + 1]

            self._pos += B * (T + 1)

        return x, y, self._epoch

    @property
    def epoch(self):
        return self._epoch

    def close(self):
        pass  # mmap cleaned up by GC
