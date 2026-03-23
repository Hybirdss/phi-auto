"""
Streaming data loader for phi-auto.
Memory-efficient: reads data on-the-fly, no full dataset in memory.
"""

import json
import os
import numpy as np


class DataLoader:
    """Streaming data loader with BPE tokenization."""

    def __init__(self, data_path, tokenizer, batch_size, seq_len, shuffle=True):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self._buffer = []
        self._token_buffer = []
        self._file = None
        self._epoch = 0

    def _open_file(self):
        if self._file is not None:
            self._file.close()
        self._file = open(self.data_path, 'r')
        self._epoch += 1

    def _fill_token_buffer(self, min_tokens):
        """Read enough text to fill token buffer."""
        if self._file is None:
            self._open_file()

        while len(self._token_buffer) < min_tokens:
            line = self._file.readline()
            if not line:
                self._open_file()
                line = self._file.readline()
                if not line:
                    break
            try:
                text = json.loads(line.strip())["text"]
                tokens = self.tokenizer.encode(text)
                self._token_buffer.extend(tokens)
            except (json.JSONDecodeError, KeyError):
                continue

    def get_batch(self):
        """Get one batch of (input, target) pairs."""
        needed = self.batch_size * (self.seq_len + 1)
        self._fill_token_buffer(needed)

        if len(self._token_buffer) < needed:
            # not enough data, wrap around
            self._fill_token_buffer(needed)

        x = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        y = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)

        for i in range(self.batch_size):
            start = i * (self.seq_len + 1)
            chunk = self._token_buffer[start:start + self.seq_len + 1]
            if len(chunk) < self.seq_len + 1:
                # pad with zeros if needed
                chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
            x[i] = chunk[:self.seq_len]
            y[i] = chunk[1:self.seq_len + 1]

        # remove consumed tokens
        consumed = self.batch_size * (self.seq_len + 1)
        self._token_buffer = self._token_buffer[consumed:]

        return x, y, self._epoch

    @property
    def epoch(self):
        return self._epoch

    def close(self):
        if self._file is not None:
            self._file.close()
