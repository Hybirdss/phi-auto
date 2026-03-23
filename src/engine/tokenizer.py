"""
Simple byte-level BPE tokenizer for phi-auto.
Starts with 256 byte tokens + merges up to vocab_size.
"""

import json
import os
import re
from collections import Counter


class ByteBPETokenizer:
    """Byte-level BPE tokenizer with small vocabulary."""

    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.merges = {}       # (a, b) -> merged_id
        self.vocab = {}        # id -> bytes
        self.inverse_vocab = {}  # bytes -> id
        self._build_base_vocab()

    def _build_base_vocab(self):
        """Initialize with 256 byte tokens."""
        for i in range(256):
            self.vocab[i] = bytes([i])
            self.inverse_vocab[bytes([i])] = i

    def train(self, texts, verbose=True):
        """Train BPE merges from text data. Uses incremental pair counting for speed."""
        # encode all text to byte sequences
        all_tokens = []
        for text in texts:
            tokens = list(text.encode('utf-8'))
            all_tokens.append(tokens)

        num_merges = self.vocab_size - 256
        if verbose:
            print(f"Training BPE: {num_merges} merges from {len(all_tokens)} texts")

        # initial pair counts
        pair_counts = Counter()
        for tokens in all_tokens:
            for j in range(len(tokens) - 1):
                pair_counts[(tokens[j], tokens[j + 1])] += 1

        import time
        t0 = time.time()

        for i in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break

            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.inverse_vocab[self.vocab[new_id]] = new_id
            bp_a, bp_b = best_pair

            # merge in all sequences and update pair counts incrementally
            for idx in range(len(all_tokens)):
                tokens = all_tokens[idx]
                if len(tokens) < 2:
                    continue

                new_tokens = []
                j = 0
                while j < len(tokens):
                    if j < len(tokens) - 1 and tokens[j] == bp_a and tokens[j + 1] == bp_b:
                        # remove old pairs touching this position
                        if j > 0:
                            old_left = (tokens[j - 1] if not new_tokens or new_tokens[-1] != new_id else new_id, bp_a)
                            # use the actual left token (which might have been merged already)
                        if j + 2 < len(tokens):
                            old_right = (bp_b, tokens[j + 2])
                            pair_counts[old_right] -= 1
                            if pair_counts[old_right] <= 0:
                                del pair_counts[old_right]
                        # add new pairs
                        if new_tokens:
                            new_left = (new_tokens[-1], new_id)
                            # remove old (new_tokens[-1], bp_a) — already counted in original
                            old_before = (new_tokens[-1], bp_a)
                            pair_counts[old_before] -= 1
                            if pair_counts[old_before] <= 0:
                                del pair_counts[old_before]
                            pair_counts[new_left] += 1

                        new_tokens.append(new_id)
                        j += 2

                        # add new right pair
                        if j < len(tokens):
                            pair_counts[(new_id, tokens[j])] += 1
                    else:
                        new_tokens.append(tokens[j])
                        j += 1
                all_tokens[idx] = new_tokens

            # remove the merged pair from counts
            if best_pair in pair_counts:
                del pair_counts[best_pair]

            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  merge {i + 1}/{num_merges}: "
                      f"{self.vocab[bp_a]!r} + {self.vocab[bp_b]!r} "
                      f"-> {self.vocab[new_id]!r} ({elapsed:.0f}s)")

        if verbose:
            total = time.time() - t0
            print(f"Tokenizer trained: {len(self.vocab)} tokens ({total:.1f}s)")

    def encode(self, text):
        """Encode text to token ids. Heap-based O(N log N) algorithm."""
        import heapq

        tokens = list(text.encode('utf-8'))
        if not self.merges or len(tokens) < 2:
            return tokens

        # doubly-linked list via prev/next arrays
        n = len(tokens)
        vals = list(tokens)
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = -1
        alive = [True] * n

        # min-heap: (merge_rank, position) — merge at pos means merge vals[pos] + vals[nxt[pos]]
        heap = []
        for i in range(n - 1):
            pair = (vals[i], vals[i + 1])
            rank = self.merges.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        while heap:
            rank, pos = heapq.heappop(heap)

            # skip if node was consumed or next is dead
            if not alive[pos]:
                continue
            right = nxt[pos]
            if right == -1 or not alive[right]:
                continue
            # verify pair still matches (may have changed)
            pair = (vals[pos], vals[right])
            r = self.merges.get(pair)
            if r is None or r != rank:
                continue

            # merge: pos absorbs right
            vals[pos] = rank  # rank == merged token id
            alive[right] = False
            nxt[pos] = nxt[right]
            if nxt[right] != -1:
                prev[nxt[right]] = pos

            # add new pairs to heap
            lft = prev[pos]
            rgt = nxt[pos]
            if lft >= 0 and alive[lft]:
                new_pair = (vals[lft], vals[pos])
                mr = self.merges.get(new_pair)
                if mr is not None:
                    heapq.heappush(heap, (mr, lft))
            if rgt != -1 and alive[rgt]:
                new_pair = (vals[pos], vals[rgt])
                mr = self.merges.get(new_pair)
                if mr is not None:
                    heapq.heappush(heap, (mr, pos))

        return [vals[i] for i in range(n) if alive[i]]

    def decode(self, ids):
        """Decode token ids to text."""
        byte_seq = b''.join(self.vocab[i] for i in ids if i in self.vocab)
        return byte_seq.decode('utf-8', errors='replace')

    def save(self, path):
        """Save tokenizer to JSON."""
        data = {
            'vocab_size': self.vocab_size,
            'merges': {f"{a},{b}": v for (a, b), v in self.merges.items()},
            'vocab': {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)
        tok = cls(data['vocab_size'])
        tok.merges = {(int(k.split(',')[0]), int(k.split(',')[1])): v
                      for k, v in data['merges'].items()}
        tok.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        return tok


if __name__ == "__main__":
    # quick test
    texts = [
        "Once upon a time, there was a little girl named Lily.",
        "She loved to play in the garden with her dog Max.",
        "One sunny day, Lily found a beautiful butterfly.",
    ]
    tok = ByteBPETokenizer(vocab_size=300)
    tok.train(texts, verbose=True)
    for t in texts:
        ids = tok.encode(t)
        decoded = tok.decode(ids)
        print(f"  [{len(ids)} tokens] {t[:50]}... -> {decoded[:50]}...")
        assert decoded == t, f"Roundtrip failed: {t!r} != {decoded!r}"
    print("All roundtrip tests passed!")
