"""
phi-auto GPT Model - CPU-optimized tiny transformer.
Pure numpy implementation, no PyTorch needed.
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    n_embd: int = 192
    n_head: int = 6
    n_layer: int = 6
    seq_len: int = 256
    dropout: float = 0.0


class Linear:
    """Dense layer with optional bias."""
    def __init__(self, in_f, out_f, bias=False):
        scale = (2.0 / (in_f + out_f)) ** 0.5
        self.w = np.random.randn(in_f, out_f).astype(np.float32) * scale
        self.b = np.zeros(out_f, dtype=np.float32) if bias else None
        self.x = None  # cached input for backward
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = x @ self.w
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, dout):
        # x: (..., in_f), dout: (..., out_f)
        x_2d = self.x.reshape(-1, self.x.shape[-1])
        d_2d = dout.reshape(-1, dout.shape[-1])
        self.dw = x_2d.T @ d_2d
        if self.b is not None:
            self.db = dout.reshape(-1, dout.shape[-1]).sum(axis=0)
        dx = dout @ self.w.T
        return dx

    def params(self):
        if self.b is not None:
            return [(self.w, 'dw'), (self.b, 'db')]
        return [(self.w, 'dw')]


class Embedding:
    """Token embedding lookup."""
    def __init__(self, num_embeddings, embedding_dim):
        self.w = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        self.indices = None
        self.dw = None

    def forward(self, idx):
        self.indices = idx
        return self.w[idx]

    def backward(self, dout, accumulate=False):
        if self.dw is None:
            self.dw = np.zeros_like(self.w)
        elif not accumulate:
            self.dw[:] = 0
        np.add.at(self.dw, self.indices, dout)

    def params(self):
        return [(self.w, 'dw')]


class RMSNorm:
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        self.g = np.ones(dim, dtype=np.float32)
        self.eps = eps
        self.x = None
        self.rms = None
        self.dg = None

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / self.rms) * self.g

    def backward(self, dout):
        x = self.x
        rms = self.rms
        x_norm = x / rms
        self.dg = np.sum(dout * x_norm, axis=tuple(range(dout.ndim - 1)))
        d_xnorm = dout * self.g
        d = x.shape[-1]
        dx = (d_xnorm / rms) - (x_norm / d) * np.sum(d_xnorm * x_norm, axis=-1, keepdims=True)
        return dx

    def params(self):
        return [(self.g, 'dg')]


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def precompute_rope(seq_len, head_dim, base=10000.0):
    """Precompute rotary position embedding sin/cos."""
    d = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, d, dtype=np.float32) / d))
    pos = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(pos, inv_freq)  # (seq_len, d)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: (B, n_head, T, head_dim)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    T = x.shape[2]
    c = cos[:T][None, None, :, :]  # (1, 1, T, d)
    s = sin[:T][None, None, :, :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def apply_rope_backward(dx, cos, sin):
    """Backward through RoPE — inverse rotation (negate sin)."""
    d = dx.shape[-1] // 2
    dx1, dx2 = dx[..., :d], dx[..., d:]
    T = dx.shape[2]
    c = cos[:T][None, None, :, :]
    s = sin[:T][None, None, :, :]
    # inverse rotation: transpose of rotation matrix = negate sin
    return np.concatenate([dx1 * c + dx2 * s, -dx1 * s + dx2 * c], axis=-1)


class CausalAttention:
    """Multi-head causal self-attention with RoPE."""
    def __init__(self, config):
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = Linear(config.n_embd, 3 * config.n_embd)
        self.proj = Linear(config.n_embd, config.n_embd)
        self.rope_cos, self.rope_sin = precompute_rope(config.seq_len, self.head_dim)
        # pre-allocate causal mask (never recreated)
        self._causal_mask = np.triu(np.full((config.seq_len, config.seq_len), -1e9, dtype=np.float32), k=1)
        # cached
        self.q = self.k = self.v = self.attn_weights = None

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv.forward(x)
        q, k, v = np.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn += self._causal_mask[None, None, :T, :T]
        attn = softmax(attn, axis=-1)

        self.q, self.k, self.v, self.attn_weights = q, k, v, attn

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj.forward(out)

    def backward(self, dout):
        B, T, C = dout.shape
        dout_proj = self.proj.backward(dout)
        dout_proj = dout_proj.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # d(attn @ v)
        dv = self.attn_weights.transpose(0, 1, 3, 2) @ dout_proj
        dattn = dout_proj @ self.v.transpose(0, 1, 3, 2)

        # softmax backward
        dattn = self.attn_weights * (dattn - (dattn * self.attn_weights).sum(axis=-1, keepdims=True))

        scale = 1.0 / math.sqrt(self.head_dim)
        dq = (dattn @ self.k) * scale
        dk = (dattn.transpose(0, 1, 3, 2) @ self.q) * scale

        # RoPE backward: inverse rotation (negate sin)
        dq = apply_rope_backward(dq, self.rope_cos, self.rope_sin)
        dk = apply_rope_backward(dk, self.rope_cos, self.rope_sin)

        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)

        dqkv = np.concatenate([dq, dk, dv], axis=-1)
        return self.qkv.backward(dqkv)

    def params(self):
        return self.qkv.params() + self.proj.params()


class MLP:
    """SiLU-gated MLP with fused gate+up projection."""
    def __init__(self, config):
        hidden = 4 * config.n_embd
        self.hidden = hidden
        # fused gate+up: single matmul instead of two
        self.gate_up = Linear(config.n_embd, 2 * hidden)
        self.down = Linear(hidden, config.n_embd)
        self.gate_out = None
        self.up_out = None

    def forward(self, x):
        gu = self.gate_up.forward(x)
        g = gu[..., :self.hidden]
        u = gu[..., self.hidden:]
        sig_g = 1.0 / (1.0 + np.exp(-np.clip(g, -20, 20)))
        self.gate_out = g
        self.up_out = u
        self.sig_g = sig_g
        h = (g * sig_g) * u  # SwiGLU
        return self.down.forward(h)

    def backward(self, dout):
        dh = self.down.backward(dout)
        g, u, sig_g = self.gate_out, self.up_out, self.sig_g
        silu_g = g * sig_g

        du = dh * silu_g
        dsilu = dh * u
        dg = dsilu * sig_g * (1.0 + g * (1.0 - sig_g))

        dgu = np.concatenate([dg, du], axis=-1)
        return self.gate_up.backward(dgu)

    def params(self):
        return self.gate_up.params() + self.down.params()


class TransformerBlock:
    """Pre-norm transformer block."""
    def __init__(self, config):
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.x = None

    def forward(self, x):
        self.x = x
        h = x + self.attn.forward(self.ln1.forward(x))
        self.h = h
        out = h + self.mlp.forward(self.ln2.forward(h))
        return out

    def backward(self, dout):
        # MLP residual
        dmlp = self.mlp.backward(self.ln2.backward(dout))
        dh = dout + dmlp
        # Attention residual
        dattn = self.attn.backward(self.ln1.backward(dh))
        dx = dh + dattn
        return dx

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.mlp.params()


class TiedLinear:
    """Linear layer that shares weights with an Embedding (weight tying).
    Eliminates separate lm_head weight: saves params + speeds up backward.
    """
    def __init__(self, embedding):
        self.emb = embedding  # shared weight: (vocab, dim)
        self.x = None
        # no own weight — gradient accumulates into embedding.dw

    def forward(self, x):
        self.x = x
        return x @ self.emb.w.T  # (B, T, dim) @ (dim, vocab) -> (B, T, vocab)

    def backward(self, dout):
        # dout: (B, T, vocab)
        dx = dout @ self.emb.w  # (B, T, vocab) @ (vocab, dim) -> (B, T, dim)
        # accumulate weight grad into embedding's dw
        x_2d = self.x.reshape(-1, self.x.shape[-1])
        d_2d = dout.reshape(-1, dout.shape[-1])
        grad = d_2d.T @ x_2d  # (vocab, B*T) @ (B*T, dim) -> (vocab, dim)
        if self.emb.dw is None:
            self.emb.dw = grad
        else:
            self.emb.dw += grad
        return dx

    def params(self):
        return []  # no own params — shared via embedding


class GPT:
    """Full GPT model."""
    def __init__(self, config: GPTConfig, tie_weights=True):
        self.config = config
        self.tok_emb = Embedding(config.vocab_size, config.n_embd)
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = RMSNorm(config.n_embd)

        # weight tying: lm_head shares weights with tok_emb (standard practice)
        self.tie_weights = tie_weights
        if tie_weights:
            self.lm_head = TiedLinear(self.tok_emb)
        else:
            self.lm_head = Linear(config.n_embd, config.vocab_size)

        # count params
        self.n_params = self._count_params()
        print(f"Model: {self.n_params:,} parameters "
              f"({self.n_params * 4 / 1024 / 1024:.1f} MB fp32)"
              f"{' (weight-tied)' if tie_weights else ''}")

    def _count_params(self):
        total = 0
        for p, _ in self.all_params():
            total += p.size
        return total

    def all_params(self):
        ps = self.tok_emb.params()
        for block in self.blocks:
            ps += block.params()
        ps += self.ln_f.params()
        ps += self.lm_head.params()
        return ps

    def forward(self, idx, targets=None):
        """Forward pass. idx: (B, T) int array."""
        x = self.tok_emb.forward(idx)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        logits = self.lm_head.forward(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.reshape(B * T, V)
            targets_flat = targets.reshape(B * T)
            # cross-entropy
            probs = softmax(logits_flat, axis=-1)
            # clamp for log stability
            probs_clamp = np.clip(probs, 1e-9, 1.0)
            log_probs = np.log(probs_clamp)
            loss = -log_probs[np.arange(B * T), targets_flat].mean()
            # cache for backward
            self._logits_shape = (B, T, V)
            self._probs = probs
            self._targets_flat = targets_flat

        return logits, loss

    def backward(self):
        """Backward pass from cached loss."""
        B, T, V = self._logits_shape
        N = B * T
        dlogits = self._probs.copy()
        dlogits[np.arange(N), self._targets_flat] -= 1.0
        dlogits /= N
        dlogits = dlogits.reshape(B, T, V)

        dx = self.lm_head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        # when weights are tied, lm_head already wrote to tok_emb.dw — accumulate
        self.tok_emb.backward(dx, accumulate=self.tie_weights)

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.config.seq_len:]
            logits, _ = self.forward(idx_crop)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                topk_idx = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
                mask = np.full_like(logits, -1e9)
                np.put_along_axis(mask, topk_idx, np.take_along_axis(logits, topk_idx, axis=-1), axis=-1)
                logits = mask
            probs = softmax(logits, axis=-1)
            next_tok = np.array([[np.random.choice(probs.shape[-1], p=probs[b])
                                  for b in range(probs.shape[0])]])
            idx = np.concatenate([idx, next_tok.T], axis=1)
        return idx


if __name__ == "__main__":
    cfg = GPTConfig(vocab_size=256, n_embd=64, n_head=4, n_layer=2, seq_len=32)
    model = GPT(cfg)
    # sanity check
    x = np.random.randint(0, 256, (2, 32))
    y = np.random.randint(0, 256, (2, 32))
    logits, loss = model.forward(x, y)
    print(f"Logits shape: {logits.shape}, Loss: {loss:.4f}")
    model.backward()
    print("Backward pass OK")
    print(f"Param groups: {len(model.all_params())}")
