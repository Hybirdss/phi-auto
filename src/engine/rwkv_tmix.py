"""
RWKV Time-Mixing block for phi-auto.
Replaces O(T²) causal attention with O(T) linear recurrence.

Based on RWKV-7 "Goose" (March 2025, arxiv:2503.14456).
Key insight: WKV recurrence replaces attention matrix with a (H,D,D) state,
which is 64x smaller than (H,T,T) attention for T=256, D=32.

Training: parallel mode (loop over T with vectorized heads)
Inference: recurrent mode (O(1) per token)
"""

import math
import numpy as np


class TimeMixing:
    """RWKV-style time-mixing — drop-in replacement for CausalAttention.

    Instead of Q @ K^T attention matrix, uses a linear recurrence:
        state[t] = decay * state[t-1] + k[t] ⊗ v[t]
        output[t] = sigmoid(r[t]) * (state[t] @ k[t] + bonus * v[t])

    Memory: O(n_head * head_dim²) vs O(n_head * T²) for attention.
    """

    def __init__(self, config):
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        D = self.n_embd

        # projections (same structure as attention: R, K, V, O)
        scale = (2.0 / (D + D)) ** 0.5
        self.w_r = np.random.randn(D, D).astype(np.float32) * scale
        self.w_k = np.random.randn(D, D).astype(np.float32) * scale
        self.w_v = np.random.randn(D, D).astype(np.float32) * scale
        self.w_o = np.random.randn(D, D).astype(np.float32) * scale

        # time decay: learnable per-head per-dim, initialized negative
        self.time_decay = -np.abs(np.random.randn(self.n_head, self.head_dim).astype(np.float32)) - 0.5
        # bonus for current token
        self.time_first = np.random.randn(self.n_head, self.head_dim).astype(np.float32) * 0.02

        # token shift (simple: x[t] = lerp(x[t-1], x[t], mu))
        self.mu = np.ones(D, dtype=np.float32) * 0.5

        # gradient storage
        self.dw_r = self.dw_k = self.dw_v = self.dw_o = None
        self.dtime_decay = self.dtime_first = self.dmu = None

        # cached for backward
        self._x = None
        self._x_shifted = None
        self._r = self._k = self._v = None
        self._sr = None
        self._outputs = None
        self._states_hist = None

    def forward(self, x):
        """Parallel training mode. x: (B, T, C)"""
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        # token shift: mix current with previous
        x_shifted = np.zeros_like(x)
        x_shifted[:, 1:, :] = x[:, :-1, :]
        x_shifted[:, 0, :] = 0
        x_mix = x * self.mu + x_shifted * (1 - self.mu)

        # projections
        r = x_mix @ self.w_r  # receptance (gate)
        k = x_mix @ self.w_k  # key
        v = x_mix @ self.w_v  # value

        r = r.reshape(B, T, H, D)
        k = k.reshape(B, T, H, D)
        v = v.reshape(B, T, H, D)

        sr = 1.0 / (1.0 + np.exp(-np.clip(r, -20, 20)))  # sigmoid gate

        decay = np.exp(self.time_decay)  # (H, D)
        bonus = np.exp(self.time_first)  # (H, D)

        # WKV recurrence (loop over T, vectorized over B and H)
        state = np.zeros((B, H, D, D), dtype=np.float32)
        outputs = np.zeros((B, T, H, D), dtype=np.float32)
        states_hist = [state.copy()]

        for t in range(T):
            kt = k[:, t]  # (B, H, D)
            vt = v[:, t]  # (B, H, D)

            # wkv = state @ k + bonus * v
            # state @ k: (B, H, D, D) @ (B, H, D) -> (B, H, D)
            wkv = np.einsum('bhde,bhd->bhe', state, kt) + bonus[None] * vt

            outputs[:, t] = sr[:, t] * wkv

            # state update: decay + outer(k, v)
            state = state * decay[None, :, :, None] + np.einsum('bhd,bhe->bhde', kt, vt)
            states_hist.append(state.copy())

        out = outputs.reshape(B, T, C)
        out = out @ self.w_o

        # cache for backward
        self._x = x
        self._x_shifted = x_shifted
        self._x_mix = x_mix
        self._r, self._k, self._v = r, k, v
        self._sr = sr
        self._outputs = outputs
        self._states_hist = states_hist
        self._decay = decay
        self._bonus = bonus

        return out

    def backward(self, dout):
        """Backward pass through time-mixing."""
        B, T, C = dout.shape
        H, D = self.n_head, self.head_dim

        # d(out @ w_o)
        dout_pre = dout  # (B, T, C)
        self.dw_o = self._outputs.reshape(B * T, C).T @ dout_pre.reshape(B * T, C)
        d_outputs = (dout_pre @ self.w_o.T).reshape(B, T, H, D)

        # backward through WKV recurrence
        d_sr = d_outputs * self._outputs / (self._sr + 1e-8)  # rough
        d_wkv = d_outputs * self._sr

        dr = np.zeros_like(self._r)
        dk = np.zeros_like(self._k)
        dv = np.zeros_like(self._v)
        d_state = np.zeros((B, H, D, D), dtype=np.float32)
        self.dtime_decay = np.zeros_like(self.time_decay)
        self.dtime_first = np.zeros_like(self.time_first)

        decay = self._decay
        bonus = self._bonus

        for t in reversed(range(T)):
            kt = self._k[:, t]
            vt = self._v[:, t]
            state_prev = self._states_hist[t]

            d_wkv_t = d_wkv[:, t]  # (B, H, D)

            # d(sr * wkv) -> d_sr, d_wkv done above
            # sr = sigmoid(r), d(sigmoid) = sr * (1 - sr)
            dr[:, t] = d_sr[:, t] * self._sr[:, t] * (1 - self._sr[:, t])

            # d(state @ k)
            d_state += np.einsum('bhe,bhd->bhde', d_wkv_t, kt)
            dk[:, t] += np.einsum('bhde,bhe->bhd', state_prev, d_wkv_t)

            # d(bonus * v)
            dv[:, t] += d_wkv_t * bonus[None]
            self.dtime_first += np.sum(d_wkv_t * vt * bonus[None], axis=0)

            # d(state update): state = decay * state_prev + outer(k, v)
            # d_state_prev from this step
            d_state_prev = d_state * decay[None, :, :, None]
            self.dtime_decay += np.sum(
                np.sum(d_state * state_prev * decay[None, :, :, None], axis=(0, 3)),
                axis=0
            )

            # d(outer(k, v))
            dk[:, t] += np.einsum('bhde,bhe->bhd', d_state, vt)
            dv[:, t] += np.einsum('bhde,bhd->bhe', d_state, kt)

            d_state = d_state_prev

        # reshape for projection backward
        dr_flat = dr.reshape(B, T, C)
        dk_flat = dk.reshape(B, T, C)
        dv_flat = dv.reshape(B, T, C)

        x_mix = self._x_mix
        x_mix_flat = x_mix.reshape(B * T, C)

        self.dw_r = x_mix_flat.T @ dr_flat.reshape(B * T, C)
        self.dw_k = x_mix_flat.T @ dk_flat.reshape(B * T, C)
        self.dw_v = x_mix_flat.T @ dv_flat.reshape(B * T, C)

        dx_mix = (dr_flat @ self.w_r.T + dk_flat @ self.w_k.T + dv_flat @ self.w_v.T)

        # token shift backward
        dx = dx_mix * self.mu
        dx_shifted = dx_mix * (1 - self.mu)
        dx[:, :-1] += dx_shifted[:, 1:]
        self.dmu = np.sum(dx_mix * (self._x - self._x_shifted), axis=(0, 1))

        return dx

    def params(self):
        """Return (param, grad_name) pairs for optimizer."""
        return [
            (self.w_r, 'dw_r'), (self.w_k, 'dw_k'),
            (self.w_v, 'dw_v'), (self.w_o, 'dw_o'),
            (self.time_decay, 'dtime_decay'),
            (self.time_first, 'dtime_first'),
            (self.mu, 'dmu'),
        ]


if __name__ == "__main__":
    from model import GPTConfig

    cfg = GPTConfig(vocab_size=256, n_embd=64, n_head=4, n_layer=2, seq_len=32)
    tmix = TimeMixing(cfg)

    x = np.random.randn(2, 32, 64).astype(np.float32)
    out = tmix.forward(x)
    print(f"Forward: input {x.shape} -> output {out.shape}")

    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = tmix.backward(dout)
    print(f"Backward: dx {dx.shape}")
    print(f"Grad shapes: dw_r={tmix.dw_r.shape}, dtime_decay={tmix.dtime_decay.shape}")
    print("RWKV TimeMixing OK!")
