"""
Optimizers for phi-auto.
Lion (memory-efficient) + AdamW (baseline) + Schedule-Free AdamW.
"""

import math
import numpy as np


class Lion:
    """Lion optimizer — 50% less memory than AdamW.
    From 'Symbolic Discovery of Optimization Algorithms' (Google Brain, 2023).
    Uses sign of momentum for updates. Only 1 buffer per param vs 2 for AdamW.

    Key: use 3-10x smaller LR than AdamW, 3-10x larger weight_decay.
    """
    def __init__(self, lr=3e-5, betas=(0.9, 0.99), weight_decay=1.0, grad_clip=1.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.m_dict = {}

    def step(self, param_grad_pairs):
        if self.grad_clip > 0:
            total_norm = math.sqrt(
                sum(np.sum(g ** 2) for _, g in param_grad_pairs if g is not None)
            )
            clip_coef = self.grad_clip / (total_norm + 1e-8)
            if clip_coef < 1.0:
                for _, g in param_grad_pairs:
                    if g is not None:
                        g *= clip_coef

        for p, g in param_grad_pairs:
            if g is None:
                continue
            pid = id(p)
            if pid not in self.m_dict:
                self.m_dict[pid] = np.zeros_like(p)

            m = self.m_dict[pid]

            # decoupled weight decay
            if self.weight_decay > 0 and p.ndim >= 2:
                p -= self.lr * self.weight_decay * p

            # Lion update: sign(interpolated momentum)
            update = np.sign(self.beta1 * m + (1 - self.beta1) * g)
            p -= self.lr * update

            # update momentum
            m[:] = self.beta2 * m + (1 - self.beta2) * g


class AdamW:
    """Standard AdamW optimizer."""
    def __init__(self, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, grad_clip=1.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.t = 0
        self.m_dict = {}
        self.v_dict = {}

    def step(self, param_grad_pairs):
        self.t += 1

        if self.grad_clip > 0:
            total_norm = math.sqrt(
                sum(np.sum(g ** 2) for _, g in param_grad_pairs if g is not None)
            )
            clip_coef = self.grad_clip / (total_norm + 1e-8)
            if clip_coef < 1.0:
                for _, g in param_grad_pairs:
                    if g is not None:
                        g *= clip_coef

        for p, g in param_grad_pairs:
            if g is None:
                continue
            pid = id(p)
            if pid not in self.m_dict:
                self.m_dict[pid] = np.zeros_like(p)
                self.v_dict[pid] = np.zeros_like(p)

            m, v = self.m_dict[pid], self.v_dict[pid]

            if self.weight_decay > 0 and p.ndim >= 2:
                p -= self.lr * self.weight_decay * p

            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ScheduleFreeAdamW:
    """Schedule-Free AdamW — no LR schedule needed.
    Matches cosine+warmup without any schedule tuning.
    From Meta Research (Defazio et al., 2024).

    Important: call eval_mode() before validation, train_mode() before training.
    """
    def __init__(self, lr=0.025, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.1, warmup_steps=0, grad_clip=1.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.t = 0
        self.states = {}

    def step(self, param_grad_pairs):
        self.t += 1

        lr = self.lr
        if self.warmup_steps > 0 and self.t <= self.warmup_steps:
            lr = self.lr * self.t / self.warmup_steps

        if self.grad_clip > 0:
            total_norm = math.sqrt(
                sum(np.sum(g ** 2) for _, g in param_grad_pairs if g is not None)
            )
            clip_coef = self.grad_clip / (total_norm + 1e-8)
            if clip_coef < 1.0:
                for _, g in param_grad_pairs:
                    if g is not None:
                        g *= clip_coef

        ck = min(1.0 - 1.0 / (self.t + 1), self.beta1)

        for p, g in param_grad_pairs:
            if g is None:
                continue
            pid = id(p)
            if pid not in self.states:
                self.states[pid] = {
                    'z': p.copy(),
                    'v': np.zeros_like(p),
                }

            state = self.states[pid]
            z, v = state['z'], state['v']

            if self.weight_decay > 0 and p.ndim >= 2:
                z -= lr * self.weight_decay * z

            v[:] = self.beta2 * v + (1 - self.beta2) * g ** 2
            v_hat = v / (1 - self.beta2 ** self.t)
            z -= lr * g / (np.sqrt(v_hat) + self.eps)
            p[:] = (1 - ck) * z + ck * p

    def eval_mode(self, param_grad_pairs):
        """Switch to averaged weights for evaluation."""
        for p, _ in param_grad_pairs:
            pid = id(p)
            if pid in self.states:
                self.states[pid]['p_train'] = p.copy()
                p[:] = self.states[pid]['z']

    def train_mode(self, param_grad_pairs):
        """Switch back to training weights."""
        for p, _ in param_grad_pairs:
            pid = id(p)
            if pid in self.states and 'p_train' in self.states[pid]:
                p[:] = self.states[pid]['p_train']
