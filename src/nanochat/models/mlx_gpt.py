"""
MLX GPT model — port of models/gpt.py for Apple Silicon training.

Training forward pass only (no KV cache, no generate).
Shares GPTConfig from models/config.py.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from nanochat.models.config import GPTConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def norm(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x, weight=None, eps=1e-5)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def _ve_key(layer_idx: int) -> str:
    return f"ve_{layer_idx}"


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, N, T, D]  cos/sin: [1, 1, T, D//2]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return mx.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1)


def causal_window_mask(T: int, window: int) -> mx.array:
    i = mx.arange(T)[:, None]
    j = mx.arange(T)[None, :]
    return (j <= i) & ((i - j) < window)


# ---------------------------------------------------------------------------
# KV-cache for autoregressive generation
# ---------------------------------------------------------------------------


class MLXKVCache:
    """Per-layer KV-cache for MLX autoregressive generation.

    Keys and values are stored as a list of (k, v) tuples, one per layer.
    Each tuple is (k: [B, n_kv_head, pos, head_dim], v: same shape).
    `pos` grows by concatenation on each decode step — MLX has no in-place updates.
    Call mx.eval after each decode step to prevent unbounded lazy graph accumulation.
    """

    def __init__(self, layer_kvs: list[tuple[mx.array, mx.array]]) -> None:
        self.layer_kvs = layer_kvs  # list[tuple[k, v]] per layer

    @property
    def pos(self) -> int:
        """Current sequence position (length of cached K)."""
        if not self.layer_kvs:
            return 0
        return int(self.layer_kvs[0][0].shape[2])

    @classmethod
    def empty(cls) -> "MLXKVCache":
        return cls([])


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.relu(self.c_fc(x)) ** 2)


class CausalSelfAttention(nn.Module):
    VE_GATE_CHANNELS = 12

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.ve_gate: Optional[nn.Linear] = (
            nn.Linear(self.VE_GATE_CHANNELS, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def __call__(
        self,
        x: mx.array,  # [B, T, C]
        ve: Optional[mx.array],  # [B, T, n_kv_head * head_dim] or None
        cos: mx.array,  # [1, 1, T, head_dim//2]
        sin: mx.array,  # [1, 1, T, head_dim//2]
        mask,  # "causal" string or boolean [T, T] array
    ) -> mx.array:
        B, T, _ = x.shape

        # Project — layout [B, T, N, D] then transpose to [B, N, T, D] for mx.fast.sdpa
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

        # Value residual
        if ve is not None:
            assert self.ve_gate is not None
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            gate = 3 * mx.sigmoid(self.ve_gate(x[..., : self.VE_GATE_CHANNELS]))  # [B, T, n_kv_head]
            gate = gate.transpose(0, 2, 1)[..., None]  # [B, n_kv_head, T, 1]
            v = v + gate * ve

        # RoPE + QK norm
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q) * 1.2
        k = norm(k) * 1.2

        # Attention — mx.fast.sdpa expects [B, N, T, D], supports GQA natively
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)

        # [B, N, T, D] → [B, T, C]
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y)

    def forward_cached(
        self,
        x: mx.array,   # [B, T, C]
        ve: Optional[mx.array],
        cos: mx.array,
        sin: mx.array,
        mask,          # causal mask for prefill, None for single-token decode
        past_kv: Optional[tuple[mx.array, mx.array]] = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """Attention with KV-cache support. Returns (output, (new_k, new_v))."""
        B, T, _ = x.shape

        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

        if ve is not None:
            assert self.ve_gate is not None
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            gate = 3 * mx.sigmoid(self.ve_gate(x[..., : self.VE_GATE_CHANNELS]))
            gate = gate.transpose(0, 2, 1)[..., None]
            v = v + gate * ve

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q) * 1.2
        k = norm(k) * 1.2

        # Concatenate past KV with current KV
        if past_kv is not None:
            past_k, past_v = past_kv
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y), (k, v)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLPBlock(config.n_embd)

    def __call__(
        self,
        x: mx.array,
        ve: Optional[mx.array],
        cos: mx.array,
        sin: mx.array,
        mask,
    ) -> mx.array:
        x = x + self.attn(norm(x), ve, cos, sin, mask)
        x = x + self.mlp(norm(x))
        return x

    def forward_cached(
        self,
        x: mx.array,
        ve: Optional[mx.array],
        cos: mx.array,
        sin: mx.array,
        mask,
        past_kv: Optional[tuple[mx.array, mx.array]] = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        attn_out, new_kv = self.attn.forward_cached(norm(x), ve, cos, sin, mask, past_kv)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------


class GPT(nn.Module):
    SOFTCAP = 15
    SMEAR_GATE_CHANNELS = 24
    ROTARY_OVERSHOOT = 10

    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64) -> None:
        super().__init__()
        self.config = config

        padded_vocab = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to * pad_vocab_size_to
        )
        self.padded_vocab = padded_vocab

        self.wte = nn.Embedding(padded_vocab, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]

        # Per-layer residual scalars
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))

        # Smear gate
        self.smear_gate = nn.Linear(self.SMEAR_GATE_CHANNELS, 1, bias=False)
        self.smear_lambda = mx.zeros((1,))

        # Backout
        self.backout_lambda = 0.2 * mx.ones((1,))

        # Value embeddings (alternating layers)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            _ve_key(i): nn.Embedding(padded_vocab, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        }

        # Rotary embeddings — precomputed, [1, 1, rotary_seq_len, head_dim//2]
        rotary_seq_len = config.sequence_len * self.ROTARY_OVERSHOOT
        self.cos, self.sin = self._precompute_rotary(rotary_seq_len, head_dim)

        # Per-layer window sizes and masks
        self.window_sizes: List[Tuple[int, int]] = self._compute_window_sizes(config)
        self._masks = self._build_masks(config)

        # Freeze non-trainable buffers
        self.freeze(keys=["cos", "sin"])

    def _precompute_rotary(self, seq_len: int, head_dim: int, base: int = 100000) -> Tuple[mx.array, mx.array]:
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)  # [seq_len, head_dim//2]
        cos = mx.cos(freqs)[None, None, :, :]  # [1, 1, seq_len, head_dim//2]
        sin = mx.sin(freqs)[None, None, :, :]
        return cos, sin

    def _compute_window_sizes(self, config: GPTConfig) -> List[Tuple[int, int]]:
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_w = config.sequence_len
        short_w = -(-long_w // 4 // 128) * 128
        char_to_window = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def _build_masks(self, config: GPTConfig):
        """Precompute per-layer masks (string "causal" or boolean array)."""
        long_w = config.sequence_len
        masks = []
        for left, _ in self.window_sizes:
            if left == long_w:
                masks.append("causal")
            else:
                masks.append(causal_window_mask(config.sequence_len, left))
        return masks

    def __call__(
        self,
        idx: mx.array,  # [B, T] int32
        targets: Optional[mx.array] = None,  # [B, T] int32 or None; -1 = masked
        loss_reduction: str = "mean",  # "mean" → scalar; "none" → (B, T) per-token CE
    ) -> mx.array:
        B, T = idx.shape

        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]

        x = self.wte(idx)  # [B, T, C]
        x = norm(x)

        # Smear gate
        gate = self.smear_lambda * mx.sigmoid(self.smear_gate(x[:, 1:, : self.SMEAR_GATE_CHANNELS]))
        x = mx.concatenate([x[:, :1], x[:, 1:] + gate * x[:, :-1]], axis=1)

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[_ve_key(i)](idx) if has_ve(i, self.config.n_layer) else None
            mask = self._masks[i] if T == self.config.sequence_len else (
                "causal" if self.window_sizes[i][0] == self.config.sequence_len
                else causal_window_mask(T, self.window_sizes[i][0])
            )
            x = block(x, ve, cos, sin, mask)
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda * x_backout
        x = norm(x)

        logits = self.lm_head(x)[..., : self.config.vocab_size]
        logits = self.SOFTCAP * mx.tanh(logits / self.SOFTCAP)

        if targets is None:
            return logits

        # Masked CE: positions with target == -1 are padding or non-assistant tokens.
        # clip(0) avoids out-of-bounds index; mask zeros their contribution.
        flat_targets = targets.reshape(-1)
        flat_logits = logits.reshape(-1, self.config.vocab_size)
        mask = (flat_targets >= 0).astype(mx.float32)
        ce = nn.losses.cross_entropy(flat_logits, mx.maximum(flat_targets, 0))

        if loss_reduction == "none":
            # Return per-token CE, zeroed at masked positions, shape (B, T).
            # In RL: logp = -model(inputs, targets, loss_reduction="none")
            return (ce * mask).reshape(idx.shape)

        # Default: masked mean (backward-compatible with base pretraining — all
        # base targets are >= 0, so mask is all-ones and result equals mx.mean).
        return mx.sum(ce * mask) / mx.maximum(mask.sum(), mx.array(1.0))

    def forward_with_kv_cache(
        self,
        idx: mx.array,           # [B, T] int32
        kv_cache: Optional["MLXKVCache"] = None,
    ) -> tuple[mx.array, "MLXKVCache"]:
        """Forward pass with KV-cache for autoregressive generation.

        When kv_cache is None (prefill): runs full forward pass over the prompt,
        builds and returns a new MLXKVCache.

        When kv_cache is provided (decode): runs single-token forward pass
        (T=1), appends to the cache, returns updated cache.

        Returns (logits, new_kv_cache). Logits shape: [B, T, vocab_size].
        """
        B, T = idx.shape
        pos = kv_cache.pos if kv_cache is not None else 0

        cos = self.cos[:, :, pos:pos + T, :]
        sin = self.sin[:, :, pos:pos + T, :]

        x = self.wte(idx)
        x = norm(x)

        # Smear gate — only meaningful when T > 1 (prefill); no-op for T=1.
        if T > 1:
            gate = self.smear_lambda * mx.sigmoid(self.smear_gate(x[:, 1:, : self.SMEAR_GATE_CHANNELS]))
            x = mx.concatenate([x[:, :1], x[:, 1:] + gate * x[:, :-1]], axis=1)

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None

        new_layer_kvs: list[tuple[mx.array, mx.array]] = []
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[_ve_key(i)](idx) if has_ve(i, self.config.n_layer) else None

            if kv_cache is None:
                # Prefill: use the same causal/window masks as training
                mask = self._masks[i] if T == self.config.sequence_len else (
                    "causal" if self.window_sizes[i][0] == self.config.sequence_len
                    else causal_window_mask(T, self.window_sizes[i][0])
                )
                past_kv = None
            else:
                # Decode: attend to all past tokens — no causal mask needed
                mask = None
                past_kv = kv_cache.layer_kvs[i]

            x, new_kv = block.forward_cached(x, ve, cos, sin, mask, past_kv)
            new_layer_kvs.append(new_kv)

            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda * x_backout
        x = norm(x)

        logits = self.lm_head(x)[..., : self.config.vocab_size]
        logits = self.SOFTCAP * mx.tanh(logits / self.SOFTCAP)

        return logits, MLXKVCache(new_layer_kvs)
