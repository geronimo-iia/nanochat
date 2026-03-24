"""
MLXEngine — autoregressive generation with KV-cache for Apple Silicon.

Mirrors the PyTorch Engine interface so that run_chat_eval, rl/eval.py,
and rl/rollout.py can call generate_batch identically on both backends.

Key differences from the PyTorch Engine:
- No in-place KV-cache updates; MLX builds new tensors per step (lazy concatenation).
- mx.eval() is called after every decode step to prevent unbounded lazy graph growth.
- No tool-use (calculator) logic — generation is token-sequence only.
- Single-device only (Apple Silicon unified memory, no DDP).
"""

from __future__ import annotations

import numpy as np


class MLXEngine:
    """MLX-based autoregressive engine with KV-cache.

    Usage::

        engine = MLXEngine(mlx_model, tokenizer)
        seqs, masks = engine.generate_batch(
            tokens=[1, 2, 3],
            num_samples=4,
            max_tokens=64,
            temperature=0.8,
            top_k=50,
            seed=42,
        )
    """

    def __init__(self, model: object, tokenizer: object) -> None:
        self._model = model       # MLXGPT instance
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        tokens: list[int],
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: int = 42,
    ):
        """Generator that yields (token_column, mask_column) per decode step.

        token_column: list[int] of length num_samples — the token chosen at
            this step for each sample (or a forced/EOS token).
        mask_column: list[int] of length num_samples — 1 for sampled tokens,
            0 for EOS/forced tokens.

        Matches the PyTorch Engine.generate() interface so that generate_batch
        can be shared between both engines.
        """
        import mlx.core as mx
        from nanochat.models.mlx_gpt import MLXKVCache

        rng = np.random.default_rng(seed)

        assistant_end = self._tokenizer.encode_special("<|assistant_end|>")
        bos = self._tokenizer.get_bos_token_id()

        # --- Prefill: encode the prompt once (batch=1), build KV cache ---
        prefix = mx.array([tokens], dtype=mx.int32)   # (1, prefix_len)
        logits, kv_cache = self._model.forward_with_kv_cache(prefix, kv_cache=None)
        mx.eval(logits)
        # Replicate last-position logits for all samples: (num_samples, vocab_size)
        last_logits = np.array(logits[:, -1, :].astype(mx.float32))  # (1, V)
        last_logits = np.repeat(last_logits, num_samples, axis=0)    # (B, V)

        # Replicate KV cache for each sample by repeating along batch axis
        replicated_kvs = []
        for k, v in kv_cache.layer_kvs:
            # k, v shape: (1, n_kv_head, pos, head_dim) → (B, ...)
            k_b = mx.repeat(k, num_samples, axis=0)
            v_b = mx.repeat(v, num_samples, axis=0)
            replicated_kvs.append((k_b, v_b))
        kv_cache = MLXKVCache(replicated_kvs)

        # Track per-sample current tokens and completion status
        current_tokens = [tokens.copy() for _ in range(num_samples)]
        completed = [False] * num_samples

        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(completed):
                break

            # Sample next tokens from last_logits
            sampled = self._sample_batch(last_logits, temperature, top_k, rng)  # (B,) int

            token_column = []
            mask_column = []
            for i in range(num_samples):
                tok = int(sampled[i])
                token_column.append(tok)
                if completed[i]:
                    mask_column.append(0)
                elif tok == assistant_end or tok == bos:
                    completed[i] = True
                    mask_column.append(0)
                else:
                    current_tokens[i].append(tok)
                    mask_column.append(1)

            yield token_column, mask_column
            num_generated += 1

            if all(completed):
                break

            # Decode step: feed the sampled tokens as a (B, 1) batch
            next_ids = mx.array([[tok] for tok in token_column], dtype=mx.int32)  # (B, 1)
            logits, kv_cache = self._model.forward_with_kv_cache(next_ids, kv_cache=kv_cache)
            mx.eval(logits)   # CRITICAL: prevent unbounded lazy graph accumulation
            last_logits = np.array(logits[:, -1, :].astype(mx.float32))  # (B, V)

    def generate_batch(
        self,
        tokens: list[int],
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: int = 42,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Non-streaming batch generation.

        Returns (sequences, masks) where:
        - sequences[i]: full token sequence (prefix + generated) for sample i
        - masks[i]: 1 for sampled tokens, 0 for prefix/EOS tokens

        Terminal tokens (assistant_end, bos) are not appended to sequences.
        """
        assistant_end = self._tokenizer.encode_special("<|assistant_end|>")
        bos = self._tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        for token_column, mask_column in self.generate(
            tokens, num_samples, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, seed=seed,
        ):
            for i, (tok, m) in enumerate(zip(token_column, mask_column)):
                if not completed[i]:
                    if tok == assistant_end or tok == bos:
                        completed[i] = True
                    else:
                        results[i].append(tok)
                        masks[i].append(m)
            if all(completed):
                break

        return results, masks

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_batch(
        logits: np.ndarray,    # (B, vocab_size) float32
        temperature: float,
        top_k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample one token per batch row. Returns (B,) int array."""
        if temperature == 0.0:
            return np.argmax(logits, axis=-1)

        logits = logits / temperature

        if top_k > 0:
            # Zero out all but the top-k logits per row
            topk_indices = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
            mask = np.full_like(logits, -np.inf)
            np.put_along_axis(mask, topk_indices, np.take_along_axis(logits, topk_indices, axis=-1), axis=-1)
            logits = mask

        # Softmax → sample
        logits -= logits.max(axis=-1, keepdims=True)   # stability
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)

        B, V = probs.shape
        samples = np.array([rng.choice(V, p=probs[i]) for i in range(B)])
        return samples
