"""Tests for MLXEngine — KV-cache autoregressive generation on Apple Silicon."""

import pytest

pytest.importorskip("mlx", reason="MLX not installed")

import mlx.core as mx
import numpy as np

from nanochat.models.config import GPTConfig
from nanochat.models.mlx_gpt import GPT as MLXGPT
from nanochat.evaluation.mlx_engine import MLXEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_CONFIG = GPTConfig(
    sequence_len=64,
    vocab_size=128,
    n_layer=2,
    n_embd=64,
    n_head=2,
    n_kv_head=2,
    window_pattern="L",
)


class _StubTokenizer:
    """Minimal tokenizer stub — no real vocabulary needed."""

    def encode_special(self, s: str) -> int:
        return {"<|assistant_end|>": 2, "<|python_start|>": 3, "<|python_end|>": 4,
                "<|output_start|>": 5, "<|output_end|>": 6}[s]

    def get_bos_token_id(self) -> int:
        return 1

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in text.split() if t.isdigit()]


def _make_engine() -> MLXEngine:
    model = MLXGPT(TINY_CONFIG)
    mx.eval(model.parameters())
    return MLXEngine(model, _StubTokenizer())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_batch_shape():
    engine = _make_engine()
    seqs, masks = engine.generate_batch([10, 20, 30], num_samples=2, max_tokens=5, seed=0)
    assert len(seqs) == 2
    assert len(masks) == 2
    # Each sequence starts with the prefix
    for s in seqs:
        assert s[:3] == [10, 20, 30], f"prefix not preserved: {s}"
    # Mask and seq have same length
    for s, m in zip(seqs, masks):
        assert len(m) == len(s)


def test_generate_batch_respects_max_tokens():
    engine = _make_engine()
    prefix = [10, 20]
    max_tokens = 4
    seqs, _ = engine.generate_batch(prefix, num_samples=3, max_tokens=max_tokens, seed=1)
    for s in seqs:
        assert len(s) <= len(prefix) + max_tokens, f"sequence too long: {len(s)}"


def test_generate_batch_deterministic_seed():
    engine = _make_engine()
    tokens = [5, 6, 7]
    seqs1, _ = engine.generate_batch(tokens, num_samples=2, max_tokens=6, temperature=0.8, seed=99)
    seqs2, _ = engine.generate_batch(tokens, num_samples=2, max_tokens=6, temperature=0.8, seed=99)
    assert seqs1 == seqs2, "same seed must produce same output"


def test_generate_batch_different_seeds():
    engine = _make_engine()
    tokens = [5, 6, 7]
    seqs1, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=8, temperature=1.0, seed=1)
    seqs2, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=8, temperature=1.0, seed=2)
    # With temperature>0 and different seeds, almost certainly different
    assert seqs1 != seqs2, "different seeds should (very likely) produce different output"


def test_generate_batch_greedy():
    """temperature=0 → all samples are identical (greedy is deterministic)."""
    engine = _make_engine()
    tokens = [10, 11, 12]
    seqs, _ = engine.generate_batch(tokens, num_samples=3, max_tokens=5, temperature=0.0)
    assert seqs[0] == seqs[1] == seqs[2], "greedy samples must be identical"


def test_no_nan_in_generated_tokens():
    engine = _make_engine()
    seqs, _ = engine.generate_batch([1, 2, 3], num_samples=2, max_tokens=8, seed=42)
    for s in seqs:
        for tok in s:
            assert isinstance(tok, int) and 0 <= tok < TINY_CONFIG.vocab_size


def test_kv_cache_prefill_updates_pos():
    """After prefill, the KV-cache should have pos == prefix_len."""
    model = MLXGPT(TINY_CONFIG)
    mx.eval(model.parameters())
    prefix = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    _, kv_cache = model.forward_with_kv_cache(prefix, kv_cache=None)
    mx.eval([k for k, v in kv_cache.layer_kvs] + [v for k, v in kv_cache.layer_kvs])
    assert kv_cache.pos == 4, f"expected pos=4, got {kv_cache.pos}"


def test_kv_cache_decode_grows_pos():
    """Each decode step should advance the cache position by 1."""
    model = MLXGPT(TINY_CONFIG)
    mx.eval(model.parameters())
    prefix = mx.array([[1, 2, 3]], dtype=mx.int32)
    _, kv_cache = model.forward_with_kv_cache(prefix, kv_cache=None)
    mx.eval([k for k, v in kv_cache.layer_kvs] + [v for k, v in kv_cache.layer_kvs])
    assert kv_cache.pos == 3

    single = mx.array([[10]], dtype=mx.int32)
    _, kv_cache = model.forward_with_kv_cache(single, kv_cache=kv_cache)
    mx.eval([k for k, v in kv_cache.layer_kvs] + [v for k, v in kv_cache.layer_kvs])
    assert kv_cache.pos == 4
