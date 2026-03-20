"""Parity tests: compression_math numpy vs torch reference."""

import numpy as np
import pytest
import torch

from nanochat.training.compression_math import (
    compute_compression_ratio,
    compute_conditional_entropy,
    compute_entropy,
    compute_gzip_compression,
)

RNG = np.random.default_rng(42)
B, T, V = 2, 16, 256


@pytest.fixture
def batch():
    logits_np = RNG.standard_normal((B, T, V)).astype(np.float32)
    tokens_np = RNG.integers(0, V, size=(B, T)).astype(np.int64)
    return logits_np, tokens_np


def _torch_conditional_entropy(logits_np: np.ndarray, tokens_np: np.ndarray) -> float:
    logits = torch.from_numpy(logits_np)
    tokens = torch.from_numpy(tokens_np)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    return (-token_log_probs.mean().item()) / np.log(2)


def test_conditional_entropy_matches_torch(batch):
    logits_np, tokens_np = batch
    numpy_result = compute_conditional_entropy(logits_np, tokens_np)
    torch_result = _torch_conditional_entropy(logits_np, tokens_np)
    assert abs(numpy_result - torch_result) < 1e-5


def test_entropy_positive(batch):
    _, tokens_np = batch
    assert compute_entropy(tokens_np) > 0.0


def test_compression_ratio_gt_one(batch):
    logits_np, tokens_np = batch
    # random logits → model barely better than uniform → ratio near 1 or slightly above/below
    ratio = compute_compression_ratio(logits_np, tokens_np)
    assert ratio > 0.0


def test_gzip_compression_gt_one():
    # repeated tokens compress well
    tokens = np.zeros((2, 64), dtype=np.int64)
    assert compute_gzip_compression(tokens) > 1.0
