"""Tests for compression metrics."""

import numpy as np
import pytest

from nanochat.training.compression_metrics import CompressionMetrics

RNG = np.random.default_rng(0)


@pytest.fixture
def compression_tracker():
    return CompressionMetrics(vocab_size=1000)


def test_compute_entropy(compression_tracker):
    tokens_uniform = RNG.integers(0, 100, size=(4, 128)).astype(np.int64)
    tokens_concentrated = np.full((4, 128), 42, dtype=np.int64)

    assert compression_tracker.compute_entropy(tokens_uniform) > compression_tracker.compute_entropy(tokens_concentrated)
    assert compression_tracker.compute_entropy(tokens_concentrated) == 0.0
    assert compression_tracker.compute_entropy(tokens_uniform) > 0.0


def test_compute_conditional_entropy(compression_tracker):
    tokens = RNG.integers(0, 100, size=(2, 64)).astype(np.int64)

    logits_perfect = np.zeros((2, 64, 1000), dtype=np.float32)
    logits_perfect[np.arange(2)[:, None], np.arange(64)[None, :], tokens] = 10.0

    logits_random = RNG.standard_normal((2, 64, 1000)).astype(np.float32) * 0.1

    assert compression_tracker.compute_conditional_entropy(logits_perfect, tokens) < \
           compression_tracker.compute_conditional_entropy(logits_random, tokens)


def test_compute_compression_ratio(compression_tracker):
    tokens = RNG.integers(0, 100, size=(2, 64)).astype(np.int64)

    logits_good = np.zeros((2, 64, 1000), dtype=np.float32)
    logits_good[np.arange(2)[:, None], np.arange(64)[None, :], tokens] = 5.0

    logits_bad = RNG.standard_normal((2, 64, 1000)).astype(np.float32) * 0.1

    ratio_good = compression_tracker.compute_compression_ratio(logits_good, tokens)
    ratio_bad = compression_tracker.compute_compression_ratio(logits_bad, tokens)

    assert ratio_good > ratio_bad
    assert ratio_good > 1.0


def test_compute_gzip_compression(compression_tracker):
    tokens_repetitive = np.full((4, 128), 42, dtype=np.int64)
    tokens_random = RNG.integers(0, 1000, size=(4, 128)).astype(np.int64)

    assert compression_tracker.compute_gzip_compression(tokens_repetitive) > \
           compression_tracker.compute_gzip_compression(tokens_random)
    assert compression_tracker.compute_gzip_compression(tokens_repetitive) > 1.0


def test_compute_pattern_diversity(compression_tracker):
    activations_uniform = np.ones((2, 64, 128), dtype=np.float32)
    activations_diverse = RNG.standard_normal((2, 64, 128)).astype(np.float32)

    diversity_uniform = compression_tracker.compute_pattern_diversity(activations_uniform)
    diversity_diverse = compression_tracker.compute_pattern_diversity(activations_diverse)

    assert diversity_diverse > diversity_uniform
    assert 0.0 <= diversity_uniform <= 1.0
    assert 0.0 <= diversity_diverse <= 1.0


def test_log_metrics(compression_tracker):
    tokens = RNG.integers(0, 100, size=(2, 64)).astype(np.int64)
    logits = RNG.standard_normal((2, 64, 1000)).astype(np.float32)

    metrics = compression_tracker.log_metrics(step=100, tokens=tokens, logits=logits, loss=2.5)

    assert metrics["step"] == 100
    assert metrics["loss"] == 2.5
    assert metrics["entropy"] > 0
    assert metrics["compression_ratio"] > 0
    assert all(k in metrics for k in ("conditional_entropy", "gzip_compression", "compression_efficiency"))
    assert len(compression_tracker.history) == 1
    assert compression_tracker.history[0] == metrics


def test_log_metrics_with_activations(compression_tracker):
    tokens = RNG.integers(0, 100, size=(2, 64)).astype(np.int64)
    logits = RNG.standard_normal((2, 64, 1000)).astype(np.float32)
    activations = {
        "layer_0": RNG.standard_normal((2, 64, 128)).astype(np.float32),
        "layer_1": RNG.standard_normal((2, 64, 128)).astype(np.float32),
    }

    metrics = compression_tracker.log_metrics(step=100, tokens=tokens, logits=logits, loss=2.5, activations=activations)

    assert "layer_0_diversity" in metrics
    assert "layer_1_diversity" in metrics
    assert 0.0 <= metrics["layer_0_diversity"] <= 1.0


def test_detect_overfitting(compression_tracker):
    tokens = RNG.integers(0, 100, size=(2, 64)).astype(np.int64)

    for step in range(250):
        logits = RNG.standard_normal((2, 64, 1000)).astype(np.float32)
        logits[np.arange(2)[:, None], np.arange(64)[None, :], tokens] += step * 0.01
        compression_tracker.log_metrics(step=step, tokens=tokens, logits=logits, loss=2.5 - step * 0.01)

    assert not compression_tracker.detect_overfitting(window=100)

    for step in range(250, 350):
        logits = RNG.standard_normal((2, 64, 1000)).astype(np.float32)
        logits[np.arange(2)[:, None], np.arange(64)[None, :], tokens] += 2.5
        compression_tracker.log_metrics(step=step, tokens=tokens, logits=logits, loss=2.0)

    assert compression_tracker.detect_overfitting(window=50)


def test_get_summary(compression_tracker):
    tokens = RNG.integers(0, 100, size=(2, 64)).astype(np.int64)

    for step in range(10):
        logits = RNG.standard_normal((2, 64, 1000)).astype(np.float32)
        compression_tracker.log_metrics(step=step, tokens=tokens, logits=logits, loss=2.5)

    summary = compression_tracker.get_summary()

    assert all(k in summary for k in (
        "compression_ratio_mean", "compression_ratio_std",
        "compression_ratio_min", "compression_ratio_max",
        "compression_efficiency_mean", "compression_efficiency_std",
    ))
    assert summary["compression_ratio_mean"] > 0
    assert summary["compression_ratio_min"] <= summary["compression_ratio_mean"] <= summary["compression_ratio_max"]


def test_empty_summary(compression_tracker):
    assert compression_tracker.get_summary() == {}
