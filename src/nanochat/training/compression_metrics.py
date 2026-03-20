"""
Compression-based metrics for training optimization.

This module implements information compression metrics to track training
efficiency, predict overfitting, and evaluate dataset quality.

Based on the principle that intelligence emerges from information compression
through pattern discovery and unification.
"""

from typing import Dict, List, Optional

import numpy as np

from nanochat.training.compression_math import (
    compute_compression_ratio,
    compute_conditional_entropy,
    compute_entropy,
    compute_gzip_compression,
)


class CompressionMetrics:
    """Track information compression during training."""

    def __init__(self, vocab_size: int):
        """
        Initialize compression metrics tracker.

        Args:
            vocab_size: Size of the vocabulary
        """
        self.vocab_size = vocab_size
        self.history: List[Dict[str, float]] = []

    def compute_entropy(self, tokens: np.ndarray) -> float:
        """Shannon entropy of token distribution in bits. tokens: (B, T)."""
        return compute_entropy(tokens)

    def compute_conditional_entropy(self, logits: np.ndarray, tokens: np.ndarray) -> float:
        """Conditional entropy H(X|Model) in bits. logits: (B, T, V), tokens: (B, T)."""
        return compute_conditional_entropy(logits, tokens)

    def compute_compression_ratio(self, logits: np.ndarray, tokens: np.ndarray) -> float:
        """H(X) / H(X|Model). Higher = better compression."""
        return compute_compression_ratio(logits, tokens)

    def compute_gzip_compression(self, tokens: np.ndarray) -> float:
        """Gzip compression ratio: original_size / compressed_size. tokens: (B, T)."""
        return compute_gzip_compression(tokens)

    def compute_pattern_diversity(self, activations: np.ndarray, window_size: int = 5) -> float:
        """
        Count unique activation patterns (n-grams).

        activations: (B, T, C). Returns diversity score (unique patterns / total patterns).
        """
        quantized = (activations * 100).astype(np.int64)
        B, T, C = quantized.shape
        sample_channels = min(10, C)
        ngrams = set()
        for b in range(B):
            for t in range(T - window_size + 1):
                ngram = tuple(quantized[b, t : t + window_size, :sample_channels].flatten().tolist())
                ngrams.add(ngram)
        total_patterns = B * (T - window_size + 1)
        return len(ngrams) / max(total_patterns, 1)

    def log_metrics(
        self,
        step: int,
        tokens: np.ndarray,
        logits: np.ndarray,
        loss: float,
        activations: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Compute and log all compression metrics.

        Args:
            step: Current training step
            tokens: Target tokens
            logits: Model predictions
            loss: Training loss
            activations: Optional dict of layer activations

        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            "step": step,
            "loss": loss,
            "entropy": self.compute_entropy(tokens),
            "conditional_entropy": self.compute_conditional_entropy(logits, tokens),
            "compression_ratio": self.compute_compression_ratio(logits, tokens),
            "gzip_compression": self.compute_gzip_compression(tokens),
        }

        # Add pattern diversity if activations provided
        if activations:
            for layer_name, acts in activations.items():
                if acts.ndim == 3:  # (B, T, C)
                    metrics[f"{layer_name}_diversity"] = self.compute_pattern_diversity(acts)

        # Compression efficiency: compression per unit loss
        metrics["compression_efficiency"] = metrics["compression_ratio"] / max(loss, 1e-6)

        self.history.append(metrics)
        return metrics

    def detect_overfitting(self, window: int = 100) -> bool:
        """
        Detect overfitting via compression plateau.

        Args:
            window: Number of steps to compare

        Returns:
            True if compression has plateaued (possible overfitting)
        """
        if len(self.history) < window * 2:
            return False

        recent = [h["compression_ratio"] for h in self.history[-window:]]
        previous = [h["compression_ratio"] for h in self.history[-2 * window : -window]]

        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)

        improvement = (recent_mean - previous_mean) / previous_mean

        # Overfitting if compression improvement < 1%
        return bool(improvement < 0.01)

    def get_summary(self) -> Dict[str, object]:
        """
        Get summary statistics of compression metrics.

        Returns:
            Dictionary with mean, std, min, max for key metrics
        """
        if not self.history:
            return {}

        compression_ratios = [h["compression_ratio"] for h in self.history]
        efficiencies = [h["compression_efficiency"] for h in self.history]

        return {
            "compression_ratio_mean": np.mean(compression_ratios),
            "compression_ratio_std": np.std(compression_ratios),
            "compression_ratio_min": np.min(compression_ratios),
            "compression_ratio_max": np.max(compression_ratios),
            "compression_efficiency_mean": np.mean(efficiencies),
            "compression_efficiency_std": np.std(efficiencies),
        }
