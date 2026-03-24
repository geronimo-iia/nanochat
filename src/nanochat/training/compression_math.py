"""
Pure numpy compression math functions.

Backend-agnostic. Callers are responsible for converting tensors to numpy
before calling these functions (boundary: forward_logits on each trainer).
"""

import gzip
from collections import Counter

import numpy as np
from scipy.special import logsumexp


def compute_entropy(tokens: np.ndarray) -> float:
    """Shannon entropy of token distribution in bits. tokens: (B, T)."""
    flat = tokens.flatten()
    counts = Counter(flat)
    total = len(flat)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def compute_conditional_entropy(logits: np.ndarray, tokens: np.ndarray) -> float:
    """
    Conditional entropy H(X|Model) in bits. logits: (B, T, V), tokens: (B, T).

    Uses cross-entropy as upper bound. Equivalent to torch.log_softmax + gather.
    """
    log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)  # (B, T, V)
    B, T = tokens.shape
    token_log_probs = log_probs[np.arange(B)[:, None], np.arange(T)[None, :], tokens]
    return -token_log_probs.mean() / np.log(2)


def compute_compression_ratio(logits: np.ndarray, tokens: np.ndarray) -> float:
    """H(X) / H(X|Model). Higher = better compression."""
    conditional = compute_conditional_entropy(logits, tokens)
    if conditional < 1e-6:
        return float("inf")
    return compute_entropy(tokens) / conditional


def compute_gzip_compression(tokens: np.ndarray) -> float:
    """Gzip compression ratio: original_size / compressed_size. tokens: (B, T)."""
    raw = tokens.tobytes()
    return len(raw) / len(gzip.compress(raw, compresslevel=9))
