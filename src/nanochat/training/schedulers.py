"""Learning rate and optimizer schedulers for training."""

import math
from typing import Callable


def create_lr_scheduler(
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> Callable[[int], float]:
    """Create learning rate multiplier scheduler (linear warmup, constant, linear warmdown).

    Args:
        num_iterations: Total number of training iterations
        warmup_steps: Number of steps for linear warmup
        warmdown_ratio: Ratio of iterations for linear warmdown
        final_lr_frac: Final LR as fraction of initial LR

    Returns:
        Function that takes iteration number and returns LR multiplier
    """
    warmdown_iters = round(warmdown_ratio * num_iterations)

    def get_lr_multiplier(it: int) -> float:
        if it < warmup_steps:
            return (it + 1) / warmup_steps
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac

    return get_lr_multiplier


def create_muon_momentum_scheduler(
    num_iterations: int,
    warmdown_ratio: float,
) -> Callable[[int], float]:
    """Create momentum scheduler for Muon optimizer (base training).

    Warms up to 0.97 over first 400 steps, holds, then warms down to 0.90
    during the LR warmdown phase.

    Args:
        num_iterations: Total number of training iterations
        warmdown_ratio: Ratio of iterations for warmdown phase

    Returns:
        Function that takes iteration number and returns momentum value
    """
    warmdown_iters = round(warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters

    def get_muon_momentum(it: int) -> float:
        if it < 400:
            frac = it / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif it >= warmdown_start:
            progress = (it - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97

    return get_muon_momentum


def create_sft_muon_momentum_scheduler() -> Callable[[int], float]:
    """Create momentum scheduler for Muon optimizer (SFT).

    Simple warmup to 0.95 over first 400 steps, no warmdown.
    Matches upstream's SFT behavior.

    Returns:
        Function that takes iteration number and returns momentum value
    """

    def get_muon_momentum(it: int) -> float:
        frac = min(it / 400, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    return get_muon_momentum


def create_weight_decay_scheduler(
    weight_decay_scaled: float,
    num_iterations: int,
) -> Callable[[int], float]:
    """Create weight decay scheduler for Muon optimizer (cosine decay to zero).

    Args:
        weight_decay_scaled: Initial weight decay value
        num_iterations: Total number of training iterations

    Returns:
        Function that takes iteration number and returns weight decay value
    """

    def get_weight_decay(it: int) -> float:
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

    return get_weight_decay
