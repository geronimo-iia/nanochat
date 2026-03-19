import math
from collections.abc import Callable


def base_lr_scheduler(
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> Callable[[int], float]:
    """Linear warmup, constant, linear warmdown LR multiplier scheduler."""
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


def base_muon_momentum_scheduler(
    num_iterations: int,
    warmdown_ratio: float,
) -> Callable[[int], float]:
    """Muon momentum: warmup to 0.97 over 400 steps, hold, warmdown to 0.90."""
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


def base_weight_decay_scheduler(
    weight_decay_scaled: float,
    num_iterations: int,
) -> Callable[[int], float]:
    """Cosine weight decay to zero over num_iterations."""

    def get_weight_decay(it: int) -> float:
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

    return get_weight_decay
