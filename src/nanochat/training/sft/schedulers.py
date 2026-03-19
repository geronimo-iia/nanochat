from typing import Callable


def sft_lr_scheduler(
    warmup_ratio: float,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> Callable[[float], float]:
    """Progress-based LR multiplier (progress 0→1): linear warmup, constant, linear warmdown."""

    def get_lr_multiplier(progress: float) -> float:
        if progress < warmup_ratio:
            return (progress + 1e-8) / warmup_ratio
        elif progress <= 1.0 - warmdown_ratio:
            return 1.0
        else:
            decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
            return (1 - decay) * 1.0 + decay * final_lr_frac

    return get_lr_multiplier


def sft_muon_momentum_scheduler() -> Callable[[int], float]:
    """Muon momentum: warmup to 0.95 over 400 steps, no warmdown."""

    def get_muon_momentum(it: int) -> float:
        frac = min(it / 400, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    return get_muon_momentum
