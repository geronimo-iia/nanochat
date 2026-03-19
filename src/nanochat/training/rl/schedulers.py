from collections.abc import Callable


def rl_lr_scheduler(num_steps: int) -> Callable[[int], float]:
    """Linear rampdown to zero over num_steps."""

    def get_lr_multiplier(it: int) -> float:
        return 1.0 - it / num_steps

    return get_lr_multiplier
