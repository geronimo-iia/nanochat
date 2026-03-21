"""MLX utility functions for device info, dtype resolution, and memory reporting.

All mlx.core imports are lazy so this module is importable on non-MLX machines.
Only called from the MLX backend path.
"""

import os


def get_mlx_compute_dtype():
    """Return the MLX compute dtype. Defaults to bfloat16, overridable via NANOCHAT_DTYPE."""
    import mlx.core as mx

    _MAP = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
    env = os.environ.get("NANOCHAT_DTYPE")
    return _MAP[env] if env in _MAP else mx.bfloat16


def get_mlx_peak_memory() -> int:
    """Return peak MLX memory usage in bytes."""
    import mlx.core as mx

    return mx.get_peak_memory()


def get_mlx_device_info() -> dict:
    """Return MLX device info dict: device_name, memory_size, architecture, etc."""
    import mlx.core as mx

    return mx.device_info()
