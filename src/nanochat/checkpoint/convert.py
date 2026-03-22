"""Array conversion between torch, mlx, and numpy for checkpoint interop."""

from typing import Any

import numpy as np
import torch


def to_numpy(state_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert a state dict of torch tensors or mlx arrays to numpy arrays."""
    out: dict[str, np.ndarray] = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            # mlx array — np.array() triggers evaluation and copies to CPU.
            # numpy has no bfloat16; upcast to float32 first.
            import mlx.core as mx
            if hasattr(v, 'dtype') and v.dtype == mx.bfloat16:
                v = v.astype(mx.float32)
            out[k] = np.array(v)
    return out


def from_numpy_torch(state_dict: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert numpy state dict to torch tensors."""
    return {k: torch.from_numpy(v.copy()) for k, v in state_dict.items()}


def from_numpy_mlx(state_dict: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert numpy state dict to mlx arrays. Only callable on Darwin."""
    import mlx.core as mx

    return {k: mx.array(v) for k, v in state_dict.items()}
