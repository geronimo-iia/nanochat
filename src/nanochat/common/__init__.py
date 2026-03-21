"""Common utilities for nanochat."""

from nanochat.common.distributed import (
    autodetect_backend,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_dist_info,
    is_ddp_initialized,
    is_ddp_requested,
    mlx_compute_init,
    torch_compute_cleanup,
    torch_compute_init,
)
from nanochat.common.dtype import get_compute_dtype, get_compute_dtype_reason
from nanochat.common.hardware import clear_device_cache, get_device_sync, get_peak_flops
from nanochat.common.io import download_file_with_lock, download_single_file, print0, print_banner
from nanochat.common.logging import ColoredFormatter, setup_default_logging
from nanochat.common.mlx import get_mlx_compute_dtype, get_mlx_device_info, get_mlx_peak_memory
from nanochat.common.wandb import WandbProtocol, init_wandb

__all__ = [
    # dtype
    "get_compute_dtype",
    "get_compute_dtype_reason",
    # logging
    "ColoredFormatter",
    "setup_default_logging",
    # distributed
    "is_ddp_requested",
    "is_ddp_initialized",
    "get_dist_info",
    "autodetect_backend",
    "autodetect_device_type",
    "torch_compute_init",
    "torch_compute_cleanup",
    "mlx_compute_init",
    # backward-compatible aliases
    "compute_init",
    "compute_cleanup",
    # io
    "download_file_with_lock",
    "download_single_file",
    "print0",
    "print_banner",
    # hardware
    "get_device_sync",
    "get_peak_flops",
    "clear_device_cache",
    # mlx
    "get_mlx_compute_dtype",
    "get_mlx_peak_memory",
    "get_mlx_device_info",
    # wandb
    "WandbProtocol",
    "init_wandb",
]
