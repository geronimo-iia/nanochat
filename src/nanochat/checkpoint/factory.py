"""Factory for creating CheckpointManager instances."""

from nanochat.checkpoint.logger import CheckpointLogger
from nanochat.checkpoint.protocol import CheckpointManager
from nanochat.checkpoint.safetensors_manager import SafetensorsCheckpointManager
from nanochat.checkpoint.torch_manager import TorchCheckpointManager
from nanochat.config.checkpoint import CheckpointConfig


def make_checkpoint_manager(
    checkpoint_dir: str,
    config: CheckpointConfig,
    logger: CheckpointLogger | None = None,
) -> CheckpointManager:
    if config.format == "torch":
        return TorchCheckpointManager(checkpoint_dir, config, logger)
    if config.format == "safetensors":
        return SafetensorsCheckpointManager(checkpoint_dir, config, logger)
    raise ValueError(f"Unsupported checkpoint format: {config.format!r}")
