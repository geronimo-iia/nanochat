"""Safetensors-based checkpoint manager. Model weights as .safetensors, optimizer as .pt.

Note: optimizer state is currently saved with torch.save/torch.load. This works for both
TorchTrainer and MLXTrainer (pickle handles mlx arrays) but introduces a hidden torch
dependency. A cleaner approach would be to split optimizer state into tensor buffers
(.safetensors) and scalar metadata (JSON), removing the torch dependency entirely.
"""

import glob
import os
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from nanochat.checkpoint.convert import to_numpy
from nanochat.checkpoint.discovery import find_last_step
from nanochat.checkpoint.logger import CheckpointLogger, RankZeroLogger
from nanochat.checkpoint.protocol import Checkpoint, CheckpointMetadata, CheckpointStateProtocol
from nanochat.config.checkpoint import CheckpointConfig


class SafetensorsCheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        config: CheckpointConfig,
        logger: CheckpointLogger | None = None,
    ) -> None:
        self._dir = checkpoint_dir
        self._config = config
        self._logger = logger or RankZeroLogger()

    def save(
        self,
        state: CheckpointStateProtocol,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any] | None,
        rank: int = 0,
    ) -> None:
        step = state.step
        os.makedirs(self._dir, exist_ok=True)

        if rank == 0:
            model_path = os.path.join(self._dir, f"model_{step:06d}.safetensors")
            save_file(to_numpy(model_state), model_path)
            self._logger.info(f"Saved model parameters to: {model_path}")

            meta_path = os.path.join(self._dir, f"meta_{step:06d}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(state.to_metadata().to_json())
            self._logger.info(f"Saved metadata to: {meta_path}")

        if optimizer_state is not None:
            optim_path = os.path.join(self._dir, f"optim_{step:06d}_rank{rank:d}.pt")
            torch.save(optimizer_state, optim_path)
            self._logger.info(f"Saved optimizer state to: {optim_path}")

        if rank == 0 and self._config.keep_last_n > 0:
            self._prune(step)

    def load(
        self,
        step: int,
        device: Any = None,
        load_optimizer: bool = False,
        rank: int = 0,
    ) -> Checkpoint:
        model_path = os.path.join(self._dir, f"model_{step:06d}.safetensors")
        model_state: dict[str, np.ndarray] = load_file(model_path)

        optimizer_state: dict[str, Any] | None = None
        if load_optimizer:
            optim_path = os.path.join(self._dir, f"optim_{step:06d}_rank{rank:d}.pt")
            optimizer_state = torch.load(optim_path, map_location="cpu", weights_only=False)

        meta_path = os.path.join(self._dir, f"meta_{step:06d}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = CheckpointMetadata.from_json(f.read())

        return Checkpoint(step=step, model_state=model_state, optimizer_state=optimizer_state, metadata=metadata)

    def find_last_step(self) -> int:
        return find_last_step(self._dir)

    def exists(self, step: int) -> bool:
        return os.path.exists(os.path.join(self._dir, f"model_{step:06d}.safetensors"))

    @property
    def checkpoint_dir(self) -> str:
        return self._dir

    def _prune(self, current_step: int) -> None:
        pattern = os.path.join(self._dir, "model_*.safetensors")
        steps = sorted(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in glob.glob(pattern))
        to_remove = steps[: -self._config.keep_last_n]
        for step in to_remove:
            for path in [
                os.path.join(self._dir, f"model_{step:06d}.safetensors"),
                os.path.join(self._dir, f"meta_{step:06d}.json"),
            ]:
                if os.path.exists(path):
                    os.remove(path)
            for optim_path in glob.glob(os.path.join(self._dir, f"optim_{step:06d}_rank*.pt")):
                os.remove(optim_path)
            self._logger.info(f"Pruned checkpoint at step {step}")
