"""Torch-based checkpoint manager."""

import glob
import os
from typing import Any

import torch

from nanochat.checkpoint.discovery import find_last_step
from nanochat.checkpoint.logger import CheckpointLogger, RankZeroLogger
from nanochat.checkpoint.protocol import Checkpoint, CheckpointMetadata, CheckpointStateProtocol
from nanochat.config.checkpoint import CheckpointConfig


class TorchCheckpointManager:
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
            model_path = os.path.join(self._dir, f"model_{step:06d}.pt")
            torch.save(model_state, model_path)
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
        device: Any,
        load_optimizer: bool = False,
        rank: int = 0,
    ) -> Checkpoint:
        model_path = os.path.join(self._dir, f"model_{step:06d}.pt")
        model_state = torch.load(model_path, map_location=device)

        optimizer_state = None
        if load_optimizer:
            optim_path = os.path.join(self._dir, f"optim_{step:06d}_rank{rank:d}.pt")
            optimizer_state = torch.load(optim_path, map_location=device)

        meta_path = os.path.join(self._dir, f"meta_{step:06d}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = CheckpointMetadata.from_json(f.read())

        return Checkpoint(step=step, model_state=model_state, optimizer_state=optimizer_state, metadata=metadata)

    def find_last_step(self) -> int:
        return find_last_step(self._dir)

    def exists(self, step: int) -> bool:
        return os.path.exists(os.path.join(self._dir, f"model_{step:06d}.pt"))

    @property
    def checkpoint_dir(self) -> str:
        return self._dir

    def _prune(self, current_step: int) -> None:
        """Remove oldest checkpoints, keeping only the last keep_last_n."""
        pattern = os.path.join(self._dir, "model_*.pt")
        steps = sorted(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in glob.glob(pattern))
        to_remove = steps[: -self._config.keep_last_n]
        for step in to_remove:
            for path in [
                os.path.join(self._dir, f"model_{step:06d}.pt"),
                os.path.join(self._dir, f"meta_{step:06d}.json"),
            ]:
                if os.path.exists(path):
                    os.remove(path)
            # Remove all rank shards for this step
            for optim_path in glob.glob(os.path.join(self._dir, f"optim_{step:06d}_rank*.pt")):
                os.remove(optim_path)
            self._logger.info(f"Pruned checkpoint at step {step}")
