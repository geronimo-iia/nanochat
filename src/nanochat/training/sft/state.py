from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass
class SFTState:
    """Mutable loop state for SFT training. Owns checkpoint serialization.

    Fields shared with the dataloader (last_step, approx_progress, current_epoch)
    replace the nonlocal variables used in the original train_sft.py.
    """

    step: int
    val_bpb: float | None
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float
    progress: float
    # Shared with dataloader — mutated by sft_data_generator_bos_bestfit
    last_step: bool
    approx_progress: float
    current_epoch: int

    @classmethod
    def fresh(cls) -> Self:
        """Initial state for a new SFT run."""
        return cls(
            step=0,
            val_bpb=None,
            min_val_bpb=float("inf"),
            smooth_train_loss=0.0,
            total_training_time=0.0,
            progress=0.0,
            last_step=False,
            approx_progress=0.0,
            current_epoch=1,
        )

    @classmethod
    def from_checkpoint(cls, meta_data: dict[str, object]) -> Self:
        """Restore state from a checkpoint meta_data dict."""
        return cls(
            step=meta_data["step"],
            val_bpb=meta_data["val_bpb"],
            min_val_bpb=float("inf"),
            smooth_train_loss=0.0,
            total_training_time=0.0,
            progress=0.0,
            last_step=False,
            approx_progress=0.0,
            current_epoch=1,
        )

    def to_checkpoint(
        self,
        model_config: dict[str, object],
        user_config: dict[str, object],
    ) -> dict[str, object]:
        """Produce the meta_data dict passed to save_checkpoint()."""
        return {
            "step": self.step,
            "val_bpb": self.val_bpb,
            "model_config": model_config,
            "user_config": user_config,
        }
