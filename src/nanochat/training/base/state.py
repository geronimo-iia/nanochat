from dataclasses import dataclass, field
from typing import Any, Self, cast

from nanochat.checkpoint.protocol import CheckpointMetadata, LoopState


@dataclass
class PretrainingState:
    """Mutable loop state for base pretraining. Owns checkpoint serialization."""

    step: int
    val_bpb: float | None
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float
    dataloader_state_dict: dict[str, object] | None
    model_config: dict[str, Any] = field(default_factory=dict)
    user_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fresh(cls) -> Self:
        """Initial state for a new training run."""
        return cls(
            step=0,
            val_bpb=None,
            min_val_bpb=float("inf"),
            smooth_train_loss=0.0,
            total_training_time=0.0,
            dataloader_state_dict=None,
        )

    def to_metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            step=self.step,
            model_config=self.model_config,
            user_config=self.user_config,
            val_bpb=self.val_bpb,
            loop_state=LoopState(
                min_val_bpb=self.min_val_bpb,
                smooth_train_loss=self.smooth_train_loss,
                total_training_time=self.total_training_time,
            ),
            dataloader_state_dict=self.dataloader_state_dict,
        )

    @classmethod
    def from_metadata(cls, meta: CheckpointMetadata) -> Self:
        loop = meta.loop_state
        return cls(
            step=meta.step,
            val_bpb=meta.val_bpb,
            min_val_bpb=loop.min_val_bpb if loop else float("inf"),
            smooth_train_loss=loop.smooth_train_loss if loop else 0.0,
            total_training_time=loop.total_training_time if loop else 0.0,
            dataloader_state_dict=meta.dataloader_state_dict,
            model_config=meta.model_config,
            user_config=meta.user_config,
        )

    @classmethod
    def from_checkpoint(cls, meta_data: dict[str, object]) -> Self:
        """Restore state from a checkpoint meta_data dict."""
        loop = cast(dict[str, object], meta_data["loop_state"])
        return cls(
            step=meta_data["step"],
            val_bpb=meta_data["val_bpb"],
            min_val_bpb=loop["min_val_bpb"],
            smooth_train_loss=loop["smooth_train_loss"],
            total_training_time=loop["total_training_time"],
            dataloader_state_dict=meta_data["dataloader_state_dict"],
        )

    def to_checkpoint(
        self, model_config: dict[str, object], user_config: dict[str, object], batch_config: dict[str, object]
    ) -> dict[str, object]:
        """Produce the full meta_data dict passed to save_checkpoint()."""
        return {
            "step": self.step,
            "val_bpb": self.val_bpb,
            "model_config": model_config,
            "user_config": user_config,
            **batch_config,
            "dataloader_state_dict": self.dataloader_state_dict,
            "loop_state": {
                "min_val_bpb": self.min_val_bpb,
                "smooth_train_loss": self.smooth_train_loss,
                "total_training_time": self.total_training_time,
            },
        }
