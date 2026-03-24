from dataclasses import dataclass, field
from typing import Any, Self

from nanochat.checkpoint.protocol import CheckpointMetadata


@dataclass
class RLState:
    """Mutable loop state for RL training.

    Minimal — RL does not checkpoint loop state.
    Exists for consistency and future extension.
    """

    step: int
    model_config: dict[str, Any] = field(default_factory=dict)
    user_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fresh(cls) -> Self:
        return cls(step=0)

    def to_metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            step=self.step,
            model_config=self.model_config,
            user_config=self.user_config,
        )

    @classmethod
    def from_metadata(cls, meta: CheckpointMetadata) -> Self:
        return cls(
            step=meta.step,
            model_config=meta.model_config,
            user_config=meta.user_config,
        )

    @classmethod
    def from_checkpoint(cls, meta_data: dict[str, object]) -> Self:
        return cls(step=int(meta_data["step"]))

    def to_checkpoint(self, model_config: dict[str, object]) -> dict[str, object]:
        return {
            "step": self.step,
            "model_config": model_config,
        }
