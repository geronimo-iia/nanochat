from dataclasses import dataclass
from typing import Self


@dataclass
class RLState:
    """Mutable loop state for RL training.

    Minimal — RL does not checkpoint loop state.
    Exists for consistency and future extension.
    """

    step: int

    @classmethod
    def fresh(cls) -> Self:
        return cls(step=0)

    @classmethod
    def from_checkpoint(cls, meta_data: dict[str, object]) -> Self:
        return cls(step=int(meta_data["step"]))

    def to_checkpoint(self, model_config: dict[str, object]) -> dict[str, object]:
        return {
            "step": self.step,
            "model_config": model_config,
        }
