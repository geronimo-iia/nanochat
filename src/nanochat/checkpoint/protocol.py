"""Protocols and data classes for checkpoint I/O."""

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Self, runtime_checkable


@dataclass
class LoopState:
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float


@dataclass
class CheckpointMetadata:
    step: int
    model_config: dict[str, Any]
    user_config: dict[str, Any]
    val_bpb: float | None = None
    loop_state: LoopState | None = None
    dataloader_state_dict: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.loop_state is None:
            d["loop_state"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        loop = d.get("loop_state")
        return cls(
            step=d["step"],
            model_config=d["model_config"],
            user_config=d["user_config"],
            val_bpb=d.get("val_bpb"),
            loop_state=LoopState(**loop) if loop else None,
            dataloader_state_dict=d.get("dataloader_state_dict"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> Self:
        return cls.from_dict(json.loads(s))


@dataclass
class Checkpoint:
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    metadata: CheckpointMetadata


@runtime_checkable
class CheckpointStateProtocol(Protocol):
    step: int

    @classmethod
    def fresh(cls) -> Self: ...

    def to_metadata(self) -> CheckpointMetadata: ...

    @classmethod
    def from_metadata(cls, meta: CheckpointMetadata) -> Self: ...


class CheckpointManager(Protocol):
    def save(
        self,
        state: CheckpointStateProtocol,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any] | None,
        rank: int = 0,
    ) -> None: ...

    def load(
        self,
        step: int,
        device: Any,
        load_optimizer: bool = False,
        rank: int = 0,
    ) -> Checkpoint: ...

    def find_last_step(self) -> int: ...

    def exists(self, step: int) -> bool: ...
