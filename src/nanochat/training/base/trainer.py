"""
BaseTrainer protocol and StepResult for the dual-trainer architecture.

See docs/trainer-implementation-plan.md for design rationale.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class StepResult:
    loss: float
    dataloader_state_dict: dict[str, object]


class BaseTrainer(Protocol):
    def forward_backward(self) -> StepResult: ...
    def step(self, lr_multiplier: float, momentum: float, weight_decay: float) -> None: ...
    def forward_logits(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]: ...
    def model_state_dict(self) -> dict[str, Any]: ...
    def optimizer_state_dict(self) -> dict[str, Any]: ...
    def load_state_dicts(self, model_state: dict[str, Any], optimizer_state: dict[str, Any]) -> None: ...

    @contextmanager
    def eval_context(self) -> Iterator[Any]: ...
