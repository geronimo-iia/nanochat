"""
BaseTrainer protocol, StepResult, and TorchTrainer implementation.

See docs/trainer-implementation-plan.md for design rationale.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

import numpy as np
import torch
import torch.distributed as dist

from nanochat.checkpoint.convert import from_numpy_torch
from nanochat.common import get_compute_dtype, is_ddp_initialized
from nanochat.models.gpt import GPT
from nanochat.training.base.fp8 import disable_fp8


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


class TorchTrainer:
    """Torch backend trainer. Owns the train loader, accumulation loop, and optimizer step."""

    def __init__(
        self,
        orig_model: GPT,
        model: GPT,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        grad_accum_steps: int,
        device_type: str,
        train_loader: Any,
    ) -> None:
        for group in optimizer.param_groups:
            assert "initial_lr" in group, (
                f"param group '{group.get('kind', '?')}' missing 'initial_lr' — "
                "optimizer must be constructed via model.setup_optimizer()"
            )

        self._orig_model = orig_model
        self._model = model
        self._optimizer = optimizer
        self._scaler = scaler
        self._grad_accum_steps = grad_accum_steps
        self._device_type = device_type
        self._train_loader = train_loader

        # Prime the loader — first batch ready before the loop starts
        self._x: torch.Tensor
        self._y: torch.Tensor
        self._loader_state: dict[str, object]
        self._x, self._y, self._loader_state = next(train_loader)
        self._last_x = self._x
        self._last_y = self._y

    def forward_backward(self) -> StepResult:
        # Snapshot the batch at the start of the step for forward_logits
        self._last_x = self._x
        self._last_y = self._y

        train_loss = torch.zeros(1, device=self._x.device)
        for _ in range(self._grad_accum_steps):
            with torch.amp.autocast(device_type=self._device_type, dtype=get_compute_dtype()):
                loss = self._model(self._x, self._y)
            train_loss = loss.detach()
            loss = loss / self._grad_accum_steps
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()
            self._x, self._y, self._loader_state = next(self._train_loader)

        return StepResult(loss=train_loss.item(), dataloader_state_dict=self._loader_state)

    def step(self, lr_multiplier: float, momentum: float, weight_decay: float) -> None:
        for group in self._optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = weight_decay

        if self._scaler is not None:
            self._scaler.unscale_(self._optimizer)
            if is_ddp_initialized():
                for v in self._scaler._found_inf_per_device(self._optimizer).values():
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()

        self._model.zero_grad(set_to_none=True)

    def forward_logits(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        with torch.no_grad(), disable_fp8(self._orig_model), torch.amp.autocast(
            device_type=self._device_type, dtype=get_compute_dtype()
        ):
            logits = self._orig_model(self._last_x)
        return logits.float().cpu().numpy(), self._last_y.cpu().numpy()

    def model_state_dict(self) -> dict[str, Any]:
        return self._orig_model.state_dict()

    def optimizer_state_dict(self) -> dict[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dicts(self, model_state: dict[str, Any], optimizer_state: dict[str, Any]) -> None:
        if any(isinstance(v, np.ndarray) for v in model_state.values()):
            model_state = from_numpy_torch(model_state)
        self._orig_model.load_state_dict(model_state, strict=True, assign=True)
        self._optimizer.load_state_dict(optimizer_state)

    @contextmanager
    def eval_context(self) -> Iterator[GPT]:
        self._orig_model.eval()
        try:
            with disable_fp8(self._orig_model):
                yield self._orig_model
        finally:
            self._model.train()
