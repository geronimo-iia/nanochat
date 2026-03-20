"""
MLXTrainer — BaseTrainer implementation for Apple Silicon via MLX.

Single-device only. No FP8, no DDP, no GradScaler.
See docs/mlx-training-patterns.md for grad accumulation and mx.eval() cadence.
"""

from contextlib import contextmanager
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from nanochat.checkpoint.convert import from_numpy_mlx
from nanochat.models.mlx_gpt import GPT
from nanochat.training.base.trainer import StepResult
from nanochat.training.mlx_optimizer import MuonAdamW


class MLXTrainer:
    """MLX backend trainer. Owns the train loader, accumulation loop, and optimizer step."""

    def __init__(
        self,
        model: GPT,
        optimizer: MuonAdamW,
        grad_accum_steps: int,
        train_loader: Any,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._grad_accum_steps = grad_accum_steps
        self._train_loader = train_loader
        self._loss_and_grad = nn.value_and_grad(model, model)

        # Snapshot initial LRs for scheduler multiplication
        for group in optimizer._groups:
            group["initial_lr"] = group["lr"]

        # Prime the loader
        self._x: mx.array
        self._y: mx.array
        self._loader_state: dict[str, object]
        self._x, self._y, self._loader_state = next(train_loader)

    def forward_backward(self) -> StepResult:
        # Snapshot batch at start of step for forward_logits
        self._last_x = self._x
        self._last_y = self._y

        accumulated_grads = None
        total_loss = 0.0

        for _ in range(self._grad_accum_steps):
            loss, grads = self._loss_and_grad(self._x, self._y)
            mx.eval(loss, grads)
            total_loss += loss.item()
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = nn.utils.tree_map(
                    lambda a, b: a + b, accumulated_grads, grads
                )
            self._x, self._y, self._loader_state = next(self._train_loader)

        assert accumulated_grads is not None
        if self._grad_accum_steps > 1:
            self._accumulated_grads = nn.utils.tree_map(
                lambda g: g / self._grad_accum_steps, accumulated_grads
            )
        else:
            self._accumulated_grads = accumulated_grads

        mean_loss = total_loss / self._grad_accum_steps
        return StepResult(loss=mean_loss, dataloader_state_dict=self._loader_state)

    def step(self, lr_multiplier: float, momentum: float, weight_decay: float) -> None:
        for group in self._optimizer._groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = weight_decay

        self._optimizer.update(self._model, self._accumulated_grads)
        mx.eval(self._model.parameters(), self._optimizer.state())

    def forward_logits(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        logits = mx.stop_gradient(self._model(self._last_x))
        mx.eval(logits)
        return np.array(logits), np.array(self._last_y)

    def model_state_dict(self) -> dict[str, Any]:
        flat = dict(nn.utils.tree_flatten(self._model.parameters()))
        mx.eval(list(flat.values()))
        return flat

    def optimizer_state_dict(self) -> dict[str, Any]:
        return {
            "groups": self._optimizer._groups,
            "muon_state": self._optimizer._muon_state,
            "adamw_state": [
                dict(nn.utils.tree_flatten(a.state)) if a is not None else None
                for a in self._optimizer._adamw
            ],
        }

    def load_state_dicts(self, model_state: dict[str, Any], optimizer_state: dict[str, Any]) -> None:
        if any(isinstance(v, np.ndarray) for v in model_state.values()):
            model_state = from_numpy_mlx(model_state)
        self._model.update(nn.utils.tree_unflatten(list(model_state.items())))
        mx.eval(self._model.parameters())

        self._optimizer._groups = optimizer_state["groups"]
        self._optimizer._muon_state = optimizer_state["muon_state"]
        for i, state_flat in enumerate(optimizer_state["adamw_state"]):
            if state_flat is not None and self._optimizer._adamw[i] is not None:
                self._optimizer._adamw[i].state = nn.utils.tree_unflatten(list(state_flat.items()))

    @contextmanager
    def eval_context(self) -> Iterator[GPT]:
        self._model.eval()
        try:
            yield self._model
        finally:
            self._model.train()
