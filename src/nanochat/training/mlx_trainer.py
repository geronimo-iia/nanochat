"""
MLXTrainer — BaseTrainer implementation for Apple Silicon via MLX.

Single-device only. No FP8, no DDP, no GradScaler.
"""

import math
from contextlib import contextmanager
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from nanochat.checkpoint.convert import from_numpy_mlx
from nanochat.models.mlx_gpt import GPT
from nanochat.training.base.trainer import StepResult
from nanochat.training.mlx_optimizer import MuonAdamW


def _first_nan_key(tree) -> str | None:
    """Return the first flat key whose array contains a NaN, or None."""
    for k, v in nn.utils.tree_flatten(tree):
        if isinstance(v, mx.array) and bool(mx.any(mx.isnan(v.astype(mx.float32))).item()):
            return k
    return None


class _LossAndGrad(nn.Module):
    """Thin nn.Module wrapper so mx.compile can capture model parameters as state."""

    def __init__(self, model: "GPT") -> None:
        super().__init__()
        self._lag = nn.value_and_grad(model, model)

    def __call__(self, x: mx.array, y: mx.array):
        return self._lag(x, y)



class MLXTrainer:
    """MLX backend trainer. Owns the train loader, accumulation loop, and optimizer step."""

    def __init__(
        self,
        orig_model: GPT,
        optimizer: MuonAdamW,
        grad_accum_steps: int,
        torch_loader: Any,
    ) -> None:
        self._orig_model = orig_model
        self._optimizer = optimizer
        self._grad_accum_steps = grad_accum_steps
        self._loader = torch_loader
        loss_and_grad = _LossAndGrad(orig_model)
        self._loss_and_grad = mx.compile(loss_and_grad, inputs=[orig_model], outputs=[orig_model])

        # Snapshot initial LRs for scheduler multiplication
        for group in optimizer._groups:
            group["initial_lr"] = group["lr"]

        self._accumulated_grads: dict | None = None
        self._nan_detected = False

        # Prime the loader
        self._x: mx.array
        self._y: mx.array
        self._loader_state: dict[str, object]
        self._x, self._y, self._loader_state = self._next_batch()
        self._last_x = self._x
        self._last_y = self._y

    def _next_batch(self) -> tuple[mx.array, mx.array, dict]:
        x, y, state = next(self._loader)
        return mx.array(x.numpy()), mx.array(y.numpy()), state

    def forward_backward(self) -> StepResult:
        # Snapshot first batch of this step for forward_logits.
        self._last_x = self._x
        self._last_y = self._y
        self._nan_detected = False

        # Per-step eval accumulation: materialize accumulated_grads after each step to
        # prevent deep lazy expression nesting. Without this, N=8 steps build a 7-level
        # chain of lazy additions per parameter (7×n_params extra kernel launches on eval).
        # Per-step eval keeps nesting depth ≤1 and avoids the overhead.
        #
        # Note: a fused single-call approach (_MultiStepLossAndGrad) was prototyped but
        # hits Metal's argument buffer limit on real models — N×n_params buffers in one
        # kernel exceeds the per-kernel maximum. See docs/dev/lazy-grad-accumulation.md.
        accumulated_grads = None
        losses: list[mx.array] = []

        for _ in range(self._grad_accum_steps):
            loss, grads = self._loss_and_grad(self._x, self._y)
            losses.append(loss)
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = nn.utils.tree_map(
                    lambda a, b: a + b, accumulated_grads, grads
                )
            mx.eval(accumulated_grads)  # Materialize — prevents 7-level lazy nesting at N=8
            self._x, self._y, self._loader_state = self._next_batch()

        assert accumulated_grads is not None
        if self._grad_accum_steps > 1:
            self._accumulated_grads = nn.utils.tree_map(
                lambda g: g / self._grad_accum_steps, accumulated_grads
            )
        else:
            self._accumulated_grads = accumulated_grads

        mx.eval(losses, self._accumulated_grads)

        train_loss = losses[-1].item()
        for i, loss in enumerate(losses):
            lv = loss.item()
            if not math.isfinite(lv):
                key = _first_nan_key(self._accumulated_grads)
                print(
                    f"[NaN] forward_backward accum={i}/{self._grad_accum_steps}: "
                    f"loss={lv}  first_nan_grad={key}"
                )
                self._nan_detected = True
                break

        return StepResult(loss=train_loss, dataloader_state_dict=self._loader_state)

    def reprime(self) -> None:
        self._x, self._y, self._loader_state = self._next_batch()

    def step(self, lr_multiplier: float, momentum: float, weight_decay: float) -> None:
        assert self._accumulated_grads is not None, "step() called before forward_backward()"
        for group in self._optimizer._groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = weight_decay

        if self._nan_detected:
            # NaN loss in forward_backward — skip optimizer to preserve model weights.
            print("[NaN] step: skipping optimizer update")
            return

        self._optimizer.update(self._orig_model, self._accumulated_grads)
        mx.eval(self._orig_model.parameters(), self._optimizer.state())
        # Param NaN check runs after the existing eval — no extra sync cost.
        key = _first_nan_key(self._orig_model.parameters())
        if key is not None:
            print(f"[NaN] after optimizer step: first_nan_param={key}")

    def forward_logits(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        logits = mx.stop_gradient(self._orig_model(self._last_x))
        mx.eval(logits)
        return np.array(logits.astype(mx.float32)), np.array(self._last_y)

    def model_state_dict(self) -> dict[str, Any]:
        flat = dict(nn.utils.tree_flatten(self._orig_model.parameters()))
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
        # Restore original dtypes — safetensors upcasts bfloat16 to float32 on save.
        # model.update() does not auto-cast, so we must match the model's existing dtypes.
        current_dtypes = {k: v.dtype for k, v in nn.utils.tree_flatten(self._orig_model.parameters())}
        model_state = {k: v.astype(current_dtypes[k]) if k in current_dtypes else v for k, v in model_state.items()}
        self._orig_model.update(nn.utils.tree_unflatten(list(model_state.items())))
        mx.eval(self._orig_model.parameters())

        self._optimizer._groups = optimizer_state["groups"]
        self._optimizer._muon_state = optimizer_state["muon_state"]
        for i, state_flat in enumerate(optimizer_state["adamw_state"]):
            if state_flat is not None and self._optimizer._adamw[i] is not None:
                self._optimizer._adamw[i].state = nn.utils.tree_unflatten(list(state_flat.items()))

    @contextmanager
    def eval_context(self) -> Iterator[GPT]:
        self._orig_model.eval()
        try:
            yield self._orig_model
        finally:
            self._orig_model.train()
