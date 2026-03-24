"""
MLXRLTrainer — REINFORCE RL trainer for Apple Silicon via MLX.

Satisfies the BaseTrainer protocol. Owns the rollout iterator, gradient
accumulation loop, and MuonAdamW optimizer step.

Key design decisions:
- REINFORCE loss: loss = -(logp * advantages).sum() / (num_valid * num_passes * examples_per_rank)
- num_passes is a Python int (compile-time constant for mx.compile loop unrolling)
- mx.eval(loss, grads) after each pass — prevents deep lazy expression nesting
- No DDP: examples_per_rank = examples_per_step (single device, no dist.all_reduce)
- Sequences padded to static max_len for mx.compile static-shape requirement
"""

from __future__ import annotations

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


def _first_nan_key(tree: Any) -> str | None:
    """Return the first flat key whose array contains NaN, or None."""
    for k, v in nn.utils.tree_flatten(tree):
        if isinstance(v, mx.array) and bool(mx.any(mx.isnan(v.astype(mx.float32))).item()):
            return k
    return None


class _RLLossAndGrad(nn.Module):
    """Compiled REINFORCE loss+grad. Accepts fixed-shape (B, T) batches.

    Wraps GPT inside nn.Module so mx.compile can capture model parameters as
    state via inputs=[orig_model].
    """

    def __init__(self, model: GPT, examples_per_rank: int) -> None:
        super().__init__()
        self._examples_per_rank = examples_per_rank

        def _rl_loss(inputs: mx.array, targets: mx.array, advantages: mx.array, num_passes: int):
            # logp = -CE (per-token log-probabilities, zeroed at masked positions)
            logp = -model(inputs, targets, loss_reduction="none")   # (B, T)
            pg_obj = mx.sum(logp * advantages[:, None])
            num_valid = mx.maximum(
                mx.sum(targets >= 0).astype(mx.float32),
                mx.array(1.0),
            )
            pg_obj = pg_obj / (num_valid * float(num_passes) * float(examples_per_rank))
            return -pg_obj   # minimize negative objective = maximize reward

        self._lag = nn.value_and_grad(model, _rl_loss)

    def __call__(
        self,
        inputs: mx.array,      # (B, T) int32
        targets: mx.array,     # (B, T) int32, -1 for masked/padding
        advantages: mx.array,  # (B,) float32
        num_passes: int,       # Python int — compile-time constant
    ):
        return self._lag(inputs, targets, advantages, num_passes)


class MLXRLTrainer:
    """MLX REINFORCE trainer. Owns rollout iterator, loss computation, and optimizer."""

    def __init__(
        self,
        orig_model: GPT,
        optimizer: MuonAdamW,
        batch_iterator: Iterator[Any],    # yields from mlx_rl_rollout.get_batch_mlx()
        device_batch_size: int,
        examples_per_rank: int,
    ) -> None:
        self._orig_model = orig_model
        self._optimizer = optimizer
        self._batch_iterator = batch_iterator
        self._device_batch_size = device_batch_size
        self._examples_per_rank = examples_per_rank

        rl_loss_and_grad = _RLLossAndGrad(orig_model, examples_per_rank)
        self._loss_and_grad = mx.compile(
            rl_loss_and_grad, inputs=[orig_model], outputs=[orig_model]
        )

        # Snapshot initial LRs for scheduler multiplication
        for group in optimizer._groups:
            group["initial_lr"] = group["lr"]

        self._accumulated_grads: dict[str, Any] | None = None
        self._nan_detected = False

    def forward_backward(self) -> StepResult:
        _sequences_all, inputs_all, targets_all, _rewards_all, advantages_all = next(
            self._batch_iterator
        )
        B_full = inputs_all.shape[0]
        num_passes = B_full // self._device_batch_size

        accumulated_grads = None
        losses = []
        self._nan_detected = False

        for pass_idx in range(num_passes):
            b0 = pass_idx * self._device_batch_size
            b1 = b0 + self._device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            advantages = advantages_all[b0:b1]

            loss, grads = self._loss_and_grad(inputs, targets, advantages, num_passes)
            mx.eval(loss, grads)   # Materialize — prevents deep lazy nesting

            losses.append(loss)
            accumulated_grads = (
                grads if accumulated_grads is None
                else nn.utils.tree_map(lambda a, b: a + b, accumulated_grads, grads)
            )

            if not math.isfinite(float(loss.item())):
                key = _first_nan_key(grads)
                print(
                    f"[NaN] MLXRLTrainer forward_backward pass={pass_idx}/{num_passes}: "
                    f"loss={loss.item()}  first_nan_grad={key}"
                )
                self._nan_detected = True
                break

        assert accumulated_grads is not None
        self._accumulated_grads = accumulated_grads

        train_loss = float(losses[-1].item())
        return StepResult(loss=train_loss, dataloader_state_dict={})

    def step(self, lr_multiplier: float, momentum: float, weight_decay: float) -> None:
        assert self._accumulated_grads is not None, "step() called before forward_backward()"
        for group in self._optimizer._groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = weight_decay

        if self._nan_detected:
            print("[NaN] MLXRLTrainer step: skipping optimizer update")
            return

        self._optimizer.update(self._orig_model, self._accumulated_grads)
        mx.eval(self._orig_model.parameters(), self._optimizer.state())
        key = _first_nan_key(self._orig_model.parameters())
        if key is not None:
            print(f"[NaN] MLXRLTrainer after optimizer step: first_nan_param={key}")

    def forward_logits(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        raise NotImplementedError("forward_logits not used in RL training")

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
        current_dtypes = {k: v.dtype for k, v in nn.utils.tree_flatten(self._orig_model.parameters())}
        model_state = {
            k: v.astype(current_dtypes[k]) if k in current_dtypes else v
            for k, v in model_state.items()
        }
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
