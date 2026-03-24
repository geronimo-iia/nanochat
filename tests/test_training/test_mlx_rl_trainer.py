"""Tests for MLXRLTrainer — REINFORCE RL trainer on Apple Silicon."""

import pytest

pytest.importorskip("mlx", reason="MLX not installed")

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

from nanochat.models.config import GPTConfig
from nanochat.models.mlx_gpt import GPT as MLXGPT
from nanochat.training.mlx_optimizer import MuonAdamW, build_param_groups
from nanochat.training.rl.mlx_rl_trainer import MLXRLTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_CONFIG = GPTConfig(
    sequence_len=32,
    vocab_size=64,
    n_layer=2,
    n_embd=32,
    n_head=2,
    n_kv_head=2,
    window_pattern="L",
)

B = 2   # batch size per pass
T = 8   # sequence length (static)


def _make_trainer(num_batches: int = 10):
    model = MLXGPT(TINY_CONFIG)
    mx.eval(model.parameters())

    optimizer = MuonAdamW(build_param_groups(
        model,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
    ))

    rng = np.random.default_rng(0)

    def _stub_rollouts():
        for _ in range(num_batches):
            inputs = mx.array(rng.integers(0, 64, (B, T - 1), dtype=np.int32))
            targets = mx.array(rng.integers(0, 64, (B, T - 1), dtype=np.int32))
            rewards = mx.array(rng.standard_normal(B).astype(np.float32))
            advantages = rewards - mx.mean(rewards)
            yield [], inputs, targets, rewards, advantages

    return MLXRLTrainer(
        orig_model=model,
        optimizer=optimizer,
        batch_iterator=_stub_rollouts(),
        device_batch_size=B,
        examples_per_rank=B,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rl_forward_backward_loss_finite():
    """forward_backward() must return a finite loss."""
    trainer = _make_trainer()
    result = trainer.forward_backward()
    assert math.isfinite(result.loss), f"loss is not finite: {result.loss}"


def test_rl_step_updates_params():
    """After forward_backward() + step(), model parameters must change."""
    trainer = _make_trainer()
    flat_before = {k: np.array(v) for k, v in mlx_nn.utils.tree_flatten(trainer._orig_model.parameters())}

    trainer.forward_backward()
    trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.0)

    flat_after = {k: np.array(v) for k, v in mlx_nn.utils.tree_flatten(trainer._orig_model.parameters())}
    changed = any(not np.allclose(flat_before[k], flat_after[k]) for k in flat_before)
    assert changed, "model parameters did not change after step()"


def test_rl_nan_guard_skips_step():
    """When loss is NaN, step() must not update parameters."""
    trainer = _make_trainer()
    flat_before = {k: np.array(v) for k, v in mlx_nn.utils.tree_flatten(trainer._orig_model.parameters())}

    # Force NaN by monkeypatching _loss_and_grad
    nan_loss = mx.array(float("nan"))
    nan_grads = {k: mx.zeros_like(v) for k, v in mlx_nn.utils.tree_flatten(trainer._orig_model.parameters())}
    trainer._loss_and_grad = lambda *args, **kwargs: (nan_loss, nan_grads)

    result = trainer.forward_backward()
    assert not math.isfinite(result.loss)

    trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.0)

    flat_after = {k: np.array(v) for k, v in mlx_nn.utils.tree_flatten(trainer._orig_model.parameters())}
    unchanged = all(np.allclose(flat_before[k], flat_after[k]) for k in flat_before)
    assert unchanged, "parameters changed despite NaN guard"


def test_rl_model_state_dict_roundtrip():
    """model_state_dict() returns a flat dict of arrays; keys are non-empty."""
    trainer = _make_trainer()
    sd = trainer.model_state_dict()
    assert isinstance(sd, dict)
    assert len(sd) > 0
    for k, v in sd.items():
        assert isinstance(k, str)
        assert isinstance(v, mx.array), f"expected mx.array, got {type(v)} for key {k}"


# ---------------------------------------------------------------------------
# import math needed for tests
# ---------------------------------------------------------------------------
import math
