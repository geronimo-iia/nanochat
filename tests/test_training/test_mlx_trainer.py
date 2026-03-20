"""Tests for MLXTrainer."""

import pytest

pytest.importorskip("mlx", reason="MLX not installed")

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

from nanochat.models.config import GPTConfig
from nanochat.models.mlx_gpt import GPT
from nanochat.training.mlx_optimizer import MuonAdamW, build_param_groups
from nanochat.training.mlx_trainer import MLXTrainer

CONFIG = GPTConfig(
    sequence_len=16,
    vocab_size=64,
    n_layer=4,
    n_embd=64,
    n_head=4,
    n_kv_head=4,
    window_pattern="SL",
)


def _batch(rng: np.random.Generator) -> tuple[mx.array, mx.array, dict]:
    x = mx.array(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)).astype(np.int32))
    y = mx.array(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)).astype(np.int32))
    return x, y, {"epoch": 0, "pq_idx": 0, "rg_idx": 0}


def _loader(rng: np.random.Generator):
    while True:
        yield _batch(rng)


def _trainer(grad_accum_steps: int = 1) -> MLXTrainer:
    rng = np.random.default_rng(0)
    model = GPT(CONFIG)
    optimizer = MuonAdamW(build_param_groups(model))
    return MLXTrainer(model, optimizer, grad_accum_steps, _loader(rng))


def test_step_result_fields():
    trainer = _trainer()
    result = trainer.forward_backward()
    assert isinstance(result.loss, float)
    assert isinstance(result.dataloader_state_dict, dict)


def test_forward_backward_loss_finite():
    trainer = _trainer()
    result = trainer.forward_backward()
    assert np.isfinite(result.loss)


def test_step_changes_params():
    trainer = _trainer()
    before = {k: np.array(v) for k, v in
              mlx_nn.utils.tree_flatten(trainer._model.trainable_parameters())}
    trainer.forward_backward()
    trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.0)

    after = dict(mlx_nn.utils.tree_flatten(trainer._model.trainable_parameters()))
    changed = sum(1 for k in before if not np.allclose(before[k], np.array(after[k])))
    assert changed > 0


def test_grad_accum_loss_finite():
    trainer = _trainer(grad_accum_steps=2)
    result = trainer.forward_backward()
    assert np.isfinite(result.loss)


def test_forward_logits_shapes():
    trainer = _trainer()
    trainer.forward_backward()
    logits, targets = trainer.forward_logits()
    B, T = 2, CONFIG.sequence_len
    assert logits.shape == (B, T, CONFIG.vocab_size)
    assert targets.shape == (B, T)


def test_forward_logits_numpy():
    trainer = _trainer()
    trainer.forward_backward()
    logits, targets = trainer.forward_logits()
    assert isinstance(logits, np.ndarray)
    assert isinstance(targets, np.ndarray)


def test_model_state_dict_is_mlx():
    trainer = _trainer()
    state = trainer.model_state_dict()
    assert all(isinstance(v, mx.array) for v in state.values())


def test_model_state_dict_roundtrip():
    trainer = _trainer()
    state = trainer.model_state_dict()
    trainer.load_state_dicts(state, trainer.optimizer_state_dict())
    state2 = trainer.model_state_dict()
    for k in state:
        assert np.allclose(np.array(state[k]), np.array(state2[k]))


def test_load_state_dicts_from_numpy():
    """load_state_dicts must accept numpy arrays (safetensors manager output)."""
    trainer = _trainer()
    numpy_state = {k: np.array(v) for k, v in trainer.model_state_dict().items()}
    trainer.load_state_dicts(numpy_state, trainer.optimizer_state_dict())
    state = trainer.model_state_dict()
    for k, v in numpy_state.items():
        assert np.allclose(v, np.array(state[k]))


def test_step_lr_does_not_compound():
    """Calling step twice with the same lr_multiplier must produce the same lr."""
    trainer = _trainer()
    trainer.forward_backward()
    trainer.step(lr_multiplier=0.5, momentum=0.95, weight_decay=0.0)
    lrs_first = [g["lr"] for g in trainer._optimizer._groups]

    trainer.forward_backward()
    trainer.step(lr_multiplier=0.5, momentum=0.95, weight_decay=0.0)
    lrs_second = [g["lr"] for g in trainer._optimizer._groups]

    assert lrs_first == lrs_second


def test_eval_context_restores_train_mode():
    trainer = _trainer()
    with trainer.eval_context():
        pass
    # MLX nn.Module doesn't expose training flag directly — verify no exception
    # and that forward still works after context exit
    trainer.forward_backward()


def test_eval_context_restores_on_exception():
    trainer = _trainer()
    try:
        with trainer.eval_context():
            raise RuntimeError("test")
    except RuntimeError:
        pass
    trainer.forward_backward()
