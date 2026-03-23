"""Tests for MLXTrainer."""

import pytest

pytest.importorskip("mlx", reason="MLX not installed")

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import torch

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


def _batch(rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, dict]:
    x = torch.from_numpy(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)).astype(np.int32))
    y = torch.from_numpy(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)).astype(np.int32))
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


def test_compiled_loss_sees_updated_params():
    """mx.compile must re-read model params after each optimizer step.

    Without inputs=[model] in mx.compile, the compiled graph captures parameter
    arrays at compile time and ignores subsequent model.update() calls — loss
    stays flat at initialization entropy for the entire run.
    """
    trainer = _trainer()
    result0 = trainer.forward_backward()
    trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.0)
    result1 = trainer.forward_backward()
    # Loss must differ — compiled fn must see updated weights
    assert result0.loss != result1.loss



    trainer = _trainer()
    before = {k: np.array(v) for k, v in
              mlx_nn.utils.tree_flatten(trainer._orig_model.trainable_parameters())}
    trainer.forward_backward()
    trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.0)

    after = dict(mlx_nn.utils.tree_flatten(trainer._orig_model.trainable_parameters()))
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
    with trainer.eval_context() as model:
        # Run an actual forward pass + mx.eval, matching what evaluate_bpb_mlx does.
        # Without outputs=[orig_model] in mx.compile, the subsequent forward_backward()
        # crashes with "Attempting to eval an array without a primitive".
        rng = np.random.default_rng(1)
        x = mx.array(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)).astype(np.int32))
        logits = model(x)
        mx.eval(logits)
    trainer.forward_backward()


def test_loader_state_preserved_across_forward_backward():
    """_next_batch must advance the loader monotonically across forward_backward calls."""
    call_count = 0

    def counting_loader():
        nonlocal call_count
        rng = np.random.default_rng(0)
        while True:
            call_count += 1
            yield _batch(rng)

    model = GPT(CONFIG)
    optimizer = MuonAdamW(build_param_groups(model))
    grad_accum_steps = 2
    trainer = MLXTrainer(model, optimizer, grad_accum_steps, counting_loader())

    assert call_count == 1  # primed once at init

    trainer.forward_backward()
    # forward_backward loads (grad_accum_steps - 1) more batches during pre-loading,
    # then one final advance — total N loads per step.
    assert call_count == 1 + grad_accum_steps

    trainer.forward_backward()
    assert call_count == 1 + 2 * grad_accum_steps


def test_grad_clip_large_grads_stay_finite():
    """Gradient clipping must keep weights finite when a large gradient spike occurs."""
    trainer = _trainer()
    trainer.forward_backward()
    assert trainer._accumulated_grads is not None
    # Simulate a large gradient spike
    trainer._accumulated_grads = mlx_nn.utils.tree_map(
        lambda g: g * 1e4, trainer._accumulated_grads
    )
    trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.0)
    # Model weights must remain finite after the clipped update
    params = dict(mlx_nn.utils.tree_flatten(trainer._orig_model.parameters()))
    mx.eval(list(params.values()))
    assert all(np.isfinite(np.array(v)).all() for v in params.values()), \
        "model weights contain NaN/Inf after large-gradient step"


def test_nan_guard_skips_bad_step():
    """NaN loss must not corrupt model weights (observed: step 28 NaN in exp2 run).

    Without the NaN guard, NaN loss → NaN grads → NaN weights → NaN forever.
    With the guard, forward_backward() zeros the grads and the optimizer skips the update.
    """
    trainer = _trainer()
    # Snapshot weights before the NaN step
    before = {k: np.array(v) for k, v in
              mlx_nn.utils.tree_flatten(trainer._orig_model.trainable_parameters())}

    # Monkeypatch _loss_and_grad to return NaN loss, simulating the step-28 failure.
    real_lag = trainer._loss_and_grad
    def nan_loss_and_grad(x, y):
        loss, grads = real_lag(x, y)
        return mx.array(float("nan")), grads
    trainer._loss_and_grad = nan_loss_and_grad

    result = trainer.forward_backward()
    assert not np.isfinite(result.loss)  # loss is NaN as reported

    trainer._loss_and_grad = real_lag  # restore
    trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.0)

    # Weights must be finite — the guard zeroed grads so NaN did not propagate
    # (AdamW weight_decay may still cause small changes, so we check finiteness not equality)
    after = dict(mlx_nn.utils.tree_flatten(trainer._orig_model.trainable_parameters()))
    mx.eval(list(after.values()))
    assert all(np.isfinite(np.array(v)).all() for v in after.values()), \
        "model weights contain NaN/Inf — NaN guard did not protect weights"


def test_eval_context_restores_on_exception():
    trainer = _trainer()
    try:
        with trainer.eval_context():
            raise RuntimeError("test")
    except RuntimeError:
        pass
    trainer.forward_backward()


def test_lazy_accum_matches_eager():
    """Lazy accumulation (N calls, single eval) must produce the same gradients
    as N calls with per-step mx.eval.

    Validates the lazy accumulation approach used in MLXTrainer.forward_backward:
    deferring mx.eval to after all N compiled calls produces numerically equivalent
    accumulated gradients (max_diff < 1e-3 across all parameters).
    """
    from nanochat.training.mlx_trainer import _LossAndGrad

    N = 2
    rng = np.random.default_rng(42)
    batches = [
        (
            mx.array(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)).astype(np.int32)),
            mx.array(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)).astype(np.int32)),
        )
        for _ in range(N)
    ]

    mx.random.seed(0)
    ref = GPT(CONFIG)
    ref_weights = list(mlx_nn.utils.tree_flatten(ref.parameters()))

    def make_model():
        m = GPT(CONFIG)
        m.update(mlx_nn.utils.tree_unflatten([(k, v) for k, v in ref_weights]))
        mx.eval(m.parameters())
        return m

    def accum(model, eval_each_step: bool) -> dict:
        lag = _LossAndGrad(model)
        compiled = mx.compile(lag, inputs=[model], outputs=[model])
        grads_acc = None
        for x, y in batches:
            _, g = compiled(x, y)
            if eval_each_step:
                mx.eval(g)
            grads_acc = g if grads_acc is None else mlx_nn.utils.tree_map(
                lambda a, b: a + b, grads_acc, g
            )
        result = mlx_nn.utils.tree_map(lambda g: g / N, grads_acc)
        mx.eval(result)
        return dict(mlx_nn.utils.tree_flatten(result))

    eager_flat = accum(make_model(), eval_each_step=True)
    lazy_flat  = accum(make_model(), eval_each_step=False)

    assert set(eager_flat.keys()) == set(lazy_flat.keys())
    for k in eager_flat:
        diff = mx.max(mx.abs(
            eager_flat[k].astype(mx.float32) - lazy_flat[k].astype(mx.float32)
        )).item()
        assert diff < 1e-3, f"Gradient mismatch for {k}: max_diff={diff}"
