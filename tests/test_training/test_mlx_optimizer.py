"""Tests for MLX MuonAdamW optimizer."""

import pytest

pytest.importorskip("mlx", reason="MLX not installed")

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import torch

from nanochat.models.config import GPTConfig
from nanochat.models.mlx_gpt import GPT as MLXGPT
from nanochat.models import GPT as TorchGPT
from nanochat.training.mlx_optimizer import MuonAdamW, build_param_groups, muon_step


# ---------------------------------------------------------------------------
# Shared config — tiny model, fast tests
# ---------------------------------------------------------------------------

CONFIG = GPTConfig(
    sequence_len=16,
    vocab_size=64,
    n_layer=4,
    n_embd=64,
    n_head=4,
    n_kv_head=4,
    window_pattern="SL",
)


# ---------------------------------------------------------------------------
# muon_step unit tests
# ---------------------------------------------------------------------------

def test_muon_step_output_shapes():
    K, R, C = 3, 8, 16
    g = mx.ones((K, R, C))
    p = mx.ones((K, R, C)) * 0.1
    mom = mx.zeros((K, R, C))
    v = mx.zeros((K, 1, C))  # wide: red_dim=-2

    p2, mom2, v2 = muon_step(g, p, mom, v, 0.95, 0.01, 0.0, 0.9, 5, -2)
    mx.eval(p2, mom2, v2)

    assert p2.shape == (K, R, C)
    assert mom2.shape == (K, R, C)
    assert v2.shape == (K, 1, C)


def test_muon_step_tall_matrix():
    K, R, C = 2, 16, 8
    g = mx.ones((K, R, C))
    p = mx.ones((K, R, C)) * 0.1
    mom = mx.zeros((K, R, C))
    v = mx.zeros((K, R, 1))  # tall: red_dim=-1

    p2, mom2, v2 = muon_step(g, p, mom, v, 0.95, 0.01, 0.0, 0.9, 5, -1)
    mx.eval(p2, mom2, v2)

    assert p2.shape == (K, R, C)
    assert v2.shape == (K, R, 1)


def test_muon_step_params_change():
    K, R, C = 2, 8, 16
    p = mx.ones((K, R, C))
    g = mx.ones((K, R, C)) * 0.5
    mom = mx.zeros((K, R, C))
    v = mx.zeros((K, 1, C))

    p2, _, _ = muon_step(g, p, mom, v, 0.95, 0.01, 0.0, 0.9, 5, -2)
    mx.eval(p2)

    assert not mx.allclose(p, p2).item(), "Params should change after a step"


def test_muon_step_no_nan():
    K, R, C = 4, 32, 64
    rng = np.random.default_rng(0)
    g = mx.array(rng.standard_normal((K, R, C)).astype(np.float32))
    p = mx.array(rng.standard_normal((K, R, C)).astype(np.float32))
    mom = mx.zeros((K, R, C))
    v = mx.zeros((K, 1, C))

    p2, mom2, v2 = muon_step(g, p, mom, v, 0.95, 0.01, 0.01, 0.9, 5, -2)
    mx.eval(p2, mom2, v2)

    assert not mx.any(mx.isnan(p2)).item()
    assert not mx.any(mx.isnan(mom2)).item()
    assert not mx.any(mx.isnan(v2)).item()


# ---------------------------------------------------------------------------
# MuonAdamW integration tests
# ---------------------------------------------------------------------------

def test_build_param_groups_covers_all_params():
    model = MLXGPT(CONFIG)
    groups = build_param_groups(model)

    flat = dict(mlx_nn.utils.tree_flatten(model.trainable_parameters()))
    covered = {k for g in groups for k in g["_keys"]}

    assert covered == set(flat.keys()), (
        f"Missing: {set(flat.keys()) - covered}\nExtra: {covered - set(flat.keys())}"
    )


def test_build_param_groups_muon_same_shape():
    model = MLXGPT(CONFIG)
    groups = build_param_groups(model)
    flat = dict(mlx_nn.utils.tree_flatten(model.trainable_parameters()))

    for g in groups:
        if g["kind"] == "muon":
            shapes = {tuple(flat[k].shape) for k in g["_keys"]}
            assert len(shapes) == 1, f"Muon group has mixed shapes: {shapes}"


def test_optimizer_update_changes_params():
    model = MLXGPT(CONFIG)
    groups = build_param_groups(model)
    optimizer = MuonAdamW(groups)

    idx = mx.array(np.random.randint(0, CONFIG.vocab_size, (1, CONFIG.sequence_len)))
    targets = mx.array(np.random.randint(0, CONFIG.vocab_size, (1, CONFIG.sequence_len)))

    loss_and_grad = mlx_nn.value_and_grad(model, model)
    loss, grads = loss_and_grad(idx, targets)
    mx.eval(loss, grads)

    before = {k: mx.array(v) for k, v in
              mlx_nn.utils.tree_flatten(model.trainable_parameters())}

    optimizer.update(model, grads)
    mx.eval(model.parameters())

    after = dict(mlx_nn.utils.tree_flatten(model.trainable_parameters()))

    changed = sum(1 for k in before if not mx.allclose(before[k], after[k]).item())
    assert changed > 0, "No parameters changed after optimizer step"


def test_optimizer_two_steps_no_nan():
    model = MLXGPT(CONFIG)
    groups = build_param_groups(model)
    optimizer = MuonAdamW(groups)

    loss_and_grad = mlx_nn.value_and_grad(model, model)
    rng = np.random.default_rng(1)

    for _ in range(2):
        idx = mx.array(rng.integers(0, CONFIG.vocab_size, (1, CONFIG.sequence_len)))
        tgt = mx.array(rng.integers(0, CONFIG.vocab_size, (1, CONFIG.sequence_len)))
        loss, grads = loss_and_grad(idx, tgt)
        mx.eval(loss, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

    flat = dict(mlx_nn.utils.tree_flatten(model.trainable_parameters()))
    for k, v in flat.items():
        assert not mx.any(mx.isnan(v)).item(), f"NaN in {k} after 2 steps"


def test_optimizer_loss_decreases():
    """Loss should trend down over several steps on a fixed batch."""
    model = MLXGPT(CONFIG)
    groups = build_param_groups(model)
    optimizer = MuonAdamW(groups)

    loss_and_grad = mlx_nn.value_and_grad(model, model)
    rng = np.random.default_rng(42)
    idx = mx.array(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)))
    tgt = mx.array(rng.integers(0, CONFIG.vocab_size, (2, CONFIG.sequence_len)))

    losses = []
    for _ in range(10):
        loss, grads = loss_and_grad(idx, tgt)
        mx.eval(loss, grads)
        losses.append(loss.item())
        optimizer.update(model, grads)
        mx.eval(model.parameters())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Numeric validation against PyTorch
# ---------------------------------------------------------------------------

def _copy_weights_torch_to_mlx(torch_model: TorchGPT, mlx_model: MLXGPT) -> None:
    """Copy all weights from PyTorch model to MLX model via numpy."""
    n_layer = torch_model.config.n_layer
    head_dim = torch_model.config.n_embd // torch_model.config.n_head

    def t2m(t: torch.Tensor) -> mx.array:
        return mx.array(t.detach().float().numpy())

    mlx_model.wte.weight = t2m(torch_model.transformer.wte.weight)
    mlx_model.lm_head.weight = t2m(torch_model.lm_head.weight)

    for i in range(n_layer):
        ve_key = f"ve_{i}"
        if ve_key in mlx_model.value_embeds:
            mlx_model.value_embeds[ve_key].weight = t2m(torch_model.value_embeds[str(i)].weight)

    for i, (tb, mb) in enumerate(zip(torch_model.transformer.h, mlx_model.blocks)):
        mb.attn.c_q.weight = t2m(tb.attn.c_q.weight)
        mb.attn.c_k.weight = t2m(tb.attn.c_k.weight)
        mb.attn.c_v.weight = t2m(tb.attn.c_v.weight)
        mb.attn.c_proj.weight = t2m(tb.attn.c_proj.weight)
        if mb.attn.ve_gate is not None:
            mb.attn.ve_gate.weight = t2m(tb.attn.ve_gate.weight)
        mb.mlp.c_fc.weight = t2m(tb.mlp.c_fc.weight)
        mb.mlp.c_proj.weight = t2m(tb.mlp.c_proj.weight)

    mlx_model.resid_lambdas = t2m(torch_model.resid_lambdas)
    mlx_model.x0_lambdas = t2m(torch_model.x0_lambdas)
    mlx_model.smear_gate.weight = t2m(torch_model.smear_gate.weight)
    mlx_model.smear_lambda = t2m(torch_model.smear_lambda)
    mlx_model.backout_lambda = t2m(torch_model.backout_lambda)
    mlx_model.cos, mlx_model.sin = mlx_model._precompute_rotary(
        torch_model.config.sequence_len * mlx_model.ROTARY_OVERSHOOT, head_dim
    )
    mx.eval(mlx_model.parameters())


def _mlx_params_to_numpy(mlx_model: MLXGPT) -> dict[str, np.ndarray]:
    flat = dict(mlx_nn.utils.tree_flatten(mlx_model.trainable_parameters()))
    mx.eval(list(flat.values()))
    return {k: np.array(v) for k, v in flat.items()}


def _torch_params_to_numpy(torch_model: TorchGPT) -> dict[str, np.ndarray]:
    return {k: v.detach().float().numpy() for k, v in torch_model.named_parameters()}


def _build_torch_to_mlx_key_map(torch_model: TorchGPT, mlx_model: MLXGPT) -> dict[str, str]:
    """Map torch param names to MLX flat keys for comparison."""
    n_layer = torch_model.config.n_layer
    mapping: dict[str, str] = {
        "transformer.wte.weight": "wte.weight",
        "lm_head.weight": "lm_head.weight",
        "resid_lambdas": "resid_lambdas",
        "x0_lambdas": "x0_lambdas",
        "smear_gate.weight": "smear_gate.weight",
        "smear_lambda": "smear_lambda",
        "backout_lambda": "backout_lambda",
    }
    for i in range(n_layer):
        t_pfx = f"transformer.h.{i}"
        m_pfx = f"blocks.{i}"
        for sub in ["attn.c_q.weight", "attn.c_k.weight", "attn.c_v.weight",
                    "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]:
            mapping[f"{t_pfx}.{sub}"] = f"{m_pfx}.{sub}"
        if f"transformer.h.{i}.attn.ve_gate.weight" in dict(torch_model.named_parameters()):
            mapping[f"{t_pfx}.attn.ve_gate.weight"] = f"{m_pfx}.attn.ve_gate.weight"
        t_ve = f"value_embeds.{i}.weight"
        m_ve = f"value_embeds.ve_{i}.weight"
        if t_ve in dict(torch_model.named_parameters()):
            mapping[t_ve] = m_ve
    return mapping


def test_full_step_matches_pytorch():
    """
    End-to-end validation plan (steps 1-4):
    - Identical weights copied torch → mlx
    - Same input tokens → forward+backward on PyTorch
    - PyTorch gradients injected into MLX grad tree
    - One optimizer step on both
    - All params match within 1e-2
    """
    config = CONFIG
    rng = np.random.default_rng(99)
    tokens_np = rng.integers(0, config.vocab_size, (1, config.sequence_len)).astype(np.int32)
    targets_np = rng.integers(0, config.vocab_size, (1, config.sequence_len)).astype(np.int32)

    # --- PyTorch setup ---
    torch_model = TorchGPT(config)
    torch_model.init_weights()
    torch_model.eval()
    torch_opt = torch_model.setup_optimizer()

    # --- MLX setup (identical weights) ---
    mlx_model = MLXGPT(config)
    _copy_weights_torch_to_mlx(torch_model, mlx_model)

    mlx_groups = build_param_groups(mlx_model)
    mlx_opt = MuonAdamW(mlx_groups)

    key_map = _build_torch_to_mlx_key_map(torch_model, mlx_model)

    # --- PyTorch forward+backward ---
    torch_tokens = torch.tensor(tokens_np.astype(np.int64))
    torch_targets = torch.tensor(targets_np.astype(np.int64))
    loss = torch_model(torch_tokens, torch_targets)
    loss.backward()

    # --- Inject PyTorch gradients into MLX grad tree ---
    torch_named_params = dict(torch_model.named_parameters())
    mlx_flat_params = dict(mlx_nn.utils.tree_flatten(mlx_model.trainable_parameters()))

    # Build MLX grad tree from PyTorch grads using the key map
    mlx_grads_flat: dict[str, mx.array] = {}
    for t_key, m_key in key_map.items():
        if t_key in torch_named_params and torch_named_params[t_key].grad is not None:
            mlx_grads_flat[m_key] = mx.array(torch_named_params[t_key].grad.float().numpy())

    mlx_grads = mlx_nn.utils.tree_unflatten(list(mlx_grads_flat.items()))
    mx.eval(list(mlx_grads_flat.values()))

    # --- PyTorch optimizer step ---
    torch_opt.step()

    # --- MLX optimizer step ---
    mlx_opt.update(mlx_model, mlx_grads)
    mx.eval(mlx_model.parameters())

    # --- Compare all params ---
    torch_params = _torch_params_to_numpy(torch_model)
    mlx_params = _mlx_params_to_numpy(mlx_model)

    max_diffs: dict[str, float] = {}
    for t_key, m_key in key_map.items():
        if t_key not in torch_params or m_key not in mlx_params:
            continue
        diff = np.max(np.abs(torch_params[t_key] - mlx_params[m_key]))
        max_diffs[t_key] = diff

    worst_key = max(max_diffs, key=lambda k: max_diffs[k])
    worst_diff = max_diffs[worst_key]
    assert worst_diff < 1e-2, (
        f"Worst param diff after step: {worst_key} = {worst_diff:.6f}\n"
        f"All diffs: { {k: f'{v:.4f}' for k, v in sorted(max_diffs.items(), key=lambda x: -x[1])} }"
    )


def test_full_two_steps_match_pytorch():
    """
    Validation plan step 5: second step params still track PyTorch.
    """
    config = CONFIG
    rng = np.random.default_rng(77)

    torch_model = TorchGPT(config)
    torch_model.init_weights()
    torch_model.eval()
    torch_opt = torch_model.setup_optimizer()

    mlx_model = MLXGPT(config)
    _copy_weights_torch_to_mlx(torch_model, mlx_model)
    mlx_opt = MuonAdamW(build_param_groups(mlx_model))
    key_map = _build_torch_to_mlx_key_map(torch_model, mlx_model)

    for step in range(2):
        tokens_np = rng.integers(0, config.vocab_size, (1, config.sequence_len)).astype(np.int32)
        targets_np = rng.integers(0, config.vocab_size, (1, config.sequence_len)).astype(np.int32)

        # PyTorch step
        torch_opt.zero_grad()
        loss = torch_model(
            torch.tensor(tokens_np.astype(np.int64)),
            torch.tensor(targets_np.astype(np.int64)),
        )
        loss.backward()

        # Build MLX grads from PyTorch grads
        torch_named_params = dict(torch_model.named_parameters())
        mlx_grads_flat = {
            m_key: mx.array(torch_named_params[t_key].grad.float().numpy())
            for t_key, m_key in key_map.items()
            if t_key in torch_named_params and torch_named_params[t_key].grad is not None
        }
        mlx_grads = mlx_nn.utils.tree_unflatten(list(mlx_grads_flat.items()))
        mx.eval(list(mlx_grads_flat.values()))

        torch_opt.step()
        mlx_opt.update(mlx_model, mlx_grads)
        mx.eval(mlx_model.parameters())

    torch_params = _torch_params_to_numpy(torch_model)
    mlx_params = _mlx_params_to_numpy(mlx_model)

    max_diffs = {
        t_key: np.max(np.abs(torch_params[t_key] - mlx_params[m_key]))
        for t_key, m_key in key_map.items()
        if t_key in torch_params and m_key in mlx_params
    }
    worst_key = max(max_diffs, key=lambda k: max_diffs[k])
    worst_diff = max_diffs[worst_key]
    assert worst_diff < 1e-2, (
        f"Worst param diff after 2 steps: {worst_key} = {worst_diff:.6f}"
    )
    """One Muon step on identical weights+grads should match PyTorch within 1e-2."""
    from nanochat.training.optimizer import muon_step_fused

    rng = np.random.default_rng(7)
    K, R, C = 4, 32, 64

    # Shared initial values
    params_np = rng.standard_normal((K, R, C)).astype(np.float32)
    grads_np = rng.standard_normal((K, R, C)).astype(np.float32)

    momentum = 0.95
    lr = 0.02
    wd = 0.01
    beta2 = 0.9
    ns_steps = 5
    red_dim = -2  # wide matrix

    # --- PyTorch step ---
    t_params = torch.tensor(params_np.copy())
    t_grads = torch.tensor(grads_np.copy())
    t_mom = torch.zeros(K, R, C)
    t_v = torch.zeros(K, 1, C)
    t_momentum = torch.tensor(momentum)
    t_lr = torch.tensor(lr * max(1.0, R / C) ** 0.5)
    t_wd = torch.tensor(wd)
    t_beta2 = torch.tensor(beta2)

    muon_step_fused(t_grads, t_params, t_mom, t_v, t_momentum, t_lr, t_wd, t_beta2, ns_steps, red_dim)
    torch_result = t_params.numpy()

    # --- MLX step ---
    m_params = mx.array(params_np.copy())
    m_grads = mx.array(grads_np.copy())
    m_mom = mx.zeros((K, R, C))
    m_v = mx.zeros((K, 1, C))

    m_params2, _, _ = muon_step(
        m_grads, m_params, m_mom, m_v,
        momentum, lr * max(1.0, R / C) ** 0.5, wd, beta2, ns_steps, red_dim
    )
    mx.eval(m_params2)
    mlx_result = np.array(m_params2)

    max_diff = np.max(np.abs(torch_result - mlx_result))
    assert max_diff < 1e-2, f"Max param diff after Muon step: {max_diff:.6f}"
