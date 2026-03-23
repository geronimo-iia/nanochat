"""Tests for MLX GPT model."""

import pytest

pytest.importorskip("mlx", reason="MLX not installed")

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import torch

from nanochat.models.config import GPTConfig
from nanochat.models.mlx_gpt import GPT as MLXGPT
from nanochat.models import GPT as TorchGPT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CONFIG = GPTConfig(
    sequence_len=64,
    vocab_size=128,
    n_layer=4,
    n_embd=128,
    n_head=4,
    n_kv_head=4,
    window_pattern="SL",
)

GQA_CONFIG = GPTConfig(
    sequence_len=64,
    vocab_size=128,
    n_layer=4,
    n_embd=128,
    n_head=4,
    n_kv_head=2,
    window_pattern="SL",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_weights_torch_to_mlx(torch_model: TorchGPT, mlx_model: MLXGPT) -> None:
    """Copy all weights from PyTorch model to MLX model via numpy."""
    n_layer = torch_model.config.n_layer
    head_dim = torch_model.config.n_embd // torch_model.config.n_head

    def t2m(tensor: torch.Tensor) -> mx.array:
        return mx.array(tensor.detach().float().numpy())

    # Embeddings
    mlx_model.wte.weight = t2m(torch_model.transformer.wte.weight)
    mlx_model.lm_head.weight = t2m(torch_model.lm_head.weight)

    # Value embeddings
    for i in range(n_layer):
        key = str(i)
        ve_key = f"ve_{i}"
        if ve_key in mlx_model.value_embeds:
            mlx_model.value_embeds[ve_key].weight = t2m(torch_model.value_embeds[key].weight)

    # Blocks
    for i, (torch_block, mlx_block) in enumerate(zip(torch_model.transformer.h, mlx_model.blocks)):
        a_t, a_m = torch_block.attn, mlx_block.attn
        mlx_block.attn.c_q.weight = t2m(a_t.c_q.weight)
        mlx_block.attn.c_k.weight = t2m(a_t.c_k.weight)
        mlx_block.attn.c_v.weight = t2m(a_t.c_v.weight)
        mlx_block.attn.c_proj.weight = t2m(a_t.c_proj.weight)
        if a_m.ve_gate is not None:
            mlx_block.attn.ve_gate.weight = t2m(a_t.ve_gate.weight)
        mlx_block.mlp.c_fc.weight = t2m(torch_block.mlp.c_fc.weight)
        mlx_block.mlp.c_proj.weight = t2m(torch_block.mlp.c_proj.weight)

    # Scalars
    mlx_model.resid_lambdas = t2m(torch_model.resid_lambdas)
    mlx_model.x0_lambdas = t2m(torch_model.x0_lambdas)
    mlx_model.smear_gate.weight = t2m(torch_model.smear_gate.weight)
    mlx_model.smear_lambda = t2m(torch_model.smear_lambda)
    mlx_model.backout_lambda = t2m(torch_model.backout_lambda)

    # Rotary embeddings
    mlx_model.cos, mlx_model.sin = mlx_model._precompute_rotary(
        torch_model.config.sequence_len * mlx_model.ROTARY_OVERSHOOT, head_dim
    )

    mx.eval(mlx_model.parameters())


# ---------------------------------------------------------------------------
# Shape / smoke tests (no PyTorch dependency)
# ---------------------------------------------------------------------------

def test_forward_logits_shape():
    model = MLXGPT(SMALL_CONFIG)
    idx = mx.array(np.random.randint(0, 128, (2, 64)))
    logits = model(idx)
    mx.eval(logits)
    assert logits.shape == (2, 64, 128)


def test_forward_loss_scalar():
    model = MLXGPT(SMALL_CONFIG)
    idx = mx.array(np.random.randint(0, 128, (2, 64)))
    targets = mx.array(np.random.randint(0, 128, (2, 64)))
    loss = model(idx, targets)
    mx.eval(loss)
    assert loss.shape == ()
    assert float(loss) > 0


def test_forward_gqa():
    model = MLXGPT(GQA_CONFIG)
    idx = mx.array(np.random.randint(0, 128, (2, 64)))
    logits = model(idx)
    mx.eval(logits)
    assert logits.shape == (2, 64, 128)


def test_forward_short_sequence():
    """T < sequence_len should work (mask computed on the fly)."""
    model = MLXGPT(SMALL_CONFIG)
    idx = mx.array(np.random.randint(0, 128, (1, 16)))
    logits = model(idx)
    mx.eval(logits)
    assert logits.shape == (1, 16, 128)


def test_no_nan_in_logits():
    model = MLXGPT(SMALL_CONFIG)
    idx = mx.array(np.random.randint(0, 128, (2, 64)))
    logits = model(idx)
    mx.eval(logits)
    assert not mx.any(mx.isnan(logits)).item()


def test_softcap_bounds():
    """Logits must be in (-softcap, +softcap)."""
    model = MLXGPT(SMALL_CONFIG)
    idx = mx.array(np.random.randint(0, 128, (2, 64)))
    logits = model(idx)
    mx.eval(logits)
    cap = MLXGPT.SOFTCAP
    assert float(mx.max(logits).item()) < cap
    assert float(mx.min(logits).item()) > -cap


def test_trainable_parameters_present():
    """All expected parameter groups must appear in trainable_parameters."""
    model = MLXGPT(SMALL_CONFIG)
    params = dict(mlx_nn.utils.tree_flatten(model.trainable_parameters()))
    keys = "\n".join(params.keys())
    assert "wte" in keys
    assert "lm_head" in keys
    assert "blocks" in keys
    assert "value_embeds" in keys


# ---------------------------------------------------------------------------
# Numeric validation against PyTorch
# ---------------------------------------------------------------------------

def test_numeric_match_pytorch():
    """Forward pass logits must match PyTorch within 1e-3 (bf16 tolerance)."""
    config = SMALL_CONFIG

    torch_model = TorchGPT(config)
    torch_model.init_weights()
    torch_model.eval()

    mlx_model = MLXGPT(config)
    _copy_weights_torch_to_mlx(torch_model, mlx_model)

    rng = np.random.default_rng(42)
    tokens_np = rng.integers(0, config.vocab_size, size=(1, config.sequence_len))

    # PyTorch forward (float32)
    with torch.no_grad():
        torch_logits = torch_model(torch.tensor(tokens_np, dtype=torch.long)).float().numpy()

    # MLX forward (float32)
    mlx_out = mlx_model(mx.array(tokens_np))
    mx.eval(mlx_out)
    mlx_logits = np.array(mlx_out)

    max_diff = np.max(np.abs(torch_logits - mlx_logits))
    assert max_diff < 1e-3, f"Max logit diff {max_diff:.6f} exceeds 1e-3"


# ---------------------------------------------------------------------------
# S1: Masked CE (ignore_index=-1)
# ---------------------------------------------------------------------------

def test_masked_loss_ignores_minus_one():
    """Positions with target == -1 must not affect the loss."""
    model = MLXGPT(SMALL_CONFIG)
    rng = np.random.default_rng(7)
    idx = mx.array(rng.integers(0, 128, (2, 64)))

    # All-valid targets
    targets_full = mx.array(rng.integers(0, 128, (2, 64)))
    loss_full = model(idx, targets_full)
    mx.eval(loss_full)

    # Same targets but with the second half masked (-1)
    targets_np = np.array(targets_full)
    targets_half = targets_np.copy()
    targets_half[:, 32:] = -1
    loss_half = model(idx, mx.array(targets_half))
    mx.eval(loss_full, loss_half)

    # The two losses should differ (masking changes the average)
    assert float(loss_full) != float(loss_half)


def test_all_masked_loss_stable():
    """When all targets are -1 the loss must be 0 (not NaN or inf)."""
    model = MLXGPT(SMALL_CONFIG)
    idx = mx.array(np.zeros((1, 64), dtype=np.int32))
    targets = mx.array(np.full((1, 64), -1, dtype=np.int32))
    loss = model(idx, targets)
    mx.eval(loss)
    val = float(loss)
    assert val == 0.0, f"Expected 0.0, got {val}"


def test_masked_loss_matches_unmasked_on_valid_positions():
    """sum(ce * mask) / n_valid must equal mean(ce) when all targets are valid."""
    model = MLXGPT(SMALL_CONFIG)
    rng = np.random.default_rng(42)
    idx = mx.array(rng.integers(0, 128, (1, 32)))
    targets = mx.array(rng.integers(0, 128, (1, 32)))

    # loss_reduction="mean" uses masked path; all targets >=0 so result = plain mean
    masked_loss = model(idx, targets)
    mx.eval(masked_loss)

    # Verify it equals what the old code would have produced: mx.mean(ce)
    import mlx.nn as nn_
    logits = model(idx)
    flat_ce = nn_.losses.cross_entropy(
        logits.reshape(-1, SMALL_CONFIG.vocab_size), targets.reshape(-1)
    )
    plain_mean = mx.mean(flat_ce)
    mx.eval(plain_mean)

    assert abs(float(masked_loss) - float(plain_mean)) < 1e-5


# ---------------------------------------------------------------------------
# R2: Per-token loss (loss_reduction="none")
# ---------------------------------------------------------------------------

def test_per_token_loss_shape():
    model = MLXGPT(SMALL_CONFIG)
    B, T = 2, 64
    idx = mx.array(np.random.randint(0, 128, (B, T)))
    targets = mx.array(np.random.randint(0, 128, (B, T)))
    loss = model(idx, targets, loss_reduction="none")
    mx.eval(loss)
    assert loss.shape == (B, T)


def test_per_token_loss_positive():
    """All unmasked per-token CE values must be non-negative."""
    model = MLXGPT(SMALL_CONFIG)
    idx = mx.array(np.random.randint(0, 128, (1, 32)))
    targets = mx.array(np.random.randint(0, 128, (1, 32)))
    loss = model(idx, targets, loss_reduction="none")
    mx.eval(loss)
    assert float(mx.min(loss).item()) >= 0.0


def test_per_token_loss_masked_positions_zero():
    """Positions with target == -1 must produce exactly 0 CE."""
    model = MLXGPT(SMALL_CONFIG)
    idx = mx.array(np.zeros((1, 32), dtype=np.int32))
    targets_np = np.random.randint(0, 128, (1, 32))
    targets_np[:, 16:] = -1  # mask second half
    loss = model(idx, mx.array(targets_np), loss_reduction="none")
    mx.eval(loss)
    assert float(mx.max(mx.abs(loss[:, 16:])).item()) == 0.0, "masked positions must be zero"
    assert float(mx.max(loss[:, :16]).item()) > 0.0, "unmasked positions must be positive"


def test_per_token_loss_sum_matches_mean():
    """sum(per_token) / n_valid == scalar mean loss."""
    model = MLXGPT(SMALL_CONFIG)
    rng = np.random.default_rng(0)
    idx = mx.array(rng.integers(0, 128, (1, 32)))
    targets = mx.array(rng.integers(0, 128, (1, 32)))

    scalar = model(idx, targets, loss_reduction="mean")
    tokens = model(idx, targets, loss_reduction="none")
    mx.eval(scalar, tokens)

    n_valid = 32  # all valid
    reconstructed = float(mx.sum(tokens).item()) / n_valid
    assert abs(float(scalar) - reconstructed) < 1e-5


def test_numeric_match_pytorch_gqa():  # noqa: PLR0914
    """Numeric match with GQA (n_kv_head < n_head)."""
    config = GQA_CONFIG

    torch_model = TorchGPT(config)
    torch_model.init_weights()
    torch_model.eval()

    mlx_model = MLXGPT(config)
    _copy_weights_torch_to_mlx(torch_model, mlx_model)

    rng = np.random.default_rng(0)
    tokens_np = rng.integers(0, config.vocab_size, size=(1, config.sequence_len))

    with torch.no_grad():
        torch_logits = torch_model(torch.tensor(tokens_np, dtype=torch.long)).float().numpy()

    mlx_out = mlx_model(mx.array(tokens_np))
    mx.eval(mlx_out)
    mlx_logits = np.array(mlx_out)

    max_diff = np.max(np.abs(torch_logits - mlx_logits))
    assert max_diff < 1e-3, f"Max logit diff {max_diff:.6f} exceeds 1e-3"
