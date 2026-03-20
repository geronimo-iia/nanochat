"""Tests for backward compatibility patch utilities."""

import torch

from nanochat.checkpoint.compat import patch_missing_config_keys, patch_missing_keys
from nanochat.models.config import GPTConfig


def test_patch_missing_config_keys_adds_window_pattern():
    kwargs: dict[str, object] = {}
    patch_missing_config_keys(kwargs)
    assert kwargs["window_pattern"] == "L"


def test_patch_missing_config_keys_no_op_if_present():
    kwargs: dict[str, object] = {"window_pattern": "G"}
    patch_missing_config_keys(kwargs)
    assert kwargs["window_pattern"] == "G"


def test_patch_missing_config_keys_calls_logger():
    logged: list[str] = []
    patch_missing_config_keys({}, logger_info=logged.append)
    assert any("window_pattern" in m for m in logged)


def test_patch_missing_keys_adds_resid_and_x0_lambdas():
    cfg = GPTConfig(n_layer=3, n_embd=64, n_head=2, n_kv_head=2, vocab_size=256)
    model_data: dict[str, object] = {}
    patch_missing_keys(model_data, cfg)
    assert "resid_lambdas" in model_data
    assert "x0_lambdas" in model_data
    assert torch.allclose(model_data["resid_lambdas"], torch.ones(3))  # type: ignore[arg-type]
    assert torch.allclose(model_data["x0_lambdas"], torch.zeros(3))  # type: ignore[arg-type]


def test_patch_missing_keys_no_op_if_present():
    cfg = GPTConfig(n_layer=2, n_embd=64, n_head=2, n_kv_head=2, vocab_size=256)
    existing = torch.tensor([0.5, 0.5])
    model_data: dict[str, object] = {"resid_lambdas": existing, "x0_lambdas": existing}
    patch_missing_keys(model_data, cfg)
    assert model_data["resid_lambdas"] is existing


def test_patch_missing_keys_calls_logger():
    cfg = GPTConfig(n_layer=2, n_embd=64, n_head=2, n_kv_head=2, vocab_size=256)
    logged: list[str] = []
    patch_missing_keys({}, cfg, logger_info=logged.append)
    assert any("resid_lambdas" in m for m in logged)
    assert any("x0_lambdas" in m for m in logged)
