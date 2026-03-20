"""Unit tests for TorchTrainer."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal stub model — avoids loading GPT weights
# ---------------------------------------------------------------------------

class _StubModel(nn.Module):
    """Two-layer linear model that returns a scalar loss or logits."""

    def __init__(self, vocab_size: int = 16, seq_len: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.linear = nn.Linear(vocab_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # x: (B, T) token ids → one-hot → linear → logits (B, T, V)
        one_hot = torch.nn.functional.one_hot(x, self.vocab_size).float()
        logits = self.linear(one_hot)  # (B, T, V)
        if y is None:
            return logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return loss

    # Satisfy TorchTrainer.eval_context which calls orig_model.eval() / model.train()
    def num_scaling_params(self):
        return {"total": sum(p.numel() for p in self.parameters())}


def _make_loader(vocab_size: int = 16, seq_len: int = 4, batch_size: int = 2):
    """Infinite loader yielding (x, y, state_dict) triples."""
    step = 0
    while True:
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        state = {"epoch": 0, "pq_idx": step, "rg_idx": 0}
        step += 1
        yield x, y, state


def _make_optimizer(model: nn.Module, initial_lr: float = 1e-3):
    """AdamW with initial_lr set on param groups (mimics setup_optimizer contract)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    for group in optimizer.param_groups:
        group["initial_lr"] = initial_lr
        group["kind"] = "adamw"
    return optimizer


def _make_trainer(grad_accum_steps: int = 1):
    from nanochat.training.base.trainer import TorchTrainer

    model = _StubModel()
    optimizer = _make_optimizer(model)
    loader = _make_loader()
    return TorchTrainer(
        orig_model=model,
        model=model,
        optimizer=optimizer,
        scaler=None,
        grad_accum_steps=grad_accum_steps,
        device_type="cpu",
        train_loader=loader,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_step_result_loss_is_float():
    trainer = _make_trainer()
    result = trainer.forward_backward()
    assert isinstance(result.loss, float)
    assert result.loss > 0.0


def test_step_result_dataloader_state_dict_populated():
    trainer = _make_trainer()
    result = trainer.forward_backward()
    assert "epoch" in result.dataloader_state_dict
    assert "pq_idx" in result.dataloader_state_dict


def test_step_result_state_dict_advances_each_step():
    trainer = _make_trainer()
    r1 = trainer.forward_backward()
    r2 = trainer.forward_backward()
    assert r2.dataloader_state_dict["pq_idx"] > r1.dataloader_state_dict["pq_idx"]


def test_initial_lr_assertion_fires():
    from nanochat.training.base.trainer import TorchTrainer

    model = _StubModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # deliberately omit initial_lr
    with pytest.raises(AssertionError, match="initial_lr"):
        TorchTrainer(
            orig_model=model,
            model=model,
            optimizer=optimizer,
            scaler=None,
            grad_accum_steps=1,
            device_type="cpu",
            train_loader=_make_loader(),
        )


def test_eval_context_restores_train_mode_on_clean_exit():
    trainer = _make_trainer()
    trainer.forward_backward()
    with trainer.eval_context():
        assert not trainer._orig_model.training
    assert trainer._model.training


def test_eval_context_restores_train_mode_on_exception():
    trainer = _make_trainer()
    trainer.forward_backward()
    with pytest.raises(RuntimeError):
        with trainer.eval_context():
            raise RuntimeError("simulated eval failure")
    assert trainer._model.training


def test_forward_logits_returns_numpy_arrays():
    trainer = _make_trainer()
    trainer.forward_backward()
    logits, tokens = trainer.forward_logits()
    assert isinstance(logits, np.ndarray)
    assert isinstance(tokens, np.ndarray)


def test_forward_logits_shapes_match_batch():
    vocab_size, seq_len, batch_size = 16, 4, 2
    trainer = _make_trainer()
    trainer.forward_backward()
    logits, tokens = trainer.forward_logits()
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert tokens.shape == (batch_size, seq_len)


def test_step_updates_lr_on_param_groups():
    trainer = _make_trainer()
    trainer.forward_backward()
    trainer.step(lr_multiplier=0.5, momentum=0.95, weight_decay=0.1)
    for group in trainer._optimizer.param_groups:
        assert abs(group["lr"] - group["initial_lr"] * 0.5) < 1e-9


def test_model_state_dict_roundtrip():
    trainer = _make_trainer()
    sd = trainer.model_state_dict()
    assert isinstance(sd, dict)
    assert len(sd) > 0


def test_load_state_dicts_restores_weights():
    trainer = _make_trainer()
    original_sd = {k: v.clone() for k, v in trainer.model_state_dict().items()}
    # corrupt weights
    with torch.no_grad():
        for p in trainer._orig_model.parameters():
            p.fill_(999.0)
    trainer.load_state_dicts(original_sd, trainer.optimizer_state_dict())
    for k, v in trainer.model_state_dict().items():
        assert torch.allclose(v, original_sd[k])
