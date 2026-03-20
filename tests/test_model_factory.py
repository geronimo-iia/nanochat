"""Tests for model_factory."""

import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from nanochat.checkpoint.factory import make_checkpoint_manager
from nanochat.checkpoint.logger import SilentLogger
from nanochat.checkpoint.protocol import CheckpointMetadata, LoopState
from nanochat.config.checkpoint import CheckpointConfig
from nanochat.model_factory import load_model_from_dir, load_optimizer_state
from nanochat.models import GPT, GPTConfig


class _State:
    def __init__(self, step: int, model_config: dict[str, Any]) -> None:
        self.step = step
        self._model_config = model_config

    def to_metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            step=self.step,
            model_config=self._model_config,
            user_config={},
            loop_state=LoopState(min_val_bpb=0.0, smooth_train_loss=0.0, total_training_time=0.0),
        )


_GPT_CFG = GPTConfig(sequence_len=64, vocab_size=256, n_layer=2, n_embd=64, n_head=2, n_kv_head=2)
_MODEL_CONFIG_DICT = {
    "sequence_len": 64,
    "vocab_size": 256,
    "n_layer": 2,
    "n_embd": 64,
    "n_head": 2,
    "n_kv_head": 2,
    "window_pattern": "L",
}


def _save_checkpoint(tmpdir: str, step: int, with_optimizer: bool = False) -> None:
    model = GPT(_GPT_CFG)
    model.init_weights()
    optimizer_state = None
    if with_optimizer:
        opt = model.setup_optimizer(
            matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.008, scalar_lr=0.5, weight_decay=0.0
        )
        optimizer_state = opt.state_dict()
    mgr = make_checkpoint_manager(tmpdir, CheckpointConfig(), SilentLogger())
    mgr.save(_State(step, _MODEL_CONFIG_DICT), model.state_dict(), optimizer_state, rank=0)


def _mock_tokenizer(vocab_size: int) -> MagicMock:
    tok = MagicMock()
    tok.get_vocab_size.return_value = vocab_size
    return tok


def test_load_model_from_dir_returns_model_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_checkpoint(tmpdir, step=10)

        with (
            patch("nanochat.model_factory.workspace.checkpoint_dir", return_value=tmpdir),
            patch("nanochat.model_factory.find_last_step", return_value=10),
            patch("nanochat.model_factory.get_tokenizer", return_value=_mock_tokenizer(256)),
        ):
            model, _, metadata = load_model_from_dir(
                phase="eval",
                device=torch.device("cpu"),
                config=CheckpointConfig(),
                model_tag="d2",
                step=10,
            )

        assert isinstance(model, GPT)
        assert metadata["step"] == 10


def test_load_model_from_dir_model_in_eval_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_checkpoint(tmpdir, step=5)

        with (
            patch("nanochat.model_factory.workspace.checkpoint_dir", return_value=tmpdir),
            patch("nanochat.model_factory.find_last_step", return_value=5),
            patch("nanochat.model_factory.get_tokenizer", return_value=_mock_tokenizer(256)),
        ):
            model, _, _ = load_model_from_dir(
                phase="eval",
                device=torch.device("cpu"),
                config=CheckpointConfig(),
                model_tag="d2",
                step=5,
            )

        assert not model.training


def test_load_optimizer_state_returns_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_checkpoint(tmpdir, step=20, with_optimizer=True)

        with (
            patch("nanochat.model_factory.workspace.checkpoint_dir", return_value=tmpdir),
            patch("nanochat.model_factory.find_largest_model", return_value="d2"),
            patch("nanochat.model_factory.find_last_step", return_value=20),
        ):
            result = load_optimizer_state(
                source="base",
                device=torch.device("cpu"),
                rank=0,
                config=CheckpointConfig(),
                model_tag="d2",
                step=20,
            )

        assert result is not None


def test_load_optimizer_state_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("nanochat.model_factory.workspace.checkpoint_dir", return_value=tmpdir),
            patch("nanochat.model_factory.find_largest_model", return_value="d2"),
            patch("nanochat.model_factory.find_last_step", return_value=99),
        ):
            result = load_optimizer_state(
                source="base",
                device=torch.device("cpu"),
                rank=0,
                config=CheckpointConfig(),
                model_tag="d2",
                step=99,
            )

        assert result is None
