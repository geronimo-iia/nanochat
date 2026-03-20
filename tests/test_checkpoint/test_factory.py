"""Tests for make_checkpoint_manager factory."""

import pytest

from nanochat.checkpoint.factory import make_checkpoint_manager
from nanochat.checkpoint.torch_manager import TorchCheckpointManager
from nanochat.config.checkpoint import CheckpointConfig


def test_torch_format_returns_torch_manager(tmp_path):
    mgr = make_checkpoint_manager(str(tmp_path), CheckpointConfig(format="torch"))
    assert isinstance(mgr, TorchCheckpointManager)


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        make_checkpoint_manager(str(tmp_path), CheckpointConfig(format="unknown"))
