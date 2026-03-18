"""Tests for nanochat.workspace."""

import os

import pytest

from nanochat.config import current
from nanochat.config.common import CommonConfig
from nanochat.config.config import Config
from nanochat import workspace


@pytest.fixture(autouse=True)
def reset_all(tmp_path):
    current.reset()
    workspace.reset()
    config = Config(common=CommonConfig(base_dir=str(tmp_path)))
    current.init(config)
    workspace.init()
    yield
    workspace.reset()
    current.reset()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_base_dir_before_init_raises():
    workspace.reset()
    with pytest.raises(RuntimeError, match="not initialized"):
        workspace.base_dir()


def test_init_requires_config():
    workspace.reset()
    current.reset()
    with pytest.raises(RuntimeError):
        workspace.init()


def test_base_dir_returns_tmp_path(tmp_path):
    assert workspace.base_dir() == str(tmp_path)


def test_reset_clears_state():
    workspace.reset()
    with pytest.raises(RuntimeError):
        workspace.base_dir()


# ---------------------------------------------------------------------------
# Path functions
# ---------------------------------------------------------------------------


def test_data_dir(tmp_path):
    assert workspace.data_dir() == str(tmp_path / "data" / "climbmix")
    assert os.path.isdir(workspace.data_dir())


def test_legacy_data_dir(tmp_path):
    result = workspace.legacy_data_dir()
    assert result == str(tmp_path / "data" / "fineweb")
    assert not os.path.isdir(result)  # not auto-created


def test_tokenizer_dir(tmp_path):
    assert workspace.tokenizer_dir() == str(tmp_path / "tokenizer")
    assert os.path.isdir(workspace.tokenizer_dir())


def test_checkpoint_dir_with_tag(tmp_path):
    assert workspace.checkpoint_dir("base", "d12") == str(tmp_path / "checkpoints" / "base" / "d12")
    assert workspace.checkpoint_dir("sft", "d20") == str(tmp_path / "checkpoints" / "sft" / "d20")
    assert workspace.checkpoint_dir("rl", "d24") == str(tmp_path / "checkpoints" / "rl" / "d24")
    assert os.path.isdir(workspace.checkpoint_dir("base", "d12"))


def test_checkpoint_dir_without_tag(tmp_path):
    assert workspace.checkpoint_dir("base") == str(tmp_path / "checkpoints" / "base")


def test_checkpoint_dir_invalid_phase():
    with pytest.raises(AssertionError, match="Unknown phase"):
        workspace.checkpoint_dir("pretrain")


def test_eval_dirs(tmp_path):
    assert workspace.eval_tasks_dir() == str(tmp_path / "data" / "eval_tasks")
    assert workspace.eval_results_dir() == str(tmp_path / "eval")


def test_identity_data_path(tmp_path):
    assert workspace.identity_data_path() == str(tmp_path / "identity.jsonl")


def test_report_dir(tmp_path):
    assert workspace.report_dir() == str(tmp_path / "report")
    assert os.path.isdir(workspace.report_dir())
