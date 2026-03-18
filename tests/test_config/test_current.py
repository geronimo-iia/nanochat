"""Tests for nanochat.config.current and load_and_init."""

import pytest

from nanochat.config import ConfigLoader, load_and_init
from nanochat.config import current
from nanochat.config.common import CommonConfig
from nanochat.config.config import Config


@pytest.fixture(autouse=True)
def reset_current():
    current.reset()
    yield
    current.reset()


# ---------------------------------------------------------------------------
# current.py
# ---------------------------------------------------------------------------


def test_get_before_init_raises():
    with pytest.raises(RuntimeError, match="not initialized"):
        current.get()


def test_init_and_get():
    cfg = Config(common=CommonConfig(base_dir="/tmp/test"))
    current.init(cfg)
    assert current.get() is cfg


def test_reset_clears_state():
    current.init(Config())
    current.reset()
    with pytest.raises(RuntimeError):
        current.get()


def test_init_overwrites():
    cfg1 = Config(common=CommonConfig(run="first"))
    cfg2 = Config(common=CommonConfig(run="second"))
    current.init(cfg1)
    current.init(cfg2)
    assert current.get().common.run == "second"


# ---------------------------------------------------------------------------
# load_and_init
# ---------------------------------------------------------------------------


def test_load_and_init_registers_config(tmp_path):
    cfg = load_and_init(ConfigLoader(), ["--base-dir", str(tmp_path)])
    assert current.get() is cfg


def test_load_and_init_returns_config(tmp_path):
    cfg = load_and_init(ConfigLoader().add_training(), ["--base-dir", str(tmp_path), "--depth", "6"])
    assert cfg.training.depth == 6


def test_load_and_init_resolve_stays_pure(tmp_path):
    """resolve() alone must not touch current."""
    ConfigLoader().resolve(__import__("argparse").Namespace(base_dir=str(tmp_path), config=None))
    with pytest.raises(RuntimeError):
        current.get()
