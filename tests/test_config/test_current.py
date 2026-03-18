"""Tests for nanochat.config.current."""

import pytest

from nanochat.config import ConfigLoader
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
# parse + init
# ---------------------------------------------------------------------------


def test_parse_then_init_registers_config(tmp_path):
    cfg = ConfigLoader().parse(["--base-dir", str(tmp_path)])
    current.init(cfg)
    assert current.get() is cfg


def test_parse_then_init_config_values(tmp_path):
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path), "--depth", "6"])
    current.init(cfg)
    assert current.get().training.depth == 6


def test_resolve_stays_pure(tmp_path):
    """resolve() alone must not touch current."""
    import argparse

    ConfigLoader().resolve(argparse.Namespace(base_dir=str(tmp_path), config=None))
    with pytest.raises(RuntimeError):
        current.get()
