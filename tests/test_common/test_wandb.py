"""Tests for LocalWandb offline logger and init_wandb helper."""

import json

import pytest

from nanochat.common.wandb import DummyWandb, LocalWandb, init_wandb
from nanochat.config import Config, init_config, reset_config
from nanochat.config.common import CommonConfig


@pytest.fixture(autouse=True)
def reset():
    reset_config()
    yield
    reset_config()


def _init(tmp_path, *, run="my-run", wandb="disabled"):
    init_config(Config(common=CommonConfig(run=run, wandb=wandb, base_dir=str(tmp_path))))


def test_creates_output_dir(tmp_path):
    run_dir = tmp_path / "runs" / "nanochat" / "my-run"
    assert not run_dir.exists()
    w = LocalWandb(str(tmp_path), "my-run")
    w.finish()
    assert run_dir.exists()


def test_log_writes_jsonl(tmp_path):
    w = LocalWandb(str(tmp_path), "test-run")
    w.log({"loss": 1.5}, step=0)
    w.log({"loss": 1.2}, step=1)
    w.finish()

    lines = (tmp_path / "runs" / "nanochat" / "test-run" / "wandb.jsonl").read_text().splitlines()
    assert len(lines) == 2
    entry0 = json.loads(lines[0])
    assert "timestamp" in entry0
    assert entry0["step"] == 0
    assert entry0["data"] == {"loss": 1.5}
    entry1 = json.loads(lines[1])
    assert entry1["step"] == 1
    assert entry1["data"] == {"loss": 1.2}


def test_log_step_none_when_omitted(tmp_path):
    w = LocalWandb(str(tmp_path), "test-run")
    w.log({"loss": 1.0})
    w.finish()

    entry = json.loads((tmp_path / "runs" / "nanochat" / "test-run" / "wandb.jsonl").read_text())
    assert entry["step"] is None


def test_finish_closes_file(tmp_path):
    w = LocalWandb(str(tmp_path), "test-run")
    w.finish()
    assert w._f.closed


# ---------------------------------------------------------------------------
# init_wandb
# ---------------------------------------------------------------------------


def test_init_wandb_disabled(tmp_path):
    _init(tmp_path, wandb="disabled")
    w = init_wandb({}, master_process=True)
    assert isinstance(w, DummyWandb)


def test_init_wandb_local(tmp_path):
    _init(tmp_path, wandb="local")
    w = init_wandb({}, master_process=True)
    assert isinstance(w, LocalWandb)
    w.finish()


def test_init_wandb_non_master_always_dummy(tmp_path):
    """Non-master ranks always get DummyWandb regardless of wandb mode."""
    for mode in ("local", "disabled", "online"):
        _init(tmp_path, wandb=mode)
        w = init_wandb({}, master_process=False)
        assert isinstance(w, DummyWandb), f"expected DummyWandb for mode={mode}, non-master"


def test_init_wandb_legacy_dummy_run(tmp_path):
    """run='dummy' magic maps to DummyWandb even if wandb='local'."""
    _init(tmp_path, run="dummy", wandb="local")
    w = init_wandb({}, master_process=True)
    assert isinstance(w, DummyWandb)


def test_init_wandb_wandb_mode_env(tmp_path, monkeypatch):
    """WANDB_MODE=disabled env var maps to DummyWandb."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    _init(tmp_path, wandb="local")
    w = init_wandb({}, master_process=True)
    assert isinstance(w, DummyWandb)
