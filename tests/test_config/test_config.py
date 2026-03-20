"""Tests for Config dataclasses and Config load/save."""

import tomllib

import pytest

from nanochat.config import (
    CommonConfig,
    Config,
    EvaluationConfig,
    RLConfig,
    SFTConfig,
    TrainingConfig,
)
from nanochat.config.checkpoint import CheckpointConfig

# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_common_config_defaults():
    cfg = CommonConfig()
    assert cfg.base_dir is None
    assert cfg.device_type == ""
    assert cfg.run == "unnamed"
    assert cfg.wandb == "local"
    assert cfg.wandb_project == "nanochat"
    assert cfg.model_tag is None


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.depth == 20
    assert cfg.aspect_ratio == 64
    assert cfg.fp8 is False


def test_sft_config_defaults():
    cfg = SFTConfig()
    assert cfg.load_optimizer is True
    assert cfg.num_iterations == -1
    assert cfg.source_step is None


def test_rl_config_defaults():
    cfg = RLConfig()
    assert cfg.num_epochs == 1
    assert cfg.temperature == 1.0
    assert cfg.source_step is None


def test_evaluation_config_defaults():
    cfg = EvaluationConfig()
    assert cfg.modes == "core,bpb,sample"
    assert cfg.max_per_task == -1


def test_evaluation_config_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid eval modes"):
        EvaluationConfig(modes="core,bogus")


def test_evaluation_config_valid_subset():
    cfg = EvaluationConfig(modes="core,bpb")
    assert cfg.modes == "core,bpb"


# ---------------------------------------------------------------------------
# generate_default — each section produces valid TOML
# ---------------------------------------------------------------------------


def test_common_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[common]\n" + CommonConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert "common" in data


def test_training_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "t.toml"
    p.write_text("[training]\n" + TrainingConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["training"]["depth"] == 20


def test_sft_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text("[sft]\n" + SFTConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["sft"]["mmlu_epochs"] == 3


def test_rl_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text("[rl]\n" + RLConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["rl"]["num_epochs"] == 1


def test_evaluation_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "e.toml"
    p.write_text("[evaluation]\n" + EvaluationConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["evaluation"]["device_batch_size"] == 32


def test_config_generate_default_all_sections():
    text = Config.generate_default()
    data = tomllib.loads(text)
    for section in ("common", "training", "sft", "rl", "evaluation", "checkpoint"):
        assert section in data, f"missing [{section}]"
    assert data["training"]["depth"] == 20
    assert data["sft"]["mmlu_epochs"] == 3
    assert data["rl"]["num_epochs"] == 1
    assert data["evaluation"]["device_batch_size"] == 32
    assert data["checkpoint"]["format"] == "torch"


# ---------------------------------------------------------------------------
# CheckpointConfig
# ---------------------------------------------------------------------------


def test_checkpoint_config_defaults():
    cfg = CheckpointConfig()
    assert cfg.format == "torch"
    assert cfg.save_every == -1
    assert cfg.resume_from_step == -1
    assert cfg.keep_last_n == -1


def test_checkpoint_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[checkpoint]\n" + CheckpointConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["checkpoint"]["format"] == "torch"


# ---------------------------------------------------------------------------
# Config.load / Config.save
# ---------------------------------------------------------------------------


def test_config_load(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[common]\nrun = "loaded"\n[training]\ndepth = 7\n', encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.common.run == "loaded"
    assert cfg.training.depth == 7
    assert cfg.sft.mmlu_epochs == 3  # default intact


def test_config_load_unknown_section_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[bogus]\nfoo = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        Config.load(p)


def test_save_roundtrip(tmp_path):
    cfg = Config()
    cfg.common.run = "roundtrip"
    cfg.training.depth = 8
    cfg.sft.mmlu_epochs = 5
    cfg.rl.num_epochs = 3
    cfg.evaluation.max_per_task = 100
    p = tmp_path / "config.toml"
    cfg.save(p)

    with open(p, "rb") as f:
        data = tomllib.load(f)

    assert data["common"]["run"] == "roundtrip"
    assert data["training"]["depth"] == 8
    assert data["sft"]["mmlu_epochs"] == 5
    assert data["rl"]["num_epochs"] == 3
    assert data["evaluation"]["max_per_task"] == 100


def test_save_omits_none_fields(tmp_path):
    cfg = Config()
    assert cfg.common.model_tag is None
    p = tmp_path / "config.toml"
    cfg.save(p)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    assert "model_tag" not in data["common"]
