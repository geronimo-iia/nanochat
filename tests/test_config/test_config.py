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
    assert cfg.backend == ""
    assert cfg.run == "unnamed"
    assert cfg.wandb == "local"
    assert cfg.wandb_project == "nanochat"
    assert cfg.model_tag is None


# ---------------------------------------------------------------------------
# CommonConfig.validate()
# ---------------------------------------------------------------------------


def test_validate_defaults_ok():
    CommonConfig().validate()


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_validate_valid_backends(backend):
    CommonConfig(backend=backend).validate()


@pytest.mark.parametrize("device_type", ["cuda", "mps", "cpu"])
def test_validate_valid_device_types(device_type):
    CommonConfig(device_type=device_type).validate()


def test_validate_invalid_backend_raises():
    with pytest.raises(ValueError, match="common.backend"):
        CommonConfig(backend="jax").validate()


def test_validate_invalid_device_type_raises():
    with pytest.raises(ValueError, match="common.device_type"):
        CommonConfig(device_type="gpu").validate()


def test_validate_mlx_with_device_type_raises():
    with pytest.raises(ValueError, match="MLX manages its own device"):
        CommonConfig(backend="mlx", device_type="mps").validate()


def test_validate_invalid_wandb_raises():
    with pytest.raises(ValueError, match="common.wandb"):
        CommonConfig(wandb="yes").validate()


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.depth == 20
    assert cfg.aspect_ratio == 64
    assert cfg.fp8 is False


# ---------------------------------------------------------------------------
# TrainingConfig.validate()
# ---------------------------------------------------------------------------


def test_training_validate_defaults_ok():
    TrainingConfig().validate()


@pytest.mark.parametrize("field", ["depth", "aspect_ratio", "head_dim", "max_seq_len"])
def test_training_validate_architecture_zero_raises(field):
    with pytest.raises(ValueError, match=f"training.{field}"):
        TrainingConfig(**{field: 0}).validate()


@pytest.mark.parametrize("pattern", ["L", "S", "SSSL", "LL"])
def test_training_validate_valid_window_patterns(pattern):
    TrainingConfig(window_pattern=pattern).validate()


@pytest.mark.parametrize("pattern", ["", "X", "LSX"])
def test_training_validate_invalid_window_pattern_raises(pattern):
    with pytest.raises(ValueError, match="window_pattern"):
        TrainingConfig(window_pattern=pattern).validate()


def test_training_validate_num_iterations_zero_raises():
    with pytest.raises(ValueError, match="training.num_iterations"):
        TrainingConfig(num_iterations=0).validate()


def test_training_validate_target_flops_zero_raises():
    with pytest.raises(ValueError, match="training.target_flops"):
        TrainingConfig(target_flops=0.0).validate()


def test_training_validate_device_batch_size_zero_raises():
    with pytest.raises(ValueError, match="training.device_batch_size"):
        TrainingConfig(device_batch_size=0).validate()


def test_training_validate_total_batch_size_zero_raises():
    with pytest.raises(ValueError, match="training.total_batch_size"):
        TrainingConfig(total_batch_size=0).validate()


def test_training_validate_total_batch_size_minus_one_ok():
    TrainingConfig(total_batch_size=-1).validate()


@pytest.mark.parametrize("field", ["embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"])
def test_training_validate_lr_zero_raises(field):
    with pytest.raises(ValueError, match=f"training.{field}"):
        TrainingConfig(**{field: 0.0}).validate()


def test_training_validate_negative_weight_decay_raises():
    with pytest.raises(ValueError, match="training.weight_decay"):
        TrainingConfig(weight_decay=-0.1).validate()


def test_training_validate_negative_warmup_steps_raises():
    with pytest.raises(ValueError, match="training.warmup_steps"):
        TrainingConfig(warmup_steps=-1).validate()


@pytest.mark.parametrize("val", [0.0, 1.0, 1.5])
def test_training_validate_warmdown_ratio_out_of_range_raises(val):
    with pytest.raises(ValueError, match="training.warmdown_ratio"):
        TrainingConfig(warmdown_ratio=val).validate()


@pytest.mark.parametrize("val", [0.0, 1.0])
def test_training_validate_final_lr_frac_out_of_range_raises(val):
    with pytest.raises(ValueError, match="training.final_lr_frac"):
        TrainingConfig(final_lr_frac=val).validate()


@pytest.mark.parametrize("field", ["eval_every", "core_metric_every", "sample_every"])
def test_training_validate_eval_interval_zero_raises(field):
    with pytest.raises(ValueError, match=f"training.{field}"):
        TrainingConfig(**{field: 0}).validate()


@pytest.mark.parametrize("field", ["eval_every", "core_metric_every", "sample_every"])
def test_training_validate_eval_interval_minus_one_ok(field):
    TrainingConfig(**{field: -1}).validate()


def test_training_validate_invalid_fp8_recipe_raises():
    with pytest.raises(ValueError, match="training.fp8_recipe"):
        TrainingConfig(fp8_recipe="bogus").validate()


def test_training_validate_compression_log_every_zero_raises():
    with pytest.raises(ValueError, match="training.compression_log_every"):
        TrainingConfig(compression_log_every=0).validate()


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
# EvaluationConfig.validate()
# ---------------------------------------------------------------------------


def test_evaluation_validate_defaults_ok():
    EvaluationConfig().validate()


@pytest.mark.parametrize("max_per_task", [-1, 1, 500])
def test_evaluation_validate_valid_max_per_task(max_per_task):
    EvaluationConfig(max_per_task=max_per_task).validate()


def test_evaluation_validate_invalid_max_per_task_raises():
    with pytest.raises(ValueError, match="evaluation.max_per_task"):
        EvaluationConfig(max_per_task=0).validate()


def test_evaluation_validate_invalid_device_batch_size_raises():
    with pytest.raises(ValueError, match="evaluation.device_batch_size"):
        EvaluationConfig(device_batch_size=0).validate()


def test_evaluation_validate_invalid_split_tokens_raises():
    with pytest.raises(ValueError, match="evaluation.split_tokens"):
        EvaluationConfig(split_tokens=0).validate()


def test_evaluation_validate_valid_step():
    EvaluationConfig(step=100).validate()


def test_evaluation_validate_negative_step_raises():
    with pytest.raises(ValueError, match="evaluation.step"):
        EvaluationConfig(step=-1).validate()


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
