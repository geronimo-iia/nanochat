"""Tests for ConfigLoader (CLI, TOML, overrides, autodiscovery)."""

import pytest

from nanochat.config import ConfigLoader

# ---------------------------------------------------------------------------
# CLI only (no TOML)
# ---------------------------------------------------------------------------


def test_parse_cli_only_common(tmp_path):
    cfg = ConfigLoader().parse(["--base-dir", str(tmp_path), "--run", "smoke", "--wandb", "disabled"])
    assert cfg.common.run == "smoke"
    assert cfg.common.wandb == "disabled"


def test_parse_cli_backend(tmp_path):
    cfg = ConfigLoader().parse(["--base-dir", str(tmp_path), "--backend", "mlx"])
    assert cfg.common.backend == "mlx"


def test_parse_cli_backend_torch(tmp_path):
    cfg = ConfigLoader().parse(["--base-dir", str(tmp_path), "--backend", "torch"])
    assert cfg.common.backend == "torch"


def test_parse_cli_invalid_backend_raises(tmp_path, capsys):
    with pytest.raises(SystemExit):
        ConfigLoader().parse(["--base-dir", str(tmp_path), "--backend", "jax"])


def test_parse_cli_only_training(tmp_path):
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path), "--depth", "6", "--num-iterations", "20"])
    assert cfg.training.depth == 6
    assert cfg.training.num_iterations == 20


def test_parse_cli_only_sft(tmp_path):
    cfg = ConfigLoader().add_sft().parse(["--base-dir", str(tmp_path), "--mmlu-epochs", "7", "--no-load-optimizer"])
    assert cfg.sft.mmlu_epochs == 7
    assert cfg.sft.load_optimizer is False


def test_parse_cli_only_rl(tmp_path):
    cfg = ConfigLoader().add_rl().parse(["--base-dir", str(tmp_path), "--num-epochs", "4", "--temperature", "0.8"])
    assert cfg.rl.num_epochs == 4
    assert cfg.rl.temperature == 0.8


def test_parse_cli_only_evaluation(tmp_path):
    cfg = ConfigLoader().add_evaluation().parse(["--base-dir", str(tmp_path), "--max-per-task", "50"])
    assert cfg.evaluation.max_per_task == 50


def test_parse_no_args_uses_defaults(tmp_path):
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path)])
    assert cfg.training.depth == 20
    assert cfg.common.run == "unnamed"


# ---------------------------------------------------------------------------
# TOML only (no CLI overrides)
# ---------------------------------------------------------------------------


def test_parse_toml_common(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[common]\nrun = "from-toml"\n', encoding="utf-8")
    cfg = ConfigLoader().parse(["--config", str(p)])
    assert cfg.common.run == "from-toml"


def test_parse_toml_training(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 12\n", encoding="utf-8")
    cfg = ConfigLoader().add_training().parse(["--config", str(p)])
    assert cfg.training.depth == 12


def test_parse_toml_sft(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[sft]\nmmlu_epochs = 9\n", encoding="utf-8")
    cfg = ConfigLoader().add_sft().parse(["--config", str(p)])
    assert cfg.sft.mmlu_epochs == 9


def test_parse_toml_rl(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[rl]\nnum_epochs = 5\n", encoding="utf-8")
    cfg = ConfigLoader().add_rl().parse(["--config", str(p)])
    assert cfg.rl.num_epochs == 5


def test_parse_toml_evaluation(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[evaluation]\nmax_per_task = 200\n", encoding="utf-8")
    cfg = ConfigLoader().add_evaluation().parse(["--config", str(p)])
    assert cfg.evaluation.max_per_task == 200


# ---------------------------------------------------------------------------
# CLI overrides TOML
# ---------------------------------------------------------------------------


def test_cli_overrides_toml_training(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 12\nnum_iterations = 1000\n", encoding="utf-8")
    cfg = ConfigLoader().add_training().parse(["--config", str(p), "--num-iterations", "500"])
    assert cfg.training.depth == 12
    assert cfg.training.num_iterations == 500


def test_cli_overrides_toml_common(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[common]\nrun = "toml-run"\n', encoding="utf-8")
    cfg = ConfigLoader().parse(["--config", str(p), "--run", "cli-run"])
    assert cfg.common.run == "cli-run"


def test_toml_beats_dataclass_defaults(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 99\n", encoding="utf-8")
    cfg = ConfigLoader().add_training().parse(["--config", str(p)])
    assert cfg.training.depth == 99


# ---------------------------------------------------------------------------
# base_dir autodiscovery
# ---------------------------------------------------------------------------


def test_base_dir_autodiscovery(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[common]\nrun = "discovered"\n[training]\ndepth = 3\n', encoding="utf-8")
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path)])
    assert cfg.common.run == "discovered"
    assert cfg.training.depth == 3


def test_explicit_config_overrides_base_dir(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.toml").write_text('[common]\nrun = "base"\n', encoding="utf-8")
    explicit = tmp_path / "explicit.toml"
    explicit.write_text('[common]\nrun = "explicit"\n', encoding="utf-8")
    cfg = ConfigLoader().parse(["--base-dir", str(base), "--config", str(explicit)])
    assert cfg.common.run == "explicit"


def test_base_dir_no_config_toml_uses_defaults(tmp_path):
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path)])
    assert cfg.training.depth == 20


def test_multiple_sections_raises():
    with pytest.raises(RuntimeError, match="one section"):
        ConfigLoader().add_training().add_sft()


def test_unregistered_section_in_toml_ignored(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 5\n", encoding="utf-8")
    cfg = ConfigLoader().parse(["--config", str(p)])
    assert cfg.training.depth == 20


# ---------------------------------------------------------------------------
# save then parse roundtrip
# ---------------------------------------------------------------------------


def test_save_then_parse_roundtrip(tmp_path):
    from nanochat.config import Config

    cfg = Config()
    cfg.common.run = "roundtrip"
    cfg.common.wandb = "disabled"
    cfg.training.depth = 4
    cfg.training.num_iterations = 300
    p = tmp_path / "config.toml"
    cfg.save(p)

    cfg2 = ConfigLoader().add_training().parse(["--config", str(p)])
    assert cfg2.common.run == "roundtrip"
    assert cfg2.common.wandb == "disabled"
    assert cfg2.training.depth == 4
    assert cfg2.training.num_iterations == 300
