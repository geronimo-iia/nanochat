"""Shared config fields common to all training modes (device, run name, wandb)."""

import argparse
from dataclasses import dataclass

_VALID_BACKENDS = {"", "torch", "mlx"}
_VALID_DEVICE_TYPES = {"", "cuda", "mps", "cpu"}
_VALID_WANDB_MODES = {"online", "local", "disabled"}


@dataclass
class CommonConfig:
    base_dir: str | None = None
    device_type: str = ""
    backend: str = ""  # torch | mlx (empty = autodetect)
    run: str = "unnamed"
    wandb: str = "local"  # online | local | disabled
    wandb_project: str = "nanochat"
    model_tag: str | None = None

    def validate(self) -> None:
        """Raise ValueError if any field value is invalid or contradictory."""
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(f"common.backend must be one of {sorted(_VALID_BACKENDS)}, got {self.backend!r}")
        if self.device_type not in _VALID_DEVICE_TYPES:
            raise ValueError(
                f"common.device_type must be one of {sorted(_VALID_DEVICE_TYPES)}, got {self.device_type!r}"
            )
        if self.backend == "mlx" and self.device_type != "":
            raise ValueError(
                f"common.device_type must be empty when backend='mlx' (MLX manages its own device), got {self.device_type!r}"
            )
        if self.wandb not in _VALID_WANDB_MODES:
            raise ValueError(f"common.wandb must be one of {sorted(_VALID_WANDB_MODES)}, got {self.wandb!r}")

    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            type=str,
            default=argparse.SUPPRESS,
            help="path to TOML config file (CLI args override file values)",
        )
        parser.add_argument(
            "--base-dir", type=str, default=argparse.SUPPRESS, help="override NANOCHAT_BASE_DIR env var"
        )
        parser.add_argument(
            "--device-type", type=str, default=argparse.SUPPRESS, help="cuda|cpu|mps (empty = autodetect)"
        )
        parser.add_argument(
            "--backend",
            type=str,
            default=argparse.SUPPRESS,
            choices=["torch", "mlx"],
            help="training backend: torch | mlx (empty = autodetect)",
        )
        parser.add_argument("--run", type=str, default=argparse.SUPPRESS, help="wandb run name")
        parser.add_argument(
            "--wandb",
            type=str,
            default=argparse.SUPPRESS,
            choices=["online", "local", "disabled"],
            help="wandb mode: online | local | disabled",
        )
        parser.add_argument("--wandb-project", type=str, default=argparse.SUPPRESS, help="wandb project name")
        parser.add_argument("--model-tag", type=str, default=argparse.SUPPRESS)

    @classmethod
    def generate_default(cls) -> str:
        return (
            'base_dir = ""              # override NANOCHAT_BASE_DIR env var (empty = use env var)\n'
            'device_type = ""           # cuda | cpu | mps (empty = autodetect)\n'
            'backend = ""               # torch | mlx (empty = autodetect)\n'
            'run = "unnamed"            # wandb run name\n'
            'wandb = "local"            # online | local | disabled\n'
            'wandb_project = "nanochat"\n'
            '# model_tag = ""           # empty = auto\n'
        )
