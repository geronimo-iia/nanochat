"""Module-level config store. Call init() once at startup; use get() everywhere else."""

from __future__ import annotations

from nanochat.config.config import Config

_config: Config | None = None


def init(config: Config) -> None:
    global _config
    _config = config


def get() -> Config:
    if _config is None:
        raise RuntimeError("Config not initialized — call current.init() first")
    return _config


def reset() -> None:
    """For testing."""
    global _config
    _config = None
