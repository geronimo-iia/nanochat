"""
Module-level workspace store.

Owns the directory structure and path functions for the nanochat base directory.
Call init() once at startup (after config.current.init()); use path functions everywhere else.
Replaces common/paths.py — same layout, no base_dir parameter threading.
"""

from __future__ import annotations

import os

from nanochat.config import current

_base_dir: str | None = None


def init() -> None:
    """Read base_dir from config and initialise the workspace root."""
    global _base_dir
    _base_dir = current.get().common.base_dir
    os.makedirs(_base_dir, exist_ok=True)


def base_dir() -> str:
    if _base_dir is None:
        raise RuntimeError("Workspace not initialized — call workspace.init() first")
    return _base_dir


def reset() -> None:
    """For testing."""
    global _base_dir
    _base_dir = None


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _dir(*parts: str) -> str:
    path = os.path.join(base_dir(), *parts)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Path functions
# ---------------------------------------------------------------------------


def data_dir() -> str:
    return _dir("data", "climbmix")


def legacy_data_dir() -> str:
    return os.path.join(_dir("data"), "fineweb")


def eval_tasks_dir() -> str:
    return _dir("data", "eval_tasks")


def tokenizer_dir() -> str:
    return _dir("tokenizer")


def checkpoint_dir(phase: str, model_tag: str | None = None) -> str:
    assert phase in ("base", "sft", "rl"), f"Unknown phase: {phase}"
    if model_tag is not None:
        return _dir("checkpoints", phase, model_tag)
    return _dir("checkpoints", phase)


def eval_results_dir() -> str:
    return _dir("eval")


def identity_data_path() -> str:
    return os.path.join(base_dir(), "identity.jsonl")


def report_dir() -> str:
    return _dir("report")
