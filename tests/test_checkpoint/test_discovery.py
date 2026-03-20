"""Tests for checkpoint discovery utilities."""

import os
import tempfile

import pytest

from nanochat.checkpoint.discovery import find_largest_model, find_last_step


def test_find_last_step():
    with tempfile.TemporaryDirectory() as tmpdir:
        for step in (10, 200, 30):
            open(os.path.join(tmpdir, f"model_{step:06d}.pt"), "w").close()
        assert find_last_step(tmpdir) == 200


def test_find_last_step_empty_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            find_last_step(tmpdir)


def test_find_largest_model_by_depth():
    with tempfile.TemporaryDirectory() as tmpdir:
        for tag in ("d12", "d6", "d20"):
            os.makedirs(os.path.join(tmpdir, tag))
        assert find_largest_model(tmpdir) == "d20"


def test_find_largest_model_fallback_to_mtime():
    with tempfile.TemporaryDirectory() as tmpdir:
        for tag in ("alpha", "beta"):
            os.makedirs(os.path.join(tmpdir, tag))
        # touch beta to make it newer
        os.utime(os.path.join(tmpdir, "beta"), None)
        assert find_largest_model(tmpdir) == "beta"


def test_find_largest_model_empty_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            find_largest_model(tmpdir)
