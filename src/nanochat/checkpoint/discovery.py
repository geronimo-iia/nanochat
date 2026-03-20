"""Checkpoint directory discovery utilities."""

import glob
import os
import re


def find_largest_model(checkpoints_dir: str) -> str:
    """Return the tag of the largest model in checkpoints_dir, guessed by depth."""
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            candidates.append((int(match.group(1)), model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir: str) -> int:
    """Return the highest step number found in checkpoint_dir."""
    checkpoint_files = [
        *glob.glob(os.path.join(checkpoint_dir, "model_*.pt")),
        *glob.glob(os.path.join(checkpoint_dir, "model_*.safetensors")),
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return int(max(os.path.basename(f).split("_")[1].split(".")[0] for f in checkpoint_files))
