"""Model construction and loading utilities shared by training, evaluation, and chat."""

from typing import cast

import torch

from nanochat import workspace
from nanochat.checkpoint import make_checkpoint_manager
from nanochat.checkpoint.compat import patch_missing_config_keys, patch_missing_keys
from nanochat.checkpoint.discovery import find_largest_model, find_last_step
from nanochat.checkpoint.logger import RankZeroLogger
from nanochat.config.checkpoint import CheckpointConfig
from nanochat.models.config import GPTConfig
from nanochat.models.gpt import GPT
from nanochat.tokenizer import get_tokenizer

_logger = RankZeroLogger(__name__)


def load_model_from_dir(
    phase: str,
    device: torch.device,
    config: CheckpointConfig,
    model_tag: str | None = None,
    step: int | None = None,
) -> tuple[GPT, object, dict[str, object]]:
    """Load a model from a checkpoint directory. Returns (model, tokenizer, metadata)."""
    assert phase in {"base", "train", "eval", "sft"}, f"Invalid phase: {phase}"

    if model_tag is None:
        phase_dir = workspace.checkpoint_dir(phase)
        model_tag = find_largest_model(phase_dir)
        _logger.info(f"No model tag provided, guessing model tag: {model_tag}")
    ckpt_dir = workspace.checkpoint_dir(phase, model_tag)
    if step is None:
        step = find_last_step(ckpt_dir)
    _logger.info(f"Loading model from {ckpt_dir} with step {step}")

    manager = make_checkpoint_manager(ckpt_dir, config, _logger)
    ckpt = manager.load(step, device, load_optimizer=False)
    model_data = ckpt.model_state
    metadata = ckpt.metadata

    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = cast(dict[str, object], metadata.model_config)
    patch_missing_config_keys(model_config_kwargs, _logger.info)
    _logger.info(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    patch_missing_keys(model_data, model_config, _logger.info)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval() if phase == "eval" else model.train()
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    )
    return model, tokenizer, metadata.to_dict()


def load_optimizer_state(
    source: str,
    device: torch.device,
    rank: int,
    config: CheckpointConfig,
    model_tag: str | None = None,
    step: int | None = None,
) -> dict[str, object] | None:
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    if model_tag is None:
        phase_dir = workspace.checkpoint_dir(source)
        model_tag = find_largest_model(phase_dir)
    ckpt_dir = workspace.checkpoint_dir(source, model_tag)
    if step is None:
        step = find_last_step(ckpt_dir)
    manager = make_checkpoint_manager(ckpt_dir, config, _logger)
    if not manager.exists(step):
        _logger.info(f"Optimizer checkpoint not found at step {step} in {ckpt_dir}")
        return None
    _logger.info(f"Loading optimizer state from {ckpt_dir} step {step}")
    return manager.load(step, device, load_optimizer=True, rank=rank).optimizer_state
