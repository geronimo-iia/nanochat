"""Backward compatibility patches for old checkpoints."""

import torch

from nanochat.models.config import GPTConfig


def patch_missing_config_keys(model_config_kwargs: dict[str, object], logger_info: object = None) -> None:
    """Add default values for new config keys missing in old checkpoints."""
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        if callable(logger_info):
            logger_info("Patching missing window_pattern in model config to 'L'")


def patch_missing_keys(model_data: dict[str, object], model_config: GPTConfig, logger_info: object = None) -> None:
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        if callable(logger_info):
            logger_info("Patching missing resid_lambdas in model data to 1.0")
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        if callable(logger_info):
            logger_info("Patching missing x0_lambdas in model data to 0.0")
