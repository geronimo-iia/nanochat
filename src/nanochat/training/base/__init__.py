def train_base(config: object) -> None:
    """
    Entry point for base model pretraining.

    The os.environ assignment and all torch-dependent imports are deferred
    to the function body so that PYTORCH_ALLOC_CONF is set before any torch
    module is loaded.
    """
    import os

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    from nanochat.common import compute_cleanup
    from nanochat.config import Config
    from nanochat.training.base.loop import train_loop
    from nanochat.training.base.setup import setup

    assert isinstance(config, Config)
    s = None
    try:
        s = setup(config)
        train_loop(s)
    finally:
        if s is not None:
            s.wandb_run.finish()
        compute_cleanup()
