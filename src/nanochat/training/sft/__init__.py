def train_sft(config: object) -> None:
    """Entry point for supervised fine-tuning.

    The os.environ assignment and all torch-dependent imports are deferred
    to the function body so that PYTORCH_ALLOC_CONF is set before any torch
    module is loaded.
    """
    import os

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    from nanochat.common import compute_cleanup
    from nanochat.config import Config
    from nanochat.training.sft.loop import sft_train_loop
    from nanochat.training.sft.setup import setup

    assert isinstance(config, Config)
    s = None
    try:
        s = setup(config)
        sft_train_loop(s)
    finally:
        if s is not None:
            s.wandb_run.finish()
        compute_cleanup()
