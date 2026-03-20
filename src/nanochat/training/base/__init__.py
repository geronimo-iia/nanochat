def train_base(config: object) -> None:
    """
    Entry point for base model pretraining.

    The os.environ assignment and all torch-dependent imports are deferred
    to the function body so that PYTORCH_ALLOC_CONF is set before any torch
    module is loaded.
    """
    import os

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    from nanochat import workspace
    from nanochat.checkpoint import make_checkpoint_manager
    from nanochat.common import compute_cleanup
    from nanochat.config import Config
    from nanochat.training.base.loop import train_loop
    from nanochat.training.base.setup import setup

    assert isinstance(config, Config)
    s = None
    try:
        output_dirname = config.common.model_tag if config.common.model_tag else f"d{config.training.depth}"
        ckpt_dir = workspace.checkpoint_dir("base", output_dirname)
        checkpoint_manager = make_checkpoint_manager(ckpt_dir, config.checkpoint)
        s = setup(config, checkpoint_manager)
        train_loop(s, checkpoint_manager)
    finally:
        if s is not None:
            s.wandb_run.finish()
        compute_cleanup()
