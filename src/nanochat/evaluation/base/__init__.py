def base_eval(config: object) -> None:
    """Entry point for base model evaluation."""
    import os

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    from nanochat.common import compute_cleanup
    from nanochat.config import Config
    from nanochat.evaluation.base.loop import run_base_eval

    assert isinstance(config, Config)
    try:
        run_base_eval(config)
    finally:
        compute_cleanup()
