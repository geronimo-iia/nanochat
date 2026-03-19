def chat_eval(
    config: object,
    source: str,
    task_name: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    num_samples: int = 1,
    top_k: int = 50,
    batch_size: int = 8,
    model_tag: str | None = None,
    step: int | None = None,
    max_problems: int | None = None,
) -> None:
    """Entry point for chat model evaluation."""
    import os

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    from nanochat.common import compute_cleanup
    from nanochat.config import Config
    from nanochat.evaluation.chat.loop import chat_eval as run_chat_eval

    assert isinstance(config, Config)
    try:
        run_chat_eval(
            config,
            source=source,
            task_name=task_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            top_k=top_k,
            batch_size=batch_size,
            model_tag=model_tag,
            step=step,
            max_problems=max_problems,
        )
    finally:
        compute_cleanup()
