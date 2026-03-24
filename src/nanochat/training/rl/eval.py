from collections.abc import Generator

from nanochat.config import Config
from nanochat.evaluation.engine import Engine
from nanochat.tasks.gsm8k import GSM8K


def run_gsm8k_eval_mlx(
    task: GSM8K,
    tokenizer: object,
    engine: object,   # MLXEngine — same interface as Engine
    config: Config,
    max_examples: int | None = None,
    num_samples: int = 1,
    max_completion_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 50,
) -> Generator[dict[str, object], None, None]:
    """Single-device GSM8K eval using MLXEngine (no DDP, no dist.all_reduce).

    Identical structure to run_gsm8k_eval but without rank sharding.
    MLXEngine.generate_batch() must call mx.eval internally — see mlx_engine.py.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(max_examples):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        generated_token_sequences, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        yield {"idx": idx, "outcomes": outcomes}


def run_gsm8k_eval(
    task: GSM8K,
    tokenizer: object,
    engine: Engine,
    config: Config,
    ddp_rank: int,
    ddp_world_size: int,
    max_examples: int | None = None,
    num_samples: int = 1,
    max_completion_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 50,
) -> Generator[dict[str, object], None, None]:
    """Evaluates GSM8K and yields records one by one.

    All ranks cooperate; reduction across ranks is the caller's responsibility.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= config.rl.device_batch_size
        generated_token_sequences, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        yield {"idx": idx, "outcomes": outcomes}
