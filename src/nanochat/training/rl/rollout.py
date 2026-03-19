import itertools
from collections.abc import Generator

import torch

from nanochat.config import Config
from nanochat.evaluation.engine import Engine
from nanochat.tasks.gsm8k import GSM8K
from nanochat.training.rl.state import RLState


@torch.no_grad()
def get_batch(
    state: RLState,
    config: Config,
    train_task: GSM8K,
    engine: Engine,
    model: object,
    tokenizer: object,
    device: torch.device,
    ddp_rank: int,
    ddp_world_size: int,
) -> Generator[tuple[list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """Yields batches of rollouts for RL training.

    Each yield: (sequences_all, inputs_all, targets_all, rewards_all, advantages_all)
    state.step is read each iteration for seed generation — matches original closure behaviour.
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = config.rl.num_samples // config.rl.device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((state.step, example_idx, sampling_step)) & 0x7FFFFFFF
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=config.rl.device_batch_size,
                max_tokens=config.rl.max_new_tokens,
                temperature=config.rl.temperature,
                top_k=config.rl.top_k,
                seed=seed,
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_seqs = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]

        ids = torch.tensor(padded_seqs, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1

        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
        advantages = rewards_t - rewards_t.mean()

        yield generated_token_sequences, inputs, targets, rewards_t, advantages
