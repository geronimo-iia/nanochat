"""
MLX version of rl/rollout.py — generates rollout batches using MLXEngine.

Key differences from the PyTorch version:
- Uses MLXEngine (no torch.no_grad, no torch.Tensor)
- Pads all sequences to a fixed max_len for static mx.compile shapes
- Advantage normalization done in numpy before mx.array conversion to avoid
  tracing it into the compiled REINFORCE graph
- No DDP: examples_per_rank == examples_per_step (single device)
"""

from __future__ import annotations

import itertools
from collections.abc import Generator

import mlx.core as mx
import numpy as np

from nanochat.config import Config
from nanochat.evaluation.mlx_engine import MLXEngine
from nanochat.tasks.gsm8k import GSM8K
from nanochat.training.rl.state import RLState


def get_batch_mlx(
    state: RLState,
    config: Config,
    train_task: GSM8K,
    mlx_engine: MLXEngine,
    tokenizer: object,
    max_len: int,
) -> Generator[
    tuple[list[list[int]], mx.array, mx.array, mx.array, mx.array], None, None
]:
    """MLX rollout generator — mirrors rollout.get_batch but yields mx.arrays.

    Pads all sequences to `max_len` (= max_new_tokens + max_prefix_len) so that
    mx.compile sees a fixed (B, max_len-1) shape every call.

    Each yield: (sequences_all, inputs_all, targets_all, rewards_all, advantages_all)
    - inputs_all:     (B, max_len-1) int32
    - targets_all:    (B, max_len-1) int32, -1 at masked/padding positions
    - rewards_all:    (B,) float32
    - advantages_all: (B,) float32, pre-normalized in numpy (outside compiled graph)
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    for example_idx in itertools.cycle(range(len(train_task))):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        generated_token_sequences = []
        masks_list = []
        num_sampling_steps = config.rl.num_samples // config.rl.device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((state.step, example_idx, sampling_step)) & 0x7FFFFFFF
            seqs_batch, masks_batch = mlx_engine.generate_batch(
                tokens,
                num_samples=config.rl.device_batch_size,
                max_tokens=config.rl.max_new_tokens,
                temperature=config.rl.temperature,
                top_k=config.rl.top_k,
                seed=seed,
            )
            generated_token_sequences.extend(seqs_batch)
            masks_list.extend(masks_batch)

        rewards_np = []
        for sample_tokens in generated_token_sequences:
            generated_text = tokenizer.decode(sample_tokens[prefix_length:])
            rewards_np.append(train_task.reward(conversation, generated_text))
        rewards_np = np.array(rewards_np, dtype=np.float32)

        # Advantage normalization happens in numpy — outside the compiled graph
        advantages_np = (rewards_np - rewards_np.mean()).astype(np.float32)

        # Pad all sequences to fixed max_len for static mx.compile shapes
        padded_seqs = [
            seq + [assistant_end] * (max_len - len(seq))
            for seq in generated_token_sequences
        ]
        padded_masks = [
            mask + [0] * (max_len - len(mask))
            for mask in masks_list
        ]

        ids_np = np.array(padded_seqs, dtype=np.int32)       # (B, max_len)
        mask_np = np.array(padded_masks, dtype=np.int32)     # (B, max_len)
        inputs_np = ids_np[:, :-1]                           # (B, max_len-1)
        targets_np = ids_np[:, 1:].copy()                    # (B, max_len-1)
        targets_np[mask_np[:, 1:] == 0] = -1                 # mask non-assistant tokens

        yield (
            generated_token_sequences,
            mx.array(inputs_np),
            mx.array(targets_np),
            mx.array(rewards_np),
            mx.array(advantages_np),
        )
