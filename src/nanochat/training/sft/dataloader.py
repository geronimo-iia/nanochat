from __future__ import annotations

from typing import Generator

import torch

from nanochat.config import Config
from nanochat.training.sft.state import SFTState


def sft_data_generator_bos_bestfit(
    state: SFTState,
    config: Config,
    tokenizer: object,
    ddp_rank: int,
    ddp_world_size: int,
    device: torch.device,
    device_type: str,
    train_dataset: object,
    val_dataset: object,
    split: str,
    buffer_size: int = 100,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).

    Mutates state.last_step, state.approx_progress, and state.current_epoch directly,
    replacing the nonlocal variables used in the original train_sft.py.
    """
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)  # type: ignore[arg-type]
    assert dataset_size > 0
    row_capacity = config.sft.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()  # type: ignore[union-attr]

    conv_buffer: list[tuple[list[int], list[int]]] = []
    cursor = ddp_rank
    consumed = ddp_rank
    epoch = 1
    it = 0

    def refill_buffer() -> None:
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]  # type: ignore[index]
            ids, mask = tokenizer.render_conversation(conversation)  # type: ignore[union-attr]
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        row_lengths: list[int] = []

        for _ in range(config.sft.device_batch_size):
            row: list[int] = []
            mask_row: list[int] = []
            padded = False
            content_len = 0

            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break

            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        it += 1
        if 0 < config.sft.num_iterations <= it and split == "train":
            state.last_step = True

        if split == "train":
            state.current_epoch = epoch
            if config.sft.num_iterations > 0:
                state.approx_progress = it / config.sft.num_iterations
            else:
                state.approx_progress = consumed / dataset_size
            if consumed >= dataset_size:
                state.last_step = True

        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1 :] = -1

        yield inputs, targets
