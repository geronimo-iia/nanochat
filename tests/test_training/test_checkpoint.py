"""Test checkpoint save/load."""

import os
import tempfile
from typing import Self, cast

import torch

from nanochat.checkpoint.factory import make_checkpoint_manager
from nanochat.checkpoint.protocol import CheckpointMetadata, LoopState
from nanochat.config.checkpoint import CheckpointConfig
from nanochat.models import GPT, GPTConfig
from nanochat.models.gpt import Block


class _State:
    def __init__(self, step: int) -> None:
        self.step = step

    @classmethod
    def fresh(cls) -> Self:
        return cls(0)

    def to_metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            step=self.step,
            model_config={},
            user_config={"test": "value", "step": self.step},
            loop_state=LoopState(min_val_bpb=0.0, smooth_train_loss=0.0, total_training_time=0.0),
        )

    @classmethod
    def from_metadata(cls, meta: CheckpointMetadata) -> Self:
        return cls(meta.step)


def _config() -> CheckpointConfig:
    return CheckpointConfig()


def test_checkpoint_save_load():
    """Test checkpoint can be saved and loaded."""
    gpt_config = GPTConfig(sequence_len=128, vocab_size=500, n_layer=2, n_embd=128, n_head=2, n_kv_head=2)
    model = GPT(gpt_config)
    model.init_weights()
    optimizer = model.setup_optimizer(
        matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.008, scalar_lr=0.5, weight_decay=0.0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = make_checkpoint_manager(tmpdir, _config())
        state = _State(100)
        manager.save(state, model.state_dict(), optimizer.state_dict(), rank=0)

        assert os.path.exists(os.path.join(tmpdir, "model_000100.pt"))
        assert os.path.exists(os.path.join(tmpdir, "optim_000100_rank0.pt"))
        assert os.path.exists(os.path.join(tmpdir, "meta_000100.json"))

        ckpt = manager.load(step=100, device=torch.device("cpu"), load_optimizer=True, rank=0)

        assert ckpt.model_state is not None
        assert ckpt.optimizer_state is not None
        assert ckpt.metadata.user_config["test"] == "value"
        assert ckpt.metadata.step == 100


def test_checkpoint_model_restore():
    """Test model weights are correctly restored from checkpoint."""
    gpt_config = GPTConfig(sequence_len=64, vocab_size=200, n_layer=2, n_embd=64, n_head=2, n_kv_head=2)
    model1 = GPT(gpt_config)
    model1.init_weights()
    initial_weight = cast(Block, cast(torch.nn.ModuleList, model1.transformer.h)[0]).attn.c_q.weight.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = make_checkpoint_manager(tmpdir, _config())
        manager.save(_State(50), model1.state_dict(), None, rank=0)

        model2 = GPT(gpt_config)
        model2.init_weights()

        ckpt = manager.load(step=50, device=torch.device("cpu"), load_optimizer=False)
        model2.load_state_dict(ckpt.model_state)

        restored_weight = cast(Block, cast(torch.nn.ModuleList, model2.transformer.h)[0]).attn.c_q.weight
        assert torch.allclose(initial_weight, restored_weight)
