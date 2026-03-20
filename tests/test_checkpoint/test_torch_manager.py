"""Tests for TorchCheckpointManager."""

import os
import tempfile
from typing import Self, cast

import torch

from nanochat.checkpoint.logger import SilentLogger
from nanochat.checkpoint.protocol import CheckpointMetadata, LoopState
from nanochat.checkpoint.torch_manager import TorchCheckpointManager
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
            user_config={"step": self.step},
            loop_state=LoopState(min_val_bpb=0.0, smooth_train_loss=0.0, total_training_time=0.0),
        )

    @classmethod
    def from_metadata(cls, meta: CheckpointMetadata) -> Self:
        return cls(meta.step)


def _manager(tmpdir: str, keep_last_n: int = -1) -> TorchCheckpointManager:
    return TorchCheckpointManager(tmpdir, CheckpointConfig(keep_last_n=keep_last_n), SilentLogger())


def _tiny_gpt() -> tuple[GPT, GPTConfig]:
    cfg = GPTConfig(sequence_len=64, vocab_size=200, n_layer=2, n_embd=64, n_head=2, n_kv_head=2)
    model = GPT(cfg)
    model.init_weights()
    return model, cfg


def test_save_creates_expected_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = _tiny_gpt()
        optimizer = model.setup_optimizer(
            matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.008, scalar_lr=0.5, weight_decay=0.0
        )
        _manager(tmpdir).save(_State(100), model.state_dict(), optimizer.state_dict(), rank=0)

        assert os.path.exists(os.path.join(tmpdir, "model_000100.pt"))
        assert os.path.exists(os.path.join(tmpdir, "optim_000100_rank0.pt"))
        assert os.path.exists(os.path.join(tmpdir, "meta_000100.json"))


def test_save_load_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = _tiny_gpt()
        optimizer = model.setup_optimizer(
            matrix_lr=0.02, embedding_lr=0.3, unembedding_lr=0.008, scalar_lr=0.5, weight_decay=0.0
        )
        mgr = _manager(tmpdir)
        mgr.save(_State(100), model.state_dict(), optimizer.state_dict(), rank=0)

        ckpt = mgr.load(step=100, device=torch.device("cpu"), load_optimizer=True, rank=0)

        assert ckpt.model_state is not None
        assert ckpt.optimizer_state is not None
        assert ckpt.metadata.user_config["step"] == 100
        assert ckpt.metadata.step == 100


def test_load_without_optimizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = _tiny_gpt()
        mgr = _manager(tmpdir)
        mgr.save(_State(50), model.state_dict(), None, rank=0)

        ckpt = mgr.load(step=50, device=torch.device("cpu"), load_optimizer=False)
        assert ckpt.optimizer_state is None


def test_model_weights_restored():
    with tempfile.TemporaryDirectory() as tmpdir:
        model1, cfg = _tiny_gpt()
        initial_weight = cast(Block, cast(torch.nn.ModuleList, model1.transformer.h)[0]).attn.c_q.weight.clone()

        mgr = _manager(tmpdir)
        mgr.save(_State(50), model1.state_dict(), None, rank=0)

        model2 = GPT(cfg)
        model2.init_weights()
        ckpt = mgr.load(step=50, device=torch.device("cpu"))
        model2.load_state_dict(ckpt.model_state)

        restored = cast(Block, cast(torch.nn.ModuleList, model2.transformer.h)[0]).attn.c_q.weight
        assert torch.allclose(initial_weight, restored)


def test_find_last_step():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _manager(tmpdir)
        model, _ = _tiny_gpt()
        for step in (10, 20, 30):
            mgr.save(_State(step), model.state_dict(), None, rank=0)
        assert mgr.find_last_step() == 30


def test_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _manager(tmpdir)
        model, _ = _tiny_gpt()
        mgr.save(_State(5), model.state_dict(), None, rank=0)
        assert mgr.exists(5)
        assert not mgr.exists(99)


def test_keep_last_n_prunes_old_checkpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _manager(tmpdir, keep_last_n=2)
        model, _ = _tiny_gpt()
        for step in (10, 20, 30):
            mgr.save(_State(step), model.state_dict(), None, rank=0)

        assert not mgr.exists(10)
        assert mgr.exists(20)
        assert mgr.exists(30)


def test_checkpoint_dir_property():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _manager(tmpdir)
        assert mgr.checkpoint_dir == tmpdir
