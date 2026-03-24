"""Tests for CheckpointMetadata and LoopState."""

from nanochat.checkpoint.protocol import CheckpointMetadata, LoopState


def test_metadata_round_trip_minimal():
    meta = CheckpointMetadata(step=10, model_config={"n_layer": 2}, user_config={"lr": 0.01})
    assert CheckpointMetadata.from_dict(meta.to_dict()) == meta


def test_metadata_round_trip_full():
    meta = CheckpointMetadata(
        step=42,
        model_config={"n_layer": 4, "n_embd": 128},
        user_config={"batch_size": 32},
        val_bpb=1.23,
        loop_state=LoopState(min_val_bpb=1.1, smooth_train_loss=0.9, total_training_time=300.0),
        dataloader_state_dict={"pos": 100},
    )
    assert CheckpointMetadata.from_dict(meta.to_dict()) == meta


def test_metadata_json_round_trip():
    meta = CheckpointMetadata(
        step=5,
        model_config={"vocab_size": 256},
        user_config={},
        loop_state=LoopState(min_val_bpb=2.0, smooth_train_loss=1.5, total_training_time=60.0),
    )
    assert CheckpointMetadata.from_json(meta.to_json()) == meta


def test_metadata_optional_fields_default_none():
    meta = CheckpointMetadata(step=1, model_config={}, user_config={})
    assert meta.val_bpb is None
    assert meta.loop_state is None
    assert meta.dataloader_state_dict is None


def test_loop_state_fields():
    ls = LoopState(min_val_bpb=0.5, smooth_train_loss=0.8, total_training_time=120.0)
    assert ls.min_val_bpb == 0.5
    assert ls.smooth_train_loss == 0.8
    assert ls.total_training_time == 120.0
