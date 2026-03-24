"""
End-to-end parity test: two runs from the same checkpoint with identical seeds
must produce identical loss trajectories (self-consistency).

Verifies that TorchTrainer + loop refactor is deterministic.

Usage:
    uv run python dev/verify_e2e_parity.py

Requires the experiment checkpoint at:
    /Users/geronimo/build/sp_theory/experiments/nanochat/checkpoints/base/d6
"""

import sys
import torch

EXPERIMENT_BASE = "/Users/geronimo/build/sp_theory/experiments/nanochat"
CHECKPOINT_STEP = 1
STEPS = 10
SEED = 42
ATOL = 1e-6

# Small batch to keep it fast — matches checkpoint model's seq_len
B, T = 2, 512


def _build_trainer(checkpoint, device, tokenizer):
    from nanochat.models.gpt import GPT, GPTConfig
    from nanochat.training.base.trainer import TorchTrainer
    from nanochat.training.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

    cfg = checkpoint.metadata.model_config
    model = GPT(GPTConfig(**cfg))
    model.to(device)
    model.load_state_dict(checkpoint.model_state, strict=False, assign=True)

    optimizer = model.setup_optimizer(
        unembedding_lr=0.008,
        embedding_lr=0.3,
        matrix_lr=0.02,
        scalar_lr=0.5,
        weight_decay=0.28,
    )
    # optimizer state not loaded — checkpoint is from older model version

    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, B, T, split="train", device=device,
        resume_state_dict=checkpoint.metadata.dataloader_state_dict,
    )

    return TorchTrainer(
        orig_model=model,
        model=model,
        optimizer=optimizer,
        scaler=None,
        grad_accum_steps=1,
        device_type=str(device),
        train_loader=loader,
    )


def _run(trainer, steps: int) -> list[float]:
    losses = []
    for _ in range(steps):
        result = trainer.forward_backward()
        trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.28)
        losses.append(result.loss)
    return losses


def main() -> None:
    from nanochat import workspace
    from nanochat.checkpoint.factory import make_checkpoint_manager
    from nanochat.checkpoint.torch_manager import TorchCheckpointManager
    from nanochat.config import current
    from nanochat.config.checkpoint import CheckpointConfig
    from nanochat.config.common import CommonConfig
    from nanochat.config.config import Config
    from nanochat.tokenizer import get_tokenizer

    config = Config(common=CommonConfig(base_dir=EXPERIMENT_BASE, device_type="cpu"))
    current.init(config)
    workspace.init()

    tokenizer = get_tokenizer()
    device = torch.device("cpu")

    ckpt_dir = f"{EXPERIMENT_BASE}/checkpoints/base/d6"
    ckpt_manager = TorchCheckpointManager(ckpt_dir, CheckpointConfig())

    # --- run A ---
    print(f"Run A: loading checkpoint step {CHECKPOINT_STEP}...")
    torch.manual_seed(SEED)
    ckpt_a = ckpt_manager.load(CHECKPOINT_STEP, device, load_optimizer=False, rank=0)
    trainer_a = _build_trainer(ckpt_a, device, tokenizer)
    losses_a = _run(trainer_a, STEPS)
    print(f"  losses: {[f'{l:.6f}' for l in losses_a]}")

    # --- run B (identical seed, same checkpoint) ---
    print(f"Run B: loading checkpoint step {CHECKPOINT_STEP}...")
    torch.manual_seed(SEED)
    ckpt_b = ckpt_manager.load(CHECKPOINT_STEP, device, load_optimizer=False, rank=0)
    trainer_b = _build_trainer(ckpt_b, device, tokenizer)
    losses_b = _run(trainer_b, STEPS)
    print(f"  losses: {[f'{l:.6f}' for l in losses_b]}")

    # --- assert parity ---
    print()
    all_match = True
    for i, (a, b) in enumerate(zip(losses_a, losses_b)):
        diff = abs(a - b)
        ok = diff < ATOL
        status = "✓" if ok else "❌"
        print(f"  step {i+1:02d}: A={a:.6f} B={b:.6f} diff={diff:.2e} {status}")
        if not ok:
            all_match = False

    print()
    if all_match:
        print(f"✅ Parity: all {STEPS} steps match within {ATOL}")
        print("All checks passed.")
    else:
        print(f"❌ Parity failed: loss trajectories diverge")
        sys.exit(1)


if __name__ == "__main__":
    main()
