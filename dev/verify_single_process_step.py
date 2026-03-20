"""
Verify that TorchTrainer.step completes without hang on a single-process (no DDP) run.

The risk: scaler path calls dist.all_reduce on inf flags. Without the
is_ddp_initialized() guard this hangs on single-process runs.

Usage:
    uv run python dev/verify_single_process_step.py
"""

import sys
import signal
import torch
import torch.nn as nn
import numpy as np

TIMEOUT_SECONDS = 10
STEPS = 5


def _timeout_handler(signum, frame):
    print(f"❌ TIMEOUT: step() did not complete within {TIMEOUT_SECONDS}s — likely hanging on dist.all_reduce")
    sys.exit(1)


class _StubModel(nn.Module):
    def __init__(self, vocab_size: int = 16, seq_len: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(vocab_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        one_hot = nn.functional.one_hot(x, self.vocab_size).float()
        logits = self.linear(one_hot)
        if y is None:
            return logits
        return nn.functional.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))


def _make_loader(vocab_size: int = 16, seq_len: int = 4, batch_size: int = 2):
    step = 0
    while True:
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        yield x, y, {"epoch": 0, "pq_idx": step, "rg_idx": 0}
        step += 1


def main() -> None:
    from nanochat.training.base.trainer import TorchTrainer

    signal.signal(signal.SIGALRM, _timeout_handler)

    model = _StubModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for group in optimizer.param_groups:
        group["initial_lr"] = 1e-3
        group["kind"] = "adamw"

    # Test 1: no scaler (most common path)
    print("Test 1: no scaler (fp32)")
    signal.alarm(TIMEOUT_SECONDS)
    trainer = TorchTrainer(
        orig_model=model,
        model=model,
        optimizer=optimizer,
        scaler=None,
        grad_accum_steps=1,
        device_type="cpu",
        train_loader=_make_loader(),
    )
    for i in range(STEPS):
        trainer.forward_backward()
        trainer.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.01)
        print(f"  step {i+1} ✓")
    signal.alarm(0)
    print("✅ No scaler: completed without hang\n")

    # Test 2: with scaler (fp16 path — the risky one with dist.all_reduce)
    # GradScaler on CPU raises, so we mock _found_inf_per_device to exercise the guard
    print("Test 2: with scaler (mocked — exercises is_ddp_initialized() guard)")

    class _MockScaler:
        """Minimal scaler mock that exercises the inf all-reduce guard path."""
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def _found_inf_per_device(self, optimizer):
            # Return a fake inf tensor — if all_reduce is called without the guard this hangs
            return {"cpu": torch.tensor(0.0)}

    model2 = _StubModel()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    for group in optimizer2.param_groups:
        group["initial_lr"] = 1e-3
        group["kind"] = "adamw"

    signal.alarm(TIMEOUT_SECONDS)
    trainer2 = TorchTrainer(
        orig_model=model2,
        model=model2,
        optimizer=optimizer2,
        scaler=_MockScaler(),
        grad_accum_steps=1,
        device_type="cpu",
        train_loader=_make_loader(),
    )
    for i in range(STEPS):
        trainer2.forward_backward()
        trainer2.step(lr_multiplier=1.0, momentum=0.95, weight_decay=0.01)
        print(f"  step {i+1} ✓")
    signal.alarm(0)
    print("✅ With scaler: completed without hang (is_ddp_initialized() guard working)\n")

    print("All checks passed.")


if __name__ == "__main__":
    main()
