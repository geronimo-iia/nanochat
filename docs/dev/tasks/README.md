# MLX SFT + RL — Implementation Tasks

See [../mlx-sft-rl-plan.md](../mlx-sft-rl-plan.md) for the full design rationale.

**Status: ALL TASKS COMPLETE** — 339 tests pass, 22 pyright errors (below pre-existing 25).
Commits on branch `feat/mlx-gpt`: `e19b4ac`, `0d0fbfb`, `9e27bc2`, `e8f7ad4`, `9d7e524`.

## SFT track

| Task | File | Depends on | Description | Status |
|---|---|---|---|---|
| S1 | [s1-mlx-gpt-masked-ce.md](s1-mlx-gpt-masked-ce.md) | — | Add `ignore_index=-1` to `mlx_gpt.py` | ✅ done |
| S2 | [s2-sft-mlx-setup.md](s2-sft-mlx-setup.md) | S1 | `mlx_sft_setup()` in `sft/setup.py` | ✅ done |
| S3 | [s3-sft-loop-basetrainer.md](s3-sft-loop-basetrainer.md) | S2 | Refactor `sft/loop.py` to use `BaseTrainer` | ✅ done |

## RL track

| Task | File | Depends on | Description | Status |
|---|---|---|---|---|
| R1 | [r1-mlx-engine.md](r1-mlx-engine.md) | — | `MLXEngine` with KV-cache generation | ✅ done |
| R2 | [r2-mlx-gpt-per-token-loss.md](r2-mlx-gpt-per-token-loss.md) | S1 | `loss_reduction="none"` in `mlx_gpt.py` | ✅ done |
| R3 | [r3-mlx-rl-trainer.md](r3-mlx-rl-trainer.md) | R1 + R2 | `MLXRLTrainer` (REINFORCE) | ✅ done |
| R4 | [r4-rl-mlx-loop.md](r4-rl-mlx-loop.md) | R3 | RL loop and setup for MLX | ✅ done |

## Dependency graph

```
S1 ──► S2 ──► S3        (SFT complete ✅)
│
└────► R2 ──┐
            ├──► R3 ──► R4    (RL complete ✅)
R1 ─────────┘
```

R1 and S1 were implemented in the same commit (both independent of prior work).
