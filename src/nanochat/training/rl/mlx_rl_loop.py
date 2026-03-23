"""
MLX RL training loop — single-device REINFORCE without DDP.

A dedicated file rather than a refactor of rl/loop.py because:
- rl/loop.py has deep DDP logic (dist.all_reduce for rewards, pass@k aggregation,
  rank-sharded rollout iteration) that is difficult to untangle safely.
- The MLX loop is structurally simpler and cleaner as a standalone module.

The PyTorch rl/loop.py is completely unchanged by this module.
"""

from typing import cast

from nanochat.checkpoint import make_checkpoint_manager
from nanochat.common import print0
from nanochat.report import get_report
from nanochat.training.rl.eval import run_gsm8k_eval_mlx
from nanochat.training.rl.setup import RLTrainingSetup


def mlx_rl_train_loop(s: RLTrainingSetup) -> None:
    """Run the MLX RL training loop. Mutates s.state in place.

    s.trainer must be an MLXRLTrainer (set by mlx_rl_setup).
    No DDP — all aggregation is local (single device).
    """
    assert s.trainer is not None, "mlx_rl_train_loop requires s.trainer to be set (MLX path)"
    config = s.config
    state = s.state
    checkpoint_manager = make_checkpoint_manager(s.ckpt_dir, config.checkpoint)

    for step in range(s.num_steps):
        state.step = step

        # Evaluate pass@k (local, no dist.all_reduce)
        if step % config.rl.eval_every == 0:
            with s.trainer.eval_context() as model:  # type: ignore[union-attr]
                records = list(
                    run_gsm8k_eval_mlx(
                        task=s.val_task,
                        tokenizer=s.tokenizer,
                        engine=s.engine,
                        config=config,
                        max_examples=config.rl.eval_examples,
                        num_samples=config.rl.device_batch_size,
                        temperature=1.0,
                    )
                )
            num_records = len(records)
            if num_records > 0:
                log_passk = {}
                for k in range(1, config.rl.device_batch_size + 1):
                    pass_k = sum(
                        any(o["is_correct"] for o in cast(list[dict[str, object]], r["outcomes"])[:k])
                        for r in records
                    ) / num_records
                    log_passk[f"pass@{k}"] = pass_k
                print_str = ", ".join(f"Pass@{k}: {v:.4f}" for k, v in log_passk.items())
                print0(f"Step {step} | {print_str}")
                s.wandb_run.log(log_passk, step=step)

        # Forward/backward on rollouts
        rewards_list = []
        for _example_step in range(s.examples_per_rank):
            result = s.trainer.forward_backward()  # type: ignore[union-attr]
            rewards_list.append(result.loss)   # trainer stores reward in loss field for logging
            print0(
                f"Step {step}/{s.num_steps} | example {_example_step} | loss: {result.loss:.6f}"
            )

        # Optimizer step
        lrm = s.get_lr_multiplier(step)
        s.trainer.step(lrm, momentum=0.95, weight_decay=config.rl.weight_decay)  # type: ignore[union-attr]
        s.wandb_run.log({"lrm": lrm, "train/loss": result.loss}, step=step)  # type: ignore[possibly-undefined]

        # Checkpoint
        if (
            step > 0
            and config.checkpoint.save_every > 0
            and step % config.checkpoint.save_every == 0
        ) or step == s.num_steps - 1:
            checkpoint_manager.save(
                state,
                s.trainer.model_state_dict(),   # type: ignore[union-attr]
                s.trainer.optimizer_state_dict(),  # type: ignore[union-attr]
                rank=0,
            )
            print0(f"Saved checkpoint to {s.ckpt_dir}")

    get_report().log(section="Chat RL (MLX)", data=[s.user_config])
