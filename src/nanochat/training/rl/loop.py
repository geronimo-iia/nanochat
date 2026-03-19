from typing import cast

import torch
import torch.distributed as dist

from nanochat import workspace
from nanochat.common import print0
from nanochat.report import get_report
from nanochat.training.checkpoint import save_checkpoint
from nanochat.training.rl.eval import run_gsm8k_eval
from nanochat.training.rl.rollout import get_batch
from nanochat.training.rl.setup import RLTrainingSetup


def rl_train_loop(s: RLTrainingSetup) -> None:
    config = s.config
    state = s.state

    batch_iterator = get_batch(
        state=state,
        config=config,
        train_task=s.train_task,
        engine=s.engine,
        model=s.model,
        tokenizer=s.tokenizer,
        device=s.device,
        ddp_rank=s.ddp_rank,
        ddp_world_size=s.ddp_world_size,
    )

    for step in range(s.num_steps):
        state.step = step

        # Evaluate
        if step % config.rl.eval_every == 0:
            s.model.eval()
            passk = torch.zeros(config.rl.device_batch_size, device=s.device)
            records = list(
                run_gsm8k_eval(
                    task=s.val_task,
                    tokenizer=s.tokenizer,
                    engine=s.engine,
                    config=config,
                    ddp_rank=s.ddp_rank,
                    ddp_world_size=s.ddp_world_size,
                    num_samples=config.rl.device_batch_size,
                    max_examples=config.rl.eval_examples,
                    temperature=1.0,
                )
            )
            for k in range(1, config.rl.device_batch_size + 1):
                passk[k - 1] = sum(
                    any(o["is_correct"] for o in cast(list[dict[str, object]], r["outcomes"])[:k]) for r in records
                )
            num_records = torch.tensor(len(records), dtype=torch.long, device=s.device)
            if s.ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
                dist.all_reduce(passk, op=dist.ReduceOp.SUM)
            passk = passk / num_records.item()
            print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, config.rl.device_batch_size + 1)]
            print0(f"Step {step} | {', '.join(print_passk)}")
            log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, config.rl.device_batch_size + 1)}
            s.wandb_run.log(log_passk, step=step)

        # Forward/backward on rollouts
        rewards_list = []
        sequence_lengths = []
        for example_step in range(s.examples_per_rank):
            sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
            s.model.train()
            assert inputs_all.size(0) % config.rl.device_batch_size == 0
            num_passes = inputs_all.size(0) // config.rl.device_batch_size
            for pass_idx in range(num_passes):
                b0 = pass_idx * config.rl.device_batch_size
                b1 = b0 + config.rl.device_batch_size
                inputs = inputs_all[b0:b1]
                targets = targets_all[b0:b1]
                rewards = rewards_all[b0:b1]
                advantages = advantages_all[b0:b1]
                logp = -s.model(inputs, targets, loss_reduction="none").view_as(inputs)
                pg_obj = (logp * advantages.unsqueeze(-1)).sum()
                num_valid = (targets >= 0).sum().clamp(min=1)
                pg_obj = pg_obj / (num_valid * num_passes * s.examples_per_rank)
                loss = -pg_obj
                loss.backward()
                print0(
                    f"Step {step}/{s.num_steps} | Example step {example_step} | Pass {pass_idx}"
                    f" | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}"
                )
            rewards_list.append(rewards_all.mean().item())
            sequence_lengths.extend(len(seq) for seq in sequences_all)

        # Aggregate and log rollout stats
        mean_reward = sum(rewards_list) / len(rewards_list)
        mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
        if s.ddp:
            mean_reward_t = torch.tensor(mean_reward, dtype=torch.float, device=s.device)
            mean_seq_len_t = torch.tensor(mean_sequence_length, dtype=torch.float, device=s.device)
            dist.all_reduce(mean_reward_t, op=dist.ReduceOp.AVG)
            dist.all_reduce(mean_seq_len_t, op=dist.ReduceOp.AVG)
            mean_reward = mean_reward_t.item()
            mean_sequence_length = mean_seq_len_t.item()
        print0(
            f"Step {step}/{s.num_steps} | Average reward: {mean_reward}"
            f" | Average sequence length: {mean_sequence_length:.2f}"
        )
        s.wandb_run.log({"reward": mean_reward, "sequence_length": mean_sequence_length}, step=step)

        # Optimizer step
        lrm = s.get_lr_multiplier(step)
        for group in s.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        s.optimizer.step()
        s.model.zero_grad(set_to_none=True)
        s.wandb_run.log({"lrm": lrm}, step=step)

        # Checkpoint
        if s.master_process and ((step > 0 and step % config.rl.save_every == 0) or step == s.num_steps - 1):
            depth = s.model.config.n_layer
            output_dirname = config.rl.model_tag if config.rl.model_tag else f"d{depth}"
            ckpt_dir = workspace.checkpoint_dir("rl", output_dirname)
            save_checkpoint(
                ckpt_dir,
                step,
                s.model.state_dict(),
                None,
                state.to_checkpoint(s.model.config.__dict__),
            )
            print(f"✅ Saved model checkpoint to {ckpt_dir}")

    get_report().log(section="Chat RL", data=[s.user_config])
