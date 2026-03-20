import gc
import time

import torch
import torch.distributed as dist

from nanochat.checkpoint import make_checkpoint_manager
from nanochat.common import is_ddp_initialized, print0
from nanochat.evaluation.chat.loop import run_chat_eval
from nanochat.evaluation.engine import Engine
from nanochat.evaluation.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.training.sft.setup import SFTTrainingSetup


def sft_train_loop(s: SFTTrainingSetup) -> None:
    """Run the SFT training loop. Mutates s.state in place."""
    checkpoint_manager = make_checkpoint_manager(s.ckpt_dir, s.config.checkpoint)
    x, y = next(s.train_loader)  # type: ignore[call-overload]

    while True:
        state = s.state
        flops_so_far = s.num_flops_per_token * s.config.sft.total_batch_size * state.step

        # Sync last_step across all ranks to avoid hangs in distributed setting
        if s.ddp:
            last_step_tensor = torch.tensor(state.last_step, dtype=torch.int32, device=s.device)
            dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
            state.last_step = bool(last_step_tensor.item())

        # Validation bpb
        if state.last_step or (s.config.sft.eval_every > 0 and state.step % s.config.sft.eval_every == 0):
            s.model.eval()  # type: ignore[union-attr]
            val_loader = s.build_val_loader()
            eval_steps = s.config.sft.eval_tokens // (
                s.config.sft.device_batch_size * s.config.sft.max_seq_len * s.ddp_world_size
            )
            state.val_bpb = evaluate_bpb(s.model, val_loader, eval_steps, s.token_bytes)
            print0(f"Step {state.step:05d} | Validation bpb: {state.val_bpb:.4f}")
            if state.val_bpb < state.min_val_bpb:
                state.min_val_bpb = state.val_bpb
            s.wandb_run.log(
                {
                    "total_training_flops": flops_so_far,
                    "total_training_time": state.total_training_time,
                    "val/bpb": state.val_bpb,
                },
                step=state.step,
            )
            s.model.train()  # type: ignore[union-attr]

        # ChatCORE metric
        if s.config.sft.chatcore_every > 0 and (
            state.last_step or (state.step > 0 and state.step % s.config.sft.chatcore_every == 0)
        ):
            s.model.eval()  # type: ignore[union-attr]
            engine = Engine(s.orig_model, s.tokenizer)
            all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
            categorical_tasks = {"ARC-Easy", "ARC-Challenge", "MMLU"}
            baseline_accuracies = {
                "ARC-Easy": 0.25,
                "ARC-Challenge": 0.25,
                "MMLU": 0.25,
                "GSM8K": 0.0,
                "HumanEval": 0.0,
                "SpellingBee": 0.0,
            }
            task_results: dict[str, float] = {}
            for task_name in all_tasks:
                limit = (
                    s.config.sft.chatcore_max_cat
                    if task_name in categorical_tasks
                    else s.config.sft.chatcore_max_sample
                )
                max_problems = None if limit < 0 else limit
                acc = run_chat_eval(
                    task_name,
                    s.orig_model,
                    s.tokenizer,
                    engine,
                    batch_size=s.config.sft.device_batch_size,
                    max_problems=max_problems,
                )
                task_results[task_name] = acc
                print0(f"  {task_name}: {100 * acc:.2f}%")

            def centered_mean(tasks: set[str]) -> float:
                return sum(
                    (task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks
                ) / len(tasks)

            chatcore = centered_mean(all_tasks)
            chatcore_cat = centered_mean(categorical_tasks)
            print0(f"Step {state.step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
            s.wandb_run.log(
                {
                    "total_training_flops": flops_so_far,
                    "chatcore_metric": chatcore,
                    "chatcore_cat": chatcore_cat,
                    **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
                },
                step=state.step,
            )
            s.model.train()  # type: ignore[union-attr]

        # Checkpoint — SFT only saves at last_step
        if state.last_step:
            checkpoint_manager.save(
                state,
                s.orig_model.state_dict(),  # type: ignore[union-attr]
                s.optimizer.state_dict(),  # type: ignore[union-attr]
                rank=s.ddp_rank,
            )

        if state.last_step:
            break

        if s.device_type == "mps":
            torch.mps.empty_cache()

        # Training step
        s.synchronize()
        t0 = time.time()
        train_loss = torch.zeros(1, device=s.device)
        for _ in range(s.grad_accum_steps):
            loss = s.model(x, y)  # type: ignore[operator]
            train_loss = loss.detach()
            loss = loss / s.grad_accum_steps
            if s.scaler is not None:
                s.scaler.scale(loss).backward()
            else:
                loss.backward()
            x, y = next(s.train_loader)  # type: ignore[call-overload]
            state.progress = max(state.progress, state.approx_progress)

        lrm = s.get_lr_multiplier(state.progress)
        muon_momentum = s.get_muon_momentum(state.step)
        for group in s.optimizer.param_groups:  # type: ignore[union-attr]
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
        if s.scaler is not None:
            s.scaler.unscale_(s.optimizer)
            if is_ddp_initialized():
                for v in s.scaler._found_inf_per_device(s.optimizer).values():  # type: ignore[arg-type]
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            s.scaler.step(s.optimizer)
            s.scaler.update()
        else:
            s.optimizer.step()  # type: ignore[union-attr]
        s.model.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        s.synchronize()
        dt = time.time() - t0

        state.step += 1

        # Logging
        ema_beta = 0.9
        state.smooth_train_loss = ema_beta * state.smooth_train_loss + (1 - ema_beta) * train_loss.item()
        debiased_smooth_loss = state.smooth_train_loss / (1 - ema_beta ** (state.step + 1))
        pct_done = 100 * state.progress
        tok_per_sec = int(s.config.sft.total_batch_size / dt)
        flops_per_sec = s.num_flops_per_token * s.config.sft.total_batch_size / dt
        mfu = 100 * flops_per_sec / (s.gpu_peak_flops * s.ddp_world_size)
        if state.step > 10:
            state.total_training_time += dt
        print0(
            f"step {state.step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
            f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | "
            f"epoch: {state.current_epoch} | total time: {state.total_training_time / 60:.2f}m"
        )
        s.wandb_run.log(
            {
                "total_training_flops": flops_so_far,
                "total_training_time": state.total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/epoch": state.current_epoch,
            },
            step=state.step,
        )

        if state.step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif state.step % 5000 == 0:
            gc.collect()

    # Final stats
    print0(f"Peak memory usage: {s.get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {state.total_training_time / 60:.2f}m")

    get_report().log(
        section="SFT",
        data=[
            s.user_config,
            {
                "Number of iterations": state.step,
                "DDP world size": s.ddp_world_size,
            },
            {
                "Minimum validation bpb": state.min_val_bpb,
            },
        ],
    )
