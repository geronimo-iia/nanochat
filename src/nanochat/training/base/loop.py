import gc
import time

import torch

from nanochat.checkpoint import CheckpointManager
from nanochat.common import print0
from nanochat.evaluation.core_benchmark import evaluate_core
from nanochat.evaluation.engine import Engine
from nanochat.evaluation.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.training.base.setup import BaseTrainingSetup
from nanochat.training.compression_metrics import CompressionMetrics


def train_loop(s: BaseTrainingSetup, checkpoint_manager: CheckpointManager) -> None:
    """Run the base pretraining loop. Mutates s.state in place."""
    compression_tracker = None
    if s.config.training.track_compression:
        compression_tracker = CompressionMetrics(vocab_size=s.tokenizer.get_vocab_size())
        print0("✓ Compression metrics tracking enabled")

    mfu = 0.0
    flops_so_far = 0.0
    results: dict[str, object] = {}

    while True:
        state = s.state
        last_step = state.step == s.num_iterations
        flops_so_far = s.num_flops_per_token * s.total_batch_size * state.step

        # Validation bpb
        if s.config.training.eval_every > 0 and (last_step or state.step % s.config.training.eval_every == 0):
            val_loader = s.build_val_loader()
            eval_steps = s.config.training.eval_tokens // (
                s.config.training.device_batch_size * s.config.training.max_seq_len * s.ddp_world_size
            )
            with s.trainer.eval_context() as model:
                state.val_bpb = evaluate_bpb(model, val_loader, eval_steps, s.token_bytes)
            print0(f"Step {state.step:05d} | Validation bpb: {state.val_bpb:.6f}")
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

        # CORE metric
        if s.config.training.core_metric_every > 0 and (
            last_step or (state.step > 0 and state.step % s.config.training.core_metric_every == 0)
        ):
            with s.trainer.eval_context() as model:
                results = dict(
                    evaluate_core(
                        model=model,
                        tokenizer=s.tokenizer,
                        device=s.device,
                        max_per_task=s.config.training.core_metric_max_per_task,
                    )
                )
            print0(f"Step {state.step:05d} | CORE metric: {results['core_metric']:.4f}")
            s.wandb_run.log(
                {
                    "total_training_flops": flops_so_far,
                    "core_metric": results["core_metric"],
                    "centered_results": results["centered_results"],
                },
                step=state.step,
            )

        # Sampling
        if (
            s.config.training.sample_every > 0
            and s.master_process
            and (last_step or (state.step > 0 and state.step % s.config.training.sample_every == 0))
        ):
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            with s.trainer.eval_context() as model:
                engine = Engine(model, s.tokenizer)
                for prompt in prompts:
                    tokens = s.tokenizer(prompt, prepend="<|bos|>")
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                    print0(s.tokenizer.decode(sample[0]))

        # Checkpoint
        if last_step or (
            state.step > 0
            and state.step != s.config.checkpoint.resume_from_step
            and s.config.checkpoint.save_every > 0
            and state.step % s.config.checkpoint.save_every == 0
        ):
            checkpoint_manager.save(
                state,
                s.trainer.model_state_dict(),
                s.trainer.optimizer_state_dict(),
                rank=s.ddp_rank,
            )

        if last_step:
            break

        if s.device_type == "mps":
            torch.mps.empty_cache()

        # Compression logits (before forward_backward so we use the current batch)
        logits_np = None
        tokens_np = None
        if compression_tracker and state.step % s.config.training.compression_log_every == 0:
            logits_np, tokens_np = s.trainer.forward_logits()

        # Training step
        s.synchronize()
        t0 = time.time()
        result = s.trainer.forward_backward()
        state.dataloader_state_dict = result.dataloader_state_dict
        train_loss_f = result.loss

        lrm = s.get_lr_multiplier(state.step)
        muon_momentum = s.get_muon_momentum(state.step)
        muon_weight_decay = s.get_weight_decay(state.step)
        s.trainer.step(lrm, muon_momentum, muon_weight_decay)
        s.synchronize()
        dt = time.time() - t0

        # Logging
        ema_beta = 0.9
        state.smooth_train_loss = ema_beta * state.smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = state.smooth_train_loss / (1 - ema_beta ** (state.step + 1))
        tok_per_sec = int(s.total_batch_size / dt)
        flops_per_sec = s.num_flops_per_token * s.total_batch_size / dt
        mfu = 100 * flops_per_sec / (s.gpu_peak_flops * s.ddp_world_size)
        if state.step > 0:
            state.total_training_time += dt
        steps_done = state.step - 1
        eta_str = ""
        if steps_done > 0:
            eta_seconds = (s.num_iterations - state.step) * (state.total_training_time / steps_done)
            eta_str = f" | eta: {eta_seconds / 60:.1f}m"
        assert state.dataloader_state_dict is not None
        epoch = f"{state.dataloader_state_dict['epoch']} pq: {state.dataloader_state_dict['pq_idx']} rg: {state.dataloader_state_dict['rg_idx']}"
        print0(
            f"step {state.step:05d}/{s.num_iterations:05d} ({100 * state.step / s.num_iterations:.2f}%) | "
            f"loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | "
            f"tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | "
            f"total time: {state.total_training_time / 60:.2f}m{eta_str}"
        )

        if compression_tracker and logits_np is not None and tokens_np is not None:
            compression_metrics = compression_tracker.log_metrics(
                step=state.step,
                tokens=tokens_np,
                logits=logits_np,
                loss=train_loss_f,
            )
            if s.config.training.compression_early_stop and compression_tracker.detect_overfitting():
                print0(f"[Step {state.step}] Compression plateau detected - possible overfitting")
            print0(
                f"[compression] step {state.step:05d} | entropy: {compression_metrics['entropy']:.4f} | "
                f"ratio: {compression_metrics['compression_ratio']:.4f} | "
                f"gzip: {compression_metrics['gzip_compression']:.4f} | "
                f"efficiency: {compression_metrics['compression_efficiency']:.4f}"
            )
            if s.master_process:
                s.wandb_run.log({f"compression/{k}": v for k, v in compression_metrics.items()}, step=state.step)

        s.wandb_run.log(
            {
                "total_training_flops": flops_so_far,
                "total_training_time": state.total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/epoch": epoch,
            },
            step=state.step,
        )

        first_step_of_run = (state.step == 0) or (s.resuming and state.step == s.config.checkpoint.resume_from_step)
        state.step += 1

        if first_step_of_run:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif state.step % 5000 == 0:
            gc.collect()

    # Final stats
    print0(f"Peak memory usage: {s.get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {state.total_training_time / 60:.2f}m")
    if state.val_bpb is not None:
        print0(f"Minimum validation bpb: {state.min_val_bpb:.6f}")

    get_report().log(
        section="Base model training",
        data=[
            s.user_config,
            {
                "Number of parameters": s.num_params,
                "Number of FLOPs per token": f"{s.num_flops_per_token:e}",
                "Calculated number of iterations": s.num_iterations,
                "Number of training tokens": s.total_tokens,
                "Tokens : Scaling params ratio": s.total_batch_size * s.num_iterations / s.num_scaling_params,
                "DDP world size": s.ddp_world_size,
                "warmup_steps": s.config.training.warmup_steps,
                "warmdown_ratio": s.config.training.warmdown_ratio,
                "final_lr_frac": s.config.training.final_lr_frac,
            },
            {
                "Minimum validation bpb": state.min_val_bpb if state.val_bpb is not None else None,
                "Final validation bpb": state.val_bpb,
                "CORE metric estimate": results.get("core_metric", None),
                "MFU %": f"{mfu:.2f}%",
                "Total training flops": f"{flops_so_far:e}",
                "Total training time": f"{state.total_training_time / 60:.2f}m",
                "Peak memory usage": f"{s.get_max_memory() / 1024 / 1024:.2f}MiB",
            },
        ],
    )
