import gc
import time

import torch
import torch.distributed as dist

from nanochat.checkpoint import make_checkpoint_manager
from nanochat.common import get_compute_dtype, is_ddp_initialized, print0
from nanochat.evaluation.core_benchmark import evaluate_core
from nanochat.evaluation.engine import Engine
from nanochat.evaluation.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.training.base.fp8 import disable_fp8
from nanochat.training.base.setup import BaseTrainingSetup
from nanochat.training.compression_metrics import CompressionMetrics


def train_loop(s: BaseTrainingSetup) -> None:
    """Run the base pretraining loop. Mutates s.state in place."""
    checkpoint_manager = make_checkpoint_manager(s.ckpt_dir, s.config.checkpoint)
    compression_tracker = None
    if s.config.training.track_compression:
        compression_tracker = CompressionMetrics(vocab_size=s.tokenizer.get_vocab_size())
        print0("✓ Compression metrics tracking enabled")

    x, y, s.state.dataloader_state_dict = next(s.train_loader)
    mfu = 0.0
    flops_so_far = 0.0
    results: dict[str, object] = {}

    while True:
        state = s.state
        last_step = state.step == s.num_iterations
        flops_so_far = s.num_flops_per_token * s.total_batch_size * state.step

        # Validation bpb
        if s.config.training.eval_every > 0 and (last_step or state.step % s.config.training.eval_every == 0):
            s.model.eval()
            val_loader = s.build_val_loader()
            eval_steps = s.config.training.eval_tokens // (
                s.config.training.device_batch_size * s.config.training.max_seq_len * s.ddp_world_size
            )
            with disable_fp8(s.model):
                state.val_bpb = evaluate_bpb(s.model, val_loader, eval_steps, s.token_bytes)
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
            s.model.train()

        # CORE metric
        if s.config.training.core_metric_every > 0 and (
            last_step or (state.step > 0 and state.step % s.config.training.core_metric_every == 0)
        ):
            s.model.eval()
            with disable_fp8(s.orig_model):
                results = dict(
                    evaluate_core(
                        model=s.orig_model,
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
            s.model.train()

        # Sampling
        if (
            s.config.training.sample_every > 0
            and s.master_process
            and (last_step or (state.step > 0 and state.step % s.config.training.sample_every == 0))
        ):
            s.model.eval()
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(s.orig_model, s.tokenizer)
            for prompt in prompts:
                tokens = s.tokenizer(prompt, prepend="<|bos|>")
                with disable_fp8(s.orig_model):
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                print0(s.tokenizer.decode(sample[0]))
            s.model.train()

        # Checkpoint
        if last_step or (
            state.step > 0
            and state.step != s.config.checkpoint.resume_from_step
            and s.config.checkpoint.save_every > 0
            and state.step % s.config.checkpoint.save_every == 0
        ):
            checkpoint_manager.save(
                state,
                s.orig_model.state_dict(),
                s.optimizer.state_dict(),
                rank=s.ddp_rank,
            )

        if last_step:
            break

        if s.device_type == "mps":
            torch.mps.empty_cache()

        # Training step
        s.synchronize()
        t0 = time.time()
        logits_for_compression = None
        train_loss = torch.zeros(1, device=s.device)
        for micro_step in range(s.grad_accum_steps):
            if compression_tracker and state.step % s.config.training.compression_log_every == 0 and micro_step == 0:
                with torch.amp.autocast(device_type=s.device_type, dtype=get_compute_dtype()):
                    logits_for_compression = s.model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits_for_compression.view(-1, logits_for_compression.size(-1)), y.view(-1)
                )
            else:
                with torch.amp.autocast(device_type=s.device_type, dtype=get_compute_dtype()):
                    loss = s.model(x, y)
            train_loss = loss.detach()
            loss = loss / s.grad_accum_steps
            if s.scaler is not None:
                s.scaler.scale(loss).backward()
            else:
                loss.backward()
            x, y, state.dataloader_state_dict = next(s.train_loader)

        lrm = s.get_lr_multiplier(state.step)
        muon_momentum = s.get_muon_momentum(state.step)
        muon_weight_decay = s.get_weight_decay(state.step)
        for group in s.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        if s.scaler is not None:
            s.scaler.unscale_(s.optimizer)
            if is_ddp_initialized():
                for v in s.scaler._found_inf_per_device(s.optimizer).values():
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            s.scaler.step(s.optimizer)
            s.scaler.update()
        else:
            s.optimizer.step()
        s.model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
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

        if (
            compression_tracker
            and state.step % s.config.training.compression_log_every == 0
            and logits_for_compression is not None
        ):
            with torch.no_grad():
                compression_metrics = compression_tracker.log_metrics(
                    step=state.step,
                    tokens=y,
                    logits=logits_for_compression,
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
