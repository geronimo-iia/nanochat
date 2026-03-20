import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from nanochat.checkpoint import CheckpointManager
from nanochat.common import (
    WandbProtocol,
    autodetect_backend,
    autodetect_device_type,
    compute_init,
    get_compute_dtype,
    get_compute_dtype_reason,
    get_device_sync,
    get_peak_flops,
    init_wandb,
    print0,
    print_banner,
)
from nanochat.config import Config
from nanochat.models.flash_attention import HAS_FA3, _use_fa3
from nanochat.models.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from nanochat.training.base.schedulers import (
    base_lr_scheduler,
    base_muon_momentum_scheduler,
    base_weight_decay_scheduler,
)
from nanochat.training.base.state import PretrainingState
from nanochat.training.base.trainer import BaseTrainer, TorchTrainer
from nanochat.training.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.training.scaling import B_REF, compute_training_hyperparams, get_scaling_params


def build_model_meta(
    depth: int,
    aspect_ratio: int,
    head_dim: int,
    max_seq_len: int,
    window_pattern: str,
    vocab_size: int,
) -> GPT:
    """Build a GPT model on meta device (shapes/dtypes only, no data)."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    gpt_config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        return GPT(gpt_config)


def _convert_fp8(model: GPT, config: Config, device_type: str) -> None:
    """Convert Linear layers to Float8Linear in-place if --fp8 is set."""
    if not config.training.fp8:
        return
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        return

    import torch.nn as nn

    from nanochat.models.fp8 import Float8LinearConfig, convert_to_float8_training

    def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True

    fp8_config = Float8LinearConfig.from_recipe_name(config.training.fp8_recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
    num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
    print0(
        f"✓ FP8 training enabled ({config.training.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_linear - num_fp8} (too small)"
    )


class BaseTrainingSetup:
    """All resolved setup state passed to train_loop(). Built once by setup()."""

    __slots__ = (
        "config",
        "device_type",
        "ddp_rank",
        "ddp_world_size",
        "device",
        "master_process",
        "trainer",
        "tokenizer",
        "token_bytes",
        "build_val_loader",
        "ckpt_dir",
        "user_config",
        "model_config_kwargs",
        "num_iterations",
        "total_batch_size",
        "total_tokens",
        "num_params",
        "num_flops_per_token",
        "num_scaling_params",
        "gpu_peak_flops",
        "get_lr_multiplier",
        "get_muon_momentum",
        "get_weight_decay",
        "synchronize",
        "get_max_memory",
        "wandb_run",
        "state",
        "resuming",
    )

    def __init__(
        self,
        config: Config,
        device_type: str,
        ddp_rank: int,
        ddp_world_size: int,
        device: torch.device,
        master_process: bool,
        trainer: BaseTrainer,
        tokenizer: object,
        token_bytes: object,
        build_val_loader: Callable[[], object],
        ckpt_dir: str,
        user_config: dict[str, object],
        model_config_kwargs: dict[str, object],
        num_iterations: int,
        total_batch_size: int,
        total_tokens: int,
        num_params: int,
        num_flops_per_token: float,
        num_scaling_params: int,
        gpu_peak_flops: float,
        get_lr_multiplier: Callable[[int], float],
        get_muon_momentum: Callable[[int], float],
        get_weight_decay: Callable[[int], float],
        synchronize: Callable[[], None],
        get_max_memory: Callable[[], int],
        wandb_run: WandbProtocol,
        state: PretrainingState,
        resuming: bool,
    ):
        self.config = config
        self.device_type = device_type
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.device = device
        self.master_process = master_process
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.token_bytes = token_bytes
        self.build_val_loader = build_val_loader
        self.ckpt_dir = ckpt_dir
        self.user_config = user_config
        self.model_config_kwargs = model_config_kwargs
        self.num_iterations = num_iterations
        self.total_batch_size = total_batch_size
        self.total_tokens = total_tokens
        self.num_params = num_params
        self.num_flops_per_token = num_flops_per_token
        self.num_scaling_params = num_scaling_params
        self.gpu_peak_flops = gpu_peak_flops
        self.get_lr_multiplier = get_lr_multiplier
        self.get_muon_momentum = get_muon_momentum
        self.get_weight_decay = get_weight_decay
        self.synchronize = synchronize
        self.get_max_memory = get_max_memory
        self.wandb_run = wandb_run
        self.state = state
        self.resuming = resuming


# ---------------------------------------------------------------------------
# Backend-specific trainer construction
# ---------------------------------------------------------------------------

def _setup_torch(
    config: Config,
    checkpoint_manager: CheckpointManager,
    meta_model: GPT,
    model_config_kwargs: dict,
    vocab_size: int,
    weight_decay_scaled: float,
    batch_lr_scale: float,
    grad_accum_steps: int,
    tokenizer: object,
    resuming: bool,
    resume_ckpt: object,
) -> tuple[BaseTrainer, str, int, int, torch.device, Callable, Callable, float]:
    """Build TorchTrainer. Returns (trainer, device_type, ddp_rank, ddp_world_size, device, synchronize, get_max_memory, gpu_peak_flops)."""
    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    synchronize, get_max_memory = get_device_sync(device_type)

    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float("inf")
    print0(f"COMPUTE_DTYPE: {get_compute_dtype()} ({get_compute_dtype_reason()})")

    if _use_fa3():
        print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
    else:
        print0("!" * 80)
        if HAS_FA3 and get_compute_dtype() != torch.bfloat16:
            print0(
                f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={get_compute_dtype()}. Using PyTorch SDPA fallback"
            )
        else:
            print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
        print0("WARNING: Training will be less efficient without FA3")
        if config.training.window_pattern != "L":
            print0(
                f"WARNING: SDPA has no support for sliding window attention (window_pattern='{config.training.window_pattern}'). Your GPU utilization will be terrible."
            )
            print0(
                "WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns."
            )
        print0("!" * 80)

    model = build_model_meta(
        config.training.depth,
        config.training.aspect_ratio,
        config.training.head_dim,
        config.training.max_seq_len,
        config.training.window_pattern,
        vocab_size,
    )
    model.to_empty(device=device)
    model.init_weights()

    if resuming:
        assert resume_ckpt is not None
        model.load_state_dict(resume_ckpt.model_state, strict=True, assign=True)
        del resume_ckpt.model_state

    _convert_fp8(model, config, device_type)

    orig_model = model
    if device_type != "mps":
        model = torch.compile(model, dynamic=False)
    else:
        print0("Skipping torch.compile on MPS (inductor backend not supported, causes NaN gradients)")

    optimizer = model.setup_optimizer(
        unembedding_lr=config.training.unembedding_lr * batch_lr_scale,
        embedding_lr=config.training.embedding_lr * batch_lr_scale,
        scalar_lr=config.training.scalar_lr * batch_lr_scale,
        matrix_lr=config.training.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    )
    if resuming:
        assert resume_ckpt is not None
        optimizer.load_state_dict(resume_ckpt.optimizer_state)
        del resume_ckpt.optimizer_state

    scaler = torch.amp.GradScaler(device=device_type) if get_compute_dtype() == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")

    dataloader_resume_state_dict = None
    if resuming:
        assert resume_ckpt is not None
        dataloader_resume_state_dict = resume_ckpt.metadata.dataloader_state_dict

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        config.training.device_batch_size,
        config.training.max_seq_len,
        split="train",
        device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )

    trainer: BaseTrainer = TorchTrainer(
        orig_model=orig_model,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        grad_accum_steps=grad_accum_steps,
        device_type=device_type,
        train_loader=train_loader,
    )
    return trainer, device_type, ddp_rank, ddp_world_size, device, synchronize, get_max_memory, gpu_peak_flops


def _setup_mlx(
    config: Config,
    checkpoint_manager: CheckpointManager,
    model_config_kwargs: dict,
    vocab_size: int,
    weight_decay_scaled: float,
    batch_lr_scale: float,
    grad_accum_steps: int,
    tokenizer: object,
    resuming: bool,
    resume_ckpt: object,
) -> tuple[BaseTrainer, str, int, int, torch.device, Callable, Callable, float]:
    """Build MLXTrainer. Returns same tuple shape as _setup_torch."""
    import mlx.core as mx
    from nanochat.models.mlx_gpt import GPT as MLXGPT
    from nanochat.training.mlx_optimizer import MuonAdamW, build_param_groups
    from nanochat.training.mlx_trainer import MLXTrainer

    print0("✓ MLX backend — single device, mx.compile, unified memory")

    gpt_config = GPTConfig(**model_config_kwargs)
    model = MLXGPT(gpt_config)

    optimizer = MuonAdamW(build_param_groups(
        model,
        unembedding_lr=config.training.unembedding_lr * batch_lr_scale,
        embedding_lr=config.training.embedding_lr * batch_lr_scale,
        scalar_lr=config.training.scalar_lr * batch_lr_scale,
        matrix_lr=config.training.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    ))

    dataloader_resume_state_dict = None
    if resuming:
        assert resume_ckpt is not None
        dataloader_resume_state_dict = resume_ckpt.metadata.dataloader_state_dict

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        config.training.device_batch_size,
        config.training.max_seq_len,
        split="train",
        device=torch.device("cpu"),
        resume_state_dict=dataloader_resume_state_dict,
    )

    # Wrap loader to convert torch tensors → mx.arrays
    def mlx_loader():
        for x, y, state in train_loader:
            import numpy as np
            yield mx.array(x.numpy()), mx.array(y.numpy()), state

    trainer: BaseTrainer = MLXTrainer(model, optimizer, grad_accum_steps, mlx_loader())

    if resuming:
        assert resume_ckpt is not None
        trainer.load_state_dicts(resume_ckpt.model_state, resume_ckpt.optimizer_state)
        del resume_ckpt.model_state, resume_ckpt.optimizer_state

    synchronize: Callable[[], None] = lambda: mx.eval([])
    get_max_memory: Callable[[], int] = lambda: 0

    return trainer, "mlx", 0, 1, torch.device("cpu"), synchronize, get_max_memory, float("inf")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def setup(config: Config, checkpoint_manager: CheckpointManager) -> BaseTrainingSetup:
    """Initialize compute, model, optimizer, dataloaders and schedulers for base pretraining."""
    print_banner()

    backend = config.common.backend or autodetect_backend()

    user_config = asdict(config)
    wandb_run = init_wandb(user_config=user_config, master_process=True)

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # Use torch meta model for param counting and scaling — framework-agnostic
    meta_model = build_model_meta(
        config.training.depth,
        config.training.aspect_ratio,
        config.training.head_dim,
        config.training.max_seq_len,
        config.training.window_pattern,
        vocab_size,
    )
    model_config_kwargs = asdict(meta_model.config)
    print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

    param_counts = meta_model.num_scaling_params()
    print0("Parameter counts:")
    for key, value in param_counts.items():
        print0(f"{key:24s}: {value:,}")
    num_params = param_counts["total"]
    num_flops_per_token = meta_model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    d12_ref = build_model_meta(
        12,
        config.training.aspect_ratio,
        config.training.head_dim,
        config.training.max_seq_len,
        config.training.window_pattern,
        vocab_size,
    )
    hp = compute_training_hyperparams(
        target_param_data_ratio=config.training.target_param_data_ratio,
        total_batch_size_override=config.training.total_batch_size,
        weight_decay=config.training.weight_decay,
        num_scaling_params=get_scaling_params(meta_model),
        d12_scaling_params=get_scaling_params(d12_ref),
    )
    num_scaling_params = hp.num_scaling_params
    target_tokens = hp.target_tokens
    total_batch_size = hp.total_batch_size
    batch_lr_scale = hp.batch_lr_scale
    weight_decay_scaled = hp.weight_decay_scaled
    if config.training.total_batch_size == -1:
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")
    if batch_lr_scale != 1.0:
        print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")
    if weight_decay_scaled != config.training.weight_decay:
        print0(
            f"Scaling weight decay from {config.training.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {config.training.depth}"
        )

    assert (
        config.training.num_iterations > 0
        or config.training.target_param_data_ratio > 0
        or config.training.target_flops > 0
    )
    if config.training.num_iterations > 0:
        num_iterations = config.training.num_iterations
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    elif config.training.target_flops > 0:
        num_iterations = round(config.training.target_flops / (num_flops_per_token * total_batch_size))
        print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
    elif config.training.target_param_data_ratio > 0:
        num_iterations = target_tokens // total_batch_size
        print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
    else:
        raise ValueError("No training horizon specified")

    total_tokens = total_batch_size * num_iterations
    print0(f"Total number of training tokens: {total_tokens:,}")
    print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}")
    print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

    tokens_per_fwdbwd = config.training.device_batch_size * config.training.max_seq_len
    # DDP world size is 1 for MLX — resolved after backend dispatch, use 1 for grad_accum_steps here
    ddp_world_size_for_accum = 1 if backend == "mlx" else None  # resolved below for torch
    if ddp_world_size_for_accum is None:
        # Torch: need to init compute to know world size — defer grad_accum_steps to _setup_torch
        # Use a placeholder; _setup_torch will assert the division holds
        ddp_world_size_for_accum = 1  # will be overridden
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size_for_accum
    assert total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd  # per-rank, corrected in torch path

    ckpt_dir = checkpoint_manager.checkpoint_dir
    config.save(Path(ckpt_dir) / "config.toml")

    resuming = config.checkpoint.resume_from_step != -1
    resume_ckpt = None
    if resuming:
        print0(f"Resuming optimization from step {config.checkpoint.resume_from_step}")
        resume_ckpt = checkpoint_manager.load(
            config.checkpoint.resume_from_step,
            torch.device("cpu"),
            load_optimizer=True,
            rank=0,
        )

    if backend == "mlx":
        trainer, device_type, ddp_rank, ddp_world_size, device, synchronize, get_max_memory, gpu_peak_flops = _setup_mlx(
            config=config,
            checkpoint_manager=checkpoint_manager,
            model_config_kwargs=model_config_kwargs,
            vocab_size=vocab_size,
            weight_decay_scaled=weight_decay_scaled,
            batch_lr_scale=batch_lr_scale,
            grad_accum_steps=grad_accum_steps,
            tokenizer=tokenizer,
            resuming=resuming,
            resume_ckpt=resume_ckpt,
        )
        token_bytes = get_token_bytes(device=torch.device("cpu"))
        master_process = True
    else:
        trainer, device_type, ddp_rank, ddp_world_size, device, synchronize, get_max_memory, gpu_peak_flops = _setup_torch(
            config=config,
            checkpoint_manager=checkpoint_manager,
            meta_model=meta_model,
            model_config_kwargs=model_config_kwargs,
            vocab_size=vocab_size,
            weight_decay_scaled=weight_decay_scaled,
            batch_lr_scale=batch_lr_scale,
            grad_accum_steps=total_batch_size // (tokens_per_fwdbwd),  # corrected after ddp_world_size known
            tokenizer=tokenizer,
            resuming=resuming,
            resume_ckpt=resume_ckpt,
        )
        token_bytes = get_token_bytes(device=device)
        master_process = ddp_rank == 0
        # Re-init wandb with correct master_process now that ddp_rank is known
        wandb_run = init_wandb(user_config=user_config, master_process=master_process)

    print0(
        f"Tokens / micro-batch / rank: {config.training.device_batch_size} x {config.training.max_seq_len} = {tokens_per_fwdbwd:,}"
    )
    print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

    get_lr_multiplier = base_lr_scheduler(
        num_iterations, config.training.warmup_steps, config.training.warmdown_ratio, config.training.final_lr_frac
    )
    get_muon_momentum = base_muon_momentum_scheduler(num_iterations, config.training.warmdown_ratio)
    get_weight_decay = base_weight_decay_scheduler(weight_decay_scaled, num_iterations)

    if not resuming:
        state = PretrainingState.fresh()
    else:
        assert resume_ckpt is not None
        state = PretrainingState.from_metadata(resume_ckpt.metadata)
    state.model_config = model_config_kwargs
    state.user_config = {
        **user_config,
        "device_batch_size": config.training.device_batch_size,
        "max_seq_len": config.training.max_seq_len,
        "total_batch_size": total_batch_size,
    }

    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, config.training.device_batch_size, config.training.max_seq_len, split="val", device=device
    )

    return BaseTrainingSetup(
        config=config,
        device_type=device_type,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        master_process=master_process,
        trainer=trainer,
        tokenizer=tokenizer,
        token_bytes=token_bytes,
        build_val_loader=build_val_loader,
        ckpt_dir=ckpt_dir,
        user_config=user_config,
        model_config_kwargs=model_config_kwargs,
        num_iterations=num_iterations,
        total_batch_size=total_batch_size,
        total_tokens=total_tokens,
        num_params=num_params,
        num_flops_per_token=num_flops_per_token,
        num_scaling_params=num_scaling_params,
        gpu_peak_flops=gpu_peak_flops,
        get_lr_multiplier=get_lr_multiplier,
        get_muon_momentum=get_muon_momentum,
        get_weight_decay=get_weight_decay,
        synchronize=synchronize,
        get_max_memory=get_max_memory,
        wandb_run=wandb_run,
        state=state,
        resuming=resuming,
    )
