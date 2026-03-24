from dataclasses import asdict
from typing import Callable

import torch

from nanochat import workspace
from nanochat.checkpoint import make_checkpoint_manager
from nanochat.checkpoint.compat import patch_missing_config_keys
from nanochat.checkpoint.discovery import find_last_step
from nanochat.checkpoint.logger import RankZeroLogger
from nanochat.common import (
    WandbProtocol,
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
from nanochat.model_factory import load_model_from_dir, load_optimizer_state
from nanochat.models.config import GPTConfig
from nanochat.models.flash_attention import HAS_FA3
from nanochat.tasks.base import TaskMixture
from nanochat.tasks.customjson import CustomJSON
from nanochat.tasks.gsm8k import GSM8K
from nanochat.tasks.mmlu import MMLU
from nanochat.tasks.smoltalk import SmolTalk
from nanochat.tasks.spellingbee import SimpleSpelling, SpellingBee
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from nanochat.training.sft.dataloader import sft_data_generator_bos_bestfit
from nanochat.training.sft.schedulers import sft_lr_scheduler, sft_muon_momentum_scheduler
from nanochat.training.sft.state import SFTState


class SFTTrainingSetup:
    """All resolved setup state passed to sft_train_loop(). Built once by setup().

    Uses __slots__ for faster attribute access — the setup object is accessed on
    every loop iteration (s.model, s.wandb_run, s.config, etc.).
    """

    __slots__ = (
        "config",
        "device_type",
        "ddp_rank",
        "ddp_world_size",
        "ddp",
        "device",
        "master_process",
        "orig_model",
        "model",
        "tokenizer",
        "token_bytes",
        "optimizer",
        "scaler",
        "train_loader",
        "build_val_loader",
        "ckpt_dir",
        "user_config",
        "model_config",
        "num_flops_per_token",
        "gpu_peak_flops",
        "grad_accum_steps",
        "get_lr_multiplier",
        "get_muon_momentum",
        "synchronize",
        "get_max_memory",
        "wandb_run",
        "state",
        "trainer",  # BaseTrainer | None — set for MLX, None for PyTorch path
    )

    def __init__(
        self,
        config: Config,
        device_type: str,
        ddp_rank: int,
        ddp_world_size: int,
        ddp: bool,
        device: torch.device,
        master_process: bool,
        orig_model: object,
        model: object,
        tokenizer: object,
        token_bytes: object,
        optimizer: object,
        scaler: torch.amp.GradScaler | None,
        train_loader: object,
        build_val_loader: Callable[[], object],
        ckpt_dir: str,
        user_config: dict[str, object],
        model_config: dict[str, object],
        num_flops_per_token: float,
        gpu_peak_flops: float,
        grad_accum_steps: int,
        get_lr_multiplier: Callable[[float], float],
        get_muon_momentum: Callable[[int], float],
        synchronize: Callable[[], None],
        get_max_memory: Callable[[], int],
        wandb_run: WandbProtocol,
        state: SFTState,
        trainer: object = None,
    ):
        self.config = config
        self.device_type = device_type
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.ddp = ddp
        self.device = device
        self.master_process = master_process
        self.orig_model = orig_model
        self.model = model
        self.tokenizer = tokenizer
        self.token_bytes = token_bytes
        self.optimizer = optimizer
        self.scaler = scaler
        self.train_loader = train_loader
        self.build_val_loader = build_val_loader
        self.ckpt_dir = ckpt_dir
        self.user_config = user_config
        self.model_config = model_config
        self.num_flops_per_token = num_flops_per_token
        self.gpu_peak_flops = gpu_peak_flops
        self.grad_accum_steps = grad_accum_steps
        self.get_lr_multiplier = get_lr_multiplier
        self.get_muon_momentum = get_muon_momentum
        self.synchronize = synchronize
        self.get_max_memory = get_max_memory
        self.wandb_run = wandb_run
        self.state = state
        self.trainer = trainer


def _build_sft_task_mixture(config: Config, tokenizer: object):
    """Build the SFT task mixture and val dataset (backend-agnostic)."""
    from nanochat import workspace as _ws
    identity_conversations_filepath = _ws.identity_data_path()
    train_tasks = [
        SmolTalk(split="train"),
        CustomJSON(filepath=identity_conversations_filepath),
        CustomJSON(filepath=identity_conversations_filepath),
        *[MMLU(subset="auxiliary_train", split="train") for _ in range(config.sft.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(config.sft.gsm8k_epochs)],
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ]
    train_dataset = TaskMixture(train_tasks)
    val_dataset = TaskMixture(
        [
            SmolTalk(split="test"),
            MMLU(subset="all", split="test", stop=5200),
            GSM8K(subset="main", split="test", stop=420),
        ]
    )
    return train_dataset, val_dataset


def mlx_sft_setup(config: Config) -> "SFTTrainingSetup":
    """Build SFT training state for the MLX backend (Apple Silicon)."""
    from nanochat.common import get_mlx_compute_dtype, get_mlx_device_info, mlx_compute_init
    from nanochat.training.base.mlx_trainer import MLXTrainer
    from nanochat.training.mlx_optimizer import MuonAdamW, build_param_groups

    mlx_compute_init()
    info = get_mlx_device_info()
    print0(
        f"MLX device: {info['device_name']} | RAM: {info['memory_size'] / 1024**3:.0f}GB | arch: {info['architecture']}"
    )
    print0("✓ MLX backend — single device, mx.compile, unified memory")

    # --- Load base checkpoint (model weights, metadata) ---
    _logger = RankZeroLogger(__name__)
    ckpt_dir_base = workspace.checkpoint_dir("base", config.common.model_tag if config.common.model_tag else None)
    manager = make_checkpoint_manager(ckpt_dir_base, config.checkpoint, _logger)
    source_step = config.sft.source_step
    if source_step is None:
        source_step = find_last_step(ckpt_dir_base)
    print0(f"Loading base checkpoint from {ckpt_dir_base} step {source_step}")
    ckpt = manager.load(source_step, torch.device("cpu"), load_optimizer=False)
    metadata = ckpt.metadata
    meta_dict = metadata.to_dict()

    # --- Inherit hyperparameters from checkpoint metadata ---
    pretrain_user_config = meta_dict.get("user_config", {})
    for name, fallback, source in [
        ("max_seq_len", 2048, meta_dict),
        ("device_batch_size", 32, meta_dict),
        ("total_batch_size", 524288, meta_dict),
        ("embedding_lr", 0.3, pretrain_user_config),
        ("unembedding_lr", 0.004, pretrain_user_config),
        ("matrix_lr", 0.02, pretrain_user_config),
    ]:
        arg_val = getattr(config.sft, name)
        pretrain_val = source.get(name) if isinstance(source, dict) else None
        if arg_val is None:
            resolved = pretrain_val if pretrain_val is not None else fallback
            setattr(config.sft, name, resolved)
            print0(f"Inherited {name}={resolved} from pretrained checkpoint")
        elif pretrain_val is not None and arg_val != pretrain_val:
            print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
        else:
            print0(f"Using {name}={arg_val}")

    # --- Build MLX model ---
    model_config_kwargs = dict(metadata.model_config)
    patch_missing_config_keys(model_config_kwargs, _logger.info)
    gpt_config = GPTConfig(**model_config_kwargs)

    from nanochat.models.mlx_gpt import GPT as MLXGPT
    model = MLXGPT(gpt_config)
    compute_dtype = get_mlx_compute_dtype()
    print0(f"COMPUTE_DTYPE: {compute_dtype} (MLX)")
    model.set_dtype(compute_dtype)

    # --- Resolve derived quantities ---
    depth = gpt_config.n_layer
    tokens_per_fwdbwd = config.sft.device_batch_size * config.sft.max_seq_len
    grad_accum_steps = config.sft.total_batch_size // tokens_per_fwdbwd
    print0(
        f"Tokens / micro-batch: {config.sft.device_batch_size} x {config.sft.max_seq_len} = {tokens_per_fwdbwd:,}"
    )
    print0(f"Total batch size {config.sft.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

    # --- Build optimizer (apply init_lr_frac before MLXTrainer so initial_lr is scaled) ---
    optimizer = MuonAdamW(build_param_groups(
        model,
        unembedding_lr=config.sft.unembedding_lr,
        embedding_lr=config.sft.embedding_lr,
        matrix_lr=config.sft.matrix_lr,
        weight_decay=0.0,
    ))
    for group in optimizer._groups:
        group["lr"] = group["lr"] * config.sft.init_lr_frac

    # --- Build tokenizer and task mixture ---
    tokenizer = get_tokenizer()
    user_config = asdict(config)
    train_dataset, val_dataset = _build_sft_task_mixture(config, tokenizer)
    print0(
        f"Training mixture: {len(train_dataset):,} rows "
        f"(MMLU x{config.sft.mmlu_epochs}, GSM8K x{config.sft.gsm8k_epochs})"
    )

    state = SFTState.fresh()

    ckpt_dir_name = config.common.model_tag if config.common.model_tag else f"d{depth}"
    ckpt_dir_sft = workspace.checkpoint_dir("sft", ckpt_dir_name)

    model_config = {
        "sequence_len": config.sft.max_seq_len,
        "vocab_size": tokenizer.get_vocab_size(),
        "n_layer": depth,
        "n_head": gpt_config.n_head,
        "n_kv_head": gpt_config.n_kv_head,
        "n_embd": gpt_config.n_embd,
        "window_pattern": gpt_config.window_pattern,
    }
    state.model_config = model_config
    state.user_config = user_config

    def make_loader(split: str):
        return sft_data_generator_bos_bestfit(
            state=state,
            config=config,
            tokenizer=tokenizer,
            ddp_rank=0,
            ddp_world_size=1,
            device=torch.device("cpu"),
            device_type="mlx",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            split=split,
        )

    train_loader = make_loader("train")
    build_val_loader = lambda: make_loader("val")

    # --- Build MLXTrainer and load base checkpoint weights ---
    trainer = MLXTrainer(model, optimizer, grad_accum_steps, train_loader)
    model_state = {k.removeprefix("_orig_mod."): v for k, v in ckpt.model_state.items()}
    # Load model weights only; optimizer starts fresh (base checkpoint was likely PyTorch)
    import mlx.nn as mlx_nn
    from nanochat.checkpoint.convert import from_numpy_mlx
    import numpy as np
    if any(isinstance(v, np.ndarray) for v in model_state.values()):
        mlx_state = from_numpy_mlx(model_state)
    else:
        mlx_state = model_state
    current_dtypes = {k: v.dtype for k, v in mlx_nn.utils.tree_flatten(model.parameters())}
    mlx_state = {k: v.astype(current_dtypes[k]) if k in current_dtypes else v for k, v in mlx_state.items()}
    model.update(mlx_nn.utils.tree_unflatten(list(mlx_state.items())))
    import mlx.core as mx
    mx.eval(model.parameters())
    print0(f"Loaded base checkpoint weights (step {source_step}) into MLX model")

    if config.sft.load_optimizer:
        print0("WARNING: MLX SFT optimizer warm-start from PyTorch base checkpoint not supported; starting fresh")

    get_lr_multiplier = sft_lr_scheduler(config.sft.warmup_ratio, config.sft.warmdown_ratio, config.sft.final_lr_frac)
    get_muon_momentum = sft_muon_momentum_scheduler()

    wandb_run = init_wandb(user_config=user_config, master_process=True, project_suffix="sft")

    token_bytes = get_token_bytes(device="cpu")

    return SFTTrainingSetup(
        config=config,
        device_type="mlx",
        ddp_rank=0,
        ddp_world_size=1,
        ddp=False,
        device=torch.device("cpu"),
        master_process=True,
        orig_model=model,
        model=None,          # unused in MLX path (loop uses trainer)
        tokenizer=tokenizer,
        token_bytes=token_bytes,
        optimizer=None,      # unused in MLX path (trainer owns optimizer)
        scaler=None,
        train_loader=None,   # unused in MLX path (trainer owns loader)
        build_val_loader=build_val_loader,
        ckpt_dir=ckpt_dir_sft,
        user_config=user_config,
        model_config=model_config,
        num_flops_per_token=float("inf"),
        gpu_peak_flops=float("inf"),
        grad_accum_steps=grad_accum_steps,
        get_lr_multiplier=get_lr_multiplier,
        get_muon_momentum=get_muon_momentum,
        synchronize=lambda: None,
        get_max_memory=lambda: 0,
        wandb_run=wandb_run,
        state=state,
        trainer=trainer,
    )


def setup(config: Config) -> SFTTrainingSetup:
    """Initialize compute, model, optimizer, dataloaders and schedulers for SFT."""
    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    if device_type == "mlx":
        return mlx_sft_setup(config)
    return _torch_sft_setup(config)


def _torch_sft_setup(config: Config) -> SFTTrainingSetup:
    """PyTorch SFT setup (CUDA/CPU/MPS)."""
    print_banner()

    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    ddp, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    synchronize, get_max_memory = get_device_sync(device_type)
    print0(f"COMPUTE_DTYPE: {get_compute_dtype()} ({get_compute_dtype_reason()})")
    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float("inf")

    user_config = asdict(config)
    master_process = ddp_rank == 0
    wandb_run = init_wandb(user_config=user_config, master_process=master_process, project_suffix="sft")

    if not HAS_FA3:
        print0(
            "WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient."
        )

    model, tokenizer, meta = load_model_from_dir(
        device=device, phase="base", config=config.checkpoint, model_tag=config.common.model_tag, step=config.sft.source_step
    )

    # Inherit training hyperparameters from pretrained checkpoint
    pretrain_user_config = meta.get("user_config", {})
    for name, fallback, source in [
        ("max_seq_len", 2048, meta),
        ("device_batch_size", 32, meta),
        ("total_batch_size", 524288, meta),
        ("embedding_lr", 0.3, pretrain_user_config),
        ("unembedding_lr", 0.004, pretrain_user_config),
        ("matrix_lr", 0.02, pretrain_user_config),
    ]:
        arg_val = getattr(config.sft, name)
        pretrain_val = source.get(name)
        if arg_val is None:
            resolved = pretrain_val if pretrain_val is not None else fallback
            setattr(config.sft, name, resolved)
            print0(f"Inherited {name}={resolved} from pretrained checkpoint")
        elif pretrain_val is not None and arg_val != pretrain_val:
            print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
        else:
            print0(f"Using {name}={arg_val}")

    orig_model = model
    model = torch.compile(model, dynamic=False)
    depth = model.config.n_layer
    num_flops_per_token = model.estimate_flops()
    tokens_per_fwdbwd = config.sft.device_batch_size * config.sft.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert config.sft.total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = config.sft.total_batch_size // world_tokens_per_fwdbwd
    print0(
        f"Tokens / micro-batch / rank: {config.sft.device_batch_size} x {config.sft.max_seq_len} = {tokens_per_fwdbwd:,}"
    )
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(f"Total batch size {config.sft.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
    token_bytes = get_token_bytes(device=device)

    optimizer = model.setup_optimizer(
        unembedding_lr=config.sft.unembedding_lr,
        embedding_lr=config.sft.embedding_lr,
        matrix_lr=config.sft.matrix_lr,
        weight_decay=0.0,
    )

    if config.sft.load_optimizer:
        optimizer_data = load_optimizer_state(
            source="base",
            device=device,
            rank=ddp_rank,
            config=config.checkpoint,
            model_tag=config.common.model_tag,
            step=config.sft.source_step,
        )
        if optimizer_data is not None:
            base_lrs = [group["lr"] for group in optimizer.param_groups]
            optimizer.load_state_dict(optimizer_data)
            del optimizer_data
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr
            print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
        else:
            print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

    scaler = torch.amp.GradScaler(device=device_type) if get_compute_dtype() == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")

    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * config.sft.init_lr_frac
        group["initial_lr"] = group["lr"]

    identity_conversations_filepath = workspace.identity_data_path()
    train_tasks = [
        SmolTalk(split="train"),
        CustomJSON(filepath=identity_conversations_filepath),
        CustomJSON(filepath=identity_conversations_filepath),
        *[MMLU(subset="auxiliary_train", split="train") for _ in range(config.sft.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(config.sft.gsm8k_epochs)],
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ]
    train_dataset = TaskMixture(train_tasks)
    print0(
        f"Training mixture: {len(train_dataset):,} rows (MMLU x{config.sft.mmlu_epochs}, GSM8K x{config.sft.gsm8k_epochs})"
    )
    val_dataset = TaskMixture(
        [
            SmolTalk(split="test"),
            MMLU(subset="all", split="test", stop=5200),
            GSM8K(subset="main", split="test", stop=420),
        ]
    )

    state = SFTState.fresh()

    ckpt_dir_name = config.common.model_tag if config.common.model_tag else f"d{depth}"
    ckpt_dir = workspace.checkpoint_dir("sft", ckpt_dir_name)

    model_config = {
        "sequence_len": config.sft.max_seq_len,
        "vocab_size": tokenizer.get_vocab_size(),
        "n_layer": depth,
        "n_head": model.config.n_head,
        "n_kv_head": model.config.n_kv_head,
        "n_embd": model.config.n_embd,
        "window_pattern": model.config.window_pattern,
    }
    state.model_config = model_config
    state.user_config = user_config

    def make_loader(split: str):
        return sft_data_generator_bos_bestfit(
            state=state,
            config=config,
            tokenizer=tokenizer,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            device=device,
            device_type=device_type,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            split=split,
        )

    train_loader = make_loader("train")
    build_val_loader = lambda: make_loader("val")

    get_lr_multiplier = sft_lr_scheduler(config.sft.warmup_ratio, config.sft.warmdown_ratio, config.sft.final_lr_frac)
    get_muon_momentum = sft_muon_momentum_scheduler()

    return SFTTrainingSetup(
        config=config,
        device_type=device_type,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        ddp=ddp,
        device=device,
        master_process=master_process,
        orig_model=orig_model,
        model=model,
        tokenizer=tokenizer,
        token_bytes=token_bytes,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        build_val_loader=build_val_loader,
        ckpt_dir=ckpt_dir,
        user_config=user_config,
        model_config=model_config,
        num_flops_per_token=num_flops_per_token,
        gpu_peak_flops=gpu_peak_flops,
        grad_accum_steps=grad_accum_steps,
        get_lr_multiplier=get_lr_multiplier,
        get_muon_momentum=get_muon_momentum,
        synchronize=synchronize,
        get_max_memory=get_max_memory,
        wandb_run=wandb_run,
        state=state,
    )
