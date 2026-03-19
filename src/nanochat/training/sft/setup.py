from dataclasses import asdict
from typing import Callable

import torch

from nanochat import workspace
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
from nanochat.models.flash_attention import HAS_FA3
from nanochat.tasks.base import TaskMixture
from nanochat.tasks.customjson import CustomJSON
from nanochat.tasks.gsm8k import GSM8K
from nanochat.tasks.mmlu import MMLU
from nanochat.tasks.smoltalk import SmolTalk
from nanochat.tasks.spellingbee import SimpleSpelling, SpellingBee
from nanochat.tokenizer import get_token_bytes
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


def setup(config: Config) -> SFTTrainingSetup:
    """Initialize compute, model, optimizer, dataloaders and schedulers for SFT."""
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
        device=device, phase="base", model_tag=config.common.model_tag, step=config.sft.source_step
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
