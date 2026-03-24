from collections.abc import Callable
from dataclasses import asdict

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
    init_wandb,
    print0,
)
from nanochat.config import Config
from nanochat.evaluation.engine import Engine
from nanochat.model_factory import load_model_from_dir
from nanochat.models.config import GPTConfig
from nanochat.tasks.gsm8k import GSM8K
from nanochat.tokenizer import get_tokenizer
from nanochat.training.rl.schedulers import rl_lr_scheduler
from nanochat.training.rl.state import RLState


class RLTrainingSetup:
    """All resolved setup state passed to rl_train_loop(). Built once by setup().

    Uses __slots__ for faster attribute access on every loop iteration.
    """

    __slots__ = (
        "config",
        "ddp",
        "ddp_rank",
        "ddp_world_size",
        "device",
        "master_process",
        "model",
        "tokenizer",
        "engine",
        "train_task",
        "val_task",
        "optimizer",
        "get_lr_multiplier",
        "num_steps",
        "examples_per_rank",
        "user_config",
        "wandb_run",
        "state",
        "ckpt_dir",
        "trainer",  # MLXRLTrainer | None — set for MLX, None for PyTorch path
    )

    def __init__(
        self,
        config: Config,
        ddp: bool,
        ddp_rank: int,
        ddp_world_size: int,
        device: torch.device,
        master_process: bool,
        model: object,
        tokenizer: object,
        engine: Engine,
        train_task: GSM8K,
        val_task: GSM8K,
        optimizer: object,
        get_lr_multiplier: Callable[[int], float],
        num_steps: int,
        examples_per_rank: int,
        user_config: dict[str, object],
        wandb_run: WandbProtocol,
        state: RLState,
        ckpt_dir: str,
        trainer: object = None,
    ):
        self.config = config
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.device = device
        self.master_process = master_process
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.train_task = train_task
        self.val_task = val_task
        self.optimizer = optimizer
        self.get_lr_multiplier = get_lr_multiplier
        self.num_steps = num_steps
        self.examples_per_rank = examples_per_rank
        self.user_config = user_config
        self.wandb_run = wandb_run
        self.state = state
        self.ckpt_dir = ckpt_dir
        self.trainer = trainer


def mlx_rl_setup(config: Config) -> RLTrainingSetup:
    """Build RL training state for the MLX backend (Apple Silicon)."""
    from nanochat.common import get_mlx_compute_dtype, get_mlx_device_info, mlx_compute_init
    from nanochat.evaluation.mlx_engine import MLXEngine
    from nanochat.training.mlx_optimizer import MuonAdamW, build_param_groups
    from nanochat.training.rl.mlx_rl_rollout import get_batch_mlx
    from nanochat.training.rl.mlx_rl_trainer import MLXRLTrainer

    mlx_compute_init()
    info = get_mlx_device_info()
    print0(
        f"MLX device: {info['device_name']} | RAM: {info['memory_size'] / 1024**3:.0f}GB | arch: {info['architecture']}"
    )
    print0("✓ MLX RL backend — single device, REINFORCE, no DDP")

    # --- Load SFT checkpoint ---
    _logger = RankZeroLogger(__name__)
    ckpt_dir_sft = workspace.checkpoint_dir("sft", config.common.model_tag if config.common.model_tag else None)
    manager = make_checkpoint_manager(ckpt_dir_sft, config.checkpoint, _logger)
    source_step = config.rl.source_step
    if source_step is None:
        source_step = find_last_step(ckpt_dir_sft)
    print0(f"Loading SFT checkpoint from {ckpt_dir_sft} step {source_step}")
    ckpt = manager.load(source_step, torch.device("cpu"), load_optimizer=False)
    metadata = ckpt.metadata

    # --- Build MLX model ---
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import numpy as np
    from nanochat.checkpoint.convert import from_numpy_mlx
    from nanochat.models.mlx_gpt import GPT as MLXGPT

    model_config_kwargs = dict(metadata.model_config)
    patch_missing_config_keys(model_config_kwargs, _logger.info)
    gpt_config = GPTConfig(**model_config_kwargs)
    model = MLXGPT(gpt_config)
    compute_dtype = get_mlx_compute_dtype()
    print0(f"COMPUTE_DTYPE: {compute_dtype} (MLX)")
    model.set_dtype(compute_dtype)

    # Load SFT weights
    model_state = {k.removeprefix("_orig_mod."): v for k, v in ckpt.model_state.items()}
    if any(isinstance(v, np.ndarray) for v in model_state.values()):
        mlx_state = from_numpy_mlx(model_state)
    else:
        mlx_state = model_state
    current_dtypes = {k: v.dtype for k, v in mlx_nn.utils.tree_flatten(model.parameters())}
    mlx_state = {k: v.astype(current_dtypes[k]) if k in current_dtypes else v for k, v in mlx_state.items()}
    model.update(mlx_nn.utils.tree_unflatten(list(mlx_state.items())))
    mx.eval(model.parameters())
    print0(f"Loaded SFT checkpoint weights (step {source_step}) into MLX model")

    # --- Build tokenizer and tasks ---
    tokenizer = get_tokenizer()
    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
    num_steps = (len(train_task) // config.rl.examples_per_step) * config.rl.num_epochs
    examples_per_rank = config.rl.examples_per_step  # no DDP rank division
    print0(f"Calculated number of steps: {num_steps}")
    print0(f"Examples per rank (no DDP): {examples_per_rank}")

    # --- Build optimizer ---
    optimizer = MuonAdamW(build_param_groups(
        model,
        unembedding_lr=config.rl.unembedding_lr,
        embedding_lr=config.rl.embedding_lr,
        matrix_lr=config.rl.matrix_lr,
        weight_decay=config.rl.weight_decay,
    ))
    for group in optimizer._groups:
        group["lr"] = group["lr"] * config.rl.init_lr_frac

    # --- Build MLXEngine ---
    mlx_engine = MLXEngine(model, tokenizer)

    # --- Build rollout iterator ---
    # max_len: pad to fixed shape for mx.compile static shapes
    # 512 is a conservative upper bound for GSM8K prompt prefixes
    max_prefix_len = 512
    max_len = config.rl.max_new_tokens + max_prefix_len

    depth = gpt_config.n_layer
    state = RLState.fresh()
    user_config = asdict(config)
    state.model_config = dict(model_config_kwargs)
    state.user_config = user_config

    batch_iterator = get_batch_mlx(
        state=state,
        config=config,
        train_task=train_task,
        mlx_engine=mlx_engine,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    # --- Build MLXRLTrainer ---
    trainer = MLXRLTrainer(
        orig_model=model,
        optimizer=optimizer,
        batch_iterator=batch_iterator,
        device_batch_size=config.rl.device_batch_size,
        examples_per_rank=examples_per_rank,
    )

    get_lr_multiplier = rl_lr_scheduler(num_steps)
    ckpt_dir_rl = workspace.checkpoint_dir("rl", config.common.model_tag or f"d{depth}")

    user_config_log = asdict(config)
    wandb_run = init_wandb(user_config=user_config_log, master_process=True, project_suffix="rl")

    return RLTrainingSetup(
        config=config,
        ddp=False,
        ddp_rank=0,
        ddp_world_size=1,
        device=torch.device("cpu"),
        master_process=True,
        model=model,
        tokenizer=tokenizer,
        engine=mlx_engine,   # type: ignore[arg-type]
        train_task=train_task,
        val_task=val_task,
        optimizer=None,      # unused in MLX path (trainer owns optimizer)
        get_lr_multiplier=get_lr_multiplier,
        num_steps=num_steps,
        examples_per_rank=examples_per_rank,
        user_config=user_config_log,
        wandb_run=wandb_run,
        state=state,
        ckpt_dir=ckpt_dir_rl,
        trainer=trainer,
    )


def setup(config: Config) -> RLTrainingSetup:
    """Initialize compute, model, optimizer, tasks and scheduler for RL."""
    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    if device_type == "mlx":
        return mlx_rl_setup(config)
    return _torch_rl_setup(config)


def _torch_rl_setup(config: Config) -> RLTrainingSetup:
    """PyTorch RL setup (CUDA/CPU/MPS)."""
    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    ddp, ddp_rank, _, ddp_world_size, device = compute_init(device_type)

    user_config = asdict(config)
    master_process = ddp_rank == 0
    wandb_run = init_wandb(user_config=user_config, master_process=master_process, project_suffix="rl")

    model, tokenizer, _ = load_model_from_dir(
        phase="sft",
        device=device,
        config=config.checkpoint,
        model_tag=config.common.model_tag,
        step=config.rl.source_step,
    )
    engine = Engine(model, tokenizer)

    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
    num_steps = (len(train_task) // config.rl.examples_per_step) * config.rl.num_epochs
    print0(f"Calculated number of steps: {num_steps}")

    optimizer = model.setup_optimizer(
        unembedding_lr=config.rl.unembedding_lr,
        embedding_lr=config.rl.embedding_lr,
        matrix_lr=config.rl.matrix_lr,
        weight_decay=config.rl.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * config.rl.init_lr_frac
        group["initial_lr"] = group["lr"]

    get_lr_multiplier = rl_lr_scheduler(num_steps)

    print0(f"Total sequences per step: {config.rl.examples_per_step * config.rl.num_samples}")
    assert config.rl.examples_per_step % ddp_world_size == 0, (
        "Desired examples per step must be divisible by the number of ranks"
    )
    examples_per_rank = config.rl.examples_per_step // ddp_world_size
    print0(f"Calculated examples per rank: {examples_per_rank}")

    depth = model.config.n_layer
    ckpt_dir = workspace.checkpoint_dir("rl", config.common.model_tag or f"d{depth}")

    state = RLState.fresh()
    state.model_config = model.config.__dict__
    state.user_config = user_config

    return RLTrainingSetup(
        config=config,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        master_process=master_process,
        model=model,
        tokenizer=tokenizer,
        engine=engine,
        train_task=train_task,
        val_task=val_task,
        optimizer=optimizer,
        get_lr_multiplier=get_lr_multiplier,
        num_steps=num_steps,
        examples_per_rank=examples_per_rank,
        user_config=user_config,
        wandb_run=wandb_run,
        state=state,
        ckpt_dir=ckpt_dir,
    )
