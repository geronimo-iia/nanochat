from collections.abc import Callable
from dataclasses import asdict

import torch

from nanochat import workspace
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
from nanochat.tasks.gsm8k import GSM8K
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


def setup(config: Config) -> RLTrainingSetup:
    """Initialize compute, model, optimizer, tasks and scheduler for RL."""
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
