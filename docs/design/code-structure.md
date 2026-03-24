---
title: "Code Structure"
summary: "Package map, responsibilities, and key cross-package flows for the nanochat codebase."
read_when:
  - Onboarding to the codebase
  - Looking for where to add a feature or fix a bug
  - Understanding how packages relate to each other
status: active
last_updated: "2025-07-24"
---

# Code Structure

All source code lives under `src/nanochat/`. Each package has a single responsibility and a clean `__init__.py` that defines its public API.

## Package Map

```
src/nanochat/
├── cli.py            # Subcommand dispatch — entry point for the nanochat CLI
├── __main__.py       # Delegates to cli.main (enables python -m nanochat)
├── workspace.py      # Module-level path store — owns all paths under NANOCHAT_BASE_DIR
├── model_factory.py  # Model construction and loading — shared by training/, evaluation/, chat/
├── execution.py      # Sandboxed Python code execution (used by HumanEval)
│
├── config/           # Configuration dataclasses, TOML loading, CLI arg parsing
├── common/           # Shared utilities: dtype, distributed, wandb, I/O
├── checkpoint/       # CheckpointManager protocol, TorchCheckpointManager, discovery, compat
├── models/           # GPT model architecture
├── tokenizer/        # BPE tokenizer: training, inference, evaluation
├── dataset/          # ClimbMix dataset download and shard iteration
├── tasks/            # Evaluation task definitions (MMLU, ARC, GSM8K, ...)
├── training/         # Training loops, optimizer, scheduler
├── evaluation/       # Evaluation engine and eval entry points
├── chat/             # Interactive CLI and web server
└── report/           # Training report generation
```

## Package Responsibilities

### `workspace.py`
Module-level path store initialized once at startup (after config). Owns all filesystem paths under `NANOCHAT_BASE_DIR` — replaces the old `common/paths.py`. Call `workspace.init()` once; use `workspace.data_dir()`, `workspace.checkpoint_dir(phase, tag)`, etc. everywhere. See [data-layout.md](data-layout.md).

### `model_factory.py`
Model construction and loading shared by `training/`, `evaluation/`, and `chat/`. Keeps model-building logic out of the checkpoint package and avoids cross-package dependencies from `chat/` into `training/`.

| Function               | Responsibility                                                              |
| ---------------------- | --------------------------------------------------------------------------- |
| `load_model_from_dir`  | Load a GPT model + tokenizer from a checkpoint directory                    |
| `load_optimizer_state` | Load just the optimizer shard for a given rank, without re-loading the model |

### `config/`
Dataclasses for each training mode (`CommonConfig`, `TrainingConfig`, `SFTConfig`, `RLConfig`, `EvaluationConfig`, `CheckpointConfig`, `TokenizerConfig`) plus `Config` (the aggregate) and `ConfigLoader` (TOML + CLI resolution). See [configuration.md](configuration.md).

Key files: `loader.py`, `config.py`, `common.py`, `checkpoint.py`, `cli.py`

### `checkpoint/`
Checkpoint I/O abstracted behind a `CheckpointManager` protocol. Two implementations: `TorchCheckpointManager` (`.pt`) for single-backend runs, `SafetensorsCheckpointManager` (`.safetensors`) for cross-backend interop. See [checkpoint-interop.md](checkpoint-interop.md).

| Module                      | Responsibility                                                              |
| --------------------------- | --------------------------------------------------------------------------- |
| `protocol.py`               | `CheckpointManager`, `CheckpointStateProtocol`, `Checkpoint`, `CheckpointMetadata`, `LoopState` |
| `torch_manager.py`          | `TorchCheckpointManager` — `.pt` + JSON, `keep_last_n` pruning              |
| `safetensors_manager.py`    | `SafetensorsCheckpointManager` — `.safetensors` + JSON, cross-backend interop |
| `convert.py`                | `to_numpy`, `from_numpy_torch`, `from_numpy_mlx` — array conversion boundary |
| `factory.py`                | `make_checkpoint_manager` — selects implementation from `CheckpointConfig.format` |
| `logger.py`                 | `CheckpointLogger`, `RankZeroLogger`, `SilentLogger`                        |
| `discovery.py`              | `find_largest_model`, `find_last_step` — supports `.pt` and `.safetensors`  |
| `compat.py`                 | `patch_missing_config_keys`, `patch_missing_keys` — old checkpoint patching |

### `common/`
Shared utilities used across all packages. Nothing in `common/` imports from other nanochat packages (except `wandb.py` which reads from `config/`).

| Module           | Responsibility                                                   |
| ---------------- | ---------------------------------------------------------------- |
| `distributed.py` | DDP init/cleanup, device autodetection                           |
| `dtype.py`       | Compute dtype selection (bf16/fp16/fp32) per device              |
| `hardware.py`    | Peak FLOPS, device sync                                          |
| `wandb.py`       | `LocalWandb`, `DummyWandb`, `init_wandb`                         |
| `io.py`          | File download with locking, `print0`                             |
| `logging.py`     | Colored log formatter                                            |

### `models/`
GPT model definitions. `GPTConfig` holds architecture hyperparameters (depth, dim, heads) and is shared by both backends.

| File | Responsibility |
| --- | --- |
| `gpt.py` | PyTorch GPT — `torch.compile`, FA3/SDPA, FP8, DDP |
| `mlx_gpt.py` | MLX GPT — `mx.fast.sdpa`, sliding window masks, unified memory |
| `config.py` | `GPTConfig` — shared by both backends |
| `attention.py` | FA3 / SDPA fallback logic for PyTorch |

### `tokenizer/`
Custom Rust BPE tokenizer (`RustBPETokenizer`) and a HuggingFace wrapper (`HuggingFaceTokenizer`) for evaluation against external models. `tokenizer_train` trains from ClimbMix data; `tokenizer_eval` measures compression.

Key files: `rust_tokenizer.py`, `train.py`, `eval.py`, `utils.py`

### `dataset/`
ClimbMix-400B dataset: shard download (`climbmix_download`) and Parquet iteration utilities. The last shard is always the validation split.

Key files: `climbmix.py`, `utils.py`

### `tasks/`
Evaluation task definitions. Each task subclasses `Task` and implements prompt formatting and scoring. Used by both `training/` (inline eval during training) and `evaluation/` (standalone eval runs).

Key files: `base.py`, `mmlu.py`, `arc.py`, `gsm8k.py`, `humaneval.py`, `smoltalk.py`

### `training/`
Training loops and supporting infrastructure.

| Module                   | Responsibility                                                       |
| ------------------------ | -------------------------------------------------------------------- |
| `base/`                  | Base model pretraining — `PretrainingState`, setup, loop             |
| `base/trainer.py`        | `BaseTrainer` protocol, `StepResult`, `TorchTrainer`                 |
| `base/loop.py`           | Backend-agnostic training loop — calls only protocol methods         |
| `base/setup.py`          | Backend dispatch — constructs `TorchTrainer` or `MLXTrainer`         |
| `sft/`                   | Supervised fine-tuning — `SFTState`, dataloader, setup, loop         |
| `rl/`                    | GRPO reinforcement learning — `RLState`, rollout, eval, loop         |
| `optimizer.py`           | `MuonAdamW` and `DistMuonAdamW` — PyTorch optimizers                 |
| `mlx_optimizer.py`       | `MuonAdamW` — MLX optimizer, see [mlx-muon-design.md](mlx-muon-design.md) |
| `base/mlx_trainer.py`    | `MLXTrainer` — `BaseTrainer` implementation for Apple Silicon        |
| `scaling.py`             | Scaling law utilities (compute-optimal step count, total batch size) |
| `dataloader.py`          | Distributed token dataloader over ClimbMix shards                    |
| `compression_math.py`    | Pure numpy compression functions — backend-agnostic                  |
| `compression_metrics.py` | Stateful compression tracker, delegates to `compression_math`        |

### `evaluation/`
Model evaluation infrastructure.

| Module              | Responsibility                                        |
| ------------------- | ----------------------------------------------------- |
| `engine.py`         | `Engine` — autoregressive generation with KV cache    |
| `loss_eval.py`      | BPB (bits-per-byte) evaluation over validation shards |
| `core_eval.py`      | Single-task evaluation runner                         |
| `core_benchmark.py` | Multi-task CORE benchmark runner                      |
| `base/`             | Base model evaluation — `BaseEvalResult`, loop        |
| `chat/`             | Chat model evaluation — `ChatEvalResult`, loop        |
| `hf_model.py`       | HuggingFace model wrapper for cross-model comparison  |

### `chat/`
User-facing interfaces. Both only need `CommonConfig` — no training config.

- `chat_cli.py` — interactive terminal chat
- `chat_web.py` + `server/` — FastAPI web server with worker pool

### `report/`
Training report generation and management (`nanochat report generate/reset`).

### `execution.py`
Sandboxed Python code execution for HumanEval scoring. Runs generated code in a subprocess with timeout, memory limits, and disabled destructive syscalls.

## Key Flows

### CLI → training

```
nanochat train base --depth 12
  └── cli.main()
        └── ConfigLoader().add_training().resolve(args) → Config
              └── current.init(config) + workspace.init()
                    └── train_base(config)                  # training/base/__init__.py
                          ├── make_checkpoint_manager()     # checkpoint/factory.py
                          ├── setup(config, manager)        # training/base/setup.py
                          │     ├── autodetect_backend()    # common/distributed.py
                          │     ├── get_tokenizer()         # tokenizer/utils.py
                          │     ├── [torch] compute_init()  # common/distributed.py
                          │     │         GPT + TorchTrainer
                          │     └── [mlx]  MLXGPT + MLXTrainer
                          └── train_loop(s, manager)        # training/base/loop.py
                                ├── s.trainer.eval_context()
                                ├── s.trainer.forward_backward()
                                ├── s.trainer.step()
                                └── manager.save()         # checkpoint/{torch,safetensors}_manager.py
```

### Config resolution

```
ConfigLoader().add_training().parse(["--depth", "12"])
  1. dataclass defaults (TrainingConfig.depth = 20)
  2. TOML file values   (depth = 15, if config.toml present)
  3. CLI flags          (--depth 12  → depth = 12)
  └── Config with merged values
```

### Evaluation engine

```
Engine(model, tokenizer, device)
  └── engine.generate(prompt_tokens, max_new_tokens)
        └── KVCache — stores past key/value tensors
              └── model.forward() called one token at a time
```

## Dependency Rules

- `common/` has no intra-nanochat imports
- `config/` has no intra-nanochat imports
- `workspace.py` imports only from `config/`
- `checkpoint/` imports only from `common/`, `config/`, `workspace` (convert.py also imports `torch`; `from_numpy_mlx` lazily imports `mlx`)
- `models/` imports only from `common/`
- `tasks/` imports only from `common/`
- `tokenizer/` imports from `common/`, `workspace`
- `dataset/` imports from `common/`, `workspace`
- `model_factory.py` imports from `checkpoint/`, `config/`, `models/`, `tokenizer/`, `workspace`
- `training/base/`, `training/sft/`, `training/rl/` import from `common/`, `workspace`, `checkpoint/`, `model_factory`, `models/`, `tokenizer/`, `dataset/`, `tasks/`, `evaluation/`
- `training/base/mlx_trainer.py` additionally imports from `mlx` (Darwin only)
- `evaluation/base/`, `evaluation/chat/` import from `common/`, `workspace`, `model_factory`, `models/`, `tokenizer/`, `tasks/`
- `chat/` imports from `common/`, `config/`, `model_factory`, `evaluation/`, `tokenizer/`
- `cli.py` imports from all packages (top-level wiring only)
