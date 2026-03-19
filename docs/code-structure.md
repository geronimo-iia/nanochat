---
title: "Code Structure"
summary: "Package map, responsibilities, and key cross-package flows for the nanochat codebase."
read_when:
  - Onboarding to the codebase
  - Looking for where to add a feature or fix a bug
  - Understanding how packages relate to each other
status: active
last_updated: "2025-07-16"
---

# Code Structure

All source code lives under `src/nanochat/`. Each package has a single responsibility and a clean `__init__.py` that defines its public API.

## Package Map

```
src/nanochat/
├── cli.py            # Subcommand dispatch — entry point for the nanochat CLI
├── __main__.py       # Delegates to cli.main (enables python -m nanochat)
├── workspace.py      # Module-level path store — owns all paths under NANOCHAT_BASE_DIR
├── execution.py      # Sandboxed Python code execution (used by HumanEval)
│
├── config/           # Configuration dataclasses, TOML loading, CLI arg parsing
├── common/           # Shared utilities: dtype, distributed, wandb, I/O
├── models/           # GPT model architecture
├── tokenizer/        # BPE tokenizer: training, inference, evaluation
├── dataset/          # ClimbMix dataset download and shard iteration
├── tasks/            # Evaluation task definitions (MMLU, ARC, GSM8K, ...)
├── training/         # Training loops, optimizer, scheduler, checkpoint
├── evaluation/       # Evaluation engine and eval entry points
├── chat/             # Interactive CLI and web server
└── report/           # Training report generation
```

## Package Responsibilities

### `workspace.py`
Module-level path store initialized once at startup (after config). Owns all filesystem paths under `NANOCHAT_BASE_DIR` — replaces the old `common/paths.py`. Call `workspace.init()` once; use `workspace.data_dir()`, `workspace.checkpoint_dir(phase, tag)`, etc. everywhere. See [data-layout.md](data-layout.md).

### `config/`
Dataclasses for each training mode (`CommonConfig`, `TrainingConfig`, `SFTConfig`, `RLConfig`, `EvaluationConfig`, `TokenizerConfig`) plus `Config` (the aggregate) and `ConfigLoader` (TOML + CLI resolution). See [configuration.md](configuration.md).

Key files: `loader.py`, `config.py`, `common.py`, `cli.py`

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
GPT model definition. `GPTConfig` holds architecture hyperparameters (depth, dim, heads). `GPT` is the model class. `CausalSelfAttention` falls back to SDPA when Flash Attention 3 is unavailable.

Key files: `gpt.py`, `config.py`, `attention.py`

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
| `sft/`                   | Supervised fine-tuning — `SFTState`, dataloader, setup, loop         |
| `rl/`                    | GRPO reinforcement learning — `RLState`, rollout, eval, loop         |
| `optimizer.py`           | `MuonAdamW` and `DistMuonAdamW` optimizers                           |
| `scaling.py`             | Scaling law utilities (compute-optimal step count, total batch size) |
| `checkpoint.py`          | Save/load model, optimizer, and training metadata                    |
| `dataloader.py`          | Distributed token dataloader over ClimbMix shards                    |
| `compression_metrics.py` | Optional FP8 compression tracking                                    |

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
                    └── train_base(config)           # training/base/__init__.py
                          ├── setup(config)           # training/base/setup.py
                          │     ├── compute_init()    # common/distributed.py
                          │     ├── get_tokenizer()   # tokenizer/utils.py
                          │     ├── GPT(GPTConfig())  # models/gpt.py
                          │     └── MuonAdamW(...)    # training/optimizer.py
                          └── train_loop(s)           # training/base/loop.py
                                ├── evaluate_bpb()   # evaluation/loss_eval.py
                                ├── evaluate_core()  # evaluation/core_benchmark.py
                                └── save_checkpoint() # training/checkpoint.py
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
- `models/` imports only from `common/`
- `tasks/` imports only from `common/`
- `tokenizer/` imports from `common/`, `workspace`
- `dataset/` imports from `common/`, `workspace`
- `training/base/`, `training/sft/`, `training/rl/` import from `common/`, `workspace`, `models/`, `tokenizer/`, `dataset/`, `tasks/`, `evaluation/`
- `evaluation/base/`, `evaluation/chat/` import from `common/`, `workspace`, `models/`, `tokenizer/`, `tasks/`
- `chat/` imports from `common/`, `config/`, `evaluation/`, `tokenizer/`
- `cli.py` imports from all packages (top-level wiring only)
