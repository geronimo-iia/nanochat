# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- `checkpoint/` package — `CheckpointManager` protocol, `TorchCheckpointManager`, typed `CheckpointMetadata`/`LoopState`, `CheckpointLogger`/`RankZeroLogger`/`SilentLogger`, `make_checkpoint_manager` factory, `find_largest_model`/`find_last_step` discovery, `patch_missing_config_keys`/`patch_missing_keys` compat utilities
- `model_factory.py` — `load_model_from_dir` and `load_optimizer_state` promoted from `training/checkpoint.py`; shared by `training/`, `evaluation/`, and `chat/`
- `config/checkpoint.py` — `CheckpointConfig` with `format`, `save_every`, `resume_from_step`, `keep_last_n`
- `model_tag` on `CommonConfig` — single source of truth across all training modes
- `checkpoint_dir` property on `CheckpointManager` protocol and `TorchCheckpointManager`

### Changed

- All training loops (`base`, `sft`, `rl`) use `make_checkpoint_manager` + `manager.save()` instead of bare `save_checkpoint`/`load_checkpoint`
- `training/base/__init__.py` creates `CheckpointManager` and passes it to both `setup()` and `train_loop()`
- `load_model_from_dir` and `load_optimizer_state` accept `config: CheckpointConfig` instead of individual checkpoint fields
- `SFTConfig.model_step` → `source_step`; `RLConfig.model_step` → `source_step`
- `model_tag` removed from `TrainingConfig`, `SFTConfig`, `RLConfig`, `EvaluationConfig` — consolidated into `CommonConfig`
- `save_every`, `resume_from_step` removed from `TrainingConfig`, `SFTConfig`, `RLConfig` — moved to `CheckpointConfig`

### Removed

- `training/checkpoint.py` — replaced by `checkpoint/` package and `model_factory.py`

---

## [0.4.0] — Phase 2: Codebase Refactor (2025-07-15)

### Added

- `workspace.py` — module-level path store replacing `common/paths.py`; initialized once at startup via `workspace.init()`, provides all path functions (`data_dir`, `tokenizer_dir`, `checkpoint_dir`, `eval_results_dir`, etc.) without `base_dir` parameter threading
- `docs/guides/tuning-guide.md` — parameter recommendations for tokenizer, pretraining, and SFT across hardware tiers
- `pyright` `executionEnvironments` in `pyproject.toml` — suppresses `reportMissingParameterType` in `tests/` and `dev/`

### Changed

- `training/train_base.py`, `training/train_sft.py`, `training/train_rl.py` split into `training/base/`, `training/sft/`, `training/rl/` sub-packages with co-located state dataclasses (`PretrainingState`, `SFTState`, `RLState`)
- `evaluation/base_eval.py`, `evaluation/chat_eval.py` split into `evaluation/base/`, `evaluation/chat/` sub-packages with result dataclasses (`BaseEvalResult`, `ChatEvalResult`)
- `train_base` closure eliminated — loop promoted to module-level function
- `train_sft` `nonlocal` hack eliminated — `SFTState` passed explicitly into dataloader
- All `base_dir` parameter threading removed across `tokenizer/`, `dataset/`, `report/`, `training/`, `evaluation/`, `chat/`, `tasks/` — all path resolution goes through `workspace`
- `download_file_with_lock` signature: `base_dir` → `data_dir`
- Scheduler functions co-located in training scripts; `training/schedulers.py` deleted
- `init_wandb` no longer takes `CommonConfig` param — reads from `current.get().common` internally
- `get_report()` and `manage_report()` no longer take path params — use `workspace.report_dir()` internally
- `cli.py` import paths updated to use new sub-packages

### Removed

- `common/paths.py` — replaced by `workspace.py`
- `training/schedulers.py` — scheduler functions co-located in training scripts

### Fixed

- `GradScaler` missing `device=device_type` in `train_sft.py` — was silently disabled on MPS, leaving fp16 gradients unscaled
- `init_wandb` called with full `Config` instead of `CommonConfig` in `train_sft.py` and `train_rl.py`
- `load_optimizer_state` called with `model_name=` instead of `source=` in `train_sft.py`
- `phase="train"` → `phase="base"` when loading pretrained checkpoint in `train_sft.py`
- `SpellingBee` and `SimpleSpelling` missing `base_dir` — `download_file_with_lock` called without required arg
- `run_chat_eval` missing `base_dir` for `SpellingBee` instantiation in `chat_eval.py`
- `chat_cli` missing `base_dir` in `load_model_from_dir` call
- Unused import `load_model` removed from `train_sft.py`

---

## [0.3.0] — Phase 1.5: Data Layout, Config & Compression (2026-03-15)

### Added

- Config system — `Config`, `CommonConfig`, `TrainingConfig`, `SFTConfig`, `RLConfig`, `EvaluationConfig` dataclasses with TOML loading and CLI override via `ConfigLoader`
- `config.toml` support — auto-discovered from working directory; CLI args override file values
- `--base-dir` flag and `NANOCHAT_BASE_DIR` env var to set the data/checkpoint root
- Hierarchical directory layout under `NANOCHAT_BASE_DIR` (see `docs/data-layout.md`)
- `LocalWandb` — local-only wandb backend for offline training runs
- `docs/configuration.md` — full config reference
- `docs/data-layout.md` — `NANOCHAT_BASE_DIR` directory structure reference
- `docs/code-structure.md` — package map, key flows, and dependency rules
- `docs/m3-max-guide.md` — MPS backend guide for Apple Silicon

### Changed

- Data layout restructured under `NANOCHAT_BASE_DIR`:
  - `base_data_climbmix/` → `data/climbmix/`
  - `eval_bundle/` → `data/eval_tasks/`
  - `base_checkpoints/` → `checkpoints/base/`
  - `chatsft_checkpoints/` → `checkpoints/sft/`
  - `chatrl_checkpoints/` → `checkpoints/rl/`
  - `base_eval/` → `eval/`
  - `identity_conversations.jsonl` → `identity.jsonl`
- Upgraded to Python 3.13

### Fixed

- argparse `SUPPRESS` fix — default values no longer override TOML values when CLI flag is absent
- Compression metrics console output formatting
- `torch.compile` skipped on MPS — inductor backend unsupported, caused NaN gradients during gradient accumulation

---

## [0.2.0] — Phase 0.5 & Phase 1: Modular Refactor + Architecture Experiments (2026-03-13)

### Added

- `src/nanochat/` package layout — all source under a proper package with `__init__.py`
- Unified `nanochat` CLI — single entry point replacing all per-script entry points:
  - `nanochat config init / show`
  - `nanochat data download / tokenizer train / tokenizer eval`
  - `nanochat train base / sft / rl`
  - `nanochat eval base / chat`
  - `nanochat chat / serve`
  - `nanochat report generate / reset`
- `pyright` strict mode — full static type checking across the codebase
- CI/CD — GitHub Actions for lint, type-check, and test on push
- `docs/guides/` — five guides mirrored from upstream GitHub discussions:
  - `introducing-nanochat.md`
  - `infusing-identity.md`
  - `counting-letters-adding-abilities.md`
  - `miniseries-v1.md`
  - `beating-gpt2-nanochat-journey.md`
- `docs/guides/quickstart.md` — step-by-step setup guide

### Changed

- SwiGLU, MoE, MTP experiments — all showed negative results; reverted to baseline architecture

---

## [0.1.0] — Phase 0: Optimizations (2026-03-05)

### Added

- Flash Attention 3 with automatic fallback to PyTorch SDPA on non-Hopper GPUs
- FP8 training via `torchao` (H100 only, opt-in with `--fp8`)
- Learnable per-layer residual scalars (`resid_lambdas`, `x0_lambdas`)
- Sliding window attention with configurable patterns (default `SSSL`: 3 short, 1 long)
- BOS-aligned dataloaders with BestFit-Crop packing and epoch tracking
- Pretraining resumption from checkpoints (`--resume-from-step`)
- `--save-every` flag for checkpoint cadence control
- Muon optimizer with Polar Express orthogonalization and Adafactor-style variance reduction
- Cautious weight decay with linear schedule to zero
- SpellingBee task for character counting and spelling ability
- Python calculator tool support in the inference engine
- Identity/personality system via synthetic data generation (`dev/gen_synthetic_data.py`)
- Web UI: slash commands, click-to-edit messages, click-to-regenerate responses
- Multi-GPU inference (data parallel)
- CPU and MPS (Apple Silicon) support with automatic device detection
- CORE score evaluation for HuggingFace models
- Miniseries and scaling laws training scripts (`runs/miniseries.sh`, `runs/scaling_laws.sh`)

### Changed

- Switched pretraining dataset from FineWeb-EDU to NVIDIA ClimbMix — time to GPT-2 reduced from 2.76h to 1.80h
- Vocab size default: 50K → 32K
- D:N ratio: 20 → 8 (compute-optimal for nanochat)
- Warmdown ratio: 0.2 → 0.4
- Embedding learning rate: 0.2 → 0.3
- Adam beta1: 0.8 → 0.96
- MPS backend uses `float16` instead of `float32` (~10–30% faster, halves memory)
- Upgraded to PyTorch 2.9.1
- Upgraded synthetic data generation to Gemini 3

### Removed

- Midtraining as a separate stage — replaced by BOS-aligned dataloader + `chat_sft`
- `base_loss` script — functionality merged into `base_eval`
- Gradient clipping — not necessary, costs 2% MFU
- Numpy dependency
- Pandas dependency in `base_eval`

### Fixed

- Grad clip bug — was clipping per GPU before gradient synchronization
- Completion-only loss masking in SFT dataloader
- KV-cache decode respecting sliding window
- Distributed Parquet dataloader resume for multi-epoch training
- Tok/sec calculation when `grad_accum_steps > 1`
- Memory leak in Rust tokenizer
- CPU bfloat16 tensor loading
- Learning rate multiplier ramping direction
- Missing `val_bpb` on resume

### Security

- Hardened eval: calculator tool blocked from accessing globals/locals
