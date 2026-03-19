# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- `training/train_base.py`, `training/train_sft.py`, `training/train_rl.py` split into
  `training/base/`, `training/sft/`, `training/rl/` sub-packages with co-located state dataclasses
- `evaluation/base_eval.py`, `evaluation/chat_eval.py` split into
  `evaluation/base/`, `evaluation/chat/` sub-packages
- Mutable loop state extracted into `PretrainingState`, `SFTState`, `RLState`, `BaseEvalResult`, `ChatEvalResult`
- `train_base` closure eliminated — loop promoted to module-level function
- `train_sft` `nonlocal` hack eliminated — `SFTState` passed explicitly into dataloader
- `cli.py` import paths updated to use new sub-packages

### Added

- `workspace.py` — module-level path store replacing `common/paths.py`; initialized once at startup via `workspace.init()`, provides all path functions (`data_dir`, `tokenizer_dir`, `checkpoint_dir`, `eval_results_dir`, etc.) without `base_dir` parameter threading

### Changed

- All `base_dir` parameter threading removed across `tokenizer/`, `dataset/`, `report/`, `training/`, `evaluation/`, `chat/`, and `tasks/` packages — all path resolution now goes through `workspace`
- `common/paths.py` deleted — replaced by `workspace.py`
- `download_file_with_lock` signature changed: `base_dir` → `data_dir` (caller passes the target directory directly)
- Scheduler functions co-located in training scripts (`train_base.py`, `train_sft.py`, `train_rl.py`); `training/schedulers.py` deleted
- `init_wandb` no longer takes `CommonConfig` param — reads from `current.get().common` internally
- `get_report()` and `manage_report()` no longer take path params — use `workspace.report_dir()` internally

### Removed

- `common/paths.py` — all path functions moved to `workspace.py`
- `training/schedulers.py` — scheduler functions co-located in training scripts

### Added

- `docs/guides/tuning-guide.md` — parameter recommendations for tokenizer, pretraining, and SFT across hardware tiers
- `pyright` `executionEnvironments` in `pyproject.toml` — suppresses `reportMissingParameterType` in `tests/` and `dev/`

### Fixed

- `GradScaler` missing `device=device_type` in `train_sft.py` — was silently disabled on MPS, leaving fp16 gradients unscaled
- `init_wandb` called with full `Config` instead of `CommonConfig` in `train_sft.py` and `train_rl.py`
- `load_optimizer_state` called with `model_name=` instead of `source=` in `train_sft.py`
- `phase="train"` → `phase="base"` when loading pretrained checkpoint in `train_sft.py`
- `SpellingBee` and `SimpleSpelling` missing `base_dir` parameter — `download_file_with_lock` called without required `base_dir` arg
- `run_chat_eval` missing `base_dir` for `SpellingBee` instantiation in `chat_eval.py`
- `chat_cli` missing `base_dir` in `load_model_from_dir` call
- Unused import `load_model` removed from `train_sft.py`

- Unified `nanochat` CLI — single entry point replacing all per-script entry points
  - `nanochat config init / show`
  - `nanochat data download / tokenizer train / tokenizer eval`
  - `nanochat train base / sft / rl`
  - `nanochat eval base / chat`
  - `nanochat chat / serve`
  - `nanochat report generate / reset`
- `config.toml` support — TOML config file auto-discovered from working directory; CLI args override file values
- `--base-dir` flag and `NANOCHAT_BASE_DIR` env var to set the data/checkpoint root
- Flash Attention 3 with automatic fallback to PyTorch SDPA on non-Hopper GPUs
- FP8 training via `torchao` (H100 only, opt-in with `--fp8`)
- Learnable per-layer residual scalars (`resid_lambdas`, `x0_lambdas`)
- Sliding window attention with configurable patterns (default `SSSL`: 3 short, 1 long)
- BOS-aligned dataloaders with BestFit-Crop packing and epoch tracking
- Pretraining resumption from checkpoints (`--resume-from-step`)
- `--save-every` flag for checkpoint cadence control
- SpellingBee task for character counting and spelling ability
- Python calculator tool support in the inference engine
- Identity/personality system via synthetic data generation (`dev/gen_synthetic_data.py`)
- Web UI: slash commands, click-to-edit messages, click-to-regenerate responses
- Multi-GPU inference (data parallel)
- CPU and MPS (Apple Silicon) support with automatic device detection
- Muon optimizer with Polar Express orthogonalization and Adafactor-style variance reduction
- Cautious weight decay with linear schedule to zero
- CORE score evaluation for HuggingFace models
- Miniseries and scaling laws training scripts (`runs/miniseries.sh`, `runs/scaling_laws.sh`)
- `docs/guides/` — five guides mirrored from upstream GitHub discussions:
  - `introducing-nanochat.md` — original Oct 2025 post
  - `infusing-identity.md` — synthetic identity data guide
  - `counting-letters-adding-abilities.md` — SpellingBee task walkthrough
  - `miniseries-v1.md` — scaling laws and miniseries v1 results
  - `beating-gpt2-nanochat-journey.md` — Jan 2026 GPT-2 speedrun deep dive
- `docs/guides/quickstart.md` — step-by-step setup guide
- `docs/configuration.md` — full config reference with all fields and defaults
- `docs/data-layout.md` — `NANOCHAT_BASE_DIR` directory structure reference
- `docs/code-structure.md` — package map, key flows, and dependency rules
- `docs/m3-max-guide.md` — MPS backend guide for Apple Silicon

### Changed

- Switched pretraining dataset from FineWeb-EDU to NVIDIA ClimbMix — time to GPT-2 reduced from 2.76h to 1.80h
- Data layout restructured under `NANOCHAT_BASE_DIR`:
  - `base_data_climbmix/` → `data/climbmix/`
  - `eval_bundle/` → `data/eval_tasks/`
  - `base_checkpoints/` → `checkpoints/base/`
  - `chatsft_checkpoints/` → `checkpoints/sft/`
  - `chatrl_checkpoints/` → `checkpoints/rl/`
  - `base_eval/` → `eval/`
  - `identity_conversations.jsonl` → `identity.jsonl`
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
