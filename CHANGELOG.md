# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Unified `nanochat` CLI — single entry point for all commands:
  - `nanochat config init / show`
  - `nanochat data download / tokenizer train / tokenizer eval`
  - `nanochat train base / sft / rl`
  - `nanochat eval base / chat`
  - `nanochat chat / serve`
  - `nanochat report generate / reset`
- `config.toml` support — auto-discovered from working directory; CLI args override file values
- `--base-dir` flag and `NANOCHAT_BASE_DIR` env var to set the data/checkpoint root
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

### Removed

- Midtraining as a separate stage — replaced by BOS-aligned dataloader + `chat_sft`
- Gradient clipping — not necessary, costs 2% MFU

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
- `torch.compile` skipped on MPS — caused NaN gradients during gradient accumulation

### Security

- Hardened eval: calculator tool blocked from accessing globals/locals
