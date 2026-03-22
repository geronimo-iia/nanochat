# Design

Stable architecture and reference documentation for the nanochat codebase.

| Document | Description |
|---|---|
| [code-structure.md](code-structure.md) | Package map, responsibilities, and key cross-package flows |
| [configuration.md](configuration.md) | Config fields, TOML files, CLI overrides, resolution order |
| [data-layout.md](data-layout.md) | Where nanochat stores data, tokenizers, and checkpoints (`NANOCHAT_BASE_DIR`) |
| [trainer-protocol.md](trainer-protocol.md) | `BaseTrainer` protocol reference, `loop.py` call sequence, implementing a new backend |
| [checkpoint-interop.md](checkpoint-interop.md) | `CheckpointManager` protocol, safetensors cross-backend interop, conversion boundary |
| [mlx-backend.md](mlx-backend.md) | MLX training stack overview with diagrams |
| [mlx-gpt-design.md](mlx-gpt-design.md) | MLX GPT architecture reference |
| [mlx-muon-design.md](mlx-muon-design.md) | MLX Muon optimizer reference |
| [mlx-training-patterns.md](mlx-training-patterns.md) | Reference patterns for grad accumulation and `mx.eval()` cadence |
