# Changelog

See `dev/LOG.md` for detailed experiment notes and `dev/LEADERBOARD.md` for the GPT-2 speedrun leaderboard.

---

## 2026-03 — MLX training stack; codebase restructure (fork work)

Complete Apple Silicon training pipeline and a major codebase restructure on top of the upstream fork.

### MLX backend (pretraining → SFT → RL)

- Full SFT and RL training on Apple Silicon via MLX — iterate without cloud costs after a one-time H100 pretrain (~$70)
- `MLXTrainer`, `MLXRLTrainer`, `MLXEngine` (KV-cache generation), `mlx_rl_loop`
- SFT and RL loops share the same `BaseTrainer` protocol as the PyTorch path
- Fixed Muon NaN at step ~25: Polar Express must run in float32, not bfloat16
- Removed `_clip_grads` sync barrier: ~22k → ~45k tok/sec on M3 Max

### Codebase restructure

- Unified `nanochat` CLI replacing scattered scripts; `config.toml` support; `NANOCHAT_BASE_DIR`
- `workspace.py` — single source of truth for all filesystem paths
- `config/` — typed `Config` dataclass hierarchy with TOML loading and CLI overrides
- `checkpoint/` — `CheckpointManager` protocol; safetensors format; numpy conversion boundary for cross-backend weight exchange
- `BaseTrainer` protocol — shared by PyTorch and MLX; `model_state_dict()`, `step()`, `eval_context()`
- `model_factory.py` — shared model construction/loading across training, eval, and chat
- `LocalWandb` — offline metric logging to JSONL, no external dependency
- `torch.compile` disabled on MPS (caused NaN gradients)

### Documentation

- `docs/design/` — architecture reference: code structure, configuration, data layout, checkpoint interop, trainer protocol, MLX backend
- `docs/guides/` — quickstart, tuning guide, MLX guide, miniseries, identity/SFT, upstream integration workflow
- `docs/dev/` — roadmap, MLX SFT/RL design plan, per-task specs

---

## 2026-01-10 to 2026-03-04 — Architecture experiments (upstream)

Key results from the GPT-2 speedrun. See `dev/LOG.md` for full details.

- **ClimbMix dataset** (Mar 4): FineWeb-EDU → NVIDIA ClimbMix 400B — speedrun 2h 46m → 2h 01m; model depth d26 → d24
- **Explicit dtype management** (Mar 4): removed `torch.amp.autocast`; single `COMPUTE_DTYPE` global; fp16 + GradScaler path
- **Logit softcap** (Mar 2): `20 * tanh(x/20)` — small but consistent val loss improvement
- **FP8 training** (Feb 2): `torchao Float8Linear`, H100 only, opt-in `--fp8`
- **Value embeddings** (Jan 28): alternating-layer token→KV embeddings — +600M params at near-zero FLOP cost
- **GQA** (Jan 28): `n_kv_head` independent of `n_head`
- **BOS-aligned dataloader** (Jan 13): BestFit-Crop packing; replaces midtraining as separate stage
- **Flash Attention 3** (Jan 11): auto-fallback to PyTorch SDPA on non-Hopper GPUs
- **Sliding window** (Jan 11): configurable `SSSL` pattern (3 short + 1 full context, tiled)
- **Per-layer residual scalars** (Jan 11): `resid_lambdas`, `x0_lambdas`, smear gate
- **Muon optimizer** (Jan 10): Nesterov + Polar Express orthogonalization; cautious weight decay; gradient clipping removed

---

## 2025-10-13 to 2026-01-06 — Baseline (upstream)

Fork of `karpathy/nanochat` at initial public release. Upstream contributions in this period:

- Pretraining checkpoint resumption (`--save-every` + `--resume-from-step`)
- CPU and MPS backends; bf16 load fix on MPS
- SpellingBee eval task; calculator tool hardened against code injection
- Identity/personality synthetic data system
- `rustbpe` extracted to its own PyPI package
- Hyperparameter tuning: `warmdown_ratio` 0.2 → 0.4, `embedding_lr` 0.2 → 0.3
- Various bug fixes: dataloader resume, KV-cache GQA shape, tok/sec calculation, val_bpb on resume

---

## Tried and rejected (do not re-add without new evidence)

| Feature | Reason |
|---|---|
| Mixture of Experts | Wall-clock negative at d18 — dispatch overhead > FLOP savings |
| SwiGLU | Worse than ReLU² |
| Vanilla FineWeb / FineWeb-EDU mixtures | No improvement or regression on CORE |
| MuonH / Hyperball | No improvement over Polar Express |
| Bigram hash embeddings | Neutral-to-negative at d25+ |
| Multi-token prediction (MTP) | Memory overhead, negative CORE |
| Varlen attention | Negative result |
| Gradient clipping | Always inactive, 2% MFU overhead |
| Olmo pretraining mix | Negative result |
