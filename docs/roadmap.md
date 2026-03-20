---
title: "nanochat Roadmap"
summary: "Development phases, active work, and future direction for nanochat."
read_when:
  - Planning nanochat development
  - Deciding what to implement next
  - Understanding scope and sequencing
status: active
last_updated: "2025-07-24"
---

# nanochat Roadmap

## Completed

| Phase | Completed | Summary |
| --- | --- | --- |
| Phase 0 — Optimizations | 2026-03-05 | FA3, FP8, ClimbMix (27% speedup), Muon, sliding window, auto batch |
| Phase 0.5 — Modular Refactor | 2026-03-13 | src/nanochat/ layout, unified CLI, pyright strict, CI/CD |
| Phase 1 — Architecture Experiments | 2026-02-19 | SwiGLU, MoE, MTP — all negative results |
| Phase 1.5.0 — Data Layout & Config | 2026-03-14 | Config system, centralized paths, hierarchical dirs, Python 3.13 |
| Phase 1.5.1 — Bugfixes & Tooling | 2026-03-15 | argparse SUPPRESS fix, compression console output, LocalWandb |
| Phase 2 — Codebase Refactor | 2025-07-15 | Config manager, workspace, scheduler co-location, entry point sub-packages, CLI cleanup |
| Phase 2.1 — Checkpoint Manager | 2025-07-19 | `CheckpointManager` protocol, typed metadata, `model_factory.py`, `CheckpointConfig` |
| Phase 2.2 — Dual Trainer / MLX Backend | 2025-07-24 | `BaseTrainer` protocol, `TorchTrainer`, `MLXTrainer`, backend-agnostic `loop.py`, safetensors checkpoint interop, `--backend=mlx` autodetect — see [mlx-backend.md](mlx-backend.md) |

## Active — Phase 1.5: Compression-Based Optimization

**Goal**: Validate whether compression-based optimization improves training efficiency before investing in scaling infrastructure.

| Sub-phase | Status | Notes |
| --- | --- | --- |
| 1.5.0 — Data layout & config | ✅ Done | |
| 1.5.1 — Compression metrics | ✅ Code done | [Validation checklist](phase-1.5.1-validation-checklist.md) |
| 1.5.2 — Dataset quality via compression | 🔜 Pending | |
| 1.5.3 — Compression-aware optimization | 🔜 Pending | |

**Exit criteria**:

- [ ] Compression ratio correlates with val loss (R² > 0.7)
- [ ] Compression-aware optimizer shows 10%+ faster convergence
- [ ] Overall 15%+ improvement → proceed to Phase 3

**Decision point** (after 1.5.3):

- **>15%** → Phase 3 (data pipeline), then scale to 7B
- **5–15%** → refine and iterate at small scale
- **<5%** → skip to Phase 6 (SP-Transformer hybrid)

## Future Phases

Sequencing depends on the Phase 1.5 outcome.

| Phase | Description | Plan |
| --- | --- | --- |
| Phase 3 — Data Pipeline | Compression-aware data quality and selection | [plan](phase-3-data-pipeline.md) |
| Phase 4 — Post-Training Alignment | SFT, RLHF, stable assistant behavior | — |
| Phase 5 — Capabilities | Long context, tool use, multimodal | [plan](phase-5-tools-transparency.md) |
| Phase 6 — SP-Transformer Hybrid | Combine transformer efficiency with SP Theory | [plan](phase-6-hybrid-architecture.md) |

## Deferred

- **`--resume-from-latest` flag** — auto-detect the last saved checkpoint step so you don't have to look it up manually. Uses `find_last_step()` which already exists in `checkpoint/discovery.py`.

- **MLX evaluation engine** — KV cache not implemented in `mlx_gpt.py`. Required for `nanochat eval` and `nanochat serve` on the MLX path.

- **MLX SFT / RL** — `MLXTrainer` covers base pretraining only. SFT and RL loops remain PyTorch-only.

- **Safetensors optimizer state** — optimizer state is currently saved with `torch.save` even in `SafetensorsCheckpointManager`. A future improvement would split into tensor buffers (`.safetensors`) and scalar metadata (JSON) to remove the torch dependency entirely. See note in [checkpoint-interop.md](checkpoint-interop.md).

- **`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`** — disables the MPS memory watermark. Tested on d6 (10 steps): no measurable throughput improvement. May help at larger depths where memory pressure is real.
