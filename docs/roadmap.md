---
title: "nanochat Roadmap"
summary: "Development phases, active work, and future direction for nanochat."
read_when:
  - Planning nanochat development
  - Deciding what to implement next
  - Understanding scope and sequencing
status: draft
last_updated: "2025-07-15"
---

# nanochat Roadmap

## Completed

| Phase                                                                             | Completed  | Summary                                                            |
| --------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------ |
| [Phase 0 — Optimizations](archive/phase-0-optimizations.md)                       | 2026-03-05 | FA3, FP8, ClimbMix (27% speedup), Muon, sliding window, auto batch |
| Phase 0.5 — Modular Refactor                                                      | 2026-03-13 | src/nanochat/ layout, unified CLI, pyright strict, CI/CD           |
| [Phase 1 — Architecture Experiments](archive/phase-1-architecture-experiments.md) | 2026-02-19 | SwiGLU, MoE, MTP — all negative results                            |
| [Phase 1.5.0 — Data Layout & Config](archive/phase-1.5.0-data-layout-config.md)   | 2026-03-14 | Config system, centralized paths, hierarchical dirs, Python 3.13   |
| [Phase 1.5.1 — Bugfixes & Tooling](archive/phase-1.5.1-bugfixes-tooling.md)       | 2026-03-15 | argparse SUPPRESS fix, compression console output, LocalWandb      |
| Phase 2 — Codebase Refactor                                                       | 2025-07-15 | Config manager, workspace (phase 1), scheduler co-location, CLI cleanup, circular import fix |

## Active — Phase 1.5: Compression-Based Optimization

**Goal**: Validate whether compression-based optimization improves training efficiency before investing in scaling infrastructure.

| Sub-phase                               | Status      | Notes                                                       |
| --------------------------------------- | ----------- | ----------------------------------------------------------- |
| 1.5.0 — Data layout & config            | ✅ Done      |                                                             |
| 1.5.1 — Compression metrics             | ✅ Code done | [Validation checklist](phase-1.5.1-validation-checklist.md) |
| 1.5.2 — Dataset quality via compression | 🔜 Pending   |                                                             |
| 1.5.3 — Compression-aware optimization  | 🔜 Pending   |                                                             |

**Exit criteria**:

- [ ] Compression ratio correlates with val loss (R² > 0.7)
- [ ] Compression-aware optimizer shows 10%+ faster convergence
- [ ] Overall 15%+ improvement → proceed to Phase 2

**Decision point** (after 1.5.3):

- **>15%** → Phase 2 (infrastructure), then scale to 7B
- **5–15%** → refine and iterate at small scale
- **<5%** → skip to Phase 6 (SP-Transformer hybrid)

## Future Phases

Sequencing depends on the Phase 1.5 outcome.

| Phase                             | Description                                    | Plan                                   |
| --------------------------------- | ---------------------------------------------- | -------------------------------------- |
| Phase 2 — Training Infrastructure | Scale beyond single node, enable 10B+ training | —                                      |
| Phase 3 — Data Pipeline           | Compression-aware data quality and selection   | [plan](phase-3-data-pipeline.md)       |
| Phase 4 — Post-Training Alignment | SFT, RLHF, stable assistant behavior           | —                                      |
| Phase 5 — Capabilities            | Long context, tool use, multimodal             | [plan](phase-5-tools-transparency.md)  |
| Phase 6 — SP-Transformer Hybrid   | Combine transformer efficiency with SP Theory  | [plan](phase-6-hybrid-architecture.md) |

## Deferred

### MPS Performance Investigations

Baseline measured on M3 Max, d6, 10 steps, **without** `torch.compile` (eager mode):

| metric      | value          |
| ----------- | -------------- |
| dt per step | ~20–22s        |
| tok/sec     | ~23,000–25,000 |

Note: earlier baseline (~9s/step, ~58k tok/sec) was measured with `torch.compile` active, which causes NaN gradients — those numbers are invalid.

- **`torch.compile` on MPS causes NaN gradients** — confirmed on PyTorch 2.9.1, M3 Max. The inductor backend is not supported on MPS. During gradient accumulation, every forward pass after the first backward produces NaN loss. Fixed by skipping `torch.compile` when `device_type == "mps"`. Cost: ~2.4× slower (~24k vs ~58k tok/sec at d6). See [m3-max-guide](m3-max-guide.md) for details.

- **`autocast` in the hot-path forward** — added. The normal training path now wraps `model(x, y)` in `torch.amp.autocast(device_type=device_type, dtype=get_compute_dtype())`, consistent with the compression-tracking branch.

- **`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`** — disables the MPS memory watermark, allowing the GPU to use more unified memory before falling back to CPU. Tested on d6 (10 steps): no measurable throughput improvement (~55,900–58,300 tok/sec vs baseline — but those numbers were with compile/NaN, so inconclusive). May help at larger depths where memory pressure is real.

- **Config manager** ✅ — `config/current.py` and `load_and_init` implemented. See [design](config-manager-design.md).
- **Workspace module** ✅ Phase 1 — `workspace.py` implemented alongside `common/paths.py`. See [design](workspace-design.md).
- **Scheduler placement** ✅ — schedulers co-located in training scripts, `schedulers.py` deleted. See [study](archive/scheduler-placement-study.md).
- **TrainingState refactor** — extract mutable training loop state into a dataclass, eliminate the closure in `train_base`. See [plan](training-state-refactor.md).
- **Checkpoint manager** — `CheckpointManager` protocol with typed metadata, format-agnostic I/O, and logging abstraction. Prerequisite for dual-trainer checkpoint interop. See [design](checkpoint-manager-design.md).
- **Dual trainer architecture** — `Trainer` protocol with `TorchTrainer` (current code) and `MLXTrainer` (MLX model + Muon on Apple Silicon). See [plan](dual-trainer-architecture.md).
- **`--resume-from-latest` flag** — auto-detect the last saved checkpoint step so you don't have to look it up manually. Uses `find_last_step()` which already exists in `checkpoint.py`. Document in quickstart guide.
- **MPS fp16 vs fp32 loss curves** — `GradScaler(device='mps')` works natively on Apple Silicon. A d8 comparison run (fp16 vs fp32) would confirm whether fp16 training is numerically stable in practice.
