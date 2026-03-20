---
title: "CLI Backend Integration"
summary: "Design for --backend torch|mlx flag, autodetection, and CommonConfig.validate()."
read_when:
  - Adding or reviewing backend selection logic
  - Understanding why backend and device_type are separate fields
  - Wiring setup.py backend dispatch in step 5
status: draft
last_updated: "2025-07-22"
---

# CLI Backend Integration

Step 8 of the [dual-trainer architecture](dual-trainer-architecture.md). Adds `--backend`
to `CommonConfig` so the training backend can be selected via CLI or TOML before the trainer
dispatch in `setup.py` is wired (step 5).

**Status**: ✅ implemented. `CommonConfig.backend`, `autodetect_backend()`, and `CommonConfig.validate()` are live.

---

## What was added

`CommonConfig` gains one field:

```python
backend: str = ""  # torch | mlx (empty = autodetect)
```

`autodetect_backend()` in `common/distributed.py`:

```python
def autodetect_backend() -> str:
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        return "torch"
```

On Apple Silicon with MLX installed → `"mlx"`. Everywhere else → `"torch"`. No platform sniffing — import presence is the only signal that matters.

CLI arg in `CommonConfig.update_parser`:

```
--backend torch|mlx   training backend (empty = autodetect)
```

`CommonConfig.validate()` enforces:
- `backend` in `{"", "torch", "mlx"}`
- `device_type` in `{"", "cuda", "mps", "cpu"}`
- `backend == "mlx"` and `device_type != ""` is a contradiction
- `wandb` in `{"online", "local", "disabled"}`

---

## Why `backend` and `device_type` are separate fields

`device_type` controls which PyTorch device is used (`cuda`, `mps`, `cpu`). `backend` controls
which runtime is used at all. On Apple Silicon you can run `backend=torch, device_type=mps`
or `backend=mlx` (device is implicit — MLX always uses the Apple GPU). Collapsing them would
make `device_type=mlx` a lie and break `compute_init`.

---

## What `setup.py` will do with it (step 5)

```python
backend = config.common.backend or autodetect_backend()
if backend == "mlx":
    # skip compute_init, DDP, device setup entirely
    trainer = build_mlx_trainer(config)
else:
    device_type = config.common.device_type or autodetect_device_type()
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    trainer = build_torch_trainer(config, device, ddp_rank, ddp_world_size)
```

`compute_init` is never called for the MLX path — it asserts `device_type in ["cuda", "mps", "cpu"]`
and would fail or be misleading.

---

## Resolved open questions

- ✅ `CompressionMetrics` tracker supported by both backends via `forward_logits() -> tuple[np.ndarray, np.ndarray]` — not deferred
