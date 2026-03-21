---
title: "MLX Integration Guidelines"
summary: "Fresh-eyes analysis of the MLX backend integration: four structural issues found in early testing, proposals, and implementation record."
read_when:
  - Reviewing the MLX backend design before going deeper on implementation
  - Deciding how to split backend-specific vs shared logic in setup.py and loop.py
  - Understanding why compute_init was renamed and what mlx_compute_init does
status: active
last_updated: "2025-07-25"
---

# MLX Integration Guidelines

Four structural issues surfaced from early testing. Phase 1 (Issues 1–3) is complete.
Phase 2 (Issue 4) is pending.

---

## Issue 1 — `common/` is torch-only

`compute_init`, `compute_cleanup`, and `get_device_sync` are torch-specific but live in
`common/`, which implies they are backend-agnostic. The MLX path currently skips
`compute_init` entirely and stubs out `get_device_sync` with lambdas in `_setup_mlx`.

### What's wrong

- `compute_init` asserts `device_type in ["cuda", "mps", "cpu"]` — MLX can never call it.
- `compute_cleanup` calls `dist.destroy_process_group()` — irrelevant for MLX.
- `get_device_sync` has no `"mlx"` branch, so `synchronize` and `get_max_memory` are dead
  stubs on the MLX path. Memory reporting is always 0, and the synchronize call is a no-op
  (though timing stays correct because `MLXTrainer.step()` already calls `mx.eval`).

### Proposal

**Option A — add MLX branches in-place.**
Cheapest. Add `"mlx"` branches to `get_device_sync` and leave `compute_init`/`compute_cleanup`
as torch-only (they are already skipped on the MLX path). Document the skip explicitly.

**Option B — rename to signal scope + introduce `mlx_compute_init`.**
Rename `compute_init` → `torch_compute_init`, `compute_cleanup` → `torch_compute_cleanup`.
Add `mlx_compute_init` for seeding and explicit device selection.
More honest, slightly more churn.

Recommendation: **Option B** — the rename is mechanical and the seeding gap is real
(see Issue 4 below). Do it together with the `common/mlx.py` module (Phase 3).

---

## Issue 2 — MLX utility functions are missing or scattered

The torch path has `get_compute_dtype()`, `get_peak_flops()`, and `get_device_sync()` as
named, testable utilities in `common/`. The MLX path has none of these — dtype is implicit,
device info is never printed, and memory functions are inlined as stubs or lazy one-liners.

As of the current MLX version, `mx.metal.*` functions are deprecated. The non-deprecated
API lives at `mx.*` top-level: `mx.get_peak_memory()`, `mx.get_active_memory()`,
`mx.clear_cache()`, `mx.device_info()`.

### What's worth adding

**`get_mlx_compute_dtype() -> mx.Dtype`**
Reads `NANOCHAT_DTYPE` env var, defaults to `mx.bfloat16` on Apple Silicon. Gives the same
control surface as `get_compute_dtype()` on the torch path. Currently MLX silently uses
whatever the model defaults to — no logging, no override.

**`get_mlx_peak_memory() -> int`**
One-liner wrapping `mx.get_peak_memory()`. Belongs alongside `get_device_sync` so the
loop's `get_max_memory` call is consistent across backends. The current stub returns `0`.

**`get_mlx_device_info() -> dict`**
Wraps `mx.device_info()`, returns a stable dict with `device_name`, `memory_size`,
`architecture`. Used for the banner print in `_setup_mlx` — currently nothing is printed
about the MLX device, while the torch path prints GPU name and peak FLOPS.

### What's not worth adding yet

**Peak FLOPS table for MLX.** `device_name` is `"Apple M3 Max"` and `architecture` is
`"applegpu_g15s"` — parseable, and Apple Silicon BF16 FLOPS are publicly known. But MFU
on MLX is not comparable to CUDA MFU, and `gpu_peak_flops = float("inf")` (MFU displays
as `0%`) is honest. A fake FLOPS table would show a number that misleads more than it helps.
Leave as `inf` until there is a real use for it.

### Where they live — `common/mlx.py`

Three utility functions is exactly the threshold for a dedicated module. A flat file
`src/nanochat/common/mlx.py` (not a subpackage) groups them without adding indirection.
The lazy import pattern stays — `common/mlx.py` is only imported from MLX-path code,
so non-MLX environments are unaffected.

```
common/
  distributed.py   # torch_compute_init, torch_compute_cleanup, mlx_compute_init
  dtype.py         # get_compute_dtype (torch)
  hardware.py      # get_device_sync, get_peak_flops, clear_device_cache (torch + mlx)
  mlx.py           # get_mlx_compute_dtype, get_mlx_peak_memory, get_mlx_device_info
```

`mlx_compute_init` stays in `distributed.py` — it is init/lifecycle, not a utility.

---

## Issue 3 — `device_type = "mlx"` sentinel vs torch device

`_setup_mlx` returns `device_type = "mlx"` and `device = torch.device("cpu")`. The sentinel
is fine and intentional (see `mlx-device-type-refactor.md`). The `torch.device("cpu")` is not.

### What's wrong

`device` is used in three places on the MLX path:

1. `tokenizing_distributed_data_loader_with_state_bos_bestfit(..., device=torch.device("cpu"))` —
   tensors are allocated on CPU, then converted to `mx.array` via `.numpy()`. This works but
   wastes a copy on Apple Silicon where MPS tensors can be converted without going through CPU.
2. `build_val_loader` in `setup()` uses `device=device` — same issue.
3. `get_token_bytes(device=torch.device("cpu"))` — minor, token bytes are small.

### Proposal

Use `autodetect_device_type()` to resolve the torch device on the MLX path, same as the
torch path does. On Apple Silicon this gives `"mps"`, so tensors land on the Metal GPU
before the `mx.array` conversion.

```python
# _setup_mlx — replace the hardcoded cpu device
torch_device_type = autodetect_device_type()  # "mps" on Apple Silicon
torch_device = torch.device(torch_device_type)

train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    ..., device=torch_device, ...
)
```

The `mlx_loader` wrapper already calls `.numpy()` on the tensors before `mx.array(...)`.
MPS tensors support `.numpy()` via the shared memory buffer — no extra copy.

Note: `device` in `BaseTrainingSetup` then means "torch device for the dataloader" on both
paths. Worth a one-line docstring on the slot to make this explicit.

---

## Issue 4 — `setup()` and `loop.py` methods are large

`setup()` is ~150 lines. `train_loop()` is ~130 lines. Both mix concerns that will diverge
further as MLX support deepens (e.g. multi-device MLX, MLX-specific memory pressure handling).

### What's wrong

The hidden goal of the current structure is logic reuse — schedulers, iteration counting,
checkpoint logic, and logging are shared. That goal is correct. But the methods are large
because they also contain backend-specific branches inline (`if backend == "mlx": ...`),
which makes the shared skeleton hard to see.

### Proposal

Don't split `loop.py` into two files — the loop body is genuinely shared. Instead, extract
the backend-specific *side-effects* into small focused helpers that the loop calls by name.

**In `loop.py` — extract cache management into `clear_device_cache`:**

```python
# common/hardware.py
def clear_device_cache(device_type: str) -> None:
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type == "mlx":
        import mlx.core as mx
        mx.clear_cache()  # non-deprecated top-level API
```

Lives in `hardware.py` alongside `get_device_sync` — same file, same concern.
The loop calls `clear_device_cache(s.device_type)`. New backends add one branch here.

**In `setup.py` — the shared preamble is already clean.** The issue is that `_setup_torch`
and `_setup_mlx` both return the same 8-tuple, which is an implicit contract. Consider a
small named return type:

```python
@dataclass(frozen=True)
class _BackendSetup:
    trainer: BaseTrainer
    device_type: str
    ddp_rank: int
    ddp_world_size: int
    device: torch.device
    synchronize: Callable[[], None]
    get_max_memory: Callable[[], int]
    gpu_peak_flops: float
```

This makes the contract explicit, removes the positional tuple unpacking, and makes it
obvious when a new backend returns something unexpected.

**Don't over-abstract.** The current two-backend structure doesn't need a `BackendFactory`
or plugin system. The `if backend == "mlx" / else` dispatch in `setup()` is clear and
sufficient. Add complexity only when a third backend arrives.

---

## Summary

| Issue | Root cause | Recommended fix | Effort |
|---|---|---|---|
| `common/` is torch-only | `compute_init`/`get_device_sync` never designed for multi-backend | Rename `compute_init` → `torch_compute_init`; add `mlx_compute_init`; add `"mlx"` branch to `get_device_sync` | Small |
| MLX utilities missing or scattered | MLX path never got equivalent of dtype/hardware utils | Add `common/mlx.py` with `get_mlx_compute_dtype`, `get_mlx_peak_memory`, `get_mlx_device_info` | Small |
| `device = torch.device("cpu")` on MLX path | Placeholder from initial implementation | Use `autodetect_device_type()` to resolve torch device on MLX path | Small |
| Large methods with inline backend branches | Shared logic + backend side-effects mixed in same body | `clear_device_cache` in `hardware.py`; `_BackendSetup` dataclass in setup | Medium |

Address issues 1, 2, and 3 together — they all feed into `_setup_mlx` and `common/`.
Issue 4 is independent and can follow once the first three are stable.

---

## Implementation plan

### Phase 1 — Hardware and device fixes (Issues 1 + 2 + 3) ✅

All tasks complete. 35/35 tests passing.

**Task 1.1 ✅ — `get_device_sync`: add `"mlx"` branch**

File: `src/nanochat/common/hardware.py`

- Add `if device_type == "mlx":` branch. Use `mx.get_peak_memory` (non-deprecated top-level
  API) — not `mx.metal.get_active_memory` which is deprecated.
- Import `mlx.core as mx` inside the branch (lazy, keeps the module importable without MLX).
- Remove the two stub lambdas currently assigned in `_setup_mlx` — they are replaced by
  calling `get_device_sync("mlx")`.

  ```python
  if device_type == "mlx":
      import mlx.core as mx
      return lambda: mx.eval([]), mx.get_peak_memory
  ```

**Task 1.2 ✅ — `_setup_mlx`: fix torch device and wire real sync/memory**

File: `src/nanochat/training/base/setup.py`

- Call `autodetect_device_type()` at the top of `_setup_mlx` to get `torch_device_type`
  (`"mps"` on Apple Silicon, `"cpu"` on non-Metal machines).
- Replace `torch.device("cpu")` with `torch.device(torch_device_type)` in the dataloader
  call and in the return value.
- Replace the two stub lambdas with `synchronize, get_max_memory = get_device_sync("mlx")`.
- Add a one-line comment on the `device` slot in `BaseTrainingSetup.__slots__` clarifying
  it is the torch device for the dataloader, not the compute device.

**Task 1.3 ✅ — `clear_device_cache`: extract cache-clear into `hardware.py`**

Files: `src/nanochat/common/hardware.py`, `src/nanochat/common/__init__.py`,
`src/nanochat/training/base/loop.py`

- Added `clear_device_cache(device_type: str) -> None` to `hardware.py` alongside
  `get_device_sync` — same file, same concern. Handles `"mps"` (`torch.mps.empty_cache`)
  and `"mlx"` (`mx.clear_cache`, non-deprecated top-level API), no-op for all others.
- Exported from `common/__init__.py`.
- `loop.py` imports and calls `clear_device_cache(s.device_type)`, replacing the
  previous inline `if s.device_type == "mps":` block.

**Task 1.4 ✅ — introduce `mlx_compute_init` and rename `compute_init` → `torch_compute_init`**

Files: `src/nanochat/common/distributed.py`, `src/nanochat/common/__init__.py`,
`src/nanochat/training/base/setup.py`

- Rename `compute_init` → `torch_compute_init` and `compute_cleanup` → `torch_compute_cleanup`
  in `distributed.py`. Update the exports in `__init__.py` and the two call sites in
  `setup.py` (torch path only — MLX path never called them).
- Add `mlx_compute_init()` in `distributed.py`:

  ```python
  def mlx_compute_init() -> None:
      """Seed and device init for the MLX path. No DDP, no process group."""
      import mlx.core as mx
      setup_default_logging()
      mx.random.seed(42)
      mx.set_default_device(mx.gpu)
  ```

- Call `mlx_compute_init()` at the top of `_setup_mlx()` in `setup.py`.
- Do **not** add `mlx_compute_cleanup` — it would be empty today. Add it only when there
  is something real to clean up.
- Export `mlx_compute_init`, `torch_compute_init`, `torch_compute_cleanup` from `__init__.py`.
  `compute_init` and `compute_cleanup` kept as backward-compatible aliases.

**Task 1.5 ✅ — introduce `common/mlx.py`**

File: `src/nanochat/common/mlx.py` (new)

- Create the module with three functions. All imports of `mlx.core` are at function scope
  (lazy) so the file is importable on non-MLX machines.

  ```python
  def get_mlx_compute_dtype():
      """Return mx.bfloat16 by default, overridable via NANOCHAT_DTYPE."""
      import mlx.core as mx
      import os
      _MAP = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
      env = os.environ.get("NANOCHAT_DTYPE")
      return _MAP[env] if env in _MAP else mx.bfloat16

  def get_mlx_peak_memory() -> int:
      import mlx.core as mx
      return mx.get_peak_memory()

  def get_mlx_device_info() -> dict:
      import mlx.core as mx
      return mx.device_info()
  ```

- Export all three from `common/__init__.py`.
- In `_setup_mlx`, replace the silent setup with a banner print using `get_mlx_device_info()`:

  ```python
  info = get_mlx_device_info()
  print0(f"MLX device: {info['device_name']} | RAM: {info['memory_size'] / 1024**3:.0f}GB")
  ```

- Wire `get_mlx_peak_memory` into `get_device_sync("mlx")` return value — `mx.get_peak_memory`
  is referenced directly there (same function, no wrapper overhead needed at that call site).

**Verification for Phase 1** ✅

- `pytest tests/test_training/test_mlx_trainer.py tests/test_common/` — 35/35 passed.

---

### Phase 2 — Structural cleanup (Issue 4) ✅

All tasks complete. 322 passed, 10 skipped.

**Task 2.1 ✅ — introduce `_BackendSetup` dataclass**

File: `src/nanochat/training/base/setup.py`

- Defined `_BackendSetup` as a `@dataclass(frozen=True)` with the 8 fields previously
  returned as a positional tuple by `_setup_torch` and `_setup_mlx`.
- Both functions now return `_BackendSetup(...)` with named fields.
- The dispatch block in `setup()` assigns to a single `b` variable, then unpacks by
  attribute name. The 8-variable positional unpacking line is gone.
- `device` slot carries a comment: "torch device for the dataloader, not the compute device".

**Task 2.2 ✅ — audit inline `if backend` branches in `setup()` and `__init__.py`**

Files: `src/nanochat/training/base/setup.py`, `src/nanochat/training/base/__init__.py`

- Two `if backend == "mlx":` blocks remain in `setup()`, both correct:
  1. Pre-dispatch block resolving `ddp_world_size_for_accum` and torch init vars —
     genuinely pre-dispatch shared logic, cannot move inside backend functions.
  2. The main dispatch block itself.
- `wandb_run` re-init and `master_process` are already inside the `else` branch.
- `__init__.py` was still calling `compute_cleanup()` (old alias) unconditionally on both
  paths. Fixed: now calls `torch_compute_cleanup()` guarded by `device_type != "mlx"`.

**Verification for Phase 2** ✅

- Full `pytest` suite — 322 passed, 10 skipped. No behaviour change.
