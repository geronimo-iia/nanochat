---
title: MLX Dataloader — Analysis & Optimization
summary: Analysis of the CUDA-designed dataloader running on the MLX path. Covers the three wasted CUDA-isms, transfer path options, numpy vs torch packing performance, threading opportunity, and recommendations.
read_when:
  - Investigating MLX training throughput bottlenecks
  - Considering a dedicated MLX dataloader implementation
  - Evaluating dataloader threading for the MLX path
status: draft
last_updated: 2025-07-11
---

# MLX Dataloader — Analysis & Optimization

---

## CUDA design intent

The dataloader was designed for multi-GPU CUDA training. Understanding its intent
makes the MLX mismatches clear.

### DDP sharding

`_document_batches` strides parquet row groups by `ddp_world_size`, each rank reads
a disjoint slice. On MLX (`ddp_world_size=1`) this degenerates correctly — rank 0
reads everything — but the sharding machinery runs regardless.

### Buffer layout

```
row_buffer   (B, T+1)     torch, CPU   — build rows without Python lists
cpu_buffer   (2*B*T,)     torch, CPU   — pinned staging area (CUDA only)
gpu_buffer   (2*B*T,)     torch, GPU   — persistent on-device buffer
  inputs  = gpu_buffer[:B*T].view(B, T)
  targets = gpu_buffer[B*T:].view(B, T)
```

The three-buffer design exists for one reason: a single async DMA transfer from
pinned CPU memory into a persistent GPU allocation, overlapping with the previous
step's compute.

### CUDA step sequence (intended)

```
step N compute (GPU)  ──────────────────────────────┐
                                                     │ mx.eval() / synchronize()
step N+1 dataloader (CPU)  ──────────┐               │
  pack → cpu_buffer (pinned)         │               │
  gpu_buffer.copy_(non_blocking=True)│ async DMA ────┘
                                     └──► inputs/targets on GPU, ready
```

`non_blocking=True` + pinned memory means the HtoD transfer runs on a CUDA DMA
engine concurrently with GPU compute. By the time the next step starts, the batch
is already on the GPU. The yielded `inputs`/`targets` are views into `gpu_buffer` —
the training loop touches GPU memory directly, zero extra copies.

---

## MLX path — three wasted CUDA-isms

On the MLX path `device=cpu`, so all three CUDA optimisations become dead weight.

### 1. Pinned memory — no DMA on unified memory

```python
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
```

`use_cuda = (device == "cuda")` → `False` on MLX. `cpu_buffer` is regular pageable
memory. No effect. But on CUDA, pinned memory is what enables the async DMA engine —
without it `non_blocking` is silently downgraded to synchronous anyway.

### 2. Persistent `gpu_buffer` — pointless allocation

```python
gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
```

On CUDA, `gpu_buffer` is a persistent device allocation reused every step — no
per-step `cudaMalloc`. On MLX `device=cpu`, so `gpu_buffer` is just another CPU
tensor. The persistent-allocation benefit doesn't exist.

### 3. `non_blocking` async copy — synchronous memcpy to itself

```python
gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
```

`use_cuda=False` → `non_blocking=False`. Both buffers are CPU tensors. This is a
synchronous CPU→CPU memcpy of `2 * B * T * 8` bytes (1 MB at B=64, T=1024) that
copies a tensor onto itself in the same memory space. No overlap, no transfer, pure
waste.

### 4. MLXTrainer adds two more copies

The yielded `inputs`/`targets` are CPU torch tensors. `MLXTrainer._next_batch()`
then does:

```python
x, y, state = next(self._torch_loader)
return mx.array(x.numpy()), mx.array(y.numpy()), state
```

- `.numpy()` — zero-copy, returns a numpy view of the torch tensor's memory
- `mx.array()` — copies from CPU into Metal memory (the real transfer)

So the full MLX path per micro-batch is:

```
pack → row_buffer (torch)
     → cpu_buffer.copy_(row_buffer)     ← copy 1: row→staging
     → gpu_buffer.copy_(cpu_buffer)     ← copy 2: staging→staging (pointless)
     → .numpy()                         ← zero-copy view
     → mx.array()                       ← copy 3: CPU→Metal (necessary)
```

Copy 2 is the only one that is purely redundant. Copies 1 and 3 are necessary.
Copy 1 could be eliminated with a numpy packer that writes directly into a
pre-allocated numpy buffer (Option B).

---

## Measured costs (M3 Max, B=64, T=1024)

All timings are per micro-batch (one `_next_batch()` call).

| Path | Cost |
|------|------|
| Current: torch pack + cpu→cpu copy + `.numpy()` + `mx.array()` | 3.5 ms |
| Option A: torch pack + `.numpy()` + `mx.array()` (skip copy 2) | 3.4 ms |
| Option B: numpy pack + `mx.array()` (no torch in hot path) | 1.2 ms |
| Transfer only: `mx.array()` from pre-built buffer | 0.02 ms |

The dominant cost is **torch tensor packing** (3.3 ms), not the redundant copy
(0.07 ms). `torch.tensor(doc, dtype=torch.long)` allocates a new tensor per
document fragment in the inner loop — numpy slice assignment is ~3× faster.

---

## Impact on step time

The dataloader runs sequentially on the main thread — called after `mx.eval()`
returns, not overlapped with compute. (On CUDA the async copy overlaps; on MLX
there is no equivalent mechanism.)

| Config | Step time | Accum steps | Dataloader/step (current) | Dataloader/step (numpy) |
|--------|-----------|-------------|--------------------------|------------------------|
| d6     | 11,600 ms | 8           | 28 ms (0.24%)            | 10 ms (0.09%)          |
| d8     | 19,000 ms | 16          | 56 ms (0.29%)            | 19 ms (0.10%)          |
| d12    | 45,000 ms | 32          | 112 ms (0.25%)           | 38 ms (0.08%)          |

**Conclusion**: dataloader is <0.3% of step time at all configs. Not the bottleneck.

---

## Options

### Option A — Remove the redundant copy (trivial)

Skip `gpu_buffer` allocation and `gpu_buffer.copy_()` when `device == cpu`.
Yield views into `cpu_buffer` directly.

**Savings**: ~0.07 ms per micro-batch. Negligible performance gain but removes a
semantic bug — a copy that pretends to be a HtoD transfer.

**Risk**: none. `MLXTrainer._next_batch()` is unchanged.

### Option B — Numpy packer (no torch in hot path)

Replace `row_buffer` (torch) with a numpy array. Replace `torch.tensor(doc)` slice
assignments with direct numpy slice assignments. Write directly into a pre-allocated
numpy buffer, yield numpy views, `mx.array()` in `_next_batch()` as today.

**Savings**: ~2.3 ms per micro-batch (3.5 ms → 1.2 ms). Across 8 accum steps at
d6: ~18 ms per step = 0.16% of step time.

**Tradeoff**: the dataloader becomes MLX-specific (torch path needs a wrapper).
Low ROI at current compute speeds.

### Option C — Dedicated MLX dataloader with threading

A fully MLX-native dataloader that packs into numpy directly and runs on a
background thread, prefetching the next batch while compute runs. This is the
explicit equivalent of the CUDA async DMA overlap — implemented in Python rather
than via a DMA engine.

#### Integration points

Before designing the class, the three integration points in the existing codebase:

**1. Construction** — `_setup_mlx` in `setup.py` currently builds the torch loader
and passes it to `MLXTrainer.__init__`. The MLX loader replaces it:

```python
# setup.py — _setup_mlx()
train_loader = MLXDataLoader(
    tokenizer=tokenizer,
    B=config.training.device_batch_size,
    T=config.training.max_seq_len,
    split="train",
    resume_state_dict=dataloader_resume_state_dict,
)
trainer = MLXTrainer(model, optimizer, grad_accum_steps, train_loader)
```

**2. Consumption** — `MLXTrainer._next_batch` currently does:

```python
x, y, state = next(self._torch_loader)
return mx.array(x.numpy()), mx.array(y.numpy()), state
```

With `MLXDataLoader`, the loader yields `(mx.array, mx.array, state_dict)` directly.
`_next_batch` becomes a passthrough:

```python
def _next_batch(self):
    return next(self._loader)  # already mx.array
```

**3. Resume** — `state_dict` must reflect the **consumer** position (the batch
just yielded to the trainer), not the worker's prefetch position. The worker may
be 1–2 batches ahead. On resume, the loader fast-forwards to `consumer_state_dict`
and the prefetch queue refills naturally.

#### Class design

```python
class MLXDataLoader:
    """
    Threaded MLX-native dataloader. Packs directly into numpy, yields mx.array.
    Background worker prefetches the next batch while mx.eval() runs on the
    main thread — equivalent to CUDA's async DMA overlap.
    """

    _QUEUE_DEPTH = 2  # 1 being consumed + 1 prefetched; compute/dataloader ~1,000:1 at all depths

    def __init__(
        self,
        tokenizer: object,
        B: int,
        T: int,
        split: str,
        tokenizer_threads: int = 4,
        tokenizer_batch_size: int = 128,
        buffer_size: int = 1000,
        resume_state_dict: dict | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._B = B
        self._T = T
        self._split = split
        self._tokenizer_threads = tokenizer_threads
        self._tokenizer_batch_size = tokenizer_batch_size
        self._buffer_size = buffer_size
        self._resume_state_dict = resume_state_dict

        # Pre-allocated numpy buffers — reused every batch, no per-batch malloc
        self._row_buf = np.empty((B, T + 1), dtype=np.int64)

        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_DEPTH)
        self._exc: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public interface — same as the torch loader: yields (x, y, state)
    # ------------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self) -> tuple[mx.array, mx.array, dict]:
        if self._exc is not None:
            raise self._exc
        item = self._queue.get()  # blocks until worker has a batch ready
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        """Signal the worker to stop and drain the queue."""
        self._stop.set()
        # Drain so the worker unblocks from queue.put()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Worker — runs on background thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        try:
            for batch in self._pack_loop():
                if self._stop.is_set():
                    return
                self._queue.put(batch)  # blocks if queue full — natural backpressure
        except BaseException as exc:
            self._queue.put(exc)  # surface exception to consumer on next __next__

    def _pack_loop(self):
        """Infinite generator: pack rows into numpy, yield (mx.array, mx.array, state_dict)."""
        bos = self._tokenizer.get_bos_token_id()
        doc_buffer: list = []
        pq_idx, rg_idx, epoch = 0, 0, 1
        row_buf = self._row_buf  # pre-allocated, reused

        batches = _document_batches(
            self._split, self._resume_state_dict, self._tokenizer_batch_size
        )

        def refill():
            nonlocal pq_idx, rg_idx, epoch
            doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
            tokens = self._tokenizer.encode(
                doc_batch, prepend=bos, num_threads=self._tokenizer_threads
            )
            doc_buffer.extend(tokens)

        while True:
            for row_idx in range(self._B):
                pos = 0
                capacity = self._T + 1
                while pos < capacity:
                    while len(doc_buffer) < self._buffer_size:
                        refill()
                    remaining = capacity - pos
                    # Best-fit: largest doc that fits entirely
                    best_idx, best_len = -1, 0
                    for i, doc in enumerate(doc_buffer):
                        dlen = len(doc)
                        if dlen <= remaining and dlen > best_len:
                            best_idx, best_len = i, dlen
                    if best_idx >= 0:
                        doc = doc_buffer.pop(best_idx)
                        row_buf[row_idx, pos:pos + best_len] = doc
                        pos += best_len
                    else:
                        # Crop shortest to fill exactly
                        si = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                        doc = doc_buffer.pop(si)
                        row_buf[row_idx, pos:pos + remaining] = doc[:remaining]
                        pos += remaining

            # Slice inputs/targets from the packed row buffer
            # np.ascontiguousarray ensures mx.array() gets a clean buffer to copy from
            x_np = np.ascontiguousarray(row_buf[:, :-1])  # (B, T)
            y_np = np.ascontiguousarray(row_buf[:, 1:])   # (B, T)

            state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

            # mx.array() copies from CPU into Metal memory — the real transfer.
            # This runs on the worker thread, overlapping with mx.eval() on main.
            yield mx.array(x_np), mx.array(y_np), state_dict
```

#### Threading model

```
main thread:   mx.eval() ──────────────────────┐  _next_batch() → queue.get()
                                                │  (instant — batch already ready)
worker thread: pack+transfer ──► queue.put()   │
               pack+transfer ──► queue.put() ──┘  (queue full, worker blocks)
               pack+transfer ──► ...              (resumes when main consumes)
```

The worker stays 1–2 batches ahead. `_QUEUE_DEPTH = 2` bounds memory overhead
to `2 * B * T * 2 * 8` bytes (2 MB at B=64, T=1024).

#### Depth scaling — the ratio stays ~1,000:1

As depth increases, `device_batch_size` is halved to stay within memory
(d6: B=64, d12: B=16, d20: B=4), so `grad_accum_steps` scales inversely.
Each individual microbatch stays in the same compute time range:

| Config | microbatch compute | dl per microbatch | ratio   | dl/step (sequential) |
|--------|--------------------|-------------------|---------|----------------------|
| d6     | ~1,459 ms          | 1.2 ms            | 1,216:1 | 10 ms (8 accum)      |
| d8     | ~1,214 ms          | 1.2 ms            | 1,011:1 | 19 ms (16 accum)     |
| d12    | ~1,425 ms          | 1.2 ms            | 1,187:1 | 38 ms (32 accum)     |
| d20    | ~1,649 ms          | 1.2 ms            | 1,374:1 | 154 ms (128 accum)   |

*d20 extrapolated from d12 via FLOPs ratio (4.63×). Step time ~211s, accum=128.*

The compute/dataloader ratio holds at ~1,000:1 at every depth — the threading
benefit does not erode at larger models. The sequential cost at d20 (154 ms
across 128 accum steps) is still only 0.07% of step time, but with threading
it is zero.

#### Changes to `MLXTrainer`

`_next_batch` becomes a one-liner — the loader already yields `mx.array`:

```python
def _next_batch(self) -> tuple[mx.array, mx.array, dict]:
    return next(self._loader)
```

The `torch_loader` field is renamed to `_loader` and typed as `Iterator` to
accept both the existing torch loader and `MLXDataLoader`.

#### Changes to `setup.py`

`_setup_mlx` replaces the torch loader construction with `MLXDataLoader`.
The `device=torch.device("cpu")` argument to the torch loader is removed entirely.

#### Resume correctness

The `state_dict` yielded by `_pack_loop` reflects the parquet position of the
batch being put into the queue, not the batch being consumed. The consumer is
always 1–2 batches behind the worker. On resume, `_document_batches` fast-forwards
to `resume_state_dict` and skips one row group (the `+1` advance already in
`_document_batches`), so the approximate-resume guarantee is unchanged.

This is the same approximation as the current torch loader — exact batch
reproduction is not guaranteed by design (best-fit packing is order-dependent
on the doc buffer state).

### Option D — mx.compile the transfer

`mx.array()` from CPU memory is already a direct DMA transfer into Metal memory.
Nothing to compile. Not applicable.

---

## Recommendation

| Priority | Action | Gain | Effort |
|----------|--------|------|--------|
| ✅ Done | Option A: remove redundant cpu→cpu copy | Correctness fix, ~0.07ms/batch | Trivial |
| Later | Option C: threaded MLX dataloader | Eliminates dataloader from critical path entirely | Medium |
| Skip | Option B: numpy packer without threading | 0.16% step time | Medium, low ROI standalone |

**Option A** is done — redundant `gpu_buffer.copy_()` removed on the CPU/MLX path,
`_torch_loader` renamed to `_loader` in `MLXTrainer`. The codebase is now in the
clean state for Option C: `_loader` is duck-typed, `_next_batch` is isolated,
`_setup_mlx` is the swap point.

**Option C** is the correct long-term design — it restores the async overlap that
CUDA gets for free via pinned memory + DMA engine, but implemented explicitly via a
background thread. Implement when compute gets faster or profiling shows dataloader
on the critical path. At that point, combine with Option B (numpy packer) since
threading is the multiplier that makes the packing speedup matter.

**Option B** alone is not worth the complexity — the 0.16% gain only materialises
if combined with threading.

## Conclusion

The dataloader is not the bottleneck and will not become one at any depth on
Apple Silicon. The compute/dataloader ratio is structurally ~1,000:1 at all
model sizes because microbatch compute time is depth-invariant — as depth
increases, `device_batch_size` shrinks proportionally, keeping each microbatch
in the same ~1,200–1,650 ms range while dataloader cost stays fixed at 1.2 ms.

The three CUDA-isms (pinned memory, persistent `gpu_buffer`, `non_blocking` copy)
are dead weight on the MLX path and should be removed. Option A does this
trivially. Option C (threaded `MLXDataLoader`) is the principled replacement —
it makes the overlap explicit in Python rather than relying on a DMA engine.
But given the 1,000:1 ratio, it is an engineering nicety, not a performance
necessity. Implement it when the codebase is ready for it, not before.

---

## What actually limits MLX throughput at higher depth

The narrowing MLX/MPS ratio is not the dataloader — it is compute-bound behaviour.

| Config | tok/sec (MLX) | tok/sec (MPS) | Ratio |
|--------|--------------|--------------|-------|
| d6     | 44,920       | 18,757       | 2.39× |
| d8     | ~27,000      | ~13,500      | ~2.0× |
| d12    | ~11,500      | ~7,100       | ~1.6× |

Likely causes:

1. **Memory bandwidth saturation**: d12 has 286M params in bfloat16 = ~572 MB of
   weights. At 32 accum steps the optimizer reads/writes all weights 32× per step.
   M3 Max memory bandwidth is ~400 GB/s — this may be the ceiling.

2. **mx.compile graph size**: larger graphs take longer to compile and may produce
   less optimal Metal kernels. JIT warmup at d12 is 43s vs 14s at d6.

3. **Gradient accumulation tree traversal**: at d12 with 32 accum steps,
   `nn.utils.tree_map` runs 31 times per step, each traversing the full parameter
   tree (~286M params).

These are separate investigations from the dataloader.
