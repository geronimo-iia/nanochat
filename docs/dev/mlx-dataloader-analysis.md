---
title: MLX Dataloader — Analysis & Optimization
summary: Complete analysis of the dataloader pipeline for the MLX backend. Covers the redundant CPU copy, transfer path options, numpy vs torch packing performance, threading opportunity, and a dedicated MLX dataloader design.
read_when:
  - Investigating MLX training throughput bottlenecks
  - Considering a dedicated MLX dataloader implementation
  - Evaluating dataloader threading for the MLX path
status: draft
last_updated: 2025-07-10
---

# MLX Dataloader — Analysis & Optimization

---

## Current pipeline

```
Parquet → tokenizer → doc_buffer
  → best-fit packer (torch, row_buffer)
  → cpu_buffer (torch, CPU)
  → gpu_buffer.copy_(cpu_buffer)   ← redundant CPU→CPU memcpy
  → yield (inputs, targets)        ← torch tensors on CPU

MLXTrainer._next_batch():
  x, y, state = next(self._torch_loader)
  return mx.array(x.numpy()), mx.array(y.numpy()), state
                                   ← .numpy() is zero-copy (shared memory)
                                   ← mx.array() copies into Metal memory
```

The dataloader was designed for CUDA: `gpu_buffer` lives on the GPU and
`gpu_buffer.copy_(cpu_buffer, non_blocking=True)` is an async HtoD transfer.
On the MLX path `device=cpu`, so `gpu_buffer` is also a CPU tensor — the copy
is a pointless CPU→CPU memcpy that touches 2×B×T×8 bytes for no reason.

---

## Measured costs (M3 Max, B=64, T=1024)

All timings are per micro-batch (one `_next_batch()` call).

| Path | Cost |
|------|------|
| Current: torch pack + cpu→cpu copy + `.numpy()` + `mx.array()` | 3.5 ms |
| Fixed: torch pack + `.numpy()` + `mx.array()` (skip copy) | 3.4 ms |
| Numpy pack + `mx.array()` (no torch in hot path) | 1.2 ms |
| Transfer only (`mx.array()` from pre-built buffer) | 0.02 ms |

Key insight: the CPU→CPU copy costs only ~0.07 ms. The dominant cost is
**torch tensor packing** (3.5 ms total vs 1.2 ms with numpy). Torch is ~3×
slower than numpy for the inner packing loop because each
`torch.tensor(doc, dtype=torch.long)` allocates a new tensor per document
fragment.

---

## Impact on step time

Dataloader runs **sequentially** on the main thread — it is called after
`mx.eval()` returns, not overlapped with compute.

| Config | Step time | Accum steps | Dataloader/step (current) | Dataloader/step (numpy) |
|--------|-----------|-------------|--------------------------|------------------------|
| d6     | 11,600 ms | 8           | 28 ms (0.24%)            | 10 ms (0.09%)          |
| d8     | 19,000 ms | 16          | 56 ms (0.29%)            | 19 ms (0.10%)          |
| d12    | 45,000 ms | 32          | 112 ms (0.25%)           | 38 ms (0.08%)          |

**Conclusion**: even with the current torch packer, dataloader is <0.3% of
step time. Switching to numpy saves ~0.15% of step time. Neither is a
meaningful throughput bottleneck at current model sizes.

---

## Why the gap is small

MLX compute dominates completely. A single forward+backward pass at d6 takes
~11,600 ms. The dataloader for that same step takes ~28 ms total across 8
accum steps. The ratio is ~240:1.

The dataloader would only become a bottleneck if:
1. Model compute gets much faster (e.g. 10× from future MLX improvements), or
2. Sequence length or batch size increases dramatically, or
3. The tokenizer/parquet IO becomes slow (large datasets, cold cache)

---

## Option A — Fix the redundant copy (minimal change)

Remove `gpu_buffer` from the MLX path. Yield directly from `cpu_buffer`.

**Change in `dataloader.py`**: when `device == cpu`, skip `gpu_buffer`
allocation and `gpu_buffer.copy_()`. Yield views into `cpu_buffer` directly.

**Change in `mlx_trainer.py`**: `_next_batch` already does `.numpy()` +
`mx.array()` — no change needed.

**Savings**: ~0.07 ms per micro-batch (the cpu→cpu copy). Negligible at
current scale. Worth doing for correctness (no pointless allocation/copy) but
not for performance.

---

## Option B — Numpy packer (no torch in hot path)

Replace `row_buffer` (torch) with `row_buffer_np` (numpy). Replace
`torch.tensor(doc)` slice assignments with direct numpy slice assignments.
Yield `cpu_buffer_np` views directly.

**Savings**: ~2.3 ms per micro-batch (3.5 ms → 1.2 ms). Across 8 accum steps
at d6: saves ~18 ms per step = 0.16% of step time.

**Tradeoff**: the dataloader becomes MLX-specific (no longer usable for the
torch path without a wrapper). Adds code complexity for negligible gain at
current scale.

---

## Option C — Dedicated MLX dataloader

A fully MLX-native dataloader that:
1. Packs directly into a numpy buffer (no torch in hot path)
2. Yields `mx.array` directly (no intermediate torch tensors)
3. Runs on a background thread, prefetching the next batch while compute runs

**Design**:

```python
class MLXDataLoader:
    def __init__(self, tokenizer, B, T, split, resume_state_dict=None):
        self._buf = np.empty(2 * B * T, dtype=np.int64)
        self._row_buf = np.empty((B, T + 1), dtype=np.int64)
        # background thread prefetch
        self._queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        for batch in self._pack_loop():
            self._queue.put(batch)  # blocks if queue full

    def __next__(self):
        return self._queue.get()   # mx.array already built on worker thread
```

**Threading benefit**: the packing loop (~1.2 ms per micro-batch) runs
concurrently with `mx.eval()` (~11,600 ms at d6). The next batch is ready
before `mx.eval()` returns. Dataloader cost drops to **zero** on the critical
path.

**Measured**: dataloader (1.2 ms) << compute (11,600 ms). A queue depth of 2
is sufficient — the worker stays ahead of the consumer at all model sizes.

**Tradeoff**: threading adds complexity. The worker must handle exceptions
cleanly and the queue must be drained on stop. Resume state becomes slightly
more complex (need to track which batch the consumer is on, not the worker).

---

## Option D — mx.compile the dataloader

MLX arrays support `mx.compile`. In principle the transfer
`mx.array(np_buf)` could be compiled into a persistent Metal kernel.
In practice `mx.array()` from CPU memory is already a direct DMA transfer —
there is nothing to compile. Not applicable.

---

## Recommendation

| Priority | Action | Gain | Effort |
|----------|--------|------|--------|
| Now | Option A: remove redundant cpu→cpu copy | Correctness, ~0.07ms/batch | Trivial |
| Later | Option C: threaded MLX dataloader | Eliminates dataloader from critical path | Medium |
| Skip | Option B: numpy packer without threading | 0.16% step time | Medium, low ROI |

**Option A** is a one-line fix and removes a semantic bug (cpu→cpu copy that
pretends to be a HtoD transfer). Do it now.

**Option C** is the right long-term design for the MLX path. The threading
model is simple because MLX compute is single-threaded (no GIL contention on
the compute side). Implement when model compute gets faster or when profiling
shows dataloader on the critical path.

**Option B** alone is not worth the complexity — the gain only materialises
if combined with threading (Option C), at which point you'd do both together.

---

## What actually limits MLX throughput at higher depth

The benchmark shows throughput drops with depth:

| Config | tok/sec (MLX) | tok/sec (MPS) | Ratio |
|--------|--------------|--------------|-------|
| d6     | 44,920       | 18,757       | 2.39× |
| d8     | ~27,000      | ~13,500      | ~2.0× |
| d12    | ~11,500      | ~7,100       | ~1.6× |

The MLX/MPS ratio narrows at higher depth. This is not the dataloader — it is
compute-bound behaviour. Likely causes:

1. **Memory bandwidth saturation**: d12 has 286M params in bfloat16 = ~572 MB
   of weights. At 32 accum steps, the optimizer reads/writes all weights 32×
   per step. The M3 Max memory bandwidth is ~400 GB/s — this may be the
   ceiling.

2. **mx.compile graph size**: larger graphs take longer to compile and may
   produce less optimal Metal kernels. The JIT warmup at d12 is 43s vs 14s
   at d6.

3. **Gradient accumulation overhead**: at d12 with 32 accum steps, the
   `nn.utils.tree_map` gradient accumulation loop runs 31 times per step.
   Each call traverses the full parameter tree.

These are separate investigations. The dataloader is not the bottleneck.
