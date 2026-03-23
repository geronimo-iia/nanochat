"""
Validate whether mx.compile + MuonAdamW introduces NaN during training.

Tests three scenarios in bfloat16 (the real training dtype on Apple Silicon):
  A) Plain grad + AdamW          — baseline
  B) Compiled grad + AdamW       — isolates compile effect
  C) Compiled grad + MuonAdamW   — full real training stack

The Muon optimizer does bfloat16 Polar Express matrix multiplications with
large coefficients (a~8, b~-22, c~15), which can overflow and produce NaN
when gradient magnitudes grow after several training steps.

Run with:  uv run python test_compiled_nan.py
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Put the project src on path so we can import mlx_optimizer
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from nanochat.training.mlx_optimizer import MuonAdamW  # noqa: E402

VOCAB   = 4096
SEQ     = 512
BATCH   = 4
N_EMBD  = 256
N_HEAD  = 4
N_LAYER = 4
STEPS   = 60


# ---------------------------------------------------------------------------
# Model — mirrors mlx_gpt.py: rms_norm, relu², softcap, residual scalars
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x, weight=None, eps=1e-5)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc   = nn.Linear(N_EMBD, N_EMBD * 4, bias=False)
        self.c_proj = nn.Linear(N_EMBD * 4, N_EMBD, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.relu(self.c_fc(x)) ** 2)


class Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_q    = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.c_k    = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.c_v    = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        hd = C // N_HEAD
        q = self.c_q(x).reshape(B, T, N_HEAD, hd).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, N_HEAD, hd).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, N_HEAD, hd).transpose(0, 2, 1, 3)
        q = rms_norm(q) * 1.2
        k = rms_norm(k) * 1.2
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5, mask="causal")
        return self.c_proj(y.transpose(0, 2, 1, 3).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attn()
        self.mlp  = MLP()

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wte    = nn.Embedding(VOCAB, N_EMBD)
        self.blocks = [Block() for _ in range(N_LAYER)]
        self.head   = nn.Linear(N_EMBD, VOCAB, bias=False)
        # residual scalars (same as mlx_gpt.py)
        self.resid_lambdas = mx.ones((N_LAYER,))
        self.x0_lambdas    = mx.zeros((N_LAYER,))

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        h = rms_norm(self.wte(x))
        x0 = h
        for i, b in enumerate(self.blocks):
            h = self.resid_lambdas[i] * h + self.x0_lambdas[i] * x0
            h = b(h)
        h = rms_norm(h)
        logits = 15 * mx.tanh(self.head(h) / 15)   # softcap
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, VOCAB), y.reshape(-1)
        )
        return loss.mean()


# ---------------------------------------------------------------------------
# Exact _LossAndGrad pattern from mlx_trainer.py
# ---------------------------------------------------------------------------

class _LossAndGrad(nn.Module):
    def __init__(self, model: TinyGPT) -> None:
        super().__init__()
        self._lag = nn.value_and_grad(model, model)

    def __call__(self, x: mx.array, y: mx.array):
        return self._lag(x, y)


class _MultiStepLossAndGrad(nn.Module):
    """Accumulates value and gradient over N mini-batches in a single compiled call.

    The Python for loop is unrolled at mx.compile trace time into one static graph.
    """

    def __init__(self, model: TinyGPT, grad_accum_steps: int) -> None:
        super().__init__()
        self._N = grad_accum_steps
        self._lag = nn.value_and_grad(model, model)

    def __call__(self, xs: mx.array, ys: mx.array):
        # xs: (N, B, T), ys: (N, B, T)
        accumulated_grads = None
        total_loss = mx.array(0.0)
        for i in range(self._N):
            loss_i, grads_i = self._lag(xs[i], ys[i])
            total_loss = total_loss + loss_i
            if accumulated_grads is None:
                accumulated_grads = grads_i
            else:
                accumulated_grads = nn.utils.tree_map(
                    lambda a, b: a + b, accumulated_grads, grads_i
                )
        assert accumulated_grads is not None
        return (
            total_loss / self._N,
            nn.utils.tree_map(lambda g: g / self._N, accumulated_grads),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cast_model_bf16(model: nn.Module) -> None:
    """Cast all float32 parameters to bfloat16, matching real Apple Silicon training."""
    updates = {
        k: v.astype(mx.bfloat16)
        for k, v in nn.utils.tree_flatten(model.parameters())
        if v.dtype == mx.float32
    }
    model.update(nn.utils.tree_unflatten(list(updates.items())))
    mx.eval(model.parameters())


def has_nan(tree) -> bool:
    flat = [v for _, v in nn.utils.tree_flatten(tree)]
    return any(bool(mx.any(mx.isnan(v.astype(mx.float32))).item()) for v in flat)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def run_adamw(compiled: bool, steps: int = STEPS) -> None:
    label = f"{'COMPILED' if compiled else 'PLAIN   '} + AdamW    "
    mx.random.seed(42)
    model = TinyGPT()
    cast_model_bf16(model)

    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)

    lag = _LossAndGrad(model)
    if compiled:
        lag = mx.compile(lag, inputs=[model], outputs=[model])

    for step in range(steps):
        x = mx.random.randint(0, VOCAB, (BATCH, SEQ))
        y = mx.random.randint(0, VOCAB, (BATCH, SEQ))
        loss, grads = lag(x, y)
        mx.eval(loss, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        lv = loss.item()
        gn = has_nan(grads)
        pn = has_nan(model.parameters())
        flag = " *** NaN ***" if (gn or pn or lv != lv) else ""
        print(f"{label} step {step:03d} | loss={lv:.4f} | grad_nan={gn} | param_nan={pn}{flag}")
        if gn or pn or lv != lv:
            return
    print(f"{label} => no NaN after {steps} steps")


def run_muon(compiled: bool, steps: int = STEPS) -> None:
    label = f"{'COMPILED' if compiled else 'PLAIN   '} + MuonAdamW"
    mx.random.seed(42)
    model = TinyGPT()
    cast_model_bf16(model)

    # Build param groups using build_param_groups logic adapted for TinyGPT
    flat = dict(nn.utils.tree_flatten(model.trainable_parameters()))
    block_keys = [k for k in flat if k.startswith("blocks.")]
    adamw_keys  = [k for k in flat if not k.startswith("blocks.")]

    # Group block weights by shape for Muon stacking
    shapes: dict[tuple, list[str]] = {}
    for k in block_keys:
        s = tuple(flat[k].shape)
        shapes.setdefault(s, []).append(k)

    groups = [
        dict(kind="adamw", _keys=adamw_keys, lr=3e-4,
             betas=[0.9, 0.99], eps=1e-8, weight_decay=0.1),
    ]
    for shape_keys in sorted(shapes.values(), key=lambda ks: ks[0]):
        groups.append(dict(
            kind="muon", _keys=shape_keys, lr=0.02,
            momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0,
        ))
    for g in groups:
        g["initial_lr"] = g["lr"]

    optimizer = MuonAdamW(groups)

    lag = _LossAndGrad(model)
    if compiled:
        lag = mx.compile(lag, inputs=[model], outputs=[model])

    for step in range(steps):
        x = mx.random.randint(0, VOCAB, (BATCH, SEQ))
        y = mx.random.randint(0, VOCAB, (BATCH, SEQ))
        loss, grads = lag(x, y)
        mx.eval(loss, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state())

        lv = loss.item()
        gn = has_nan(grads)
        pn = has_nan(model.parameters())
        flag = " *** NaN ***" if (gn or pn or lv != lv) else ""
        print(f"{label} step {step:03d} | loss={lv:.4f} | grad_nan={gn} | param_nan={pn}{flag}")
        if gn or pn or lv != lv:
            return
    print(f"{label} => no NaN after {steps} steps")

def check_compile_nan():
    print("Running NaN tests for compiled vs. plain gradients with AdamW and MuonAdamW optimizers.")
    print("This may take a few minutes...")
    print()    
    print("=" * 65)
    print("A) PLAIN grad + AdamW  (baseline)")
    print("=" * 65)
    run_adamw(compiled=False)

    print()
    print("=" * 65)
    print("B) COMPILED grad + AdamW  (isolates compile effect)")
    print("=" * 65)
    run_adamw(compiled=True)

    print()
    print("=" * 65)
    print("C) PLAIN grad + MuonAdamW  (Polar Express bfloat16)")
    print("=" * 65)
    run_muon(compiled=False)

    print()
    print("=" * 65)
    print("D) COMPILED grad + MuonAdamW  (full real training stack)")
    print("=" * 65)
    run_muon(compiled=True)


def check_fused_grads_match():
    mx.random.seed(42)
    model_eager = TinyGPT(); cast_model_bf16(model_eager)
    model_fused = TinyGPT(); cast_model_bf16(model_fused)
    # Copy identical weights into both models
    model_fused.update(nn.utils.tree_unflatten(list(nn.utils.tree_flatten(model_eager.parameters()))))
    mx.eval(model_eager.parameters(), model_fused.parameters())

    N = 4  # grad_accum_steps
    xs = [mx.random.randint(0, VOCAB, (BATCH, SEQ)) for _ in range(N)]
    ys = [mx.random.randint(0, VOCAB, (BATCH, SEQ)) for _ in range(N)]

    # Eager: N separate compiled calls (mirrors current MLXTrainer behaviour)
    lag_eager = _LossAndGrad(model_eager)
    compiled_eager = mx.compile(lag_eager, inputs=[model_eager], outputs=[model_eager])
    accum_eager = None
    for x, y in zip(xs, ys):
        _, g = compiled_eager(x, y)
        mx.eval(g)
        accum_eager = g if accum_eager is None else nn.utils.tree_map(
            lambda a, b: a + b, accum_eager, g
        )
    accum_eager = nn.utils.tree_map(lambda g: g / N, accum_eager)
    mx.eval(accum_eager)

    # Fused: single compiled call with all N mini-batches
    multi_lag = _MultiStepLossAndGrad(model_fused, N)
    compiled_fused = mx.compile(multi_lag, inputs=[model_fused], outputs=[model_fused])
    xs_stack = mx.stack(xs)
    ys_stack = mx.stack(ys)
    _, accum_fused = compiled_fused(xs_stack, ys_stack)
    mx.eval(accum_fused)

    # Compare: flatten both grad trees to (key, array) pairs for element-wise comparison
    eager_flat = dict(nn.utils.tree_flatten(accum_eager))
    fused_flat = dict(nn.utils.tree_flatten(accum_fused))

    assert set(eager_flat.keys()) == set(fused_flat.keys()), \
        f"Grad key mismatch: {set(eager_flat.keys()) ^ set(fused_flat.keys())}"

    max_diff_overall = 0.0
    for k in sorted(eager_flat.keys()):
        e = eager_flat[k].astype(mx.float32)
        f = fused_flat[k].astype(mx.float32)
        diff = mx.max(mx.abs(e - f)).item()
        max_diff_overall = max(max_diff_overall, diff)
        if diff >= 1e-3:
            print(f"  [FAIL] {k}: max_diff={diff:.6f}")
        else:
            print(f"  [OK]   {k}: max_diff={diff:.6f}")

    assert max_diff_overall < 1e-3, f"Fused grads diverge: max_diff={max_diff_overall:.6f}"
    print(f"\nFused grads match eager grads — OK (max_diff={max_diff_overall:.6f})")

check_fused_grads_match()


def check_weight_trajectory_match():
    """Verify that 20 optimizer steps with fused and eager accumulation produce
    identical model weight trajectories, using the same batches and initial weights."""
    STEPS = 20
    N = 4  # grad_accum_steps

    # Pre-generate all batches so both paths consume identical data.
    mx.random.seed(42)
    all_xs = [[mx.random.randint(0, VOCAB, (BATCH, SEQ)) for _ in range(N)] for _ in range(STEPS)]
    all_ys = [[mx.random.randint(0, VOCAB, (BATCH, SEQ)) for _ in range(N)] for _ in range(STEPS)]
    mx.eval(all_xs, all_ys)

    # Build a reference model whose weights both paths will start from.
    mx.random.seed(7)
    ref = TinyGPT(); cast_model_bf16(ref); mx.eval(ref.parameters())
    ref_weights = list(nn.utils.tree_flatten(ref.parameters()))

    def make_model():
        m = TinyGPT(); cast_model_bf16(m)
        m.update(nn.utils.tree_unflatten([(k, v) for k, v in ref_weights]))
        mx.eval(m.parameters())
        return m

    # --- Eager path (mirrors current MLXTrainer.forward_backward) ---
    model_e = make_model()
    opt_e = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)
    lag_e = _LossAndGrad(model_e)
    comp_e = mx.compile(lag_e, inputs=[model_e], outputs=[model_e])

    for step in range(STEPS):
        accum_grads = None
        for i in range(N):
            _, g = comp_e(all_xs[step][i], all_ys[step][i])
            mx.eval(g)
            accum_grads = g if accum_grads is None else nn.utils.tree_map(
                lambda a, b: a + b, accum_grads, g
            )
        accum_grads = nn.utils.tree_map(lambda g: g / N, accum_grads)
        opt_e.update(model_e, accum_grads)
        mx.eval(model_e.parameters(), opt_e.state)

    # --- Fused path ---
    model_f = make_model()
    opt_f = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)
    multi_lag = _MultiStepLossAndGrad(model_f, N)
    comp_f = mx.compile(multi_lag, inputs=[model_f], outputs=[model_f])

    for step in range(STEPS):
        xs = mx.stack(all_xs[step])
        ys = mx.stack(all_ys[step])
        _, grads = comp_f(xs, ys)
        mx.eval(grads)
        opt_f.update(model_f, grads)
        mx.eval(model_f.parameters(), opt_f.state)

    # Compare final weights.
    eager_flat = dict(nn.utils.tree_flatten(model_e.parameters()))
    fused_flat  = dict(nn.utils.tree_flatten(model_f.parameters()))

    max_diff = 0.0
    for k in sorted(eager_flat.keys()):
        diff = mx.max(mx.abs(
            eager_flat[k].astype(mx.float32) - fused_flat[k].astype(mx.float32)
        )).item()
        max_diff = max(max_diff, diff)
        status = "OK" if diff < 1e-2 else "FAIL"
        print(f"  [{status}]  {k}: max_diff={diff:.6f}")

    assert max_diff < 1e-2, f"Weight trajectories diverge: max_diff={max_diff:.6f}"
    print(f"\nWeight trajectories match — OK (max_diff={max_diff:.6f})")


def benchmark_fused_vs_eager():
    """Compare tok/sec for fused vs eager accumulation over N mini-batches.

    Both paths time only forward/backward + mx.eval (no optimizer step)
    to isolate the accumulation loop cost.
    Note: uses TinyGPT (small); results are relative, not absolute.
    """
    import time

    N = 4
    WARMUP = 5
    BENCH  = 15

    for use_fused, label in [(False, "Eager (N separate calls) "), (True, "Fused (1 compiled call)  ")]:
        mx.random.seed(42)
        model = TinyGPT(); cast_model_bf16(model)
        optimizer = optim.AdamW(learning_rate=3e-4)

        if use_fused:
            multi_lag = _MultiStepLossAndGrad(model, N)
            compiled = mx.compile(multi_lag, inputs=[model], outputs=[model])
        else:
            lag = _LossAndGrad(model)
            compiled = mx.compile(lag, inputs=[model], outputs=[model])

        total_time = 0.0
        for step in range(WARMUP + BENCH):
            if use_fused:
                xs = mx.stack([mx.random.randint(0, VOCAB, (BATCH, SEQ)) for _ in range(N)])
                ys = mx.stack([mx.random.randint(0, VOCAB, (BATCH, SEQ)) for _ in range(N)])
                t0 = time.perf_counter()
                loss, grads = compiled(xs, ys)
                mx.eval(loss, grads)
                dt = time.perf_counter() - t0
            else:
                xs_list = [mx.random.randint(0, VOCAB, (BATCH, SEQ)) for _ in range(N)]
                ys_list = [mx.random.randint(0, VOCAB, (BATCH, SEQ)) for _ in range(N)]
                t0 = time.perf_counter()
                accum_grads = None
                for x, y in zip(xs_list, ys_list):
                    loss, g = compiled(x, y)
                    mx.eval(loss, g)
                    accum_grads = g if accum_grads is None else nn.utils.tree_map(
                        lambda a, b: a + b, accum_grads, g
                    )
                grads = nn.utils.tree_map(lambda g: g / N, accum_grads)
                dt = time.perf_counter() - t0

            if step >= WARMUP:
                total_time += dt

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        tokens_per_step = N * BATCH * SEQ
        avg_dt = total_time / BENCH
        tok_sec = tokens_per_step / avg_dt
        print(f"  {label}: {tok_sec:>10,.0f} tok/sec  (avg {avg_dt * 1000:.1f} ms/step)")


print()
print("=" * 65)
print("F) Weight trajectory equivalence — 20 optimizer steps")
print("=" * 65)
check_weight_trajectory_match()

print()
print("=" * 65)
print("G) tok/sec benchmark — eager vs fused accumulation")
print("=" * 65)
benchmark_fused_vs_eager()
