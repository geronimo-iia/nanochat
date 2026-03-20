"""
MLX MuonAdamW optimizer — port of training/optimizer.py for Apple Silicon.

AdamW groups: one mlx.optimizers.AdamW instance per group (bias_correction=True).
Muon groups: manual Polar Express + NorMuon + cautious weight decay.

Not a subclass of mlx.optimizers.Optimizer — Muon requires stacked group ops,
incompatible with the per-leaf apply_single interface.
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Polar Express coefficients — same as PyTorch (num_iters=5, safety_factor=2e-2, cushion=2)
# https://arxiv.org/pdf/2505.16932
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _lerp(a: mx.array, b: mx.array, t: float) -> mx.array:
    return a + t * (b - a)


def _mT(x: mx.array) -> mx.array:
    return mx.swapaxes(x, -2, -1)


@mx.compile
def muon_step(
    stacked_grads: mx.array,       # (K, rows, cols)
    stacked_params: mx.array,      # (K, rows, cols)
    momentum_buffer: mx.array,     # (K, rows, cols)
    second_momentum_buffer: mx.array,  # (K, rows, 1) or (K, 1, cols)
    momentum: float,
    lr: float,
    weight_decay: float,
    beta2: float,
    ns_steps: int,
    red_dim: int,                  # -1 (tall) or -2 (wide)
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Muon step: Nesterov momentum → Polar Express → NorMuon variance reduction
    → cautious weight decay update.

    Returns updated (stacked_params, momentum_buffer, second_momentum_buffer).
    All inputs/outputs are immutable mx.arrays — no in-place mutation.
    """
    # Nesterov momentum
    momentum_buffer = _lerp(momentum_buffer, stacked_grads, 1 - momentum)
    g = _lerp(stacked_grads, momentum_buffer, momentum)

    # Polar Express orthogonalization
    x = g.astype(mx.bfloat16)
    x = x / (mx.linalg.norm(x, axis=(-2, -1), keepdims=True) * 1.01 + 1e-6)
    if g.shape[-2] > g.shape[-1]:  # tall matrix
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            m = _mT(x) @ x
            n = b * m + c * (m @ m)
            x = a * x + x @ n
    else:  # wide matrix
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            m = x @ _mT(x)
            n = b * m + c * (m @ m)
            x = a * x + n @ x
    g = x

    # NorMuon variance reduction
    g_f = g.astype(mx.float32)
    v_mean = mx.mean(mx.square(g_f), axis=red_dim, keepdims=True)
    red_dim_size = g.shape[red_dim]
    v_norm_sq = mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size
    v_norm = mx.sqrt(v_norm_sq)

    second_momentum_buffer = _lerp(
        second_momentum_buffer.astype(mx.float32),
        v_mean,
        1 - beta2,
    ).astype(second_momentum_buffer.dtype)

    step_size = mx.rsqrt(mx.maximum(second_momentum_buffer.astype(mx.float32), 1e-10))
    scaled_sq_sum = (v_mean * red_dim_size) * mx.square(step_size)
    v_norm_new = mx.sqrt(mx.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
    g = (g_f * final_scale).astype(g.dtype)

    # Cautious weight decay + parameter update
    g = g.astype(stacked_params.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params = stacked_params - lr * g - lr * weight_decay * stacked_params * mask

    return stacked_params, momentum_buffer, second_momentum_buffer


class MuonAdamW:
    """
    Combined MLX optimizer: Muon for matrix params, AdamW for embeddings/scalars.

    Param group interface mirrors PyTorch MuonAdamW, with one addition:
    each group must have a "_keys" field listing the flat parameter key strings
    (as returned by nn.utils.tree_flatten) that belong to this group.

      kind="adamw": lr, betas, eps, weight_decay, _keys
      kind="muon":  lr, momentum, ns_steps, beta2, weight_decay, _keys
                    all params in a muon group must have the same shape

    Use build_param_groups(model, ...) to construct groups with _keys injected.

    Usage:
        optimizer = MuonAdamW(build_param_groups(model, ...))
        # inside training loop:
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state())
    """

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        self._groups = param_groups
        # One mlx.optimizers.AdamW per AdamW group
        self._adamw: list[optim.AdamW | None] = []
        # Muon state keyed by group index: momentum_buffer, second_momentum_buffer
        self._muon_state: dict[int, dict[str, mx.array]] = {}

        for i, g in enumerate(param_groups):
            if g["kind"] == "adamw":
                self._adamw.append(
                    optim.AdamW(
                        learning_rate=g["lr"],
                        betas=g["betas"],
                        eps=g["eps"],
                        weight_decay=g["weight_decay"],
                        bias_correction=True,
                    )
                )
            elif g["kind"] == "muon":
                self._adamw.append(None)
            else:
                raise ValueError(f"Unknown optimizer kind: {g['kind']}")

    def state(self) -> list:
        """Return all optimizer state arrays for mx.eval()."""
        out = []
        for adamw in self._adamw:
            if adamw is not None:
                flat, _ = nn.utils.tree_flatten(adamw.state)
                out.extend(v for v in flat if isinstance(v, mx.array))
        for s in self._muon_state.values():
            out.extend(s.values())
        return out

    def update(self, model: nn.Module, grads: dict) -> None:
        """Apply gradients to model parameters in-place."""
        # Flatten model params and grads to name→array dicts
        params_flat = dict(nn.utils.tree_flatten(model.trainable_parameters()))
        grads_flat = dict(nn.utils.tree_flatten(grads))

        updates: dict[str, mx.array] = {}

        for i, group in enumerate(self._groups):
            if group["kind"] == "adamw":
                self._step_adamw(i, group, params_flat, grads_flat, updates)
            else:
                self._step_muon(i, group, params_flat, grads_flat, updates)

        model.update(nn.utils.tree_unflatten(list(updates.items())))

    def _step_adamw(
        self,
        idx: int,
        group: dict[str, Any],
        params_flat: dict[str, mx.array],
        grads_flat: dict[str, mx.array],
        updates: dict[str, mx.array],
    ) -> None:
        adamw = self._adamw[idx]
        assert adamw is not None

        # Update LR in case it was changed by the scheduler
        adamw.learning_rate = mx.array(group["lr"])

        # Build sub-trees for just this group's params
        group_params = {k: params_flat[k] for k in group["_keys"] if k in grads_flat}
        group_grads = {k: grads_flat[k] for k in group_params}

        updated = adamw.apply_gradients(group_grads, group_params)
        updates.update(updated)

    def _step_muon(
        self,
        idx: int,
        group: dict[str, Any],
        params_flat: dict[str, mx.array],
        grads_flat: dict[str, mx.array],
        updates: dict[str, mx.array],
    ) -> None:
        keys = [k for k in group["_keys"] if k in grads_flat]
        if not keys:
            return

        param_list = [params_flat[k] for k in keys]
        grad_list = [grads_flat[k] for k in keys]
        shape = param_list[0].shape

        stacked_params = mx.stack(param_list)   # (K, rows, cols)
        stacked_grads = mx.stack(grad_list)     # (K, rows, cols)

        # Lazy state init
        if idx not in self._muon_state:
            K = len(param_list)
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            v_shape = (K, shape[-2], 1) if shape[-2] >= shape[-1] else (K, 1, shape[-1])
            self._muon_state[idx] = {
                "momentum_buffer": mx.zeros((K, *shape)),
                "second_momentum_buffer": mx.zeros(v_shape),
                "red_dim": red_dim,
            }

        state = self._muon_state[idx]
        red_dim: int = state["red_dim"]

        # LR scaled by sqrt(max(1, rows/cols)) — same as PyTorch
        lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

        stacked_params, new_mom, new_v = muon_step(
            stacked_grads=stacked_grads,
            stacked_params=stacked_params,
            momentum_buffer=state["momentum_buffer"],
            second_momentum_buffer=state["second_momentum_buffer"],
            momentum=group["momentum"],
            lr=lr,
            weight_decay=group["weight_decay"],
            beta2=group["beta2"],
            ns_steps=group["ns_steps"],
            red_dim=red_dim,
        )

        state["momentum_buffer"] = new_mom
        state["second_momentum_buffer"] = new_v

        for k, p in zip(keys, stacked_params):
            updates[k] = p


def build_param_groups(
    model: nn.Module,
    unembedding_lr: float = 0.004,
    embedding_lr: float = 0.2,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.0,
    scalar_lr: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Build MuonAdamW param groups for an MLX GPT model.

    Mirrors GPT.setup_optimizer from models/gpt.py. Returns groups with
    _keys populated (flat key strings from nn.utils.tree_flatten).

    AdamW groups: lm_head, wte, value_embeds, resid_lambdas, x0_lambdas,
                  smear_gate, smear_lambda, backout_lambda
    Muon groups:  all block weights, grouped by shape
    """
    from nanochat.models.config import GPTConfig

    config: GPTConfig = model.config
    model_dim = config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    flat = dict(nn.utils.tree_flatten(model.trainable_parameters()))

    def keys_matching(prefix: str) -> list[str]:
        return [k for k in flat if k.startswith(prefix)]

    def keys_exact(names: list[str]) -> list[str]:
        return [k for k in flat if k in names]

    lm_head_keys = keys_matching("lm_head.")
    wte_keys = keys_matching("wte.")
    ve_keys = keys_matching("value_embeds.ve_")
    resid_keys = keys_exact(["resid_lambdas"])
    x0_keys = keys_exact(["x0_lambdas"])
    smear_keys = keys_exact(["smear_gate.weight", "smear_lambda", "backout_lambda"])
    block_keys = keys_matching("blocks.")

    # Group block weights by shape for Muon stacking
    shapes: dict[tuple, list[str]] = {}
    for k in block_keys:
        s = tuple(flat[k].shape)
        shapes.setdefault(s, []).append(k)

    groups: list[dict[str, Any]] = [
        dict(kind="adamw", _keys=lm_head_keys,
             lr=unembedding_lr * dmodel_lr_scale,
             betas=[0.8, 0.96], eps=1e-10, weight_decay=0.01),
        dict(kind="adamw", _keys=wte_keys,
             lr=embedding_lr * dmodel_lr_scale,
             betas=[0.8, 0.995], eps=1e-10, weight_decay=0.001),
        dict(kind="adamw", _keys=ve_keys,
             lr=embedding_lr * dmodel_lr_scale * 0.5,
             betas=[0.8, 0.995], eps=1e-10, weight_decay=0.01),
        dict(kind="adamw", _keys=resid_keys,
             lr=scalar_lr * 0.01, betas=[0.8, 0.95], eps=1e-10, weight_decay=0.05),
        dict(kind="adamw", _keys=x0_keys,
             lr=scalar_lr, betas=[0.96, 0.95], eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", _keys=smear_keys,
             lr=0.2, betas=[0.8, 0.95], eps=1e-10, weight_decay=0.0),
    ]

    for shape_keys in sorted(shapes.values(), key=lambda ks: ks[0]):
        groups.append(dict(
            kind="muon", _keys=shape_keys,
            lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9,
            weight_decay=weight_decay,
        ))

    return groups
