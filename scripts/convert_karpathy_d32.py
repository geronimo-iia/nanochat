"""Convert karpathy/nanochat-d32 upstream checkpoint to our MLX-compatible format.

The upstream checkpoint is a bare vanilla GPT (no value_embeds, smear_gate, etc.).
This script:
  1. Remaps key names  (transformer.h.N.* → blocks.N.*,  transformer.wte → wte)
  2. Adds zero-init tensors for all our architectural extras
  3. Saves as model_000650.pt (dict of numpy arrays) + meta_000650.json

Usage:
    cd forge/nanochat
    uv run python scripts/convert_karpathy_d32.py
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file

BASE = "/Users/geronimo/build/sp_theory/experiments/nanochat"
SRC = os.path.join(BASE, "checkpoints/base/d32/model_000650.pt")
DST_DIR = os.path.join(BASE, "checkpoints/base/d32")

# d32 architecture constants (from meta_000650.json)
N_LAYER = 32
N_EMBD = 2048
N_HEAD = 16
N_KV_HEAD = 16
VOCAB_SIZE = 65536
HEAD_DIM = N_EMBD // N_HEAD          # 128
KV_DIM = N_KV_HEAD * HEAD_DIM        # 2048

# has_ve(i, 32) → i % 2 == (32-1) % 2 == 1 → odd layers
VE_LAYERS = [i for i in range(N_LAYER) if i % 2 == (N_LAYER - 1) % 2]  # [1,3,...,31]
VE_GATE_CHANNELS = 12   # CausalSelfAttention.VE_GATE_CHANNELS in mlx_gpt.py
SMEAR_GATE_CHANNELS = 24  # GPT.SMEAR_GATE_CHANNELS in mlx_gpt.py

print(f"Loading upstream checkpoint from {SRC} ...")
# weights_only=False needed if file contains numpy arrays (from a previous conversion run)
upstream = torch.load(SRC, map_location="cpu", weights_only=False)
print(f"  {len(upstream)} tensors loaded")

def to_f32(v: object) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v.astype(np.float32)
    return v.float().numpy()  # type: ignore[union-attr]

state: dict[str, np.ndarray] = {}

# ------------------------------------------------------------------
# Remap core weights: transformer.h.N.* → blocks.N.*
#                     transformer.wte   → wte
# ------------------------------------------------------------------
SUFFIX_MAP = {
    "attn.c_q.weight": "attn.c_q.weight",
    "attn.c_k.weight": "attn.c_k.weight",
    "attn.c_v.weight": "attn.c_v.weight",
    "attn.c_proj.weight": "attn.c_proj.weight",
    "mlp.c_fc.weight": "mlp.c_fc.weight",
    "mlp.c_proj.weight": "mlp.c_proj.weight",
}

# Detect format: upstream uses transformer.h.N.*, converted uses blocks.N.*
already_converted = any(k.startswith("blocks.") for k in upstream)
if already_converted:
    print("  Detected already-converted format — passing core weights through")

for k, v in upstream.items():
    arr = to_f32(v)

    if already_converted:
        # Already in our naming; pass through only the 231 model keys
        # (skip our own added zero-init keys so we can re-add them fresh below)
        ZERO_INIT_PREFIXES = ("resid_lambdas", "x0_lambdas", "smear_gate", "smear_lambda",
                               "backout_lambda", "value_embeds.ve_", "blocks.")
        is_ve_gate = "attn.ve_gate" in k
        is_zero_init = k in ("resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda") \
                        or k.startswith("smear_gate") \
                        or k.startswith("value_embeds.ve_") \
                        or is_ve_gate
        if not is_zero_init:
            state[k] = arr
        else:
            pass  # will be re-added below with correct init values
    else:
        if k == "transformer.wte.weight":
            state["wte.weight"] = arr
        elif k == "lm_head.weight":
            state["lm_head.weight"] = arr
        elif k.startswith("transformer.h."):
            # e.g. transformer.h.3.attn.c_q.weight → blocks.3.attn.c_q.weight
            rest = k[len("transformer.h."):]          # "3.attn.c_q.weight"
            dot = rest.index(".")
            layer_idx = rest[:dot]
            suffix = rest[dot + 1:]
            if suffix in SUFFIX_MAP:
                new_key = f"blocks.{layer_idx}.{SUFFIX_MAP[suffix]}"
                state[new_key] = arr
            else:
                print(f"  WARN: unknown suffix {suffix!r} in {k!r} — skipping")
        else:
            print(f"  WARN: unrecognised key {k!r} — skipping")

print(f"  {len(state)} core tensors remapped")

# ------------------------------------------------------------------
# Add missing architectural parameters (zero / unit init)
# ------------------------------------------------------------------
state["resid_lambdas"] = np.ones(N_LAYER, dtype=np.float32)
state["x0_lambdas"] = np.zeros(N_LAYER, dtype=np.float32)
state["smear_gate.weight"] = np.zeros((1, SMEAR_GATE_CHANNELS), dtype=np.float32)
state["smear_lambda"] = np.zeros(1, dtype=np.float32)
state["backout_lambda"] = np.full(1, 0.2, dtype=np.float32)

for i in VE_LAYERS:
    # value embedding: nn.Embedding(vocab_size, kv_dim)  → weight [vocab_size, kv_dim]
    state[f"value_embeds.ve_{i}.weight"] = np.zeros((VOCAB_SIZE, KV_DIM), dtype=np.float32)
    # ve_gate: nn.Linear(VE_GATE_CHANNELS, n_kv_head) → weight [n_kv_head, VE_GATE_CHANNELS]
    state[f"blocks.{i}.attn.ve_gate.weight"] = np.zeros((N_KV_HEAD, VE_GATE_CHANNELS), dtype=np.float32)

print(f"  {len(state)} total tensors after patching")

# ------------------------------------------------------------------
# Save model weights (safetensors — matches config.checkpoint.format)
# ------------------------------------------------------------------
out_model = os.path.join(DST_DIR, "model_000650.safetensors")
print(f"Saving model to {out_model} ...")
save_file(state, out_model)
size_gb = os.path.getsize(out_model) / 1e9
print(f"  {size_gb:.1f} GB")

# ------------------------------------------------------------------
# Write our meta format (CheckpointMetadata)
# ------------------------------------------------------------------
meta = {
    "step": 650,
    "model_config": {
        "sequence_len": 2048,
        "vocab_size": VOCAB_SIZE,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "n_kv_head": N_KV_HEAD,
        "n_embd": N_EMBD,
        "window_pattern": "L",
    },
    "user_config": {},
    "val_bpb": 0.841,
}
out_meta = os.path.join(DST_DIR, "meta_000650.json")
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved meta to {out_meta}")
print("Done.")
