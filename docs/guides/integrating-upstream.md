---
title: "Integrating Upstream Changes"
summary: "How to bring upstream karpathy/nanochat changes into our fork, which has a different file layout."
read_when:
  - Upstream has new commits to integrate
  - Reviewing a PR from master into main
  - Understanding why rebase doesn't work for this fork
status: active
last_updated: "2025-07-14"
---

# Integrating Upstream Changes

Our fork reorganized the codebase from flat files (`nanochat/`, `scripts/`) into a package
layout (`src/nanochat/`). This means `git rebase upstream/master` will produce conflicts on
every commit that touches the old paths — files that no longer exist in our tree.

**Do not rebase. Apply changes manually.**

---

## Step-by-step

### 1. Fetch and inspect

```bash
git fetch upstream master
git log --oneline main..upstream/master
```

### 2. For each upstream commit, review the diff

```bash
git diff --stat <commit>~1..<commit>
git show <commit> -- <file> | cat
```

### 3. Map old paths to new paths

| Upstream path | Our path |
|---|---|
| `nanochat/gpt.py` | `src/nanochat/models/gpt.py` |
| `nanochat/engine.py` | `src/nanochat/evaluation/engine.py` |
| `nanochat/optim.py` | `src/nanochat/training/optimizer.py` |
| `scripts/base_train.py` | `src/nanochat/training/train_base.py` |
| `scripts/chat_sft.py` | `src/nanochat/training/train_sft.py` |
| `scripts/base_eval.py` | `src/nanochat/evaluation/base_eval.py` |
| `scripts/chat_eval.py` | `src/nanochat/evaluation/chat_eval.py` |
| `scripts/chat_cli.py` | `src/nanochat/chat/chat_cli.py` |
| `scripts/chat_web.py` | `src/nanochat/chat/chat_web.py` |
| `runs/`, `dev/`, `README.md` | same paths |

### 4. Apply changes to the correct files

Read the upstream diff, then edit our files to match. Watch for:

- **Signature changes** in shared functions — update all call sites (e.g. schedulers used
  by both `train_base.py` and `train_sft.py`)
- **New parameters/modules** — update `__init__` methods, `init_weights`, `estimate_flops`,
  `num_scaling_params`, `setup_optimizer`, and `forward` consistently
- **Upstream's inline code vs our refactored modules** — upstream may change a function
  inline in `base_train.py` that we extracted into `schedulers.py` or `checkpoint.py`

### 5. Format and test

```bash
uv run ruff format .
uv run pytest tests/ -q
```

### 6. Commit

One commit per upstream logical change, referencing the upstream commit hash:

```bash
git add -A && git commit -m "feat: integrate upstream a825e63 — autoresearch round 2

Port smear, backout, hyperparameter tuning from upstream.
Upstream commit: a825e63"
```

---

## Common pitfalls

- **Don't use `git rebase upstream/master`** — it replays 90+ commits and produces
  modify/delete conflicts on every old-path file
- **Check all call sites** when upstream changes a function signature — we may have
  split one upstream file into multiple modules
- **Compare upstream's SFT/RL with ours** — upstream may not update all scripts for
  a change (e.g. momentum scheduler only changed in base_train, not chat_sft)
- **Notebooks** (`*.ipynb`) — check for stale references (e.g. removed columns)
