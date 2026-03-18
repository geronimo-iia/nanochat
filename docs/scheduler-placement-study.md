---
title: "Scheduler Placement Study"
summary: "Study whether schedulers should live in a shared module or inline where they are used."
read_when:
  - Deciding where scheduler functions should live
  - Reviewing the schedulers.py module
  - Integrating upstream changes that touch scheduler logic
status: done
last_updated: "2025-07-15"
---

# Scheduler Placement Study

Question: should `schedulers.py` exist as a shared module, or should each scheduler be
defined inline in the training script that uses it?

---

## Current layout (our fork)

```
training/schedulers.py          # shared module, 5 factory functions
├── create_lr_scheduler()                → used by train_base
├── create_muon_momentum_scheduler()     → used by train_base
├── create_sft_muon_momentum_scheduler() → used by train_sft
├── create_weight_decay_scheduler()      → used by train_base
```

SFT also defines `get_lr_multiplier` inline (progress-based, not step-based).
RL defines `get_lr_multiplier` inline (simple linear rampdown).

## Upstream layout (karpathy/nanochat)

Each script defines its own schedulers inline as module-level functions:

```
scripts/base_train.py
├── get_lr_multiplier(it)       # warmup + warmdown, captures num_iterations etc.
├── get_muon_momentum(it)       # warmup + warmdown
├── get_weight_decay(it)        # cosine decay

scripts/chat_sft.py
├── get_lr_multiplier(progress) # progress-based (0→1), different signature
├── get_muon_momentum(it)       # simpler: warmup to 0.95, no warmdown

scripts/chat_rl.py
├── get_lr_multiplier(it)       # simple linear rampdown
```

---

## Analysis

### What's actually shared?

Very little. Each training script has different scheduler behavior:

| Scheduler | base_train | train_sft | train_rl |
|---|---|---|---|
| LR | step-based, warmup+warmdown | progress-based (0→1), warmup+warmdown | step-based, linear rampdown |
| Muon momentum | warmup→0.97, warmdown→0.90 | warmup→0.95, no warmdown | not used |
| Weight decay | cosine decay | not used | not used |

The LR schedulers have **different signatures** (`it: int` vs `progress: float`).
The Muon schedulers have **different targets** (0.97 vs 0.95) and **different phases**.
Only `create_weight_decay_scheduler` is truly single-use (base_train only).

### The upstream sync problem

This is the strongest argument for inline. When upstream changes a scheduler (as happened
with the Muon momentum warmdown in `a825e63`), the integration path is:

- **Inline**: read upstream diff, apply same change to our file. Done.
- **Shared module**: read upstream diff, figure out which factory function it maps to,
  update the factory signature, update all call sites, check if other callers are affected.

We hit exactly this problem: upstream added warmdown to base_train's momentum scheduler
but not SFT's. With a shared module, we had to create a second factory function
(`create_sft_muon_momentum_scheduler`) to avoid breaking SFT. Inline, this would have
been a non-issue — each script has its own function.

---

## Options

### Option A — Shared module (current)

Keep `schedulers.py` as-is.

- ✅ Single place to see all schedules
- ✅ Easy to unit test
- ❌ Creates coupling between scripts that don't share behavior
- ❌ Upstream sync friction (signature changes break all callers)

### Option B — Inline closures (upstream)

Define schedulers as closures inside each training function, capturing loop variables.

- ✅ Matches upstream exactly — trivial to sync
- ✅ "Data scientist" pattern — everything in one file
- ❌ Closures capture implicit state — hard to test, hard to extract for dual-trainer
- ❌ Conflicts with the TrainingState refactor goal of making state explicit

### Option C — Co-located named functions (recommended)

Define schedulers as standalone named functions in the same file as their training loop:

```
train_base.py
├── def base_lr_scheduler(...) -> Callable
├── def base_muon_momentum_scheduler(...) -> Callable
├── def base_weight_decay_scheduler(...) -> Callable
├── def train_base(config): ...

train_sft.py
├── def sft_lr_scheduler(...) -> Callable
├── def sft_muon_momentum_scheduler() -> Callable
├── def train_sft(config): ...

train_rl.py
├── def rl_lr_scheduler(...) -> Callable
├── def train_rl(config): ...
```

- ✅ Each script owns its schedules — no cross-script coupling
- ✅ Named functions with explicit args — testable by importing from the module
- ✅ Visible next to the loop — easy to read and modify
- ✅ Upstream sync is straightforward — apply diff to the right file
- ✅ Compatible with dual-trainer refactor — the trainer can call the co-located function
- ✅ No closures capturing implicit state

---

## Recommendation

**Option C — co-located named functions.** It combines the best of both:

- Locality of inline (visible next to usage, no cross-file jumps)
- Testability of a shared module (named functions with explicit parameters)
- No coupling between scripts that have different scheduler behavior
- Clean upstream sync path

---

## Migration plan

1. Move `create_lr_scheduler` + `create_muon_momentum_scheduler` + `create_weight_decay_scheduler`
   into `train_base.py` as `base_lr_scheduler`, `base_muon_momentum_scheduler`,
   `base_weight_decay_scheduler`
2. Move `create_sft_muon_momentum_scheduler` into `train_sft.py` as `sft_muon_momentum_scheduler`,
   rename the existing inline SFT LR function to `sft_lr_scheduler`
3. RL's `get_lr_multiplier` is already inline — rename to `rl_lr_scheduler` for consistency
4. Delete `schedulers.py`
5. Update `training/__init__.py` exports

---

## Decision

**Option C — co-located named functions.** Implemented:

- `train_base.py`: `base_lr_scheduler`, `base_muon_momentum_scheduler`, `base_weight_decay_scheduler`
- `train_sft.py`: `sft_lr_scheduler`, `sft_muon_momentum_scheduler`
- `train_rl.py`: `rl_lr_scheduler`
- `schedulers.py` deleted, `training/__init__.py` scheduler exports removed.
