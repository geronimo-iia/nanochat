---
title: "Config Manager Design"
summary: "Design for a ConfigManager singleton that eliminates config/base_dir parameter threading across the codebase."
read_when:
  - Implementing or reviewing the config manager refactor
  - Adding new functions that need access to config or base_dir
  - Understanding why config is passed explicitly everywhere
status: draft
last_updated: "2025-07-14"
---

# Config Manager Design

Goal: introduce a `ConfigManager` singleton so that config and base_dir don't need to be
threaded as parameters through every function call.

---

## Problem

`Config` and `base_dir` are passed explicitly through the entire call chain:

```
train_base(config)
  → get_tokenizer(base_dir=config.common.base_dir)
  → get_token_bytes(base_dir=config.common.base_dir, device=device)
  → checkpoint_dir(base_dir, "base", output_dirname)
  → save_checkpoint(ckpt_dir, step, ...)
  → evaluate_core(base_dir=base_dir, model=model, ...)
  → evaluate_bpb(model, val_loader, ...)
  → init_wandb(config.common, ...)
```

`base_dir` alone appears as a parameter in 20+ functions across tokenizer, dataset,
checkpoint, evaluation, tasks, and chat modules. Every new function that needs config
must accept it as a parameter and every caller must pass it.

### What gets threaded

| Value | Passed to |
|---|---|
| `config.common.base_dir` | tokenizer, dataset, checkpoint, evaluation, tasks, chat, report |
| `config.common` | wandb init |
| `config.training` / `config.sft` / `config.rl` | training loops (already local) |
| `config.evaluation` | eval scripts (already local) |

The training/eval-specific sections are only used in their own entry points — no threading
problem there. The issue is `base_dir` and `common` config being passed everywhere.

---

## Design

### Module-level config store

A simple module with a private variable and two functions:

```python
# nanochat/config/current.py
from nanochat.config.config import Config

_config: Config | None = None

def init(config: Config) -> None:
    global _config
    _config = config

def get() -> Config:
    if _config is None:
        raise RuntimeError("Config not initialized — call config.current.init() first")
    return _config

def base_dir() -> str:
    return get().common.base_dir

def reset() -> None:
    """For testing."""
    global _config
    _config = None
```

### Initialization

Called once in the CLI entry point:

```python
# cli.py or __main__.py
from nanochat.config import current

config = ConfigLoader().resolve(args)
current.init(config)
train_base(config)  # config still passed to top-level for explicitness
```

### Usage in leaf functions

```python
from nanochat.config import current

def get_tokenizer() -> RustBPETokenizer:
    path = os.path.join(current.base_dir(), "tokenizer.json")
    ...
```

---

## Migration strategy

### Phase 1 — Add `config/current.py`, keep existing signatures

- Add `config/current.py` with `init`, `get`, `base_dir`, `reset`
- Initialize it in CLI entry point
- Don't change any function signatures yet

### Phase 2 — Migrate leaf functions

Start with functions that only need `base_dir`:

| Module | Functions |
|---|---|
| `tokenizer/utils.py` | `get_tokenizer`, `get_token_bytes` |
| `dataset/utils.py` | `list_parquet_files`, `parquets_iter_batched` |
| `common/paths.py` | `checkpoint_dir`, `eval_results_dir` |
| `report/` | `get_report` |

Remove `base_dir` parameter, use `current.base_dir()` internally.

### Phase 3 — Migrate mid-level functions

Functions that pass `base_dir` through to leaf functions:

| Module | Functions |
|---|---|
| `training/checkpoint.py` | `load_model_from_dir`, `build_model`, `load_optimizer_state` |
| `training/dataloader.py` | data loader constructors |
| `evaluation/core_benchmark.py` | `evaluate_core` |
| `tasks/spellingbee.py` | `SpellingBee.__init__`, `SimpleSpelling.__init__` |

### Phase 4 — Clean up entry points

Training/eval entry points still receive `config` as an explicit parameter (clarity at
the top level), but stop extracting `base_dir` as a local variable:

```python
# Before
def train_base(config: Config):
    base_dir = config.common.base_dir
    tokenizer = get_tokenizer(base_dir=base_dir)
    token_bytes = get_token_bytes(base_dir=base_dir, device=device)

# After
def train_base(config: Config):
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
```

---

## Testing

`ConfigManager.reset()` clears the singleton between tests:

```python
@pytest.fixture(autouse=True)
def reset_config():
    current.reset()
    yield
    current.reset()
```

Tests that need config initialize it explicitly:

```python
def test_something():
    config = Config(common=CommonConfig(base_dir="/tmp/test"))
    current.init(config)
    ...
```

---

## What NOT to put in ConfigManager

- **Runtime state** (step, loss, etc.) — that's `TrainingState`
- **Device/dtype** — these are compute concerns, not config
- **Model config** (`GPTConfig`) — this is per-model, not global
- **Mutable config** — `Config` should be treated as frozen after `init()`

---

## Tradeoffs

| | Explicit parameters (current) | Module-level store (proposed) |
|---|---|---|
| **Clarity** | Every function declares its dependencies | Hidden dependency on module state |
| **Testability** | Easy to pass test values | Requires `reset()` fixture |
| **Boilerplate** | `base_dir=` threaded through 20+ functions | One `init()` call |
| **Upstream sync** | Upstream also threads `base_dir` everywhere | Diverges from upstream pattern |
| **Refactoring** | Adding a new config field means updating all callers | Just access it from `current` |
| **Complexity** | None | One module, two functions, one private variable |

The module-level store trades explicitness for ergonomics. The `reset()` function keeps tests clean.

---

## Dependencies

- Independent of other refactors — can be done at any time
- Simplifies the [checkpoint manager](checkpoint-manager-design.md) (no `base_dir` parameter)
- Simplifies the [dual-trainer architecture](dual-trainer-architecture.md) (trainer doesn't
  need config passed in)
