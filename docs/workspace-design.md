---
title: "Workspace Design"
summary: "Module-level workspace store that owns the directory structure and path functions, replacing explicit base_dir parameter threading."
read_when:
  - Implementing or reviewing the workspace refactor
  - Adding new path functions or directory conventions
  - Understanding how paths.py evolves into the workspace module
status: active
last_updated: "2025-07-15"
---

# Workspace Design

Goal: evolve `common/paths.py` into a module-level workspace store that reads `base_dir`
from the [config manager](archive/config-manager-design.md) instead of taking it as a parameter.

---

## Problem

`base_dir` is threaded as a parameter through 20+ functions across tokenizer, dataset,
checkpoint, evaluation, tasks, chat, and report modules. Every function in `common/paths.py`
takes `base_dir` as its first argument, and every caller must extract and pass it.

```python
# Current — base_dir threaded everywhere
base_dir = config.common.base_dir
tokenizer = get_tokenizer(base_dir=base_dir)
token_bytes = get_token_bytes(base_dir=base_dir, device=device)
ckpt_dir = checkpoint_dir(base_dir, "base", model_tag)
results = evaluate_core(base_dir=base_dir, model=model, ...)
```

---

## Design

### Module-level workspace store

Same pattern as `config/current.py` — a private variable initialized once at startup:

```python
# nanochat/workspace.py
from nanochat.config import current

_base_dir: str | None = None

def init() -> None:
    """Read base_dir from config and set up the workspace root."""
    global _base_dir
    _base_dir = current.get().common.base_dir
    os.makedirs(_base_dir, exist_ok=True)

def base_dir() -> str:
    if _base_dir is None:
        raise RuntimeError("Workspace not initialized — call workspace.init() first")
    return _base_dir

def reset() -> None:
    """For testing."""
    global _base_dir
    _base_dir = None
```

### Path functions

Same functions as `common/paths.py`, minus the `base_dir` parameter:

```python
# nanochat/workspace.py (continued)

def _dir(*parts: str) -> str:
    path = os.path.join(base_dir(), *parts)
    os.makedirs(path, exist_ok=True)
    return path

def data_dir() -> str:
    return _dir("data", "climbmix")

def legacy_data_dir() -> str:
    return os.path.join(_dir("data"), "fineweb")

def eval_tasks_dir() -> str:
    return _dir("data", "eval_tasks")

def tokenizer_dir() -> str:
    return _dir("tokenizer")

def checkpoint_dir(phase: str, model_tag: str | None = None) -> str:
    assert phase in ("base", "sft", "rl"), f"Unknown phase: {phase}"
    if model_tag is not None:
        return _dir("checkpoints", phase, model_tag)
    return _dir("checkpoints", phase)

def eval_results_dir() -> str:
    return _dir("eval")

def identity_data_path() -> str:
    return os.path.join(base_dir(), "identity.jsonl")

def report_dir() -> str:
    return _dir("report")
```

### Initialization

Called once in the CLI entry point, after config init:

```python
# cli.py or __main__.py
from nanochat.config import current
from nanochat import workspace

config = ConfigLoader().resolve(args)
current.init(config)
workspace.init()
```

### Usage in leaf functions

```python
from nanochat import workspace

def get_tokenizer() -> RustBPETokenizer:
    path = os.path.join(workspace.tokenizer_dir(), "tokenizer.json")
    ...
```

---

## Migration strategy

### Phase 1 — Add `workspace.py`, keep `common/paths.py` ✅

- `workspace.py` with `init`, `base_dir`, `reset`, and all path functions
- Both modules coexist — `paths.py` still works for callers not yet migrated

### Phase 2 — Migrate leaf functions

Functions that only need a path and currently take `base_dir`:

| Module               | Functions                                     |
| -------------------- | --------------------------------------------- |
| `tokenizer/utils.py` | `get_tokenizer`, `get_token_bytes`            |
| `dataset/utils.py`   | `list_parquet_files`, `parquets_iter_batched` |
| `report/`            | `get_report`                                  |

Remove `base_dir` parameter, use `workspace.*()` internally.

### Phase 3 — Migrate mid-level functions

Functions that pass `base_dir` through to leaf functions:

| Module                         | Functions                                                    |
| ------------------------------ | ------------------------------------------------------------ |
| `training/checkpoint.py`       | `load_model_from_dir`, `build_model`, `load_optimizer_state` |
| `training/dataloader.py`       | data loader constructors                                     |
| `evaluation/core_benchmark.py` | `evaluate_core`                                              |
| `tasks/spellingbee.py`         | `SpellingBee.__init__`, `SimpleSpelling.__init__`            |

### Phase 4 — Clean up entry points and remove `common/paths.py`

Training/eval entry points stop extracting `base_dir` as a local variable:

```python
# Before
def train_base(config: Config):
    base_dir = config.common.base_dir
    tokenizer = get_tokenizer(base_dir=base_dir)
    ckpt_dir = checkpoint_dir(base_dir, "base", model_tag)

# After
def train_base(config: Config):
    tokenizer = get_tokenizer()
    ckpt_dir = workspace.checkpoint_dir("base", model_tag)
```

Delete `common/paths.py` once all callers are migrated. Update `data-layout.md` to
reference `workspace.py` instead.

---

## Testing

```python
@pytest.fixture(autouse=True)
def reset_workspace():
    workspace.reset()
    yield
    workspace.reset()

def test_checkpoint_dir(tmp_path):
    config = Config(common=CommonConfig(base_dir=str(tmp_path)))
    current.init(config)
    workspace.init()
    assert workspace.checkpoint_dir("base", "d12") == str(tmp_path / "checkpoints" / "base" / "d12")
```

---

## What NOT to put in workspace

- **Config access** — that's `config/current.py`
- **Runtime state** (step, loss) — that's `TrainingState`
- **File I/O** (reading/writing checkpoints) — that's the checkpoint manager
- **Directory structure changes** — the layout is defined in [data-layout.md](data-layout.md), workspace just implements it

---

## Tradeoffs

|                   | Explicit `base_dir` param (current)          | Module-level workspace (proposed) |
| ----------------- | -------------------------------------------- | --------------------------------- |
| **Clarity**       | Every function declares its dependency       | Hidden dependency on module state |
| **Testability**   | Easy to pass test values                     | Requires `reset()` fixture        |
| **Boilerplate**   | `base_dir=` threaded through 20+ functions   | One `init()` call                 |
| **Upstream sync** | Upstream also threads `base_dir` everywhere  | Diverges from upstream pattern    |
| **Refactoring**   | Adding a new path means updating all callers | Just add a function to workspace  |
| **Complexity**    | None                                         | One module, one private variable  |

---

## Dependencies

- Requires [config manager](archive/config-manager-design.md) (reads `base_dir` from config)
- Replaces `common/paths.py` (same functions, no `base_dir` param)
- Simplifies the [checkpoint manager](checkpoint-manager-design.md) (no `base_dir` threading)
- Simplifies the [dual-trainer architecture](dual-trainer-architecture.md)
