"""Config for base model pre-training (architecture, optimizer, schedule, eval)."""

import argparse
from dataclasses import dataclass

_VALID_WINDOW_CHARS = frozenset("LS")
_VALID_FP8_RECIPES = frozenset({"tensorwise", "rowwise"})


@dataclass
class TrainingConfig:
    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        # Model architecture
        parser.add_argument("--depth", type=int, default=argparse.SUPPRESS)
        parser.add_argument(
            "--aspect-ratio", type=int, default=argparse.SUPPRESS, help="model_dim = depth * aspect_ratio"
        )
        parser.add_argument("--head-dim", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--max-seq-len", type=int, default=argparse.SUPPRESS)
        parser.add_argument(
            "--window-pattern", type=str, default=argparse.SUPPRESS, help="L=full, S=half context, tiled"
        )
        # Training horizon
        parser.add_argument("--num-iterations", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
        parser.add_argument("--target-flops", type=float, default=argparse.SUPPRESS, help="-1 = disabled")
        parser.add_argument("--target-param-data-ratio", type=float, default=argparse.SUPPRESS)
        # Batch
        parser.add_argument("--device-batch-size", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--total-batch-size", type=int, default=argparse.SUPPRESS, help="-1 = auto")
        # Optimizer
        parser.add_argument("--embedding-lr", type=float, default=argparse.SUPPRESS)
        parser.add_argument("--unembedding-lr", type=float, default=argparse.SUPPRESS)
        parser.add_argument("--matrix-lr", type=float, default=argparse.SUPPRESS)
        parser.add_argument("--scalar-lr", type=float, default=argparse.SUPPRESS)
        parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
        parser.add_argument("--warmup-steps", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--warmdown-ratio", type=float, default=argparse.SUPPRESS)
        parser.add_argument("--final-lr-frac", type=float, default=argparse.SUPPRESS)
        # Evaluation
        parser.add_argument("--eval-every", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
        parser.add_argument("--eval-tokens", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--core-metric-every", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
        parser.add_argument("--core-metric-max-per-task", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--sample-every", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
        # FP8
        parser.add_argument("--fp8", action="store_true", default=argparse.SUPPRESS)
        parser.add_argument("--fp8-recipe", type=str, default=argparse.SUPPRESS, choices=["tensorwise", "rowwise"])
        # Compression
        parser.add_argument("--track-compression", action="store_true", default=argparse.SUPPRESS)
        parser.add_argument("--compression-log-every", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--track-layer-compression", action="store_true", default=argparse.SUPPRESS)
        parser.add_argument("--compression-early-stop", action="store_true", default=argparse.SUPPRESS)

    @classmethod
    def generate_default(cls) -> str:
        return (
            "depth = 20\n"
            "aspect_ratio = 64          # model_dim = depth * aspect_ratio\n"
            "head_dim = 128\n"
            "max_seq_len = 2048\n"
            'window_pattern = "SSSL"    # L=full context, S=half context, tiled across layers\n'
            "num_iterations = -1        # explicit step count (-1 = disabled)\n"
            "target_flops = -1.0        # compute budget in FLOPs (-1 = disabled)\n"
            "target_param_data_ratio = 10.5  # tokens:params ratio (Chinchilla=20)\n"
            "device_batch_size = 32\n"
            "total_batch_size = -1      # -1 = auto-compute optimal\n"
            "embedding_lr = 0.3\n"
            "unembedding_lr = 0.008\n"
            "matrix_lr = 0.02\n"
            "scalar_lr = 0.5\n"
            "weight_decay = 0.28\n"
            "warmup_steps = 40\n"
            "warmdown_ratio = 0.65\n"
            "final_lr_frac = 0.05\n"
            "eval_every = 250           # -1 = disabled\n"
            f"eval_tokens = {80 * 524288}       # 80 * 524288\n"
            "core_metric_every = 2000   # -1 = disabled\n"
            "core_metric_max_per_task = 500\n"
            "sample_every = 2000        # -1 = disabled\n"
            "fp8 = false\n"
            'fp8_recipe = "tensorwise"  # tensorwise | rowwise\n'
            "track_compression = false\n"
            "compression_log_every = 100\n"
            "track_layer_compression = false\n"
            "compression_early_stop = false\n"
        )

    # Model architecture
    depth: int = 20
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"
    # Training horizon
    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 10.5
    # Batch
    device_batch_size: int = 32
    total_batch_size: int = -1
    # Optimizer
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.28
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    # Evaluation
    eval_every: int = 250
    eval_tokens: int = 80 * 524288
    core_metric_every: int = 2000
    core_metric_max_per_task: int = 500
    sample_every: int = 2000
    # FP8
    fp8: bool = False
    fp8_recipe: str = "tensorwise"
    # Compression
    track_compression: bool = False
    compression_log_every: int = 100
    track_layer_compression: bool = False
    compression_early_stop: bool = False

    def validate(self) -> None:
        """Raise ValueError if any field value is invalid or contradictory."""
        # Architecture
        for name, val in (("depth", self.depth), ("aspect_ratio", self.aspect_ratio),
                          ("head_dim", self.head_dim), ("max_seq_len", self.max_seq_len)):
            if val < 1:
                raise ValueError(f"training.{name} must be >= 1, got {val}")
        invalid_chars = set(self.window_pattern) - _VALID_WINDOW_CHARS
        if not self.window_pattern or invalid_chars:
            raise ValueError(
                f"training.window_pattern must be non-empty and contain only 'L'/'S', got {self.window_pattern!r}"
            )
        # Horizon — at least one must be active
        if self.num_iterations != -1 and self.num_iterations < 1:
            raise ValueError(f"training.num_iterations must be -1 or >= 1, got {self.num_iterations}")
        if self.target_flops != -1.0 and self.target_flops <= 0:
            raise ValueError(f"training.target_flops must be -1 or > 0, got {self.target_flops}")
        if self.target_param_data_ratio <= 0:
            raise ValueError(f"training.target_param_data_ratio must be > 0, got {self.target_param_data_ratio}")
        if self.num_iterations == -1 and self.target_flops == -1.0 and self.target_param_data_ratio <= 0:
            raise ValueError(
                "training: at least one of num_iterations, target_flops, target_param_data_ratio must be active"
            )
        # Batch
        if self.device_batch_size < 1:
            raise ValueError(f"training.device_batch_size must be >= 1, got {self.device_batch_size}")
        if self.total_batch_size != -1 and self.total_batch_size < 1:
            raise ValueError(f"training.total_batch_size must be -1 or >= 1, got {self.total_batch_size}")
        # Optimizer
        for name, val in (("embedding_lr", self.embedding_lr), ("unembedding_lr", self.unembedding_lr),
                          ("matrix_lr", self.matrix_lr), ("scalar_lr", self.scalar_lr)):
            if val <= 0:
                raise ValueError(f"training.{name} must be > 0, got {val}")
        if self.weight_decay < 0:
            raise ValueError(f"training.weight_decay must be >= 0, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"training.warmup_steps must be >= 0, got {self.warmup_steps}")
        if not 0 < self.warmdown_ratio < 1:
            raise ValueError(f"training.warmdown_ratio must be in (0, 1), got {self.warmdown_ratio}")
        if not 0 < self.final_lr_frac < 1:
            raise ValueError(f"training.final_lr_frac must be in (0, 1), got {self.final_lr_frac}")
        # Evaluation
        for name, val in (("eval_every", self.eval_every), ("core_metric_every", self.core_metric_every),
                          ("sample_every", self.sample_every)):
            if val != -1 and val < 1:
                raise ValueError(f"training.{name} must be -1 or >= 1, got {val}")
        if self.eval_tokens < 1:
            raise ValueError(f"training.eval_tokens must be >= 1, got {self.eval_tokens}")
        if self.core_metric_max_per_task < 1:
            raise ValueError(f"training.core_metric_max_per_task must be >= 1, got {self.core_metric_max_per_task}")
        # FP8
        if self.fp8_recipe not in _VALID_FP8_RECIPES:
            raise ValueError(f"training.fp8_recipe must be one of {sorted(_VALID_FP8_RECIPES)}, got {self.fp8_recipe!r}")
        # Compression
        if self.compression_log_every < 1:
            raise ValueError(f"training.compression_log_every must be >= 1, got {self.compression_log_every}")
