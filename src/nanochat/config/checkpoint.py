"""Config for checkpoint saving and resumption."""

import argparse
from dataclasses import dataclass


@dataclass
class CheckpointConfig:
    @classmethod
    def update_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--format", type=str, default=argparse.SUPPRESS, choices=["torch", "safetensors", "numpy"])
        parser.add_argument("--save-every", type=int, default=argparse.SUPPRESS, help="-1 = only at end")
        parser.add_argument("--resume-from-step", type=int, default=argparse.SUPPRESS, help="-1 = disabled")
        parser.add_argument("--keep-last-n", type=int, default=argparse.SUPPRESS, help="-1 = keep all")

    @classmethod
    def generate_default(cls) -> str:
        return (
            'format = "torch"           # torch | safetensors | numpy\n'
            "save_every = -1            # -1 = only at end\n"
            "resume_from_step = -1      # -1 = disabled\n"
            "keep_last_n = -1           # -1 = keep all\n"
        )

    format: str = "torch"
    save_every: int = -1
    resume_from_step: int = -1
    keep_last_n: int = -1
