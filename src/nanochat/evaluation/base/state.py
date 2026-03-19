from dataclasses import dataclass
from typing import Self


@dataclass
class BaseEvalResult:
    """Result state for base model evaluation."""

    core_results: dict[str, object] | None
    bpb_results: dict[str, float]
    samples: list[str]
    unconditioned_samples: list[str]

    @classmethod
    def fresh(cls) -> Self:
        return cls(
            core_results=None,
            bpb_results={},
            samples=[],
            unconditioned_samples=[],
        )

    @classmethod
    def from_checkpoint(cls, meta_data: dict[str, object]) -> Self:
        return cls(
            core_results=meta_data.get("core_results"),
            bpb_results=dict(meta_data.get("bpb_results", {})),
            samples=list(meta_data.get("samples", [])),
            unconditioned_samples=list(meta_data.get("unconditioned_samples", [])),
        )

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "core_results": self.core_results,
            "bpb_results": self.bpb_results,
            "samples": self.samples,
            "unconditioned_samples": self.unconditioned_samples,
        }
