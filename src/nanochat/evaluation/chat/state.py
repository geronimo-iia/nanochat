from dataclasses import dataclass
from typing import Self


@dataclass
class ChatEvalResult:
    """Result state for chat model evaluation."""

    results: dict[str, float]

    @classmethod
    def fresh(cls) -> Self:
        return cls(results={})

    @classmethod
    def from_checkpoint(cls, meta_data: dict[str, object]) -> Self:
        return cls(results=dict(meta_data.get("results", {})))  # type: ignore[arg-type]

    def to_checkpoint(self) -> dict[str, object]:
        return {"results": self.results}
