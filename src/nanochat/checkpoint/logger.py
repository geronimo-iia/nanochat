"""Logging abstraction for checkpoint I/O."""

import logging
import os
from typing import Protocol


class CheckpointLogger(Protocol):
    def info(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...


class RankZeroLogger:
    def __init__(self, name: str = __name__) -> None:
        self._logger = logging.getLogger(name)

    def _is_rank_zero(self) -> bool:
        return int(os.environ.get("RANK", 0)) == 0

    def info(self, message: str) -> None:
        if self._is_rank_zero():
            self._logger.info(message)

    def warning(self, message: str) -> None:
        if self._is_rank_zero():
            self._logger.warning(message)


class SilentLogger:
    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass
