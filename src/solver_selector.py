from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class SolverSelector(ABC):
    @abstractmethod
    def algorithm_select(self, instance_path: str | Path) -> int:
        raise NotImplementedError
