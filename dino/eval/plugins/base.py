from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PluginResult:
    """Standardized plugin output payload."""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    log_metrics: Dict[str, Any] = field(default_factory=dict)
    primary: Dict[str, Dict[str, float]] | None = None


class BenchmarkPlugin(ABC):
    """Benchmark plugin contract used by the Tuner orchestrator."""

    name: str
    tuning_output_dir: Optional[Path] = None

    def bind_runtime(self, tuning_output_dir: Optional[Path]) -> None:
        self.tuning_output_dir = tuning_output_dir

    def should_run(self, epoch: int) -> bool:
        return True

    @abstractmethod
    def run(self, student: Any, teacher: Any, epoch: int) -> PluginResult:
        raise NotImplementedError
