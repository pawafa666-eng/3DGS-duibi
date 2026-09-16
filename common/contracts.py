from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageDefinition:
    commands: list[list[str]]
    cwd: str = "repo"
    artifacts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentDefinition:
    id: str
    directory: str
    name: str
    category: str
    source_url: str
    source_repo_dir: str
    mask_strategy: str
    train: StageDefinition
    test: StageDefinition
    artifacts: list[str]
    notes: str = ""
    source_commit: str | None = None

    @property
    def train_command(self) -> list[str]:
        """Return the single-command shorthand used by simple adapters."""
        if len(self.train.commands) != 1:
            raise ValueError("This experiment has multiple train commands")
        return self.train.commands[0]

    @property
    def test_commands(self) -> list[list[str]]:
        return self.test.commands

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = {
            "url": value.pop("source_url"),
            "repo_dir": value.pop("source_repo_dir"),
            "commit": value.pop("source_commit"),
        }
        return value


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    train: Path
    test: Path
    train_masks: Path
    test_masks: Path


@dataclass(frozen=True)
class DatasetReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    counts: dict[str, int]


def stage_to_dict(stage: StageDefinition) -> dict[str, Any]:
    return asdict(stage)
