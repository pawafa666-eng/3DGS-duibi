from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

from .contracts import ExperimentDefinition, StageDefinition


ALLOWED_PLACEHOLDERS = {
    "root",
    "data_root",
    "train_data",
    "test_data",
    "train_masks",
    "test_masks",
    "output",
    "train_output",
    "test_output",
    "repo",
    "iterations",
}


def _commands(stage_value: Any, field_name: str) -> list[list[str]]:
    if not isinstance(stage_value, dict):
        raise ValueError(f"{field_name} must be an object")
    raw = stage_value.get("commands", stage_value.get("command"))
    if raw is None:
        raise ValueError(f"{field_name} must define command or commands")
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, list) for item in raw):
        if not all(all(isinstance(part, str) for part in command) for command in raw):
            raise ValueError(f"{field_name}.commands must contain only string arguments")
        return raw
    raise ValueError(f"{field_name}.command(s) must be a string list or list of string lists")


def _stage(stage_value: Any, field_name: str) -> StageDefinition:
    commands = _commands(stage_value, field_name)
    cwd = stage_value.get("cwd", "repo")
    artifacts = stage_value.get("artifacts", [])
    if not isinstance(cwd, str) or not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
        raise ValueError(f"{field_name}.cwd must be a string and artifacts must be a string list")
    return StageDefinition(commands=commands, cwd=cwd, artifacts=artifacts)


def _validate_placeholders(commands: list[list[str]], source: str) -> None:
    for command in commands:
        for part in command:
            for _, field_name, _, _ in string.Formatter().parse(part):
                if field_name and field_name not in ALLOWED_PLACEHOLDERS:
                    raise ValueError(f"Unknown placeholder {{{field_name}}} in {source}: {part}")


def load_experiment_definition(experiment_dir: Path) -> ExperimentDefinition:
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / "experiment.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"id", "name", "category", "source", "mask_strategy", "train", "test", "artifacts"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{config_path}: missing fields: {', '.join(missing)}")
    source = payload["source"]
    if not isinstance(source, dict) or not isinstance(source.get("url"), str):
        raise ValueError(f"{config_path}: source.url is required")
    source_repo_dir = source.get("repo_dir", "repo")
    if not isinstance(source_repo_dir, str):
        raise ValueError(f"{config_path}: source.repo_dir must be a string")
    train = _stage(payload["train"], "train")
    test = _stage(payload["test"], "test")
    _validate_placeholders(train.commands + test.commands, str(config_path))
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
        raise ValueError(f"{config_path}: artifacts must be a string list")
    experiment_id = str(payload["id"])
    if experiment_dir.name[:2] != experiment_id:
        raise ValueError(f"{config_path}: directory name must start with id {experiment_id}")
    return ExperimentDefinition(
        id=experiment_id,
        directory=experiment_dir.name,
        name=str(payload["name"]),
        category=str(payload["category"]),
        source_url=source["url"],
        source_repo_dir=source_repo_dir,
        mask_strategy=str(payload["mask_strategy"]),
        train=train,
        test=test,
        artifacts=artifacts,
        notes=str(payload.get("notes", "")),
        source_commit=source.get("commit"),
    )


def discover_experiments(root: Path) -> list[ExperimentDefinition]:
    definitions = []
    for directory in sorted(Path(root).glob("[0-9][0-9]_*/")):
        if (directory / "experiment.json").is_file():
            definitions.append(load_experiment_definition(directory))
    if not definitions:
        raise ValueError(f"No experiment.json files found under {root}")
    ids = [definition.id for definition in definitions]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment ids must be unique")
    return definitions

