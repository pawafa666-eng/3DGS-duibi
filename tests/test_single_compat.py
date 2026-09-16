from __future__ import annotations

import json
from pathlib import Path

import common.single as single_module
import common.dataset_compat as dataset_compat_module
from common.single import run_single
from test_dataset_compat import _make_dataset


def test_independent_run_entrypoint_calls_shared_dataset_preparation(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_root, _ = _make_dataset(tmp_path)
    experiment_dir = tmp_path / "02_demo"
    experiment_dir.mkdir()
    (experiment_dir / "experiment.json").write_text(
        json.dumps(
            {
                "id": "02",
                "name": "Demo",
                "category": "test",
                "source": {"url": "local", "repo_dir": "repo"},
                "mask_strategy": "native",
                "dataset_compat": "standard_colmap",
                "train": {"commands": [["python", "train.py", "{train_data}"]]},
                "test": {"commands": [["python", "render.py", "{test_data}"]]},
                "artifacts": ["train", "test"],
            }
        ),
        encoding="utf-8",
    )
    calls = []
    original = single_module.prepare_context_dataset
    prepare_calls = []
    original_prepare = dataset_compat_module.prepare_standard_colmap_dataset

    def observe(context, **kwargs):
        calls.append(context)
        return original(context, **kwargs)

    def observe_prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(single_module, "prepare_context_dataset", observe, raising=False)
    monkeypatch.setattr(
        dataset_compat_module,
        "prepare_standard_colmap_dataset",
        observe_prepare,
    )

    result = run_single(
        experiment_dir,
        [
            "--stage",
            "train",
            "--data",
            str(dataset_root),
            "--output",
            str(tmp_path / "output"),
            "--iterations",
            "1",
            "--dry-run",
        ],
    )

    assert result == 0
    assert len(calls) == 1
    assert len(prepare_calls) == 1
    command = (tmp_path / "output" / "01" / "02_demo" / "command.txt").read_text(encoding="utf-8")
    assert ".dataset_compat" in command
