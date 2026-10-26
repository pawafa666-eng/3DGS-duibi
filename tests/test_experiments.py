from __future__ import annotations

from pathlib import Path

from common.config import discover_experiments


ROOT = Path(__file__).resolve().parents[1]


def test_comparison_registers_exactly_ten_ordered_experiments() -> None:
    definitions = discover_experiments(ROOT)

    assert len(definitions) == 10
    assert [definition.id for definition in definitions] == [f"{index:02d}" for index in range(1, 11)]
    assert definitions[1].name == "2D Gaussian Splatting"
    assert definitions[-1].name == "Concrete-GS"


def test_01_3dgs_uses_the_bundled_custom_runtime() -> None:
    definitions = discover_experiments(ROOT)

    baseline = definitions[0]

    assert baseline.id == "01"
    assert baseline.source_repo_dir == "runtime"
    assert (ROOT / "01_3DGS" / "runtime" / "scene" / "colmap_loader.py").is_file()


def test_10_concrete_gs_uses_the_bundled_runtime() -> None:
    definitions = discover_experiments(ROOT)

    concrete = definitions[-1]

    assert concrete.id == "10"
    assert concrete.source_repo_dir == "runtime"
    assert (ROOT / "10_Concrete_GS" / "runtime" / "train.py").is_file()


def test_every_experiment_has_train_test_commands_and_artifacts() -> None:
    definitions = discover_experiments(ROOT)

    for definition in definitions:
        assert definition.train.commands
        assert definition.test.commands
        assert definition.artifacts == ["train", "test"]
        assert definition.source_repo_dir == "runtime"
        assert (ROOT / definition.directory / "runtime/train.py").is_file()


def test_every_experiment_uses_common_mask_aware_evaluation() -> None:
    definitions = discover_experiments(ROOT)

    for definition in definitions:
        command_text = " ".join(
            part
            for command in definition.test.commands
            for part in command
        )
        assert "common/metrics.py" in command_text
