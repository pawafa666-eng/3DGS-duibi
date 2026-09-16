from __future__ import annotations

from pathlib import Path

from .contracts import DatasetLayout, DatasetReport


REQUIRED_SPLIT_FILES = (
    Path("sparse") / "0" / "cameras.bin",
    Path("sparse") / "0" / "images.bin",
    Path("sparse") / "0" / "points3D.bin",
)


def dataset_layout(dataset_root: Path) -> DatasetLayout:
    dataset_root = Path(dataset_root).resolve()
    return DatasetLayout(
        root=dataset_root,
        train=dataset_root / "train",
        test=dataset_root / "test",
        train_masks=dataset_root / "train" / "masks",
        test_masks=dataset_root / "test" / "masks",
    )


def _validate_split(split_root: Path, split_name: str, errors: list[str], counts: dict[str, int]) -> None:
    images = split_root / "images"
    masks = split_root / "masks"
    if not split_root.is_dir():
        errors.append(f"Missing {split_name} split: {split_root}")
        return
    if not images.is_dir():
        errors.append(f"Missing {split_name} image directory: {images}")
    else:
        counts[f"{split_name}_images"] = sum(1 for item in images.iterdir() if item.is_file())
    if not masks.is_dir():
        errors.append(f"Missing {split_name} mask directory: {masks}")
    else:
        counts[f"{split_name}_masks"] = sum(1 for item in masks.iterdir() if item.is_file())
    for relative_path in REQUIRED_SPLIT_FILES:
        path = split_root / relative_path
        if not path.is_file():
            errors.append(f"Missing {split_name} COLMAP file: {path}")


def validate_dataset_layout(dataset_root: Path) -> DatasetReport:
    dataset_root = Path(dataset_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    if not dataset_root.is_dir():
        return DatasetReport(False, [f"Missing dataset directory: {dataset_root}"], [], {})
    _validate_split(dataset_root / "train", "train", errors, counts)
    _validate_split(dataset_root / "test", "test", errors, counts)
    if counts.get("train_images", 0) != counts.get("train_masks", 0):
        warnings.append("train image and mask counts differ")
    if counts.get("test_images", 0) != counts.get("test_masks", 0):
        warnings.append("test image and mask counts differ")
    return DatasetReport(not errors, errors, warnings, counts)

