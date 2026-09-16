from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import DatasetLayout


RAW_DATASET_COMPAT = "raw"
STANDARD_COLMAP_DATASET_COMPAT = "standard_colmap"

_CUSTOM_POINT_HEADER = "<QdddBBBdddd"
_STANDARD_POINT_HEADER = "<QdddBBBd"
_UINT64 = "<Q"
_REQUIRED_FILES = (
    Path("sparse") / "0" / "cameras.bin",
    Path("sparse") / "0" / "images.bin",
    Path("sparse") / "0" / "points3D.bin",
)


class DatasetCompatibilityError(RuntimeError):
    """Raised when an isolated dataset compatibility view cannot be built."""


@dataclass(frozen=True)
class DatasetCompatibilityResult:
    root: Path
    train: Path
    test: Path
    train_masks: Path
    test_masks: Path
    cache_dir: Path
    cache_hit: bool

    def as_layout(self) -> DatasetLayout:
        return DatasetLayout(
            root=self.root,
            train=self.train,
            test=self.test,
            train_masks=self.train_masks,
            test_masks=self.test_masks,
        )


def dataset_compatibility_mode(root: Path, experiment_directory: str) -> str:
    """Read the optional compatibility mode without changing the config schema parser."""
    config_path = Path(root) / experiment_directory / "experiment.json"
    if not config_path.is_file():
        return RAW_DATASET_COMPAT
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetCompatibilityError(
            f"Unable to read dataset compatibility config {config_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise DatasetCompatibilityError(f"Dataset compatibility config must be an object: {config_path}")
    value = payload.get("dataset_compat", RAW_DATASET_COMPAT)
    if not isinstance(value, str):
        raise DatasetCompatibilityError(
            f"dataset_compat must be a string in {config_path}; got {value!r}"
        )
    return value


def _dataset_layout(root: Path) -> DatasetLayout:
    root = Path(root).resolve()
    return DatasetLayout(
        root=root,
        train=root / "train",
        test=root / "test",
        train_masks=root / "train" / "masks",
        test_masks=root / "test" / "masks",
    )


def _raise_missing(split: str, path: Path, description: str) -> None:
    raise DatasetCompatibilityError(
        f"Dataset compatibility failed for {split} split: missing {description}: {path}"
    )


def _validate_sources(source_root: Path) -> DatasetLayout:
    if not source_root.is_dir():
        raise DatasetCompatibilityError(f"Dataset compatibility failed: missing dataset directory: {source_root}")
    layout = _dataset_layout(source_root)
    for split, split_root in (("train", layout.train), ("test", layout.test)):
        if not split_root.is_dir():
            _raise_missing(split, split_root, "split directory")
        for directory_name in ("images", "masks"):
            directory = split_root / directory_name
            if not directory.is_dir():
                _raise_missing(split, directory, f"{directory_name} directory")
        for relative_path in _REQUIRED_FILES:
            path = split_root / relative_path
            if not path.is_file():
                _raise_missing(split, path, "COLMAP file")
    return layout


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _directory_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for item in sorted((candidate for candidate in path.rglob("*") if candidate.is_file()), key=lambda value: value.as_posix()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        signature = _file_signature(item)
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(signature["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(signature["mtime_ns"]).encode("ascii"))
        digest.update(b"\n")
        count += 1
    return {"count": count, "digest": digest.hexdigest()}


def _fingerprint_payload(source_root: Path, layout: DatasetLayout) -> dict[str, Any]:
    files: dict[str, dict[str, dict[str, int]]] = {}
    directories: dict[str, dict[str, dict[str, Any]]] = {}
    for split, split_root in (("train", layout.train), ("test", layout.test)):
        files[split] = {
            str(relative_path): _file_signature(split_root / relative_path)
            for relative_path in _REQUIRED_FILES
        }
        directories[split] = {
            name: _directory_signature(split_root / name)
            for name in ("images", "masks")
        }
    return {
        "schema": 1,
        "mode": STANDARD_COLMAP_DATASET_COMPAT,
        "source_root": str(source_root),
        "files": files,
        "directories": directories,
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _points_to(path: Path, target: Path) -> bool:
    try:
        return path.samefile(target)
    except OSError:
        try:
            return path.resolve() == target.resolve()
        except OSError:
            return False


def _remove_cache_dir(path: Path) -> None:
    """Remove one cache entry without traversing linked source directories."""
    if not path.exists():
        return
    for child in path.iterdir():
        is_junction = getattr(child, "is_junction", lambda: False)()
        if child.is_dir() and not child.is_symlink() and not is_junction:
            shutil.rmtree(child)
        else:
            child.unlink()
    path.rmdir()


def _cache_is_valid(cache_dir: Path, fingerprint_payload: dict[str, Any], fingerprint: str) -> bool:
    metadata = _read_json(cache_dir / "metadata.json") if cache_dir.is_dir() else None
    if not metadata:
        return False
    if metadata.get("fingerprint") != fingerprint or metadata.get("source") != fingerprint_payload:
        return False
    source_root = Path(str(fingerprint_payload["source_root"]))
    directory_storage = metadata.get("directory_storage")
    if not isinstance(directory_storage, dict):
        return False
    for split in ("train", "test"):
        source_split = source_root / split
        output_split = cache_dir / split
        split_storage = directory_storage.get(split)
        if not isinstance(split_storage, dict):
            return False
        for name in ("images", "masks"):
            output_directory = output_split / name
            source_directory = source_split / name
            storage = split_storage.get(name)
            if not output_directory.is_dir():
                return False
            if storage == "link":
                if not _points_to(output_directory, source_directory):
                    return False
            elif storage == "copy":
                if _directory_signature(output_directory) != fingerprint_payload["directories"][split][name]:
                    return False
            else:
                return False
        for relative_path in _REQUIRED_FILES:
            output_path = output_split / relative_path
            if not output_path.is_file():
                return False
        expected_size = metadata.get("output_sizes", {}).get(split, {}).get("points3D.bin")
        points_path = output_split / "sparse" / "0" / "points3D.bin"
        if not isinstance(expected_size, int) or points_path.stat().st_size != expected_size:
            return False
    return True


def _link_directory(source: Path, target: Path, split: str, name: str) -> str:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(target), str(source)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise OSError(detail or f"mklink exited with {result.returncode}")
        else:
            target.symlink_to(source, target_is_directory=True)
        return "link"
    except OSError as link_error:
        try:
            shutil.copytree(source, target)
            return "copy"
        except OSError as copy_error:
            raise DatasetCompatibilityError(
                f"Dataset compatibility failed for {split} split: cannot link or copy {name} directory "
                f"from {source} to {target}: {copy_error}"
            ) from link_error


def _read_exact(handle, size: int, source: Path, field: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(
            f"truncated {field}: expected {size} bytes, got {len(data)} bytes in {source}"
        )
    return data


def convert_custom_points3d(source: Path, target: Path) -> int:
    """Convert QdddBBBddddQ records to standard COLMAP QdddBBBdQ records."""
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    total_size = 0
    with source.open("rb") as source_handle, target.open("wb") as target_handle:
        count = struct.unpack(_UINT64, _read_exact(source_handle, 8, source, "point count"))[0]
        target_handle.write(struct.pack(_UINT64, count))
        custom_header_size = struct.calcsize(_CUSTOM_POINT_HEADER)
        for _ in range(count):
            values = struct.unpack(
                _CUSTOM_POINT_HEADER,
                _read_exact(source_handle, custom_header_size, source, "custom point record"),
            )
            track_count_bytes = _read_exact(source_handle, 8, source, "track length")
            track_count = struct.unpack(_UINT64, track_count_bytes)[0]
            track_bytes = _read_exact(source_handle, 8 * track_count, source, "track elements")
            point_id, x, y, z, red, green, blue = values[:7]
            error = values[10]
            target_handle.write(
                struct.pack(_STANDARD_POINT_HEADER, point_id, x, y, z, red, green, blue, error)
            )
            target_handle.write(track_count_bytes)
            target_handle.write(track_bytes)
        if source_handle.read(1):
            raise ValueError(f"unexpected trailing bytes after custom point records in {source}")
        total_size = target_handle.tell()
    return total_size


def _copy_split(source_split: Path, target_split: Path, split: str) -> tuple[int, dict[str, str]]:
    target_split.mkdir(parents=True, exist_ok=True)
    directory_storage = {
        name: _link_directory(source_split / name, target_split / name, split, name)
        for name in ("images", "masks")
    }
    target_sparse = target_split / "sparse" / "0"
    target_sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin"):
        try:
            shutil.copy2(source_split / "sparse" / "0" / name, target_sparse / name)
        except OSError as error:
            raise DatasetCompatibilityError(
                f"Dataset compatibility failed for {split} split: cannot copy {name} "
                f"from {source_split / 'sparse' / '0' / name} to {target_sparse / name}: {error}"
            ) from error
    source_points = source_split / "sparse" / "0" / "points3D.bin"
    target_points = target_sparse / "points3D.bin"
    try:
        return convert_custom_points3d(source_points, target_points), directory_storage
    except (OSError, struct.error, ValueError) as error:
        raise DatasetCompatibilityError(
            f"Dataset compatibility failed for {split} split: cannot convert custom "
            f"points3D.bin {source_points} to standard COLMAP {target_points}: {error}"
        ) from error


def _compatibility_result(cache_dir: Path, cache_hit: bool) -> DatasetCompatibilityResult:
    return DatasetCompatibilityResult(
        root=cache_dir,
        train=cache_dir / "train",
        test=cache_dir / "test",
        train_masks=cache_dir / "train" / "masks",
        test_masks=cache_dir / "test" / "masks",
        cache_dir=cache_dir,
        cache_hit=cache_hit,
    )


def prepare_standard_colmap_dataset(
    source_root: Path, cache_root: Path, *, dry_run: bool = False
) -> DatasetCompatibilityResult:
    """Create or reuse a standard-COLMAP view without changing source data."""
    source_root = Path(source_root).resolve()
    cache_root = Path(cache_root).resolve()
    try:
        cache_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise DatasetCompatibilityError(
            f"Dataset compatibility cache must be outside the source dataset: {cache_root}"
        )

    source_layout = _validate_sources(source_root)
    source_payload = _fingerprint_payload(source_root, source_layout)
    fingerprint = _fingerprint(source_payload)
    cache_dir = cache_root / f"standard-colmap-{fingerprint}"
    if _cache_is_valid(cache_dir, source_payload, fingerprint):
        return _compatibility_result(cache_dir, True)
    if dry_run:
        return _compatibility_result(cache_dir, False)

    cache_root.mkdir(parents=True, exist_ok=True)

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}-", dir=str(cache_root)))
    output_sizes: dict[str, dict[str, int]] = {}
    try:
        train_size, train_storage = _copy_split(source_layout.train, temporary_dir / "train", "train")
        test_size, test_storage = _copy_split(source_layout.test, temporary_dir / "test", "test")
        output_sizes["train"] = {"points3D.bin": train_size}
        output_sizes["test"] = {"points3D.bin": test_size}
        metadata = {
            "schema": 1,
            "mode": STANDARD_COLMAP_DATASET_COMPAT,
            "fingerprint": fingerprint,
            "source": source_payload,
            "output_sizes": output_sizes,
            "directory_storage": {"train": train_storage, "test": test_storage},
        }
        (temporary_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if cache_dir.exists():
            if _cache_is_valid(cache_dir, source_payload, fingerprint):
                return _compatibility_result(cache_dir, True)
            _remove_cache_dir(cache_dir)
        temporary_dir.replace(cache_dir)
    except DatasetCompatibilityError:
        raise
    except OSError as error:
        raise DatasetCompatibilityError(
            f"Dataset compatibility failed while publishing cache {cache_dir}: {error}"
        ) from error
    finally:
        if temporary_dir.exists():
            try:
                _remove_cache_dir(temporary_dir)
            except OSError:
                pass

    return _compatibility_result(cache_dir, False)


def prepare_context_dataset(context: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Replace only dataset placeholders for standard methods; leave raw methods untouched."""
    mode = str(context.get("dataset_compat", RAW_DATASET_COMPAT))
    if mode == RAW_DATASET_COMPAT:
        prepared = dict(context)
        prepared["_dataset_context_prepared"] = True
        prepared["_dataset_context_dry_run"] = dry_run
        return prepared
    if mode != STANDARD_COLMAP_DATASET_COMPAT:
        raise DatasetCompatibilityError(
            f"Unsupported dataset compatibility mode {mode!r} for {context.get('experiment', 'experiment')}; "
            f"expected {RAW_DATASET_COMPAT!r} or {STANDARD_COLMAP_DATASET_COMPAT!r}"
        )
    try:
        result = prepare_standard_colmap_dataset(
            Path(context["data_root"]),
            Path(context["dataset_compat_cache_root"]),
            dry_run=dry_run,
        )
    except DatasetCompatibilityError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise DatasetCompatibilityError(
            f"Dataset compatibility failed for {context.get('experiment', 'experiment')}: {error}"
        ) from error
    prepared = dict(context)
    prepared.update(
        {
            "data_root": result.root,
            "train_data": result.train,
            "test_data": result.test,
            "train_masks": result.train_masks,
            "test_masks": result.test_masks,
            "dataset_compat_cache_dir": result.cache_dir,
            "dataset_compat_cache_hit": result.cache_hit,
            "_dataset_context_prepared": True,
            "_dataset_context_dry_run": dry_run,
        }
    )
    return prepared
