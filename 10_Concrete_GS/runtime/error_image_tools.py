import csv
import shutil
from pathlib import Path


CSV_FIELDS = [
    "image_name",
    "first_marked_iteration",
    "iteration",
    "psnr",
    "ssim",
    "lpips",
    "loss",
    "l1_loss",
    "opacity_loss",
    "copied_image",
]


def dataset_root_from_source(source_path):
    source_dir = Path(source_path).resolve()
    if source_dir.name.lower() in {"train", "test"}:
        return source_dir.parent
    return source_dir


def image_name_with_extension(source_path, image_name):
    source_dir = Path(source_path).resolve()
    images_dir = source_dir / "images"
    raw = Path(image_name)
    if raw.suffix:
        candidate = images_dir / raw.name
        return raw.name, candidate if candidate.exists() else None

    matches = sorted(images_dir.glob(f"{image_name}.*"))
    if matches:
        return matches[0].name, matches[0]
    return raw.name, None


def _read_rows(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _format_metric(value):
    if value is None:
        return ""
    return str(float(value))


def record_training_metrics_for_error_image(
    source_path,
    image_name,
    iteration,
    psnr_value,
    ssim_value,
    lpips_value,
    loss_value,
    l1_loss_value,
    opacity_loss_value,
    threshold=10.0,
):
    dataset_root = dataset_root_from_source(source_path)
    error_dir = dataset_root / "error"
    csv_path = error_dir / "bad_images.csv"
    resolved_name, source_image = image_name_with_extension(source_path, image_name)
    existing_rows = _read_rows(csv_path)
    marked_iterations = {
        (row["image_name"], row["iteration"])
        for row in existing_rows
        if row.get("image_name") and row.get("iteration")
    }
    first_marked_by_image = {
        row["image_name"]: row["first_marked_iteration"]
        for row in existing_rows
        if row.get("image_name") and row.get("first_marked_iteration")
    }

    is_already_marked = resolved_name in first_marked_by_image
    should_mark = float(psnr_value) < float(threshold)
    if not is_already_marked and not should_mark:
        return None

    error_dir.mkdir(parents=True, exist_ok=True)
    copied_image = error_dir / resolved_name
    if source_image is not None and not copied_image.exists():
        shutil.copy2(source_image, copied_image)

    first_marked_iteration = first_marked_by_image.get(resolved_name, str(iteration))
    row_key = (resolved_name, str(iteration))
    if row_key not in marked_iterations:
        _write_row(
            csv_path,
            {
                "image_name": resolved_name,
                "first_marked_iteration": first_marked_iteration,
                "iteration": str(iteration),
                "psnr": _format_metric(psnr_value),
                "ssim": _format_metric(ssim_value),
                "lpips": _format_metric(lpips_value),
                "loss": _format_metric(loss_value),
                "l1_loss": _format_metric(l1_loss_value),
                "opacity_loss": _format_metric(opacity_loss_value),
                "copied_image": str(copied_image),
            },
        )

    return copied_image if copied_image.exists() else None


def is_error_image_already_marked(source_path, image_name):
    dataset_root = dataset_root_from_source(source_path)
    csv_path = dataset_root / "error" / "bad_images.csv"
    resolved_name, _ = image_name_with_extension(source_path, image_name)
    return any(row.get("image_name") == resolved_name for row in _read_rows(csv_path))
