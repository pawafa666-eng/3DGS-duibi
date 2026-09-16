from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_image(path: Path):
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("common metrics requires numpy and Pillow") from error
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _resize(array, width: int, height: int):
    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _ssim(prediction, target):
    import numpy as np

    mu_pred = prediction.mean(axis=0)
    mu_target = target.mean(axis=0)
    sigma_pred = prediction.var(axis=0)
    sigma_target = target.var(axis=0)
    covariance = ((prediction - mu_pred) * (target - mu_target)).mean(axis=0)
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_pred * mu_target + c1) * (2 * covariance + c2)) / (
        (mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2)
    )
    return float(np.mean(score))


def _render_files(render_root: Path) -> list[Path]:
    return sorted(path for path in render_root.rglob("renders/*.png") if path.is_file())


def evaluate(render_root: Path, data_root: Path, masks_root: Path, output_path: Path) -> dict[str, Any]:
    import numpy as np

    render_root = Path(render_root)
    data_images = {path.stem: path for path in (Path(data_root) / "images").iterdir() if path.is_file()}
    render_files = _render_files(render_root)
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    per_image: list[dict[str, Any]] = []
    for render_path in render_files:
        stem = render_path.stem
        target_path = data_images.get(stem)
        mask_path = Path(masks_root) / f"{stem}.mask.png"
        if target_path is None or not mask_path.is_file():
            continue
        prediction = _load_image(render_path)
        target = _load_image(target_path)
        mask = _load_image(mask_path)[:, :, 0] > 0
        if target.shape[:2] != prediction.shape[:2]:
            target = _resize(target, prediction.shape[1], prediction.shape[0])
            mask = _resize(mask.astype(np.float32)[:, :, None], prediction.shape[1], prediction.shape[0])[:, :, 0] > 0.5
        valid = mask.reshape(-1)
        pred_valid = prediction.reshape(-1, 3)[valid]
        target_valid = target.reshape(-1, 3)[valid]
        if pred_valid.size == 0:
            continue
        mse = float(np.mean((pred_valid - target_valid) ** 2))
        psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
        ssim = _ssim(pred_valid, target_valid)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        per_image.append({"image": stem, "PSNR": psnr, "SSIM": ssim, "valid_pixels": int(valid.sum())})
    if not per_image:
        result = {
            "status": "missing_renders",
            "PSNR": None,
            "SSIM": None,
            "LPIPS": None,
            "num_images": 0,
            "mask_strategy": "evaluator",
        }
    else:
        result = {
            "status": "completed",
            "PSNR": float(np.mean(psnr_values)),
            "SSIM": float(np.mean(ssim_values)),
            "LPIPS": None,
            "num_images": len(per_image),
            "mask_strategy": "evaluator",
            "per_image": per_image,
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute common mask-aware PSNR/SSIM from rendered test images.")
    parser.add_argument("--render-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    evaluate(Path(args.render_root), Path(args.data), Path(args.masks), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

