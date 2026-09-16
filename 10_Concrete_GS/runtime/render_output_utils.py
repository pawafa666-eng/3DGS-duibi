import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def rendered_image_stem(image_name):
    path = Path(str(image_name))
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return path.with_suffix("").name
    return path.name


def prepare_clean_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
