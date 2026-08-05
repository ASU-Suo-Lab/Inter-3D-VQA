from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PIL import Image


def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def load_images(image_paths: Iterable[str]) -> List[Image.Image]:
    return [load_image(image_path) for image_path in image_paths]
