from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

IMAGE_EXTENSIONS = {
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".webp",
}


def iter_image_paths(root: Path, recursive: bool = True) -> Iterable[Path]:
    root = root.resolve()
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_image(path: Path, image: np.ndarray, jpeg_quality: int = 92) -> None:
    extension = path.suffix.lower() or ".png"
    parameters: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise ValueError(f"Не удалось закодировать изображение: {path}")
    _atomic_replace_bytes(path, encoded.tobytes())


def write_text(path: Path, text: str) -> None:
    _atomic_replace_bytes(path, text.encode("utf-8"))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Корень YAML должен быть словарём: {path}")
    return value


def project_input_path(config: dict[str, object], mode: str) -> Path:
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Раздел paths в project.yaml должен быть словарём.")
    value = paths.get(mode)
    if not value:
        raise ValueError(f"В project.yaml не задан путь paths.{mode}")
    return Path(str(value))


def project_path(config: dict[str, object], key: str, default: str) -> Path:
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        return Path(default)
    return Path(str(paths.get(key, default)))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Неподдерживаемая форма изображения: {image.shape}")


def to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    image_u8 = normalize_to_uint8(image)
    if image_u8.ndim == 2:
        return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    if image_u8.ndim == 3 and image_u8.shape[2] == 1:
        return cv2.cvtColor(image_u8[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image_u8.ndim == 3 and image_u8.shape[2] == 3:
        return image_u8.copy()
    if image_u8.ndim == 3 and image_u8.shape[2] == 4:
        return cv2.cvtColor(image_u8, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Неподдерживаемая форма изображения: {image.shape}")


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image.astype(np.float32) - minimum) * (255.0 / (maximum - minimum))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def resize_long_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def safe_slug(text: str, max_length: int = 100) -> str:
    kept: list[str] = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            kept.append(char)
        else:
            kept.append("_")
    slug = "".join(kept).strip("_") or "root"
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_length]


def file_hash(path: Path, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
