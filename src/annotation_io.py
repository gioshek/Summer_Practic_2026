from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import read_csv, read_image, write_csv, write_image, write_text

CLASS_NAMES = {
    0: "unlabeled",
    1: "background",
    2: "candy_core",
    3: "sticker",
}

CLASS_COLORS_BGR = {
    1: (255, 80, 40),
    2: (70, 220, 70),
    3: (40, 60, 255),
}

MANIFEST_FIELDS = [
    "annotation_id",
    "source_relative_path",
    "source_group",
    "selected_from_mode",
    "selection_order",
    "width",
    "height",
    "channels",
    "mean",
    "std",
    "entropy",
    "edge_density",
    "label_status",
    "pixels_background",
    "pixels_candy_core",
    "pixels_sticker",
    "mask_relative_path",
    "metadata_relative_path",
]


def normalized_relative_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Недопустимый относительный путь: {value}")
    return path


def mask_path_for(annotation_root: Path, relative_path: str | Path) -> Path:
    relative = normalized_relative_path(relative_path)
    return annotation_root / "pixel_masks" / relative.with_suffix(".png")


def metadata_path_for(annotation_root: Path, relative_path: str | Path) -> Path:
    relative = normalized_relative_path(relative_path)
    return annotation_root / "metadata" / relative.with_suffix(".json")


def preview_path_for(annotation_root: Path, relative_path: str | Path) -> Path:
    relative = normalized_relative_path(relative_path)
    return annotation_root / "previews" / relative.with_suffix(".jpg")


def count_pixels(mask: np.ndarray) -> dict[str, int]:
    return {
        "background": int(np.count_nonzero(mask == 1)),
        "candy_core": int(np.count_nonzero(mask == 2)),
        "sticker": int(np.count_nonzero(mask == 3)),
    }


def load_label_mask(
    annotation_root: Path,
    relative_path: str | Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    path = mask_path_for(annotation_root, relative_path)
    if not path.exists():
        return np.zeros(expected_shape, dtype=np.uint8)
    mask = read_image(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != expected_shape:
        raise ValueError(
            f"Размер маски {mask.shape} не совпадает с изображением {expected_shape}: {path}"
        )
    unique = set(int(value) for value in np.unique(mask))
    invalid = unique.difference(CLASS_NAMES)
    if invalid:
        raise ValueError(f"В маске есть неизвестные классы {sorted(invalid)}: {path}")
    return mask.astype(np.uint8, copy=False)


def read_metadata(annotation_root: Path, relative_path: str | Path) -> dict[str, object]:
    path = metadata_path_for(annotation_root, relative_path)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_annotation(
    annotation_root: Path,
    relative_path: str | Path,
    mask: np.ndarray,
    status: str,
    source_mode: str,
    source_root: Path,
) -> None:
    relative = normalized_relative_path(relative_path)
    mask_path = mask_path_for(annotation_root, relative)
    metadata_path = metadata_path_for(annotation_root, relative)
    write_image(mask_path, mask.astype(np.uint8, copy=False))
    counts = count_pixels(mask)
    payload = {
        "format_version": 2,
        "source_relative_path": relative.as_posix(),
        "source_mode_used_for_editing": source_mode,
        "source_root_used_for_editing": str(source_root.resolve()),
        "mask_relative_path": mask_path.relative_to(annotation_root).as_posix(),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "status": status,
        "pixel_counts": counts,
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_text(metadata_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_manifest(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def save_manifest(path: Path, records: list[dict[str, object]]) -> None:
    normalized: list[dict[str, object]] = []
    for record in records:
        normalized.append({field: record.get(field, "") for field in MANIFEST_FIELDS})
    write_csv(path, normalized, MANIFEST_FIELDS)


def update_manifest_record(
    manifest_path: Path,
    relative_path: str | Path,
    status: str,
    mask: np.ndarray,
) -> None:
    records = load_manifest(manifest_path)
    if not records:
        return
    key = normalized_relative_path(relative_path).as_posix()
    counts = count_pixels(mask)
    found = False
    for record in records:
        if record.get("source_relative_path") == key:
            record["label_status"] = status
            record["pixels_background"] = str(counts["background"])
            record["pixels_candy_core"] = str(counts["candy_core"])
            record["pixels_sticker"] = str(counts["sticker"])
            found = True
            break
    if found:
        save_manifest(manifest_path, records)
