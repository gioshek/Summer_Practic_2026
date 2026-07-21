from __future__ import annotations

import argparse
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import find_objects
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_holes
from skimage.segmentation import watershed

from radial_symmetry import RadialMarker, create_radial_markers

from common import (
    iter_image_paths,
    load_yaml,
    project_input_path,
    project_path,
    read_image,
    to_bgr_uint8,
    to_gray,
    write_csv,
    write_image,
    write_text,
)


@dataclass(frozen=True)
class StickerCandidate:
    contour: np.ndarray
    rect: tuple[tuple[float, float], tuple[float, float], float]
    area: float
    area_fraction: float
    aspect_ratio: float
    rectangularity: float
    solidity: float
    support_fraction: float
    mean_gray: float
    touches_border: bool
    score: float
    accepted: bool


@dataclass(frozen=True)
class CandidateRecord:
    label: int
    center_x: int
    center_y: int
    area: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    max_radius: float
    mean_gray: float
    touches_border: bool
    strength: str
    marker_score: float
    marker_scale_radius: float
    marker_mask_radius: float
    marker_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Очистка пиксельных масок, проверка стикера и получение кандидатов-конфет."
    )
    parser.add_argument("--project-config", default="configs/project.yaml")
    parser.add_argument("--config", default="configs/stage4.yaml")
    parser.add_argument("--mode", choices=("sample", "raw"), default="sample")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-list", type=Path)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def odd_kernel_size(value: float, minimum: int = 3, maximum: int | None = None) -> int:
    size = max(minimum, int(round(value)))
    if size % 2 == 0:
        size += 1
    if maximum is not None:
        size = min(size, maximum if maximum % 2 == 1 else maximum - 1)
    return max(1, size)


def load_requested_images(input_root: Path, list_path: Path, skip_missing: bool) -> list[Path]:
    if not list_path.exists():
        raise FileNotFoundError(f"Файл списка не найден: {list_path}")

    result: list[Path] = []
    missing: list[str] = []
    seen: set[str] = set()

    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        normalized = Path(line).as_posix().lstrip("./")
        relative = Path(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Недопустимый путь в списке: {line}")
        if normalized in seen:
            continue
        seen.add(normalized)

        source_path = input_root / relative
        if source_path.is_file():
            result.append(source_path)
        else:
            missing.append(normalized)

    if missing:
        text = "\n".join(f"- {value}" for value in missing)
        if not skip_missing:
            raise FileNotFoundError(f"Не найдены изображения из списка:\n{text}")
        print(f"Предупреждение: пропущены отсутствующие изображения:\n{text}")

    return result


def contour_touches_border(contour: np.ndarray, width: int, height: int, margin: int) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin


def rotated_rect_mask(shape: tuple[int, int], rect: tuple[tuple[float, float], tuple[float, float], float]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(np.int32)
    cv2.fillConvexPoly(result, box, 255)
    return result



def detect_sticker(
    image: np.ndarray,
    raw_sticker_mask: np.ndarray,
    raw_candy_mask: np.ndarray,
    source_group: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, StickerCandidate | None, list[StickerCandidate]]:
    """Находит калибровочный стикер по геометрии и яркости.

    Стикер допускается только в явно разрешённых группах. Для папки ``images``
    используются сразу три источника кандидатов: исходная маска класса sticker,
    яркие области с малым замыканием и яркие области с большим горизонтальным
    замыканием. Последний вариант восстанавливает прямоугольник, разрезанный
    лежащей сверху конфетой.
    """
    height, width = raw_sticker_mask.shape
    short_side = min(height, width)
    image_area = float(height * width)
    gray = to_gray(image)

    allowed_groups = {str(value) for value in config.get("allowed_groups", ["images"])}
    if source_group not in allowed_groups:
        return np.zeros((height, width), dtype=np.uint8), None, []

    def kernel_size(fraction: float, minimum: int = 3) -> int:
        return odd_kernel_size(short_side * fraction, minimum=minimum)

    raw_close_size = kernel_size(float(config.get("raw_close_fraction_short_side", 0.012)))
    raw_prepared = cv2.morphologyEx(
        raw_sticker_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (raw_close_size, raw_close_size)),
    )

    percentile = float(config.get("bright_percentile", 99.0))
    threshold = float(
        np.clip(
            np.percentile(gray, percentile),
            float(config.get("bright_threshold_min", 180.0)),
            float(config.get("bright_threshold_max", 215.0)),
        )
    )
    bright_mask = ((gray >= threshold).astype(np.uint8) * 255)
    bright_open_size = kernel_size(float(config.get("bright_open_fraction_short_side", 0.0025)))
    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bright_open_size, bright_open_size)),
    )

    close_height = kernel_size(float(config.get("bright_close_height_fraction", 0.030)))
    small_width = kernel_size(float(config.get("bright_close_width_fraction", 0.050)))
    large_width = kernel_size(float(config.get("bright_large_close_width_fraction", 0.090)))

    bright_small = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (small_width, close_height)),
    )
    bright_large = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (large_width, close_height)),
    )
    combined = cv2.bitwise_or(raw_sticker_mask, bright_mask)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (small_width, close_height)),
    )

    prepared_masks = (raw_prepared, bright_small, bright_large, combined)
    contours: list[np.ndarray] = []
    for prepared in prepared_masks:
        found, _ = cv2.findContours(prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)

    min_area_fraction = float(config.get("min_area_fraction", 0.0025))
    max_area_fraction = float(config.get("max_area_fraction", 0.060))
    min_long_fraction = float(config.get("min_long_fraction", 0.140))
    max_long_fraction = float(config.get("max_long_fraction", 0.290))
    min_short_fraction = float(config.get("min_visible_short_fraction", 0.020))
    max_short_fraction = float(config.get("max_short_fraction", 0.170))
    min_aspect_ratio = float(config.get("min_aspect_ratio", 1.15))
    max_aspect_ratio = float(config.get("max_aspect_ratio", 4.80))
    max_border_aspect_ratio = float(config.get("max_border_aspect_ratio", 10.5))
    min_rectangularity = float(config.get("min_rectangularity", 0.40))
    min_border_rectangularity = float(config.get("min_border_rectangularity", 0.55))
    min_solidity = float(config.get("min_solidity", 0.60))
    min_support_fraction = float(config.get("min_support_fraction", 0.075))
    min_mean_gray = float(config.get("min_mean_gray", 175.0))
    min_contrast = float(config.get("min_contrast", 18.0))
    expected_area_fraction = float(config.get("expected_area_fraction", 0.025))
    expected_long_fraction = float(config.get("expected_long_fraction", 0.216))
    expected_aspect_ratio = float(config.get("expected_aspect_ratio", 1.90))
    border_margin = max(1, int(round(short_side * float(config.get("border_margin_fraction", 0.003)))))
    contrast_ring = max(5, int(round(short_side * float(config.get("contrast_ring_fraction", 0.018)))))

    candidates: list[StickerCandidate] = []
    seen_signatures: set[tuple[int, int, int, int, int]] = set()

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue

        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        if rect_width <= 0 or rect_height <= 0:
            continue

        long_side = max(rect_width, rect_height)
        short_rect_side = min(rect_width, rect_height)
        rect_area = long_side * short_rect_side
        if rect_area <= 0:
            continue

        center_x, center_y = rect[0]
        signature = (
            int(round(center_x / 8)),
            int(round(center_y / 8)),
            int(round(long_side / 8)),
            int(round(short_rect_side / 8)),
            int(round(rect[2] / 5)),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        area_fraction = rect_area / image_area
        aspect_ratio = long_side / max(short_rect_side, 1e-6)
        rectangularity = min(1.0, contour_area / rect_area)

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0

        rect_mask = rotated_rect_mask((height, width), rect)
        rect_pixels = int(np.count_nonzero(rect_mask))
        if rect_pixels == 0:
            continue

        seed_support = ((raw_sticker_mask > 0) | (bright_mask > 0)) & (rect_mask > 0)
        support_fraction = int(np.count_nonzero(seed_support)) / rect_pixels
        raw_support_fraction = int(
            np.count_nonzero((raw_sticker_mask > 0) & (rect_mask > 0))
        ) / rect_pixels

        support_values = gray[seed_support]
        mean_gray = float(np.mean(support_values)) if support_values.size else 0.0

        outside_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (odd_kernel_size(2 * contrast_ring + 1), odd_kernel_size(2 * contrast_ring + 1)),
        )
        outside_ring = cv2.subtract(cv2.dilate(rect_mask, outside_kernel), rect_mask)
        inside_visible = (rect_mask > 0) & (raw_candy_mask == 0)
        outside_visible = (outside_ring > 0) & (raw_candy_mask == 0)
        inside_values = gray[inside_visible]
        outside_values = gray[outside_visible]
        inside_level = float(np.percentile(inside_values, 70)) if inside_values.size else mean_gray
        outside_level = float(np.median(outside_values)) if outside_values.size else 0.0
        contrast = inside_level - outside_level

        touches_border = contour_touches_border(contour, width, height, border_margin)
        long_fraction = long_side / max(width, height)
        short_fraction = short_rect_side / min(width, height)

        long_closeness = math.exp(
            -abs(long_fraction - expected_long_fraction) / max(expected_long_fraction * 0.30, 1e-6)
        )
        area_closeness = math.exp(
            -abs(area_fraction - expected_area_fraction) / max(expected_area_fraction, 1e-6)
        )
        aspect_closeness = math.exp(
            -abs(aspect_ratio - expected_aspect_ratio) / max(expected_aspect_ratio, 1e-6)
        )
        if touches_border:
            area_closeness = max(area_closeness, 0.55 * long_closeness)
            aspect_closeness = max(aspect_closeness, 0.45)

        score = (
            2.0 * rectangularity
            + 1.0 * solidity
            + 1.4 * min(1.0, support_fraction / 0.45)
            + 0.8 * min(1.0, mean_gray / 255.0)
            + 1.4 * long_closeness
            + 0.7 * area_closeness
            + 0.5 * aspect_closeness
            + 0.7 * min(1.0, max(0.0, contrast) / 90.0)
        )

        allowed_aspect = max_border_aspect_ratio if touches_border else max_aspect_ratio
        required_rectangularity = (
            min_border_rectangularity if touches_border else min_rectangularity
        )
        accepted = (
            min_area_fraction <= area_fraction <= max_area_fraction
            and min_long_fraction <= long_fraction <= max_long_fraction
            and min_short_fraction <= short_fraction <= max_short_fraction
            and min_aspect_ratio <= aspect_ratio <= allowed_aspect
            and rectangularity >= required_rectangularity
            and solidity >= min_solidity
            and support_fraction >= min_support_fraction
            and mean_gray >= min_mean_gray
            and (contrast >= min_contrast or raw_support_fraction >= 0.20)
        )

        candidates.append(
            StickerCandidate(
                contour=contour,
                rect=rect,
                area=rect_area,
                area_fraction=area_fraction,
                aspect_ratio=aspect_ratio,
                rectangularity=rectangularity,
                solidity=solidity,
                support_fraction=support_fraction,
                mean_gray=mean_gray,
                touches_border=touches_border,
                score=score,
                accepted=accepted,
            )
        )

    accepted_candidates = [candidate for candidate in candidates if candidate.accepted]
    best = max(accepted_candidates, key=lambda candidate: candidate.score, default=None)
    valid_mask = np.zeros((height, width), dtype=np.uint8)
    if best is not None:
        valid_mask = rotated_rect_mask((height, width), best.rect)

    return valid_mask, best, candidates

def boundary_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)

    size = odd_kernel_size(2 * radius + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    return cv2.subtract(dilated, eroded)



def remove_dark_sensor_borders(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Удаляет протяжённые тёмные полосы сенсора у границ кадра.

    В отличие от простого удаления всех пограничных компонент функция ищет
    именно узкую тёмную область, которая касается края и тянется вдоль
    значительной части изображения. Поэтому частичные конфеты у границы не
    удаляются: маска очищается только внутри найденной тёмной полосы.
    """
    if not np.any(mask):
        return mask.copy(), 0

    gray = to_gray(image)
    height, width = gray.shape
    short_side = min(height, width)
    search_width = max(
        12,
        int(round(short_side * float(config.get("sensor_border_search_fraction", 0.13)))),
    )
    max_thickness = max(
        8,
        int(round(short_side * float(config.get("sensor_border_max_thickness_fraction", 0.10)))),
    )
    min_span_fraction = float(config.get("sensor_border_min_span_fraction", 0.55))
    min_contrast = float(config.get("sensor_border_min_contrast", 55.0))
    threshold_min = float(config.get("sensor_border_threshold_min", 35.0))
    threshold_max = float(config.get("sensor_border_threshold_max", 115.0))
    dilation_radius = max(
        1,
        int(round(short_side * float(config.get("sensor_border_dilation_fraction", 0.0035)))),
    )

    removal = np.zeros_like(mask, dtype=np.uint8)

    def process_vertical(side: str) -> None:
        nonlocal removal
        band_width = min(search_width, width // 3)
        if band_width <= 0:
            return

        if side == "left":
            band = gray[:, :band_width]
            interior = gray[:, band_width : min(width, 2 * band_width)]
        else:
            band = gray[:, width - band_width :]
            interior = gray[:, max(0, width - 2 * band_width) : width - band_width]

        if interior.size == 0:
            return

        interior_level = float(np.median(interior))
        threshold = float(np.clip(interior_level - min_contrast, threshold_min, threshold_max))
        dark = ((band < threshold).astype(np.uint8) * 255)
        close_height = odd_kernel_size(height * 0.045, minimum=9)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, close_height)),
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        min_span = height * min_span_fraction
        for label in range(1, count):
            x, y, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            touches_edge = x <= 2 if side == "left" else x + component_width >= band_width - 2
            if (
                not touches_edge
                or component_height < min_span
                or component_width > max_thickness
                or area < max(20, component_height * 2)
            ):
                continue

            component = ((labels == label).astype(np.uint8) * 255)
            kernel_size = odd_kernel_size(2 * dilation_radius + 1)
            component = cv2.dilate(
                component,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            )
            if side == "left":
                target = removal[:, :band_width]
            else:
                target = removal[:, width - band_width :]
            target[component > 0] = 255

    def process_horizontal(side: str) -> None:
        nonlocal removal
        band_height = min(search_width, height // 3)
        if band_height <= 0:
            return

        if side == "top":
            band = gray[:band_height, :]
            interior = gray[band_height : min(height, 2 * band_height), :]
        else:
            band = gray[height - band_height :, :]
            interior = gray[max(0, height - 2 * band_height) : height - band_height, :]

        if interior.size == 0:
            return

        interior_level = float(np.median(interior))
        threshold = float(np.clip(interior_level - min_contrast, threshold_min, threshold_max))
        dark = ((band < threshold).astype(np.uint8) * 255)
        close_width = odd_kernel_size(width * 0.045, minimum=9)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 7)),
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        min_span = width * min_span_fraction
        for label in range(1, count):
            x, y, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            touches_edge = y <= 2 if side == "top" else y + component_height >= band_height - 2
            if (
                not touches_edge
                or component_width < min_span
                or component_height > max_thickness
                or area < max(20, component_width * 2)
            ):
                continue

            component = ((labels == label).astype(np.uint8) * 255)
            kernel_size = odd_kernel_size(2 * dilation_radius + 1)
            component = cv2.dilate(
                component,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            )
            if side == "top":
                target = removal[:band_height, :]
            else:
                target = removal[height - band_height :, :]
            target[component > 0] = 255

    for side in ("left", "right"):
        process_vertical(side)
    for side in ("top", "bottom"):
        process_horizontal(side)

    result = mask.copy()
    removed_pixels = int(np.count_nonzero((result > 0) & (removal > 0)))
    result[removal > 0] = 0
    return result, removed_pixels


def remove_thin_border_runs(
    mask: np.ndarray,
    distance: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Удаляет оставшиеся длинные тонкие следы класса candy у края кадра."""
    height, width = mask.shape
    short_side = min(height, width)
    strip_width = max(2, int(round(short_side * float(config.get("border_strip_fraction", 0.014)))))
    max_radius = short_side * float(config.get("border_strip_max_radius_fraction", 0.010))
    min_span_fraction = float(config.get("border_strip_min_span_fraction", 0.18))

    thin = (mask > 0) & (distance <= max_radius)
    remove = np.zeros_like(mask, dtype=np.uint8)

    bands = [
        (slice(0, height), slice(0, strip_width), height, "vertical"),
        (slice(0, height), slice(max(0, width - strip_width), width), height, "vertical"),
        (slice(0, strip_width), slice(0, width), width, "horizontal"),
        (slice(max(0, height - strip_width), height), slice(0, width), width, "horizontal"),
    ]

    for y_slice, x_slice, reference_length, orientation in bands:
        roi = thin[y_slice, x_slice].astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        min_span = reference_length * min_span_fraction
        for label in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[label])
            span = h if orientation == "vertical" else w
            if span < min_span or area < max(8, strip_width):
                continue
            target = remove[y_slice, x_slice]
            target[labels == label] = 255

    result = mask.copy()
    removed_pixels = int(np.count_nonzero((result > 0) & (remove > 0)))
    result[remove > 0] = 0
    return result, removed_pixels

def component_filter(
    mask: np.ndarray,
    distance: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    short_side = min(height, width)
    image_area = height * width

    min_area = max(4, int(round(image_area * float(config.get("min_component_area_fraction", 0.000015)))))
    min_radius = short_side * float(config.get("min_component_radius_fraction", 0.0025))
    border_aspect = float(config.get("border_artifact_aspect_ratio", 9.0))
    border_max_radius = short_side * float(config.get("border_artifact_max_radius_fraction", 0.0045))
    border_margin = max(1, int(round(short_side * float(config.get("border_margin_fraction", 0.003)))))

    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    result = np.zeros_like(mask, dtype=np.uint8)
    removed = 0

    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        label_roi = labels[y : y + h, x : x + w]
        component_roi = label_roi == label
        distance_roi = distance[y : y + h, x : x + w]
        max_radius = float(np.max(distance_roi[component_roi])) if np.any(component_roi) else 0.0
        aspect = max(w, h) / max(1.0, min(w, h))
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or x + w >= width - border_margin
            or y + h >= height - border_margin
        )

        tiny = area < min_area and max_radius < min_radius
        thin_border_artifact = touches_border and aspect >= border_aspect and max_radius < border_max_radius

        if tiny or thin_border_artifact:
            removed += 1
            continue

        result_roi = result[y : y + h, x : x + w]
        result_roi[component_roi] = 255

    return result, removed



def clean_candy_mask(
    image: np.ndarray,
    raw_candy_mask: np.ndarray,
    raw_sticker_mask: np.ndarray,
    valid_sticker_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = raw_candy_mask.shape
    short_side = min(height, width)
    image_area = height * width

    raw_distance = cv2.distanceTransform((raw_candy_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    raw_ring_radius = max(1, int(round(short_side * float(config.get("raw_boundary_fraction", 0.004)))))
    geometric_ring_radius = max(1, int(round(short_side * float(config.get("geometric_boundary_fraction", 0.006)))))
    fringe_max_radius = short_side * float(config.get("thin_sticker_fringe_fraction", 0.0045))

    raw_ring = boundary_ring(raw_sticker_mask, raw_ring_radius)
    geometric_ring = boundary_ring(valid_sticker_mask, geometric_ring_radius)
    suppress_region = (raw_ring > 0) | (geometric_ring > 0)
    thin_fringe = suppress_region & (raw_distance <= fringe_max_radius)

    cleaned = raw_candy_mask.copy()
    removed_fringe_pixels = int(np.count_nonzero((cleaned > 0) & thin_fringe))
    cleaned[thin_fringe] = 0

    cleaned, removed_sensor_border_pixels = remove_dark_sensor_borders(image, cleaned, config)

    close_size = odd_kernel_size(short_side * float(config.get("close_fraction_short_side", 0.0025)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    hole_area = max(8, int(round(image_area * float(config.get("hole_area_fraction", 0.00005)))))
    cleaned_bool = remove_small_holes(cleaned > 0, max_size=max(0, hole_area - 1), connectivity=2)
    cleaned = (cleaned_bool.astype(np.uint8) * 255)

    distance = cv2.distanceTransform((cleaned > 0).astype(np.uint8), cv2.DIST_L2, 5)
    cleaned, removed_border_pixels = remove_thin_border_runs(cleaned, distance, config)

    distance = cv2.distanceTransform((cleaned > 0).astype(np.uint8), cv2.DIST_L2, 5)
    cleaned, removed_components = component_filter(cleaned, distance, config)

    return cleaned, {
        "removed_fringe_pixels": removed_fringe_pixels,
        "removed_sensor_border_pixels": removed_sensor_border_pixels,
        "removed_border_pixels": removed_border_pixels,
        "removed_components": removed_components,
    }

def line_saddle_value(
    distance: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
) -> float:
    """Возвращает минимальную высоту карты расстояний между двумя максимумами."""
    first_y, first_x = first
    second_y, second_x = second
    length = max(2, int(round(math.hypot(second_y - first_y, second_x - first_x))) + 1)
    y_values = np.rint(np.linspace(first_y, second_y, length)).astype(np.int32)
    x_values = np.rint(np.linspace(first_x, second_x, length)).astype(np.int32)
    values = distance[y_values, x_values]
    trim = max(1, int(round(length * 0.15)))
    if length > 2 * trim:
        values = values[trim : length - trim]
    return float(np.min(values)) if values.size else 0.0


def suppress_duplicate_peaks(
    peaks: list[tuple[float, int, int]],
    smoothed_distance: np.ndarray,
    short_side: int,
    config: dict[str, Any],
) -> list[tuple[float, int, int]]:
    """Подавляет дубли внутри одной конфеты, сохраняя близкие разные конфеты.

    Решение принимается не только по расстоянию между точками, но и по глубине
    впадины на distance transform. Выраженная впадина означает две разные
    конфеты; неглубокая — два максимума внутри одного тела.
    """
    saddle_ratio = float(config.get("duplicate_saddle_ratio", 0.40))
    suppression_factor = float(config.get("adaptive_suppression_factor", 0.85))
    close_factor = float(config.get("very_close_factor", 0.50))
    close_saddle_ratio = float(config.get("very_close_saddle_ratio", 0.25))
    minimum_distance = max(
        3.0,
        short_side * float(config.get("minimum_suppression_fraction", 0.010)),
    )

    accepted: list[tuple[float, int, int]] = []
    for radius, y, x in sorted(peaks, reverse=True):
        keep = True
        for accepted_radius, accepted_y, accepted_x in accepted:
            point_distance = math.hypot(y - accepted_y, x - accepted_x)
            duplicate_limit = max(
                minimum_distance,
                suppression_factor * (radius + accepted_radius),
            )
            if point_distance > duplicate_limit:
                continue

            saddle = line_saddle_value(
                smoothed_distance,
                (y, x),
                (accepted_y, accepted_x),
            )
            ratio = saddle / max(1e-6, min(radius, accepted_radius))
            very_close = point_distance < close_factor * (radius + accepted_radius)
            if ratio >= saddle_ratio or (very_close and ratio >= close_saddle_ratio):
                keep = False
                break

        if keep:
            accepted.append((radius, y, x))

    return accepted


def create_markers(
    cleaned_mask: np.ndarray,
    config: dict[str, Any],
    image: np.ndarray | None = None,
    source_group: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = cleaned_mask.shape
    short_side = min(height, width)
    binary = (cleaned_mask > 0).astype(np.uint8)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    sigma = max(0.8, short_side * float(config.get("gaussian_sigma_fraction", 0.0015)))
    smoothed = cv2.GaussianBlur(distance, (0, 0), sigmaX=sigma, sigmaY=sigma)

    raw_min_distance = max(
        4,
        int(round(short_side * float(config.get("raw_min_distance_fraction", 0.006)))),
    )
    min_peak_radius = max(
        2.5,
        short_side * float(config.get("min_peak_radius_fraction", 0.0030)),
    )
    threshold_relative = float(config.get("threshold_relative", 0.04))
    max_peaks = int(config.get("max_peaks", 5000))

    raw_coordinates = peak_local_max(
        smoothed,
        min_distance=raw_min_distance,
        threshold_abs=min_peak_radius,
        threshold_rel=threshold_relative,
        exclude_border=False,
        num_peaks=max_peaks,
        labels=binary,
    )
    raw_peaks = [
        (float(smoothed[int(y), int(x)]), int(y), int(x))
        for y, x in raw_coordinates
    ]
    accepted = suppress_duplicate_peaks(raw_peaks, smoothed, short_side, config)

    # В сценах basler_120126 часть тел конфет сливается в одну широкую
    # distance-transform область. Для этой группы добавляем только те
    # текстурные максимумы исходного изображения, которые находятся далеко
    # от уже выбранных центров. Это восстанавливает центры в плотных кучах,
    # не возвращая множество дублей на одиночных конфетах.
    recovery_groups = {
        str(value) for value in config.get("texture_recovery_groups", ["basler_120126"])
    }
    if image is not None and source_group in recovery_groups and np.any(binary):
        gray = to_gray(image).astype(np.float32) / 255.0
        local_sigma = max(1.5, short_side * float(config.get("texture_local_sigma_fraction", 0.0033)))
        score_sigma = max(3.0, short_side * float(config.get("texture_score_sigma_fraction", 0.010)))
        local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
        local_square_mean = cv2.GaussianBlur(
            gray * gray,
            (0, 0),
            sigmaX=local_sigma,
            sigmaY=local_sigma,
        )
        local_std = np.sqrt(np.maximum(0.0, local_square_mean - local_mean * local_mean))
        texture = cv2.GaussianBlur(local_std, (0, 0), sigmaX=score_sigma, sigmaY=score_sigma)
        radius_scale = max(1.0, short_side * float(config.get("texture_radius_scale_fraction", 0.030)))
        normalized_radius = np.minimum(distance / radius_scale, 1.0)
        texture_score = texture * (0.25 + 0.75 * normalized_radius)
        texture_score[binary == 0] = 0.0

        values = texture_score[binary > 0]
        if values.size:
            percentile = float(config.get("texture_threshold_percentile", 40.0))
            texture_threshold = max(
                float(np.percentile(values, percentile)),
                float(config.get("texture_threshold_min", 0.004)),
            )
            texture_min_distance = max(
                8,
                int(round(short_side * float(config.get("texture_min_distance_fraction", 0.018)))),
            )
            texture_coordinates = peak_local_max(
                texture_score,
                min_distance=texture_min_distance,
                threshold_abs=texture_threshold,
                exclude_border=False,
                num_peaks=int(config.get("texture_max_peaks", 2000)),
                labels=binary,
            )

            minimum_spacing = short_side * float(
                config.get("texture_min_spacing_fraction", 0.014)
            )
            maximum_spacing = short_side * float(
                config.get("texture_max_spacing_fraction", 0.032)
            )
            radius_spacing_factor = float(config.get("texture_radius_spacing_factor", 0.75))
            spacing_offset = short_side * float(
                config.get("texture_spacing_offset_fraction", 0.008)
            )

            for y, x in texture_coordinates:
                y = int(y)
                x = int(x)
                radius = float(smoothed[y, x])
                if radius < min_peak_radius:
                    continue
                nearest = min(
                    (math.hypot(y - other_y, x - other_x) for _, other_y, other_x in accepted),
                    default=float("inf"),
                )
                required_spacing = max(
                    minimum_spacing,
                    min(maximum_spacing, radius_spacing_factor * radius + spacing_offset),
                )
                if nearest >= required_spacing:
                    accepted.append((radius, y, x))

    # Гарантируем хотя бы один центр каждой отдельной достаточно толстой
    # связной области. Это сохраняет слабые конфеты в тёмных участках.
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    fallback_multiplier = float(config.get("fallback_radius_multiplier", 0.70))
    for component_label in range(1, component_count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[component_label]
        )
        if area <= 0:
            continue
        has_marker = any(
            component_labels[marker_y, marker_x] == component_label
            for _, marker_y, marker_x in accepted
        )
        if has_marker:
            continue

        component = component_labels[y : y + component_height, x : x + component_width] == component_label
        values = np.where(component, smoothed[y : y + component_height, x : x + component_width], -1.0)
        local_y, local_x = np.unravel_index(int(np.argmax(values)), values.shape)
        radius = float(values[local_y, local_x])
        if radius >= min_peak_radius * fallback_multiplier:
            accepted.append((radius, y + int(local_y), x + int(local_x)))

    markers = np.zeros(cleaned_mask.shape, dtype=np.int32)
    for marker_index, (_, y, x) in enumerate(accepted, 1):
        markers[y, x] = marker_index

    return markers, distance, smoothed

def split_instances(
    image: np.ndarray,
    cleaned_mask: np.ndarray,
    markers: np.ndarray,
    distance: np.ndarray,
    radial_score: np.ndarray,
    marker_records: list[RadialMarker],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[CandidateRecord]]:
    if not np.any(cleaned_mask) or int(np.max(markers)) == 0:
        return np.zeros(cleaned_mask.shape, dtype=np.int32), []

    labels = watershed(
        -distance,
        markers=markers,
        mask=cleaned_mask > 0,
        connectivity=2,
        watershed_line=False,
    ).astype(np.int32)

    height, width = cleaned_mask.shape
    short_side = min(height, width)
    image_area = height * width
    gray = to_gray(image)
    marker_by_id = {record.marker_id: record for record in marker_records}

    min_area = max(4, int(round(image_area * float(config.get("min_area_fraction", 0.000025)))))
    min_radius = short_side * float(config.get("min_radius_fraction", 0.0030))
    strong_area = max(min_area, int(round(image_area * float(config.get("strong_area_fraction", 0.00010)))))
    strong_radius = short_side * float(config.get("strong_radius_fraction", 0.0060))
    border_min_area = max(2, int(round(image_area * float(config.get("border_min_area_fraction", 0.000012)))))

    records: list[CandidateRecord] = []
    filtered = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    object_slices = find_objects(labels)

    for label, region_slice in enumerate(object_slices, 1):
        if region_slice is None:
            continue

        y_slice, x_slice = region_slice
        region_labels = labels[region_slice]
        region = region_labels == label
        area = int(np.count_nonzero(region))
        if area == 0:
            continue

        x0 = int(x_slice.start)
        x1 = int(x_slice.stop)
        y0 = int(y_slice.start)
        y1 = int(y_slice.stop)
        bbox_width = x1 - x0
        bbox_height = y1 - y0

        distance_roi = distance[region_slice]
        gray_roi = gray[region_slice]
        max_radius = float(np.max(distance_roi[region]))
        mean_gray = float(np.mean(gray_roi[region]))
        touches_border = x0 == 0 or y0 == 0 or x1 == width or y1 == height

        marker_record = marker_by_id.get(label)
        valid = marker_record is not None or (
            max_radius >= min_radius
            and (area >= min_area or (touches_border and area >= border_min_area))
        )
        if not valid:
            continue

        if marker_record is None:
            local_peak_index = int(np.argmax(np.where(region, distance_roi, -1.0)))
            local_y, local_x = np.unravel_index(local_peak_index, distance_roi.shape)
            center_y = y0 + int(local_y)
            center_x = x0 + int(local_x)
            marker_score = float(radial_score[center_y, center_x])
            marker_scale_radius = max_radius
            marker_mask_radius = max_radius
            marker_source = "watershed_fallback"
            strength = "weak"
        else:
            center_y = marker_record.center_y
            center_x = marker_record.center_x
            marker_score = marker_record.score
            marker_scale_radius = marker_record.scale_radius
            marker_mask_radius = marker_record.mask_radius
            marker_source = marker_record.source
            strength = marker_record.strength

        # Уверенность определяется качеством центра, а не случайной площадью
        # watershed-региона. В плотной куче правильный центр может получить
        # очень маленькую область из-за соседних маркеров.
        filtered_roi = filtered[region_slice]
        filtered_roi[region] = next_label
        records.append(
            CandidateRecord(
                label=next_label,
                center_x=center_x,
                center_y=center_y,
                area=area,
                bbox_x=x0,
                bbox_y=y0,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                max_radius=max_radius,
                mean_gray=mean_gray,
                touches_border=touches_border,
                strength=strength,
                marker_score=marker_score,
                marker_scale_radius=marker_scale_radius,
                marker_mask_radius=marker_mask_radius,
                marker_source=marker_source,
            )
        )
        next_label += 1

    return filtered, records

def visualize(
    image: np.ndarray,
    cleaned_mask: np.ndarray,
    valid_sticker_mask: np.ndarray,
    instance_labels: np.ndarray,
    candidates: list[CandidateRecord],
    overlay_alpha: float,
    draw_ids: bool,
    draw_watershed_contours: bool,
) -> np.ndarray:
    result = to_bgr_uint8(image)
    green = np.zeros_like(result)
    green[:, :, 1] = 255
    blended = cv2.addWeighted(result, 1.0 - overlay_alpha, green, overlay_alpha, 0.0)
    result[cleaned_mask > 0] = blended[cleaned_mask > 0]

    if np.any(valid_sticker_mask):
        contours, _ = cv2.findContours(valid_sticker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 0), 3, cv2.LINE_AA)

    for candidate in candidates:
        x0 = candidate.bbox_x
        y0 = candidate.bbox_y
        x1 = x0 + candidate.bbox_width
        y1 = y0 + candidate.bbox_height
        region = (instance_labels[y0:y1, x0:x1] == candidate.label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour[:, 0, 0] += x0
            contour[:, 0, 1] += y0
        contour_color = (0, 255, 255) if candidate.strength == "strong" else (255, 0, 255)
        # Контур watershed оставляем тонким как техническую границу. Основной
        # результат этапа 4.3 — центр и масштаб тела конфеты по FRST.
        if draw_watershed_contours:
            cv2.drawContours(result, contours, -1, contour_color, 1, cv2.LINE_AA)
        marker_color = (0, 0, 255) if candidate.strength == "strong" else (255, 0, 255)
        detection_radius = max(7, int(round(candidate.marker_scale_radius * 1.15)))
        cv2.circle(
            result,
            (candidate.center_x, candidate.center_y),
            detection_radius,
            marker_color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            result,
            (candidate.center_x, candidate.center_y),
            4,
            marker_color,
            -1,
            cv2.LINE_AA,
        )
        if draw_ids:
            cv2.putText(
                result,
                str(candidate.label),
                (candidate.center_x + 5, candidate.center_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    strong_count = sum(candidate.strength == "strong" for candidate in candidates)
    weak_count = len(candidates) - strong_count
    sticker_text = "yes" if np.any(valid_sticker_mask) else "no"
    text = f"candidates={len(candidates)} strong={strong_count} weak={weak_count} sticker={sticker_text}"
    cv2.rectangle(result, (8, 8), (min(result.shape[1] - 8, 760), 42), (0, 0, 0), -1)
    cv2.putText(result, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def distance_visualization(distance: np.ndarray) -> np.ndarray:
    if not np.any(distance > 0):
        return np.zeros((*distance.shape, 3), dtype=np.uint8)
    normalized = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def sticker_debug_visualization(
    image: np.ndarray,
    candidates: list[StickerCandidate],
    accepted: StickerCandidate | None,
) -> np.ndarray:
    result = to_bgr_uint8(image)
    for candidate in candidates:
        box = np.round(cv2.boxPoints(candidate.rect)).astype(np.int32)
        color = (0, 255, 255) if candidate is accepted else (0, 0, 255)
        cv2.polylines(result, [box], True, color, 2, cv2.LINE_AA)
        center = tuple(np.round(candidate.rect[0]).astype(int))
        text = f"a={candidate.area_fraction:.3f} r={candidate.rectangularity:.2f} s={candidate.score:.2f}"
        cv2.putText(result, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return result


def main() -> int:
    args = parse_args()
    project_config = load_yaml(Path(args.project_config))
    stage_config = load_yaml(Path(args.config))

    sticker_config = dict(stage_config.get("sticker", {}))
    cleanup_config = dict(stage_config.get("candy_cleanup", {}))
    marker_config = dict(stage_config.get("markers", {}))
    candidate_config = dict(stage_config.get("candidates", {}))
    visualization_config = dict(stage_config.get("visualization", {}))

    input_root = project_input_path(project_config, args.mode).resolve()
    output_root = project_path(project_config, "output", "output").resolve()
    raw_mask_root = output_root / "masks"

    if not input_root.exists():
        raise FileNotFoundError(f"Источник изображений не найден: {input_root}")
    if not raw_mask_root.exists():
        raise FileNotFoundError(
            f"Папка сырых масок не найдена: {raw_mask_root}. "
            f"Сначала запусти ./run_stage3.sh predict {args.mode}."
        )

    cleaned_root = output_root / "cleaned_masks"
    sticker_root = output_root / "sticker_masks"
    marker_root = output_root / "marker_masks"
    marker_preview_root = output_root / "marker_previews"
    instance_root = output_root / "instance_labels"
    annotated_root = output_root / "annotated"
    report_root = output_root / "reports"
    distance_root = output_root / "distance_maps"
    sticker_debug_root = output_root / "sticker_candidates"

    if not args.no_clear:
        clear_paths = [
            cleaned_root,
            sticker_root,
            marker_root,
            marker_preview_root,
            instance_root,
            annotated_root,
            distance_root,
            sticker_debug_root,
        ]
        for path in clear_paths:
            if path.exists():
                shutil.rmtree(path)

    for path in (cleaned_root, sticker_root, marker_root, marker_preview_root, instance_root, annotated_root, report_root):
        path.mkdir(parents=True, exist_ok=True)
    if args.debug:
        distance_root.mkdir(parents=True, exist_ok=True)
        sticker_debug_root.mkdir(parents=True, exist_ok=True)

    if args.include_list is not None:
        image_paths = load_requested_images(input_root, args.include_list, args.skip_missing)
    else:
        image_paths = list(iter_image_paths(input_root, recursive=True))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError(f"В источнике нет изображений: {input_root}")

    missing_masks = [
        path.relative_to(input_root).as_posix()
        for path in image_paths
        if not (raw_mask_root / path.relative_to(input_root).with_suffix(".png")).is_file()
    ]
    if missing_masks:
        preview = "\n".join(f"- {value}" for value in missing_masks[:20])
        suffix = "" if len(missing_masks) <= 20 else f"\n... и ещё {len(missing_masks) - 20}"
        raise FileNotFoundError(
            f"Не хватает сырых масок этапа №3 для {len(missing_masks)} изображений:\n"
            f"{preview}{suffix}\n"
            f"Запусти ./run_stage3.sh predict {args.mode}, затем повтори этап №4."
        )

    overlay_alpha = float(visualization_config.get("overlay_alpha", 0.38))
    jpeg_quality = int(visualization_config.get("jpeg_quality", 92))
    draw_ids = bool(visualization_config.get("draw_candidate_ids", True))
    draw_watershed_contours = bool(
        visualization_config.get("draw_watershed_contours", False)
    )

    print("Этап №4.3: поиск центров конфет по радиальной симметрии")
    print(f"Режим: {args.mode}")
    print(f"Источник: {input_root}")
    print(f"Сырые маски: {raw_mask_root}")
    print(f"Изображений: {len(image_paths)}")
    print(f"Отладочные карты: {'да' if args.debug else 'нет'}")
    if args.include_list is not None:
        print(f"Точный список: {args.include_list}")

    image_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    sticker_rows: list[dict[str, object]] = []
    errors = 0
    total_candidates = 0
    total_strong = 0
    total_weak = 0
    detected_stickers = 0
    started_total = time.perf_counter()

    for image_index, source_path in enumerate(image_paths, 1):
        relative = source_path.relative_to(input_root)
        started = time.perf_counter()
        try:
            image = read_image(source_path)
            raw_mask_path = raw_mask_root / relative.with_suffix(".png")
            raw_mask = read_image(raw_mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Размер маски {raw_mask.shape[::-1]} не совпадает с изображением {image.shape[1]}x{image.shape[0]}"
                )

            raw_candy = ((raw_mask == 2).astype(np.uint8) * 255)
            raw_sticker = ((raw_mask == 3).astype(np.uint8) * 255)

            source_group = relative.parts[0] if len(relative.parts) > 1 else "root"
            valid_sticker, best_sticker, sticker_candidates = detect_sticker(
                image, raw_sticker, raw_candy, source_group, sticker_config
            )
            if best_sticker is not None:
                detected_stickers += 1

            cleanup_runtime_config = dict(cleanup_config)
            cleanup_runtime_config["raw_boundary_fraction"] = sticker_config.get("raw_boundary_fraction", 0.004)
            cleanup_runtime_config["geometric_boundary_fraction"] = sticker_config.get(
                "geometric_boundary_fraction", 0.006
            )
            cleaned_mask, cleanup_stats = clean_candy_mask(
                image,
                raw_candy,
                raw_sticker,
                valid_sticker,
                cleanup_runtime_config,
            )

            markers, distance, radial_score, marker_records = create_radial_markers(
                image,
                cleaned_mask,
                source_group,
                marker_config,
            )
            instance_labels, candidates = split_instances(
                image,
                cleaned_mask,
                markers,
                distance,
                radial_score,
                marker_records,
                candidate_config,
            )

            marker_visual = np.zeros_like(cleaned_mask, dtype=np.uint8)
            marker_visual[markers > 0] = 255
            preview_radius = max(2, int(round(min(cleaned_mask.shape) * float(
                visualization_config.get("marker_preview_radius_fraction", 0.0045)
            ))))
            preview_size = odd_kernel_size(2 * preview_radius + 1)
            marker_preview = cv2.dilate(
                marker_visual,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (preview_size, preview_size)),
            )

            write_image(cleaned_root / relative.with_suffix(".png"), cleaned_mask)
            write_image(sticker_root / relative.with_suffix(".png"), valid_sticker)
            write_image(marker_root / relative.with_suffix(".png"), marker_visual)
            write_image(marker_preview_root / relative.with_suffix(".png"), marker_preview)
            write_image(instance_root / relative.with_suffix(".png"), instance_labels.astype(np.uint16))

            annotated = visualize(
                image,
                cleaned_mask,
                valid_sticker,
                instance_labels,
                candidates,
                overlay_alpha,
                draw_ids,
                draw_watershed_contours,
            )
            write_image(annotated_root / relative.with_suffix(".jpg"), annotated, jpeg_quality=jpeg_quality)

            if args.debug:
                write_image(distance_root / relative.with_suffix(".jpg"), distance_visualization(radial_score))
                debug_sticker = sticker_debug_visualization(image, sticker_candidates, best_sticker)
                write_image(
                    sticker_debug_root / relative.with_suffix(".jpg"),
                    debug_sticker,
                    jpeg_quality=jpeg_quality,
                )

            strong_count = sum(candidate.strength == "strong" for candidate in candidates)
            weak_count = len(candidates) - strong_count
            total_candidates += len(candidates)
            total_strong += strong_count
            total_weak += weak_count

            for candidate in candidates:
                candidate_rows.append(
                    {
                        "source_relative_path": relative.as_posix(),
                        "source_group": relative.parts[0] if len(relative.parts) > 1 else "root",
                        "candidate_id": candidate.label,
                        "center_x": candidate.center_x,
                        "center_y": candidate.center_y,
                        "area": candidate.area,
                        "bbox_x": candidate.bbox_x,
                        "bbox_y": candidate.bbox_y,
                        "bbox_width": candidate.bbox_width,
                        "bbox_height": candidate.bbox_height,
                        "max_radius": f"{candidate.max_radius:.3f}",
                        "mean_gray": f"{candidate.mean_gray:.3f}",
                        "touches_border": "yes" if candidate.touches_border else "no",
                        "strength": candidate.strength,
                        "marker_score": f"{candidate.marker_score:.6f}",
                        "marker_scale_radius": f"{candidate.marker_scale_radius:.3f}",
                        "marker_mask_radius": f"{candidate.marker_mask_radius:.3f}",
                        "marker_source": candidate.marker_source,
                    }
                )

            for sticker_index, candidate in enumerate(sticker_candidates, 1):
                sticker_rows.append(
                    {
                        "source_relative_path": relative.as_posix(),
                        "candidate_id": sticker_index,
                        "accepted": "yes" if candidate is best_sticker else "no",
                        "area_fraction": f"{candidate.area_fraction:.6f}",
                        "aspect_ratio": f"{candidate.aspect_ratio:.4f}",
                        "rectangularity": f"{candidate.rectangularity:.4f}",
                        "solidity": f"{candidate.solidity:.4f}",
                        "support_fraction": f"{candidate.support_fraction:.4f}",
                        "mean_gray": f"{candidate.mean_gray:.3f}",
                        "touches_border": "yes" if candidate.touches_border else "no",
                        "score": f"{candidate.score:.4f}",
                    }
                )

            elapsed = time.perf_counter() - started
            image_rows.append(
                {
                    "source_relative_path": relative.as_posix(),
                    "source_group": relative.parts[0] if len(relative.parts) > 1 else "root",
                    "raw_candy_pixels": int(np.count_nonzero(raw_candy)),
                    "cleaned_candy_pixels": int(np.count_nonzero(cleaned_mask)),
                    "raw_sticker_pixels": int(np.count_nonzero(raw_sticker)),
                    "valid_sticker_pixels": int(np.count_nonzero(valid_sticker)),
                    "valid_sticker": "yes" if best_sticker is not None else "no",
                    "sticker_candidate_count": len(sticker_candidates),
                    "removed_fringe_pixels": cleanup_stats["removed_fringe_pixels"],
                    "removed_sensor_border_pixels": cleanup_stats["removed_sensor_border_pixels"],
                    "removed_border_pixels": cleanup_stats["removed_border_pixels"],
                    "removed_components": cleanup_stats["removed_components"],
                    "marker_count": len(marker_records),
                    "candidate_count": len(candidates),
                    "strong_candidates": strong_count,
                    "weak_candidates": weak_count,
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "error": "",
                }
            )
            print(
                f"[{image_index:03d}/{len(image_paths):03d}] {relative.as_posix()} | "
                f"candidates={len(candidates)} strong={strong_count} weak={weak_count} "
                f"sticker={'yes' if best_sticker is not None else 'no'} | {elapsed:.2f}s"
            )
        except Exception as error:
            errors += 1
            elapsed = time.perf_counter() - started
            image_rows.append(
                {
                    "source_relative_path": relative.as_posix(),
                    "source_group": relative.parts[0] if len(relative.parts) > 1 else "root",
                    "raw_candy_pixels": "",
                    "cleaned_candy_pixels": "",
                    "raw_sticker_pixels": "",
                    "valid_sticker_pixels": "",
                    "valid_sticker": "",
                    "sticker_candidate_count": "",
                    "removed_fringe_pixels": "",
                    "removed_sensor_border_pixels": "",
                    "removed_border_pixels": "",
                    "removed_components": "",
                    "marker_count": "",
                    "candidate_count": "",
                    "strong_candidates": "",
                    "weak_candidates": "",
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "error": str(error),
                }
            )
            print(f"[{image_index:03d}/{len(image_paths):03d}] ОШИБКА {relative.as_posix()}: {error}")

    total_elapsed = time.perf_counter() - started_total

    write_csv(
        report_root / "stage4_images.csv",
        image_rows,
        [
            "source_relative_path",
            "source_group",
            "raw_candy_pixels",
            "cleaned_candy_pixels",
            "raw_sticker_pixels",
            "valid_sticker_pixels",
            "valid_sticker",
            "sticker_candidate_count",
            "removed_fringe_pixels",
            "removed_sensor_border_pixels",
            "removed_border_pixels",
            "removed_components",
            "marker_count",
            "candidate_count",
            "strong_candidates",
            "weak_candidates",
            "elapsed_seconds",
            "error",
        ],
    )
    write_csv(
        report_root / "stage4_candidates.csv",
        candidate_rows,
        [
            "source_relative_path",
            "source_group",
            "candidate_id",
            "center_x",
            "center_y",
            "area",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "max_radius",
            "mean_gray",
            "touches_border",
            "strength",
            "marker_score",
            "marker_scale_radius",
            "marker_mask_radius",
            "marker_source",
        ],
    )
    write_csv(
        report_root / "stage4_sticker_candidates.csv",
        sticker_rows,
        [
            "source_relative_path",
            "candidate_id",
            "accepted",
            "area_fraction",
            "aspect_ratio",
            "rectangularity",
            "solidity",
            "support_fraction",
            "mean_gray",
            "touches_border",
            "score",
        ],
    )

    summary = [
        "Этап №4.3: радиально-симметричные центры конфет",
        f"Режим: {args.mode}",
        f"Источник: {input_root}",
        f"Обработано изображений: {len(image_paths)}",
        f"Ошибок: {errors}",
        f"Подтверждённых стикеров: {detected_stickers}",
        f"Найденных центров-кандидатов: {total_candidates}",
        f"Сильных кандидатов: {total_strong}",
        f"Слабых кандидатов: {total_weak}",
        f"Время: {total_elapsed:.1f} с",
        f"Очищенные маски: {cleaned_root}",
        f"Маски стикера: {sticker_root}",
        f"Маркеры: {marker_root}",
        f"Просмотровые маркеры: {marker_preview_root}",
        f"Метки экземпляров: {instance_root}",
        f"Аннотированные изображения: {annotated_root}",
        f"Отчёт по изображениям: {report_root / 'stage4_images.csv'}",
        f"Отчёт по кандидатам: {report_root / 'stage4_candidates.csv'}",
        f"Завершено UTC: {datetime.now(timezone.utc).isoformat()}",
    ]
    write_text(report_root / "stage4_summary.txt", "\n".join(summary) + "\n")

    print("\nПоиск центров завершён.")
    for line in summary[1:]:
        print(line)

    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
