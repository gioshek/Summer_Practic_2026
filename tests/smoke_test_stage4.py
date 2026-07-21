from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from postprocess_candidates import clean_candy_mask, detect_sticker, split_instances
from radial_symmetry import create_radial_markers


def make_scene() -> tuple[np.ndarray, np.ndarray]:
    height, width = 600, 900
    image = np.full((height, width), 95, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = np.ones((height, width), dtype=np.uint8)

    cv2.rectangle(image, (310, 235), (570, 365), (245, 245, 245), -1)
    cv2.rectangle(mask, (310, 235), (570, 365), 3, -1)

    centers = [(210, 180), (270, 185), (500, 170), (565, 195), (665, 420)]
    for center in centers:
        cv2.ellipse(image, center, (34, 23), 0, 0, 360, (155, 155, 155), -1)
        cv2.ellipse(image, center, (23, 14), 0, 0, 360, (205, 205, 205), 2)
        cv2.line(
            image,
            (center[0] - 15, center[1]),
            (center[0] + 15, center[1]),
            (55, 55, 55),
            3,
        )
        cv2.ellipse(mask, center, (34, 23), 0, 0, 360, 2, -1)

    cv2.rectangle(image, (865, 0), (899, 599), (8, 8, 8), -1)
    cv2.rectangle(mask, (870, 20), (899, 580), 2, -1)
    return image, mask


def main() -> int:
    image, mask = make_scene()
    raw_candy = (mask == 2).astype(np.uint8) * 255
    raw_sticker = (mask == 3).astype(np.uint8) * 255

    sticker_config = {
        "allowed_groups": ["images"],
        "raw_close_fraction_short_side": 0.012,
        "bright_percentile": 99.0,
        "bright_threshold_min": 180.0,
        "bright_threshold_max": 250.0,
        "bright_open_fraction_short_side": 0.0025,
        "bright_close_width_fraction": 0.050,
        "bright_large_close_width_fraction": 0.090,
        "bright_close_height_fraction": 0.030,
        "min_area_fraction": 0.002,
        "max_area_fraction": 0.085,
        "min_long_fraction": 0.14,
        "max_long_fraction": 0.40,
        "min_visible_short_fraction": 0.02,
        "max_short_fraction": 0.30,
        "min_aspect_ratio": 1.15,
        "max_aspect_ratio": 4.80,
        "max_border_aspect_ratio": 10.5,
        "min_rectangularity": 0.40,
        "min_border_rectangularity": 0.55,
        "min_solidity": 0.60,
        "min_support_fraction": 0.075,
        "min_mean_gray": 175.0,
        "min_contrast": 10.0,
        "expected_area_fraction": 0.025,
        "expected_long_fraction": 0.289,
        "expected_aspect_ratio": 2.0,
        "border_margin_fraction": 0.003,
        "contrast_ring_fraction": 0.018,
    }
    valid_sticker, best, _ = detect_sticker(
        image,
        raw_sticker,
        raw_candy,
        "images",
        sticker_config,
    )
    if best is None or not np.any(valid_sticker):
        raise AssertionError("Синтетический стикер не найден.")

    cleanup_config = {
        "raw_boundary_fraction": 0.004,
        "geometric_boundary_fraction": 0.006,
        "thin_sticker_fringe_fraction": 0.0045,
        "close_fraction_short_side": 0.0025,
        "hole_area_fraction": 0.00005,
        "min_component_area_fraction": 0.000015,
        "min_component_radius_fraction": 0.0025,
        "border_artifact_aspect_ratio": 9.0,
        "border_artifact_max_radius_fraction": 0.0045,
        "border_margin_fraction": 0.003,
        "sensor_border_search_fraction": 0.13,
        "sensor_border_max_thickness_fraction": 0.10,
        "sensor_border_min_span_fraction": 0.55,
        "sensor_border_min_contrast": 45.0,
        "sensor_border_threshold_min": 20.0,
        "sensor_border_threshold_max": 120.0,
        "sensor_border_dilation_fraction": 0.0035,
        "border_strip_fraction": 0.014,
        "border_strip_max_radius_fraction": 0.010,
        "border_strip_min_span_fraction": 0.18,
    }
    cleaned, stats = clean_candy_mask(
        image,
        raw_candy,
        raw_sticker,
        valid_sticker,
        cleanup_config,
    )

    if np.count_nonzero(cleaned[:, -25:]) > 0:
        raise AssertionError("Тёмная пограничная полоса не была удалена.")
    if stats["removed_sensor_border_pixels"] <= 0:
        raise AssertionError("Счётчик удаления сенсорной полосы остался нулевым.")

    marker_config = {
        "radii_fractions_short_side": [0.025, 0.035, 0.045, 0.055],
        "alpha": 2.0,
        "gradient_percentile": 70.0,
        "gradient_blur_fraction": 0.0010,
        "gaussian_scale": 0.25,
        "support_dilation_fraction": 0.006,
        "distance_weight_scale_fraction": 0.025,
        "distance_weight_power": 0.70,
        "minimum_mask_radius_fraction": 0.001,
        "score_percentile": 65.0,
        "minimum_score": 0.020,
        "strong_score_percentile": 20.0,
        "min_distance_fraction": 0.060,
        "max_peaks": 50,
        "fallback_for_empty_components": True,
        "fallback_min_radius_fraction": 0.0045,
        "fallback_max_component_area_fraction": 0.020,
    }
    markers, distance, radial_score, marker_records = create_radial_markers(
        image,
        cleaned,
        "basler_120126",
        marker_config,
    )

    candidate_config = {
        "min_area_fraction": 0.000025,
        "min_radius_fraction": 0.0030,
        "strong_area_fraction": 0.00010,
        "strong_radius_fraction": 0.0060,
        "border_min_area_fraction": 0.000012,
    }
    labels, candidates = split_instances(
        image,
        cleaned,
        markers,
        distance,
        radial_score,
        marker_records,
        candidate_config,
    )

    if not 4 <= len(candidates) <= 7:
        raise AssertionError(
            f"Неожиданное количество кандидатов в smoke-тесте: {len(candidates)}"
        )
    if labels.shape != mask.shape:
        raise AssertionError("Размер instance_labels не совпадает с исходной маской.")

    print(
        "OK: "
        f"sticker=yes, candidates={len(candidates)}, "
        f"markers={len(marker_records)}, "
        f"removed_sensor_border_pixels={stats['removed_sensor_border_pixels']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
