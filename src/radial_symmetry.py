from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.feature import peak_local_max


@dataclass(frozen=True)
class RadialMarker:
    marker_id: int
    center_x: int
    center_y: int
    score: float
    scale_radius: float
    mask_radius: float
    strength: str
    source: str


def _group_value(config: dict[str, Any], key: str, group: str, default: float) -> float:
    mapping = config.get(key, {})
    if isinstance(mapping, dict):
        if group in mapping:
            return float(mapping[group])
        if "default" in mapping:
            return float(mapping["default"])
    return float(default)


def fast_radial_symmetry(
    gray: np.ndarray,
    support_mask: np.ndarray,
    radii: list[int],
    *,
    alpha: float,
    gradient_percentile: float,
    gradient_blur_sigma: float,
    gaussian_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Вычисляет многомасштабный Fast Radial Symmetry Transform.

    Реализация следует идее Loy–Zelinsky: сильные градиентные пиксели голосуют
    вдоль и против направления градиента на нескольких радиусах. Абсолютная
    симметрия используется одновременно для светлых и тёмных тел конфет.
    """
    image = gray.astype(np.float32)
    if gradient_blur_sigma > 0:
        image = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=gradient_blur_sigma,
            sigmaY=gradient_blur_sigma,
        )

    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    roi = support_mask > 0
    roi_values = magnitude[roi]
    if roi_values.size == 0 or not radii:
        return np.zeros_like(image), np.zeros_like(image)

    threshold = float(np.percentile(magnitude, gradient_percentile))
    valid = magnitude >= max(1e-6, threshold)
    y_coordinates, x_coordinates = np.nonzero(valid)
    if y_coordinates.size == 0:
        return np.zeros_like(image), np.zeros_like(image)

    unit_x = grad_x[valid] / np.maximum(magnitude[valid], 1e-6)
    unit_y = grad_y[valid] / np.maximum(magnitude[valid], 1e-6)
    gradient_values = magnitude[valid]

    height, width = gray.shape
    responses: list[np.ndarray] = []

    for radius in radii:
        orientation = np.zeros((height, width), dtype=np.float32)
        projection = np.zeros((height, width), dtype=np.float32)

        delta_x = np.rint(unit_x * radius).astype(np.int32)
        delta_y = np.rint(unit_y * radius).astype(np.int32)

        for sign in (1, -1):
            target_x = x_coordinates + sign * delta_x
            target_y = y_coordinates + sign * delta_y
            inside = (
                (target_x >= 0)
                & (target_x < width)
                & (target_y >= 0)
                & (target_y < height)
            )
            np.add.at(
                orientation,
                (target_y[inside], target_x[inside]),
                float(sign),
            )
            np.add.at(
                projection,
                (target_y[inside], target_x[inside]),
                sign * gradient_values[inside],
            )

        normalization = 8.0 if radius <= 1 else 9.9
        normalized_orientation = np.clip(
            orientation,
            -normalization,
            normalization,
        ) / normalization

        projection_scale = float(np.max(np.abs(projection)))
        normalized_projection = np.abs(projection) / max(1e-6, projection_scale)

        response = normalized_projection * np.power(
            np.abs(normalized_orientation),
            alpha,
        )
        sigma = max(1.0, radius * gaussian_scale)
        response = cv2.GaussianBlur(
            response,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        responses.append(response)

    stack = np.stack(responses, axis=0)
    combined = np.mean(stack, axis=0)
    best_radius_indices = np.argmax(stack, axis=0)
    radius_values = np.asarray(radii, dtype=np.float32)
    scale_map = radius_values[best_radius_indices]

    combined[~roi] = 0.0
    low = float(np.min(combined))
    high = float(np.max(combined))
    combined = np.clip((combined - low) / max(1e-6, high - low), 0.0, 1.0)
    combined[~roi] = 0.0
    return combined.astype(np.float32), scale_map.astype(np.float32)


def create_radial_markers(
    image: np.ndarray,
    cleaned_mask: np.ndarray,
    source_group: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RadialMarker]]:
    """Находит центры тел конфет по радиальной симметрии исходного кадра."""
    height, width = cleaned_mask.shape
    short_side = min(height, width)
    binary = (cleaned_mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)

    support_dilation = max(
        1,
        int(round(short_side * float(config.get("support_dilation_fraction", 0.006)))),
    )
    support_kernel_size = 2 * support_dilation + 1
    support = cv2.dilate(
        binary,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (support_kernel_size, support_kernel_size),
        ),
    )

    radius_fractions = config.get(
        "radii_fractions_short_side",
        [0.015, 0.020, 0.025, 0.030, 0.035],
    )
    radii = sorted(
        {
            max(2, int(round(short_side * float(fraction))))
            for fraction in radius_fractions
        }
    )

    symmetry, scale_map = fast_radial_symmetry(
        gray,
        support,
        radii,
        alpha=float(config.get("alpha", 2.0)),
        gradient_percentile=float(config.get("gradient_percentile", 75.0)),
        gradient_blur_sigma=max(
            0.0,
            short_side * float(config.get("gradient_blur_fraction", 0.0010)),
        ),
        gaussian_scale=float(config.get("gaussian_scale", 0.25)),
    )

    distance_scale = max(
        1.0,
        short_side * float(config.get("distance_weight_scale_fraction", 0.018)),
    )
    distance_weight = np.clip(distance / distance_scale, 0.0, 1.0)
    distance_weight = np.power(
        distance_weight,
        float(config.get("distance_weight_power", 0.70)),
    )
    score = symmetry * distance_weight
    score[binary == 0] = 0.0

    score_values = score[binary > 0]
    if score_values.size == 0:
        return np.zeros_like(cleaned_mask, dtype=np.int32), distance, score, []

    threshold_percentile = _group_value(
        config,
        "score_percentile_by_group",
        source_group,
        float(config.get("score_percentile", 82.0)),
    )
    threshold = max(
        float(config.get("minimum_score", 0.075)),
        float(np.percentile(score_values, threshold_percentile)),
    )

    min_distance_fraction = _group_value(
        config,
        "min_distance_fraction_by_group",
        source_group,
        float(config.get("min_distance_fraction", 0.032)),
    )
    min_distance = max(6, int(round(short_side * min_distance_fraction)))
    coordinates = peak_local_max(
        score,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
        num_peaks=int(config.get("max_peaks", 3000)),
        labels=binary,
    )

    minimum_mask_radius = max(
        2.0,
        short_side * float(config.get("minimum_mask_radius_fraction", 0.0040)),
    )
    accepted: list[tuple[float, int, int, float, float, str]] = []
    for y_value, x_value in coordinates:
        y = int(y_value)
        x = int(x_value)
        mask_radius = float(distance[y, x])
        if mask_radius < minimum_mask_radius:
            continue
        accepted.append(
            (
                float(score[y, x]),
                y,
                x,
                float(scale_map[y, x]),
                mask_radius,
                "frst",
            )
        )

    # Резерв: отдельная связная область с заметной толщиной не должна остаться
    # совсем без центра из-за слабого контраста или глубокой тени.
    if bool(config.get("fallback_for_empty_components", True)):
        component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        fallback_radius = max(
            2.0,
            short_side * float(config.get("fallback_min_radius_fraction", 0.0045)),
        )
        fallback_max_area_fraction = float(
            config.get("fallback_max_component_area_fraction", 0.020)
        )
        image_area = height * width

        for component_label in range(1, component_count):
            x, y, component_width, component_height, area = (
                int(value) for value in stats[component_label]
            )
            if area <= 0 or area > image_area * fallback_max_area_fraction:
                continue
            if any(
                component_labels[center_y, center_x] == component_label
                for _, center_y, center_x, _, _, _ in accepted
            ):
                continue

            component = (
                component_labels[
                    y : y + component_height,
                    x : x + component_width,
                ]
                == component_label
            )
            local_distance = np.where(
                component,
                distance[y : y + component_height, x : x + component_width],
                -1.0,
            )
            local_y, local_x = np.unravel_index(
                int(np.argmax(local_distance)),
                local_distance.shape,
            )
            mask_radius = float(local_distance[local_y, local_x])
            if mask_radius < fallback_radius:
                continue

            center_y = y + int(local_y)
            center_x = x + int(local_x)
            accepted.append(
                (
                    float(score[center_y, center_x]),
                    center_y,
                    center_x,
                    float(scale_map[center_y, center_x]),
                    mask_radius,
                    "fallback",
                )
            )

    accepted.sort(reverse=True)
    if accepted:
        accepted_scores = np.asarray([item[0] for item in accepted], dtype=np.float32)
        strong_threshold = max(
            threshold,
            float(
                np.percentile(
                    accepted_scores,
                    float(config.get("strong_score_percentile", 20.0)),
                )
            ),
        )
    else:
        strong_threshold = threshold

    markers = np.zeros(cleaned_mask.shape, dtype=np.int32)
    records: list[RadialMarker] = []
    for marker_id, (marker_score, y, x, scale_radius, mask_radius, source) in enumerate(
        accepted,
        1,
    ):
        strength = (
            "strong"
            if source == "frst" and marker_score >= strong_threshold
            else "weak"
        )
        markers[y, x] = marker_id
        records.append(
            RadialMarker(
                marker_id=marker_id,
                center_x=x,
                center_y=y,
                score=marker_score,
                scale_radius=scale_radius,
                mask_radius=mask_radius,
                strength=strength,
                source=source,
            )
        )

    return markers, distance, score, records
