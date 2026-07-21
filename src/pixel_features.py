from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from common import to_bgr_uint8


@dataclass(frozen=True)
class FeatureConfig:
    gaussian_sigmas: tuple[float, ...]
    gradient_sigmas: tuple[float, ...]
    local_std_sigmas: tuple[float, ...]
    robust_low_percentile: float
    robust_high_percentile: float
    include_color_features: bool
    include_local_normalized: bool

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "FeatureConfig":
        return cls(
            gaussian_sigmas=tuple(float(x) for x in value.get("gaussian_sigmas", [1, 2, 4, 8])),
            gradient_sigmas=tuple(float(x) for x in value.get("gradient_sigmas", [1, 2, 4])),
            local_std_sigmas=tuple(float(x) for x in value.get("local_std_sigmas", [2, 4, 8])),
            robust_low_percentile=float(value.get("robust_low_percentile", 1.0)),
            robust_high_percentile=float(value.get("robust_high_percentile", 99.0)),
            include_color_features=bool(value.get("include_color_features", True)),
            include_local_normalized=bool(value.get("include_local_normalized", True)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "gaussian_sigmas": list(self.gaussian_sigmas),
            "gradient_sigmas": list(self.gradient_sigmas),
            "local_std_sigmas": list(self.local_std_sigmas),
            "robust_low_percentile": self.robust_low_percentile,
            "robust_high_percentile": self.robust_high_percentile,
            "include_color_features": self.include_color_features,
            "include_local_normalized": self.include_local_normalized,
        }


def _robust_stretch(channel: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    channel = channel.astype(np.float32, copy=False)
    low, high = np.percentile(channel, [low_percentile, high_percentile])
    if high <= low + 1e-6:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _gaussian(channel: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return channel.astype(np.float32, copy=False)
    return cv2.GaussianBlur(
        channel.astype(np.float32, copy=False),
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT101,
    )


def _gradient_magnitude(channel: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    return cv2.magnitude(gx, gy)


def _local_mean_std(channel: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    mean = _gaussian(channel, sigma)
    mean_sq = _gaussian(channel * channel, sigma)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, np.sqrt(variance).astype(np.float32, copy=False)


def extract_feature_cube(
    image: np.ndarray,
    config: FeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    """Return H x W x F float32 features with stable names."""
    bgr_u8 = to_bgr_uint8(image)
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    features: list[np.ndarray] = [
        gray,
        _robust_stretch(gray, config.robust_low_percentile, config.robust_high_percentile),
    ]
    names: list[str] = ["gray", "gray_robust"]

    if config.include_color_features:
        bgr = bgr_u8.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        for index, name in enumerate(("b", "g", "r")):
            features.append(bgr[:, :, index])
            names.append(name)
        features.extend(
            [
                hsv[:, :, 0] / 179.0,
                hsv[:, :, 1] / 255.0,
                hsv[:, :, 2] / 255.0,
                lab[:, :, 0] / 255.0,
                (lab[:, :, 1] - 128.0) / 127.0,
                (lab[:, :, 2] - 128.0) / 127.0,
            ]
        )
        names.extend(
            [
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "lab_l",
                "lab_a_centered",
                "lab_b_centered",
            ]
        )

    blur_cache: dict[float, np.ndarray] = {}
    for sigma in config.gaussian_sigmas:
        blurred = _gaussian(gray, sigma)
        blur_cache[sigma] = blurred
        features.append(blurred)
        names.append(f"gray_gaussian_{sigma:g}")
        features.append(gray - blurred)
        names.append(f"gray_detail_{sigma:g}")

    for sigma in config.gradient_sigmas:
        blurred = blur_cache.get(sigma)
        if blurred is None:
            blurred = _gaussian(gray, sigma)
        features.append(_gradient_magnitude(blurred))
        names.append(f"gradient_{sigma:g}")

    for sigma in config.local_std_sigmas:
        mean, local_std = _local_mean_std(gray, sigma)
        features.append(local_std)
        names.append(f"local_std_{sigma:g}")
        if config.include_local_normalized:
            normalized = np.clip((gray - mean) / (local_std + 0.04), -4.0, 4.0) / 4.0
            features.append(normalized.astype(np.float32, copy=False))
            names.append(f"local_normalized_{sigma:g}")

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT101)
    features.append(np.abs(laplacian))
    names.append("laplacian_abs")

    cube = np.stack(features, axis=-1).astype(np.float32, copy=False)
    cube = np.nan_to_num(cube, nan=0.0, posinf=1.0, neginf=-1.0)
    return cube, names


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray | None,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image, mask, 1.0
    scale = max_side / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    resized_mask = None
    if mask is not None:
        resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask, scale


def shrink_class_region(mask: np.ndarray, class_id: int, margin_px: int) -> np.ndarray:
    region = (mask == class_id).astype(np.uint8)
    if margin_px <= 0 or not np.any(region):
        return region.astype(bool)
    size = 2 * int(margin_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    eroded = cv2.erode(region, kernel, iterations=1)
    if np.count_nonzero(eroded) < 20:
        return region.astype(bool)
    return eroded.astype(bool)


def colorize_class_mask(mask: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    result[mask == 1] = (255, 80, 40)
    result[mask == 2] = (70, 220, 70)
    result[mask == 3] = (40, 60, 255)
    return result
