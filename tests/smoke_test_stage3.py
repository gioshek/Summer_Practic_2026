from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pixel_features import FeatureConfig, extract_feature_cube, resize_image_and_mask, shrink_class_region


def main() -> int:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:] = (90, 100, 110)
    cv2.ellipse(image, (80, 60), (28, 18), 20, 0, 360, (40, 180, 70), -1)
    cv2.rectangle(image, (115, 15), (150, 45), (245, 245, 245), -1)
    mask = np.ones((120, 160), dtype=np.uint8)
    cv2.ellipse(mask, (80, 60), (24, 14), 20, 0, 360, 2, -1)
    cv2.rectangle(mask, (120, 20), (145, 40), 3, -1)
    config = FeatureConfig.from_mapping({})
    resized, resized_mask, scale = resize_image_and_mask(image, mask, 100)
    cube, names = extract_feature_cube(resized, config)
    assert resized_mask is not None
    assert cube.shape[:2] == resized.shape[:2]
    assert cube.shape[2] == len(names)
    assert np.all(np.isfinite(cube))
    for class_id in (1, 2, 3):
        assert np.count_nonzero(shrink_class_region(resized_mask, class_id, 2)) > 0
    print(f"OK: shape={cube.shape}, features={len(names)}, scale={scale:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
