from __future__ import annotations

from pathlib import Path

import numpy as np

from annotation_io import count_pixels, mask_path_for, metadata_path_for


def main() -> int:
    root = Path("annotations")
    relative = Path("basler_120126/example.bmp")
    assert mask_path_for(root, relative) == root / "pixel_masks/basler_120126/example.png"
    assert metadata_path_for(root, relative) == root / "metadata/basler_120126/example.json"
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    assert count_pixels(mask) == {"background": 1, "candy_core": 1, "sticker": 1}
    print("Stage 2 smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
