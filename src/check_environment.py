from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    errors: list[str] = []

    print(f"Python: {sys.version.split()[0]}")
    print(f"Интерпретатор: {Path(sys.executable).resolve()}")
    virtual_environment = os.environ.get("VIRTUAL_ENV")
    if virtual_environment:
        print(f"Виртуальное окружение: {Path(virtual_environment).resolve()}")
    else:
        print("Виртуальное окружение: не активно")

    checks = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("scikit-image", "skimage"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("Pillow", "PIL"),
        ("joblib", "joblib"),
        ("tqdm", "tqdm"),
        ("PyYAML", "yaml"),
    ]

    for display_name, module_name in checks:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "версия не указана")
            print(f"{display_name}: {version}")
        except Exception as exc:
            errors.append(f"{display_name}: {exc}")

    try:
        from scipy.ndimage import distance_transform_edt
        from skimage.feature import multiscale_basic_features, peak_local_max
        from skimage.segmentation import watershed
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupKFold

        _ = (
            distance_transform_edt,
            multiscale_basic_features,
            peak_local_max,
            watershed,
            RandomForestClassifier,
            GroupKFold,
        )
        print("Необходимые функции машинного зрения и машинного обучения: доступны")
    except Exception as exc:
        errors.append(f"Проверка необходимых функций: {exc}")

    if errors:
        print("\nОшибки окружения:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nОкружение готово к работе.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
