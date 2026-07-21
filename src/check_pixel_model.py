from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from common import load_yaml, project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка файлов пиксельной модели.")
    parser.add_argument("--project-config", default="configs/project.yaml")
    args = parser.parse_args()
    config = load_yaml(Path(args.project_config))
    model_root = project_path(config, "models", "models").resolve()
    model_path = model_root / "pixel_segmenter.joblib"
    info_path = model_root / "pixel_segmenter_info.json"
    dataset_path = model_root / "pixel_training_dataset.npz"

    print(f"Модель: {model_path}")
    print(f"Существует: {'да' if model_path.exists() else 'нет'}")
    print(f"Описание: {info_path}")
    print(f"Существует: {'да' if info_path.exists() else 'нет'}")
    print(f"Датасет пикселей: {dataset_path}")
    print(f"Существует: {'да' if dataset_path.exists() else 'нет'}")
    if not model_path.exists():
        return 1
    payload = joblib.load(model_path)
    classifier = payload["classifier"]
    print(f"Классы модели: {[int(value) for value in classifier.classes_]}")
    print(f"Количество признаков: {len(payload.get('feature_names', []))}")
    print(f"Количество деревьев: {len(classifier.estimators_)}")
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        report = info.get("classification_report", {})
        print(f"Holdout accuracy: {report.get('accuracy', 'нет данных')}")
        print(f"Macro F1 трёх классов: {report.get('macro avg', {}).get('f1-score', 'нет данных')}")
        print(f"Macro F1 background+candy: {info.get('background_candy_macro_f1', 'нет данных')}")
        candy = info.get("candy_vs_rest", {})
        sticker = info.get("sticker_vs_rest", {})
        print(f"Candy-vs-rest F1: {candy.get('f1', 'нет данных')}")
        print(f"Sticker-vs-rest F1: {sticker.get('f1', 'нет данных')}")
        print(
            "Финальная модель обучена на всех размеченных изображениях: "
            f"{'да' if info.get('final_model_trained_on_all_annotations') else 'нет'}"
        )
        print(f"Изображений финального обучения: {len(info.get('final_training_images', []))}")
        print(f"Evaluation OOB accuracy: {info.get('evaluation_oob_score', 'нет данных')}")
        print(f"Final model OOB accuracy: {info.get('oob_score', 'нет данных')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
