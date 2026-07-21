from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np

from annotation_io import CLASS_NAMES
from common import iter_image_paths, load_yaml, project_input_path, project_path, read_image, write_csv, write_image, write_text
from pixel_features import FeatureConfig, colorize_class_mask, extract_feature_cube, resize_image_and_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Применение пиксельного сегментатора.")
    parser.add_argument("--project-config", default="configs/project.yaml")
    parser.add_argument("--config", default="configs/stage3.yaml")
    parser.add_argument("--mode", choices=("sample", "raw"), default="sample")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--probabilities", action="store_true")
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument(
        "--include-list",
        type=Path,
        help="Обработать только точные относительные пути из текстового файла.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Пропустить отсутствующие изображения из точного списка.",
    )
    return parser.parse_args()



def load_requested_images(input_root: Path, list_path: Path, skip_missing: bool) -> list[Path]:
    if not list_path.exists():
        raise FileNotFoundError(f"Файл списка не найден: {list_path}")
    result: list[Path] = []
    seen: set[str] = set()
    missing: list[str] = []
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
        joined = "\n".join(f"- {value}" for value in missing)
        if not skip_missing:
            raise FileNotFoundError(f"Не найдены изображения из списка:\n{joined}")
        print(f"Предупреждение: пропущены отсутствующие изображения:\n{joined}")
    return result


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def overlay_prediction(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    from common import to_bgr_uint8

    base = to_bgr_uint8(image)
    colors = colorize_class_mask(mask)
    overlay_region = (mask == 2) | (mask == 3)
    result = base.copy()
    blended = cv2.addWeighted(base, 1.0 - alpha, colors, alpha, 0.0)
    result[overlay_region] = blended[overlay_region]
    candy_count = int(np.count_nonzero(mask == 2))
    sticker_count = int(np.count_nonzero(mask == 3))
    text = f"candy_core pixels={candy_count} | sticker pixels={sticker_count}"
    cv2.rectangle(result, (8, 8), (min(result.shape[1] - 8, 680), 42), (0, 0, 0), -1)
    cv2.putText(result, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> int:
    args = parse_args()
    project_config = load_yaml(Path(args.project_config))
    stage_config = load_yaml(Path(args.config))
    prediction = stage_config.get("prediction", {})
    if not isinstance(prediction, dict):
        raise ValueError("Раздел prediction в stage3.yaml должен быть словарём.")

    input_root = project_input_path(project_config, args.mode).resolve()
    model_root = project_path(project_config, "models", "models").resolve()
    output_root = project_path(project_config, "output", "output").resolve()
    model_path = model_root / "pixel_segmenter.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}. Сначала запусти train raw.")

    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "classifier" not in payload:
        raise ValueError(f"Неверный формат модели: {model_path}")
    classifier = payload["classifier"]
    feature_config = FeatureConfig.from_mapping(dict(payload.get("feature_config", {})))
    expected_feature_names = list(payload.get("feature_names", []))
    class_ids = np.asarray(payload.get("class_ids", [1, 2, 3]), dtype=np.uint8)

    max_side = int(prediction.get("max_image_side", 1200))
    chunk_size = int(prediction.get("chunk_size", 200000))
    overlay_alpha = float(prediction.get("overlay_alpha", 0.45))
    jpeg_quality = int(prediction.get("jpeg_quality", 92))

    mask_root = output_root / "masks"
    annotated_root = output_root / "annotated"
    probability_root = output_root / "probability_maps"
    report_root = output_root / "reports"
    if not args.no_clear:
        for path in (mask_root, annotated_root, probability_root):
            if path.exists():
                shutil.rmtree(path)
    mask_root.mkdir(parents=True, exist_ok=True)
    annotated_root.mkdir(parents=True, exist_ok=True)
    if args.probabilities:
        probability_root.mkdir(parents=True, exist_ok=True)

    if args.include_list is not None:
        image_paths = load_requested_images(input_root, args.include_list, args.skip_missing)
    else:
        image_paths = list(iter_image_paths(input_root, recursive=True))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError(f"В источнике нет изображений: {input_root}")

    print(f"Режим: {args.mode}")
    print(f"Источник: {input_root}")
    print(f"Изображений: {len(image_paths)}")
    print(f"Максимальная сторона предсказания: {max_side}")
    print(f"Карты вероятностей: {'да' if args.probabilities else 'нет'}")
    if args.include_list is not None:
        print(f"Точный список: {args.include_list}")

    rows: list[dict[str, object]] = []
    total_started = time.perf_counter()
    errors = 0
    for index, source_path in enumerate(image_paths):
        relative = source_path.relative_to(input_root)
        started = time.perf_counter()
        try:
            image = read_image(source_path)
            image_small, _, scale = resize_image_and_mask(image, None, max_side)
            cube, feature_names = extract_feature_cube(image_small, feature_config)
            if expected_feature_names and expected_feature_names != feature_names:
                raise RuntimeError("Набор признаков модели не совпадает с набором признаков программы.")
            flat = cube.reshape(-1, cube.shape[-1])
            prediction_flat = np.empty(len(flat), dtype=np.uint8)
            probability_flat = None
            if args.probabilities:
                probability_flat = np.empty((len(flat), len(classifier.classes_)), dtype=np.float32)
            for start in range(0, len(flat), chunk_size):
                stop = min(start + chunk_size, len(flat))
                prediction_flat[start:stop] = classifier.predict(flat[start:stop]).astype(np.uint8)
                if probability_flat is not None:
                    probability_flat[start:stop] = classifier.predict_proba(flat[start:stop]).astype(np.float32)
            mask_small = prediction_flat.reshape(image_small.shape[:2])
            if image_small.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = mask_small

            write_image(mask_root / relative.with_suffix(".png"), mask)
            annotated = overlay_prediction(image, mask, overlay_alpha)
            write_image(annotated_root / relative.with_suffix(".jpg"), annotated, jpeg_quality=jpeg_quality)
            if probability_flat is not None:
                probabilities = probability_flat.reshape((*image_small.shape[:2], -1)).astype(np.float16)
                atomic_save_npz(
                    probability_root / relative.with_suffix(".npz"),
                    probabilities=probabilities,
                    class_ids=np.asarray(classifier.classes_, dtype=np.uint8),
                    scale=np.asarray([scale], dtype=np.float32),
                    original_shape=np.asarray(image.shape[:2], dtype=np.int32),
                )

            counts = {class_id: int(np.count_nonzero(mask == class_id)) for class_id in (1, 2, 3)}
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "source_relative_path": relative.as_posix(),
                    "source_group": relative.parts[0] if len(relative.parts) > 1 else "root",
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "prediction_width": image_small.shape[1],
                    "prediction_height": image_small.shape[0],
                    "scale": f"{scale:.6f}",
                    "pixels_background": counts[1],
                    "pixels_candy_core": counts[2],
                    "pixels_sticker": counts[3],
                    "candy_core_percent": f"{100.0 * counts[2] / mask.size:.4f}",
                    "sticker_percent": f"{100.0 * counts[3] / mask.size:.4f}",
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "error": "",
                }
            )
            print(f"[{index + 1:03d}/{len(image_paths):03d}] {relative.as_posix()} | {elapsed:.2f}s")
        except Exception as error:
            errors += 1
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "source_relative_path": relative.as_posix(),
                    "source_group": relative.parts[0] if len(relative.parts) > 1 else "root",
                    "width": "",
                    "height": "",
                    "prediction_width": "",
                    "prediction_height": "",
                    "scale": "",
                    "pixels_background": "",
                    "pixels_candy_core": "",
                    "pixels_sticker": "",
                    "candy_core_percent": "",
                    "sticker_percent": "",
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "error": str(error),
                }
            )
            print(f"[{index + 1:03d}/{len(image_paths):03d}] ОШИБКА {relative.as_posix()}: {error}")

    total_elapsed = time.perf_counter() - total_started
    write_csv(
        report_root / "pixel_predictions.csv",
        rows,
        [
            "source_relative_path",
            "source_group",
            "width",
            "height",
            "prediction_width",
            "prediction_height",
            "scale",
            "pixels_background",
            "pixels_candy_core",
            "pixels_sticker",
            "candy_core_percent",
            "sticker_percent",
            "elapsed_seconds",
            "error",
        ],
    )
    summary = [
        "Этап 3.1: применение улучшенного пиксельного сегментатора",
        f"Режим: {args.mode}",
        f"Источник: {input_root}",
        f"Обработано изображений: {len(image_paths)}",
        f"Ошибок: {errors}",
        f"Время: {total_elapsed:.1f} с",
        f"Маски: {mask_root}",
        f"Аннотированные изображения: {annotated_root}",
        f"Карты вероятностей: {'созданы' if args.probabilities else 'не создавались'}",
        f"Отчёт: {report_root / 'pixel_predictions.csv'}",
        f"Завершено UTC: {datetime.now(timezone.utc).isoformat()}",
    ]
    write_text(report_root / "pixel_prediction_summary.txt", "\n".join(summary) + "\n")
    print("\nПредсказание завершено.")
    for line in summary[1:]:
        print(line)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
