from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from annotation_io import CLASS_NAMES, load_label_mask, load_manifest, read_metadata
from common import load_yaml, project_input_path, project_path, read_image, write_csv, write_text
from pixel_features import FeatureConfig, extract_feature_cube, resize_image_and_mask, shrink_class_region

CLASS_IDS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение улучшенного пиксельного сегментатора конфет.")
    parser.add_argument("--project-config", default="configs/project.yaml")
    parser.add_argument("--config", default="configs/stage3.yaml")
    parser.add_argument("--mode", choices=("sample", "raw"), default="raw")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trees", type=int)
    parser.add_argument("--max-side", type=int)
    return parser.parse_args()


def atomic_joblib_dump(value: object, path: Path, compress: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(value, temporary_path, compress=compress)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


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


def image_class_counts(image_record: dict[str, object]) -> dict[int, int]:
    samples = image_record["samples"]
    assert isinstance(samples, dict)
    return {class_id: int(len(samples[class_id])) for class_id in CLASS_IDS}


def select_image_holdout(
    image_records: list[dict[str, object]],
    test_fraction: float,
    seed: int,
    attempts: int,
    min_sticker_train_images: int,
    min_sticker_test_images: int,
) -> tuple[list[int], list[int]]:
    image_count = len(image_records)
    if image_count < 6:
        raise ValueError("Для улучшенной проверки требуется хотя бы 6 завершённых изображений.")

    test_count = max(4, int(round(image_count * test_fraction)))
    test_count = min(test_count, image_count - 3)
    rng = np.random.default_rng(seed)
    all_indices = np.arange(image_count)
    all_groups = sorted({str(record["source_group"]) for record in image_records})
    best: tuple[float, list[int], list[int]] | None = None

    for _ in range(max(100, attempts)):
        test_indices = sorted(int(value) for value in rng.choice(all_indices, size=test_count, replace=False))
        test_set = set(test_indices)
        train_indices = [index for index in range(image_count) if index not in test_set]

        train_counts = {class_id: 0 for class_id in CLASS_IDS}
        test_counts = {class_id: 0 for class_id in CLASS_IDS}
        for index in train_indices:
            counts = image_class_counts(image_records[index])
            for class_id in CLASS_IDS:
                train_counts[class_id] += counts[class_id]
        for index in test_indices:
            counts = image_class_counts(image_records[index])
            for class_id in CLASS_IDS:
                test_counts[class_id] += counts[class_id]

        if any(train_counts[class_id] == 0 or test_counts[class_id] == 0 for class_id in CLASS_IDS):
            continue

        train_sticker_images = sum(image_class_counts(image_records[index])[3] > 0 for index in train_indices)
        test_sticker_images = sum(image_class_counts(image_records[index])[3] > 0 for index in test_indices)
        if train_sticker_images < min_sticker_train_images or test_sticker_images < min_sticker_test_images:
            continue

        train_groups = {str(image_records[index]["source_group"]) for index in train_indices}
        test_groups = {str(image_records[index]["source_group"]) for index in test_indices}
        if len(train_groups) < len(all_groups):
            continue

        total_train = sum(train_counts.values())
        total_test = sum(test_counts.values())
        train_distribution = np.array([train_counts[class_id] / total_train for class_id in CLASS_IDS])
        test_distribution = np.array([test_counts[class_id] / total_test for class_id in CLASS_IDS])
        distribution_penalty = float(np.abs(train_distribution - test_distribution).sum())
        minimum_test_class = min(test_counts.values())
        group_coverage = len(test_groups)
        sticker_reward = min(test_sticker_images, 3)
        score = (
            group_coverage * 1_000_000
            + sticker_reward * 100_000
            + min(minimum_test_class, 50_000)
            - distribution_penalty * 10_000
        )
        if best is None or score > best[0]:
            best = (score, train_indices, test_indices)

    if best is None:
        raise ValueError(
            "Не удалось сформировать holdout с пикселями всех классов и изображениями со стикером "
            "в обучающей и проверочной частях. Добавь и заверши дополнительные разметки со стикером."
        )
    return best[1], best[2]


def combine_class_samples(
    image_records: list[dict[str, object]],
    image_indices: list[int],
    class_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    for image_index in image_indices:
        record = image_records[image_index]
        samples = record["samples"]
        assert isinstance(samples, dict)
        values = samples[class_id]
        assert isinstance(values, np.ndarray)
        if values.size == 0:
            continue
        relative = str(record["relative_path"])
        feature_parts.append(values)
        group_parts.append(np.full(len(values), relative, dtype=f"<U{max(64, len(relative))}"))
    if not feature_parts:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype="<U1")
    return np.concatenate(feature_parts, axis=0), np.concatenate(group_parts, axis=0)


def balanced_training_arrays(
    image_records: list[dict[str, object]],
    train_indices: list[int],
    rng: np.random.Generator,
    target_per_class: int,
    balance_to_smallest: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    available: dict[int, int] = {}
    values_by_class: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for class_id in CLASS_IDS:
        features, groups = combine_class_samples(image_records, train_indices, class_id)
        if len(features) == 0:
            raise ValueError(f"В train отсутствует класс {CLASS_NAMES[class_id]}.")
        values_by_class[class_id] = (features, groups)
        available[class_id] = len(features)

    if balance_to_smallest:
        target = min(target_per_class, min(available.values()))
    else:
        target = target_per_class

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    selected_counts: dict[int, int] = {}
    for class_id in CLASS_IDS:
        features, groups = values_by_class[class_id]
        sample_count = min(target, len(features))
        chosen = rng.choice(len(features), size=sample_count, replace=False)
        X_parts.append(features[chosen])
        y_parts.append(np.full(sample_count, class_id, dtype=np.uint8))
        group_parts.append(groups[chosen])
        selected_counts[class_id] = sample_count

    X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)
    permutation = rng.permutation(len(y))
    return X[permutation], y[permutation], groups[permutation], available, selected_counts


def test_arrays(
    image_records: list[dict[str, object]],
    test_indices: list[int],
    rng: np.random.Generator,
    max_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    counts: dict[int, int] = {}
    for class_id in CLASS_IDS:
        features, groups = combine_class_samples(image_records, test_indices, class_id)
        if len(features) == 0:
            raise ValueError(f"В test отсутствует класс {CLASS_NAMES[class_id]}.")
        if len(features) > max_per_class:
            chosen = rng.choice(len(features), size=max_per_class, replace=False)
            features = features[chosen]
            groups = groups[chosen]
        counts[class_id] = len(features)
        X_parts.append(features)
        y_parts.append(np.full(len(features), class_id, dtype=np.uint8))
        group_parts.append(groups)
    X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)
    permutation = rng.permutation(len(y))
    return X[permutation], y[permutation], groups[permutation], counts


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true == positive_class,
        y_pred == positive_class,
        average="binary",
        zero_division=0,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def save_confusion_png(matrix: np.ndarray, path: Path) -> None:
    labels = [CLASS_NAMES[class_id] for class_id in CLASS_IDS]
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    axis.set_title("Пиксельная confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def metric_row(scope: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(CLASS_IDS),
        target_names=[CLASS_NAMES[class_id] for class_id in CLASS_IDS],
        output_dict=True,
        zero_division=0,
    )
    candy = binary_metrics(y_true, y_pred, 2)
    sticker = binary_metrics(y_true, y_pred, 3)
    return {
        "scope": scope,
        "pixels": len(y_true),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "background_f1": report[CLASS_NAMES[1]]["f1-score"],
        "candy_precision": candy["precision"],
        "candy_recall": candy["recall"],
        "candy_f1": candy["f1"],
        "sticker_precision": sticker["precision"],
        "sticker_recall": sticker["recall"],
        "sticker_f1": sticker["f1"],
    }


def main() -> int:
    args = parse_args()
    project_config = load_yaml(Path(args.project_config))
    stage_config = load_yaml(Path(args.config))

    input_root = project_input_path(project_config, args.mode).resolve()
    annotation_root = project_path(project_config, "annotations", "annotations").resolve()
    model_root = project_path(project_config, "models", "models").resolve()
    output_root = project_path(project_config, "output", "output").resolve()
    report_root = output_root / "reports"
    manifest_path = annotation_root / "manifest.csv"

    training = stage_config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("Раздел training в stage3.yaml должен быть словарём.")
    feature_mapping = stage_config.get("features", {})
    if not isinstance(feature_mapping, dict):
        raise ValueError("Раздел features в stage3.yaml должен быть словарём.")
    feature_config = FeatureConfig.from_mapping(feature_mapping)

    seed = int(args.seed if args.seed is not None else training.get("random_seed", 2026))
    max_side = int(args.max_side if args.max_side is not None else training.get("max_image_side", 1000))
    trees = int(args.trees if args.trees is not None else training.get("n_estimators", 240))
    max_depth_value = training.get("max_depth", 22)
    max_depth = None if max_depth_value in (None, "null", "None") else int(max_depth_value)
    min_samples_leaf = int(training.get("min_samples_leaf", 2))
    max_per_image = int(training.get("max_pixels_per_image_per_class", 3500))
    target_train_per_class = int(training.get("target_train_pixels_per_class", 30000))
    max_test_per_class = int(training.get("max_test_pixels_per_class", 15000))
    balance_to_smallest = bool(training.get("balance_to_smallest_class", True))
    boundary_margin = int(training.get("boundary_margin_px", 3))
    test_fraction = float(training.get("test_fraction", 0.25))
    split_attempts = int(training.get("split_attempts", 5000))
    min_sticker_train_images = int(training.get("min_sticker_train_images", 4))
    min_sticker_test_images = int(training.get("min_sticker_test_images", 1))
    n_jobs = int(training.get("n_jobs", -1))

    records = load_manifest(manifest_path)
    if not records:
        raise FileNotFoundError(f"Манифест разметки не найден или пуст: {manifest_path}")

    selected_records: list[dict[str, str]] = []
    skipped: list[str] = []
    for record in records:
        relative = record.get("source_relative_path", "")
        if not relative:
            continue
        source_path = input_root / relative
        metadata = read_metadata(annotation_root, relative)
        status = str(metadata.get("status") or record.get("label_status") or "")
        if status != "complete":
            skipped.append(f"{relative}: status={status or 'unknown'}")
            continue
        if not source_path.exists():
            skipped.append(f"{relative}: нет в режиме {args.mode}")
            continue
        selected_records.append(record)

    if len(selected_records) < 6:
        raise ValueError(
            f"Недостаточно завершённых разметок в режиме {args.mode}: {len(selected_records)}. Нужно хотя бы 6."
        )

    print(f"Режим источника: {args.mode}")
    print(f"Источник изображений: {input_root}")
    print(f"Завершённых разметок для обучения: {len(selected_records)}")
    print(f"Максимальная сторона при извлечении признаков: {max_side}")
    print(f"Пограничный отступ: {boundary_margin}px")
    print(f"Цветовые признаки: {'включены' if feature_config.include_color_features else 'отключены'}")

    rng = np.random.default_rng(seed)
    image_records: list[dict[str, object]] = []
    feature_names: list[str] | None = None
    image_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for image_index, record in enumerate(selected_records):
        relative = record["source_relative_path"]
        source_path = input_root / relative
        image = read_image(source_path)
        mask = load_label_mask(annotation_root, relative, image.shape[:2])
        image_small, mask_small, scale = resize_image_and_mask(image, mask, max_side)
        assert mask_small is not None
        feature_cube, current_names = extract_feature_cube(image_small, feature_config)
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise RuntimeError("Набор признаков изменился между изображениями.")
        flat_features = feature_cube.reshape(-1, feature_cube.shape[-1])
        samples: dict[int, np.ndarray] = {}
        class_counts: dict[int, int] = {}
        for class_id in CLASS_IDS:
            region = shrink_class_region(mask_small, class_id, boundary_margin)
            indices = np.flatnonzero(region.reshape(-1))
            if indices.size > max_per_image:
                indices = rng.choice(indices, size=max_per_image, replace=False)
            values = flat_features[indices].copy() if indices.size else np.empty((0, flat_features.shape[1]), dtype=np.float32)
            samples[class_id] = values
            class_counts[class_id] = len(values)

        source_group = Path(relative).parts[0] if Path(relative).parts else "root"
        image_records.append(
            {
                "relative_path": relative,
                "source_group": source_group,
                "samples": samples,
            }
        )
        image_rows.append(
            {
                "source_relative_path": relative,
                "source_group": source_group,
                "original_width": image.shape[1],
                "original_height": image.shape[0],
                "training_width": image_small.shape[1],
                "training_height": image_small.shape[0],
                "scale": f"{scale:.6f}",
                "background_samples": class_counts[1],
                "candy_core_samples": class_counts[2],
                "sticker_samples": class_counts[3],
                "split": "",
            }
        )
        print(
            f"[{image_index + 1:02d}/{len(selected_records):02d}] {relative} | "
            f"bg={class_counts[1]} candy={class_counts[2]} sticker={class_counts[3]}"
        )

    if feature_names is None:
        raise RuntimeError("Не удалось извлечь признаки.")

    train_image_indices, test_image_indices = select_image_holdout(
        image_records,
        test_fraction,
        seed,
        split_attempts,
        min_sticker_train_images,
        min_sticker_test_images,
    )
    train_path_set = {str(image_records[index]["relative_path"]) for index in train_image_indices}
    test_path_set = {str(image_records[index]["relative_path"]) for index in test_image_indices}
    for row in image_rows:
        relative = str(row["source_relative_path"])
        row["split"] = "train" if relative in train_path_set else "test"

    X_train, y_train, train_groups, train_available, train_selected = balanced_training_arrays(
        image_records,
        train_image_indices,
        rng,
        target_train_per_class,
        balance_to_smallest,
    )
    X_test, y_test, test_groups, test_counts = test_arrays(
        image_records,
        test_image_indices,
        rng,
        max_test_per_class,
    )

    evaluation_classifier = RandomForestClassifier(
        n_estimators=trees,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=seed,
        max_features="sqrt",
        oob_score=True,
    )
    print(
        f"Оценочное обучение Random Forest: train={len(y_train)}, test={len(y_test)}, "
        f"trees={trees}, features={X_train.shape[1]}"
    )
    print(
        "Выбрано evaluation-train по классам: "
        + ", ".join(f"{CLASS_NAMES[class_id]}={train_selected[class_id]}" for class_id in CLASS_IDS)
    )
    evaluation_classifier.fit(X_train, y_train)
    predicted = evaluation_classifier.predict(X_test).astype(np.uint8)
    matrix = confusion_matrix(y_test, predicted, labels=list(CLASS_IDS))
    report = classification_report(
        y_test,
        predicted,
        labels=list(CLASS_IDS),
        target_names=[CLASS_NAMES[class_id] for class_id in CLASS_IDS],
        output_dict=True,
        zero_division=0,
    )
    candy_metrics = binary_metrics(y_test, predicted, 2)
    sticker_metrics = binary_metrics(y_test, predicted, 3)
    bg_candy_macro_f1 = float(
        (report[CLASS_NAMES[1]]["f1-score"] + report[CLASS_NAMES[2]]["f1-score"]) / 2.0
    )

    group_metric_rows: list[dict[str, object]] = []
    for source_group in sorted({Path(str(value)).parts[0] for value in test_groups}):
        group_mask = np.array([Path(str(value)).parts[0] == source_group for value in test_groups], dtype=bool)
        group_metric_rows.append(metric_row(source_group, y_test[group_mask], predicted[group_mask]))

    image_metric_rows: list[dict[str, object]] = []
    for relative in sorted(set(str(value) for value in test_groups)):
        image_mask = test_groups == relative
        image_metric_rows.append(metric_row(relative, y_test[image_mask], predicted[image_mask]))

    all_image_indices = list(range(len(image_records)))
    X_final, y_final, final_groups, final_available, final_selected = balanced_training_arrays(
        image_records,
        all_image_indices,
        rng,
        target_train_per_class,
        balance_to_smallest,
    )
    final_classifier = RandomForestClassifier(
        n_estimators=trees,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=seed,
        max_features="sqrt",
        oob_score=True,
    )
    print(
        f"Финальное обучение на всех {len(all_image_indices)} размеченных изображениях: "
        f"pixels={len(y_final)}, trees={trees}, features={X_final.shape[1]}"
    )
    print(
        "Выбрано final-train по классам: "
        + ", ".join(f"{CLASS_NAMES[class_id]}={final_selected[class_id]}" for class_id in CLASS_IDS)
    )
    final_classifier.fit(X_final, y_final)

    model_payload = {
        "format_version": 2,
        "classifier": final_classifier,
        "class_ids": list(CLASS_IDS),
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "feature_config": feature_config.to_dict(),
        "feature_names": feature_names,
        "training_max_side": max_side,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_path = model_root / "pixel_segmenter.joblib"
    dataset_path = model_root / "pixel_training_dataset.npz"
    info_path = model_root / "pixel_segmenter_info.json"
    atomic_joblib_dump(model_payload, model_path)
    atomic_save_npz(
        dataset_path,
        X_train=X_train,
        y_train=y_train,
        train_groups=train_groups,
        X_test=X_test,
        y_test=y_test,
        test_groups=test_groups,
        X_final=X_final,
        y_final=y_final,
        final_groups=final_groups,
        feature_names=np.asarray(feature_names),
    )

    elapsed = time.perf_counter() - started
    metrics_payload = {
        "format_version": 2,
        "stage": "3.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": args.mode,
        "source_root": str(input_root),
        "annotation_count": len(selected_records),
        "skipped_annotations": skipped,
        "train_images": sorted(train_path_set),
        "test_images": sorted(test_path_set),
        "final_training_images": sorted(str(record["relative_path"]) for record in image_records),
        "final_model_trained_on_all_annotations": True,
        "train_available_class_counts": {CLASS_NAMES[key]: value for key, value in train_available.items()},
        "train_selected_class_counts": {CLASS_NAMES[key]: value for key, value in train_selected.items()},
        "test_class_counts": {CLASS_NAMES[key]: value for key, value in test_counts.items()},
        "final_available_class_counts": {CLASS_NAMES[key]: value for key, value in final_available.items()},
        "final_selected_class_counts": {CLASS_NAMES[key]: value for key, value in final_selected.items()},
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_pixel_count": int(len(y_train)),
        "test_pixel_count": int(len(y_test)),
        "final_train_pixel_count": int(len(y_final)),
        "evaluation_oob_score": float(evaluation_classifier.oob_score_),
        "oob_score": float(final_classifier.oob_score_),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "candy_vs_rest": candy_metrics,
        "sticker_vs_rest": sticker_metrics,
        "background_candy_macro_f1": bg_candy_macro_f1,
        "metrics_by_group": group_metric_rows,
        "metrics_by_test_image": image_metric_rows,
        "elapsed_seconds": elapsed,
        "random_seed": seed,
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "parameters": {
            "max_image_side": max_side,
            "boundary_margin_px": boundary_margin,
            "max_pixels_per_image_per_class": max_per_image,
            "target_train_pixels_per_class": target_train_per_class,
            "max_test_pixels_per_class": max_test_per_class,
            "balance_to_smallest_class": balance_to_smallest,
            "test_fraction": test_fraction,
            "split_attempts": split_attempts,
            "min_sticker_train_images": min_sticker_train_images,
            "min_sticker_test_images": min_sticker_test_images,
            "n_estimators": trees,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_jobs": n_jobs,
            "feature_config": feature_config.to_dict(),
        },
    }
    write_text(info_path, json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n")
    write_text(report_root / "pixel_metrics.json", json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n")
    write_csv(
        report_root / "pixel_training_images.csv",
        image_rows,
        [
            "source_relative_path",
            "source_group",
            "split",
            "original_width",
            "original_height",
            "training_width",
            "training_height",
            "scale",
            "background_samples",
            "candy_core_samples",
            "sticker_samples",
        ],
    )
    matrix_rows = []
    for row_index, true_id in enumerate(CLASS_IDS):
        matrix_rows.append(
            {
                "true_class": CLASS_NAMES[true_id],
                **{
                    f"pred_{CLASS_NAMES[pred_id]}": int(matrix[row_index, column])
                    for column, pred_id in enumerate(CLASS_IDS)
                },
            }
        )
    write_csv(
        report_root / "pixel_confusion_matrix.csv",
        matrix_rows,
        ["true_class"] + [f"pred_{CLASS_NAMES[class_id]}" for class_id in CLASS_IDS],
    )
    save_confusion_png(matrix, report_root / "pixel_confusion_matrix.png")

    metric_fields = [
        "scope",
        "pixels",
        "accuracy",
        "macro_f1",
        "background_f1",
        "candy_precision",
        "candy_recall",
        "candy_f1",
        "sticker_precision",
        "sticker_recall",
        "sticker_f1",
    ]
    write_csv(report_root / "pixel_metrics_by_group.csv", group_metric_rows, metric_fields)
    write_csv(report_root / "pixel_metrics_by_test_image.csv", image_metric_rows, metric_fields)

    train_sticker_images = sum(image_class_counts(image_records[index])[3] > 0 for index in train_image_indices)
    test_sticker_images = sum(image_class_counts(image_records[index])[3] > 0 for index in test_image_indices)
    summary_lines = [
        "Этап 3.1: улучшенное обучение пиксельного сегментатора",
        f"Режим источника: {args.mode}",
        f"Завершённых размеченных изображений: {len(selected_records)}",
        f"Изображений в оценочном train: {len(train_image_indices)}",
        f"Изображений в holdout-проверке: {len(test_image_indices)}",
        f"Финальная модель обучена на изображениях: {len(all_image_indices)}/{len(selected_records)}",
        f"Изображений со стикером в train/test: {train_sticker_images}/{test_sticker_images}",
        f"Признаков на пиксель: {len(feature_names)}",
        f"Пикселей evaluation-train: {len(y_train)}",
        f"Пикселей holdout-test: {len(y_test)}",
        f"Пикселей final-train: {len(y_final)}",
        f"Evaluation OOB accuracy: {evaluation_classifier.oob_score_:.4f}",
        f"Final model OOB accuracy: {final_classifier.oob_score_:.4f}",
        f"Holdout accuracy: {report['accuracy']:.4f}",
        f"Macro F1 трёх классов: {report['macro avg']['f1-score']:.4f}",
        f"Macro F1 background+candy: {bg_candy_macro_f1:.4f}",
        f"Candy-vs-rest: precision={candy_metrics['precision']:.4f}, "
        f"recall={candy_metrics['recall']:.4f}, f1={candy_metrics['f1']:.4f}",
        f"Sticker-vs-rest: precision={sticker_metrics['precision']:.4f}, "
        f"recall={sticker_metrics['recall']:.4f}, f1={sticker_metrics['f1']:.4f}",
    ]
    for class_id in CLASS_IDS:
        class_report = report[CLASS_NAMES[class_id]]
        summary_lines.append(
            f"{CLASS_NAMES[class_id]}: precision={class_report['precision']:.4f}, "
            f"recall={class_report['recall']:.4f}, f1={class_report['f1-score']:.4f}"
        )
    summary_lines.extend(
        [
            f"Время: {elapsed:.1f} с",
            f"Модель: {model_path}",
            f"Датасет пикселей: {dataset_path}",
            f"Метрики: {report_root / 'pixel_metrics.json'}",
        ]
    )
    write_text(report_root / "pixel_training_summary.txt", "\n".join(summary_lines) + "\n")

    print("\nОбучение завершено.")
    for line in summary_lines[2:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
