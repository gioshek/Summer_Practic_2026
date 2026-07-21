from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from annotation_io import (
    MANIFEST_FIELDS,
    load_manifest,
    mask_path_for,
    metadata_path_for,
    read_metadata,
    save_manifest,
)
from common import (
    iter_image_paths,
    load_yaml,
    project_input_path,
    project_path,
    read_csv,
    read_image,
    resize_long_side,
    to_gray,
)

FEATURE_FIELDS = [
    "mean",
    "std",
    "entropy",
    "edge_density",
    "p05",
    "p50",
    "p95",
] + [f"hist_{index:02d}" for index in range(16)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Выбор разнообразных изображений для ручной разметки без копирования исходников."
    )
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--mode", choices=["sample", "raw"], default="sample")
    parser.add_argument("--input", type=Path, help="Явный путь к исходным изображениям")
    parser.add_argument("--count", type=int, help="Сколько изображений выбрать из текущего источника")
    parser.add_argument(
        "--replace-unannotated-selection",
        action="store_true",
        help="Удалить из манифеста старые неразмеченные записи и сформировать выбор заново",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Точный относительный путь изображения, которое нужно добавить в манифест. Можно повторять.",
    )
    parser.add_argument(
        "--include-list",
        type=Path,
        help="Текстовый файл с точными относительными путями изображений, по одному на строку.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Пропустить отсутствующие пути из точного списка с предупреждением.",
    )
    return parser.parse_args()



def requested_relative_paths(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in args.include if str(value).strip()]
    if args.include_list is not None:
        if not args.include_list.exists():
            raise FileNotFoundError(f"Файл списка не найден: {args.include_list}")
        for raw_line in args.include_list.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(line)

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = Path(value).as_posix().lstrip("./")
        path = Path(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Недопустимый относительный путь: {value}")
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def entropy_uint8(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return 0.0
    probabilities = histogram[histogram > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def quick_features(path: Path, max_side: int) -> dict[str, float | int]:
    image = read_image(path)
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    small, _ = resize_long_side(image, max_side)
    gray = to_gray(small)
    p05, p50, p95 = np.percentile(gray, [5, 50, 95])
    edges = cv2.Canny(gray, 50, 150)
    histogram = cv2.calcHist([gray], [0], None, [16], [0, 256]).ravel()
    histogram = histogram / max(float(histogram.sum()), 1.0)
    result: dict[str, float | int] = {
        "width": int(width),
        "height": int(height),
        "channels": channels,
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "entropy": entropy_uint8(gray),
        "edge_density": float(np.count_nonzero(edges) / max(edges.size, 1)),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }
    for index, value in enumerate(histogram):
        result[f"hist_{index:02d}"] = float(value)
    return result


def inventory_index(path: Path) -> dict[str, dict[str, str]]:
    return {
        row.get("relative_path", ""): row
        for row in read_csv(path)
        if row.get("relative_path")
    }


def usable_inventory_features(row: dict[str, str]) -> bool:
    try:
        for field in FEATURE_FIELDS:
            float(row[field])
        int(float(row["width"]))
        int(float(row["height"]))
        int(float(row["channels"]))
        return True
    except (KeyError, TypeError, ValueError):
        return False


def feature_matrix(records: list[dict[str, object]]) -> np.ndarray:
    matrix = np.array(
        [[float(record[field]) for field in FEATURE_FIELDS] for record in records],
        dtype=np.float64,
    )
    median = np.median(matrix, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    scale = q75 - q25
    scale[scale < 1e-9] = 1.0
    matrix = (matrix - median) / scale
    return np.clip(matrix, -8.0, 8.0)


def farthest_point_indices(matrix: np.ndarray, count: int) -> list[int]:
    if count >= len(matrix):
        return list(range(len(matrix)))
    if count <= 0:
        return []
    center = matrix.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(matrix - center, axis=1)))
    selected = [first]
    minimum_distances = np.linalg.norm(matrix - matrix[first], axis=1)
    while len(selected) < count:
        candidate = int(np.argmax(minimum_distances))
        if candidate in selected:
            break
        selected.append(candidate)
        distances = np.linalg.norm(matrix - matrix[candidate], axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
    return selected


def allocate_per_group(grouped: dict[str, list[int]], count: int) -> dict[str, int]:
    groups = sorted(grouped)
    allocation = {group: 0 for group in groups}
    remaining = count
    while remaining > 0:
        progressed = False
        for group in groups:
            if remaining <= 0:
                break
            if allocation[group] < len(grouped[group]):
                allocation[group] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocation


def select_balanced(records: list[dict[str, object]], count: int) -> list[int]:
    if count >= len(records):
        return list(range(len(records)))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record["source_group"])].append(index)
    allocation = allocate_per_group(grouped, count)
    chosen: list[int] = []
    for group in sorted(grouped):
        global_indices = grouped[group]
        local_records = [records[index] for index in global_indices]
        local_matrix = feature_matrix(local_records)
        local_indices = farthest_point_indices(local_matrix, allocation[group])
        chosen.extend(global_indices[index] for index in local_indices)
    return sorted(chosen)


def existing_status(annotation_root: Path, relative_path: str) -> tuple[str, dict[str, int]]:
    metadata = read_metadata(annotation_root, relative_path)
    status = str(metadata.get("status", "not_started"))
    pixel_counts = metadata.get("pixel_counts", {})
    if not isinstance(pixel_counts, dict):
        pixel_counts = {}
    counts = {
        "background": int(pixel_counts.get("background", 0) or 0),
        "candy_core": int(pixel_counts.get("candy_core", 0) or 0),
        "sticker": int(pixel_counts.get("sticker", 0) or 0),
    }
    return status, counts


def main() -> int:
    args = parse_args()
    config = load_yaml(args.project_config)
    selection_config = config.get("annotation_selection", {})
    if not isinstance(selection_config, dict):
        selection_config = {}

    input_root = (args.input or project_input_path(config, args.mode)).resolve()
    output_root = project_path(config, "output", "output").resolve()
    annotation_root = project_path(config, "annotations", "annotations").resolve()
    manifest_path = annotation_root / "manifest.csv"

    if not input_root.exists():
        print(f"Папка изображений не найдена: {input_root}", file=sys.stderr)
        return 2

    requested = requested_relative_paths(args)
    exact_selection = bool(requested)
    if exact_selection:
        paths: list[Path] = []
        missing: list[str] = []
        for relative_text in requested:
            source_path = input_root / relative_text
            if source_path.is_file():
                paths.append(source_path)
            else:
                missing.append(relative_text)
        if missing:
            print("Не найдены изображения из точного списка:", file=sys.stderr)
            for relative_text in missing:
                print(f"- {relative_text}", file=sys.stderr)
            if not args.skip_missing:
                return 4
            print("Отсутствующие файлы пропущены.", file=sys.stderr)
        if not paths:
            print("После проверки точного списка не осталось доступных изображений.", file=sys.stderr)
            return 4
        count = len(paths)
    else:
        paths = list(iter_image_paths(input_root, recursive=True))
        if not paths:
            print(f"В папке нет изображений: {input_root}", file=sys.stderr)
            return 3
        default_count = int(selection_config.get(f"{args.mode}_count", 12 if args.mode == "sample" else 20))
        count = min(len(paths), max(1, args.count if args.count is not None else default_count))

    max_side = int(selection_config.get("max_analysis_side", 720))
    inventory = inventory_index(output_root / "reports" / "dataset.csv")

    print(f"Режим: {args.mode}")
    print(f"Источник: {input_root}")
    if exact_selection:
        print(f"Точный список изображений: {len(paths)}")
    else:
        print(f"Изображений в источнике: {len(paths)}")
        print(f"Нужно выбрать: {count}")

    records: list[dict[str, object]] = []
    reused = 0
    calculated = 0
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(input_root)
        relative_text = relative.as_posix()
        row = inventory.get(relative_text, {})
        if row and usable_inventory_features(row):
            features: dict[str, object] = {
                field: float(row[field]) for field in FEATURE_FIELDS
            }
            features.update(
                {
                    "width": int(float(row["width"])),
                    "height": int(float(row["height"])),
                    "channels": int(float(row["channels"])),
                }
            )
            reused += 1
        else:
            features = quick_features(path, max_side=max_side)
            calculated += 1
        records.append(
            {
                "relative_path": relative_text,
                "source_group": relative.parts[0] if relative.parts else ".",
                **features,
            }
        )
        print(f"[{index:03d}/{len(paths):03d}] {relative_text}")

    if exact_selection:
        selected = records
    else:
        selected_indices = select_balanced(records, count)
        selected = [records[index] for index in selected_indices]

    existing = load_manifest(manifest_path)
    if args.replace_unannotated_selection:
        kept: list[dict[str, str]] = []
        for record in existing:
            relative_path = record.get("source_relative_path", "")
            status, counts = existing_status(annotation_root, relative_path)
            has_work = status != "not_started" or any(counts.values())
            if has_work:
                kept.append(record)
        existing = kept

    by_path = {
        record.get("source_relative_path", ""): dict(record)
        for record in existing
        if record.get("source_relative_path")
    }
    next_id = 1
    if by_path:
        next_id = max(int(record.get("annotation_id", 0) or 0) for record in by_path.values()) + 1

    added = 0
    for order, record in enumerate(selected, start=1):
        relative_path = str(record["relative_path"])
        status, counts = existing_status(annotation_root, relative_path)
        mask_relative = mask_path_for(annotation_root, relative_path).relative_to(annotation_root).as_posix()
        metadata_relative = metadata_path_for(annotation_root, relative_path).relative_to(annotation_root).as_posix()
        if relative_path in by_path:
            row = by_path[relative_path]
            row.update(
                {
                    "selected_from_mode": args.mode,
                    "selection_order": order,
                    "width": record["width"],
                    "height": record["height"],
                    "channels": record["channels"],
                    "mean": record["mean"],
                    "std": record["std"],
                    "entropy": record["entropy"],
                    "edge_density": record["edge_density"],
                    "label_status": status,
                    "pixels_background": counts["background"],
                    "pixels_candy_core": counts["candy_core"],
                    "pixels_sticker": counts["sticker"],
                    "mask_relative_path": mask_relative,
                    "metadata_relative_path": metadata_relative,
                }
            )
        else:
            by_path[relative_path] = {
                "annotation_id": next_id,
                "source_relative_path": relative_path,
                "source_group": record["source_group"],
                "selected_from_mode": args.mode,
                "selection_order": order,
                "width": record["width"],
                "height": record["height"],
                "channels": record["channels"],
                "mean": record["mean"],
                "std": record["std"],
                "entropy": record["entropy"],
                "edge_density": record["edge_density"],
                "label_status": status,
                "pixels_background": counts["background"],
                "pixels_candy_core": counts["candy_core"],
                "pixels_sticker": counts["sticker"],
                "mask_relative_path": mask_relative,
                "metadata_relative_path": metadata_relative,
            }
            next_id += 1
            added += 1

    manifest_records = sorted(
        by_path.values(),
        key=lambda row: (int(row.get("annotation_id", 0) or 0), row.get("source_relative_path", "")),
    )
    annotation_root.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest_path, manifest_records)

    print("\nВыбраны изображения:")
    for record in selected:
        print(f"- {record['relative_path']}")
    print("\nПодготовка завершена.")
    print(f"Признаки взяты из dataset.csv: {reused}")
    print(f"Признаки рассчитаны заново: {calculated}")
    print(f"Новых записей добавлено: {added}")
    print(f"Всего записей в манифесте: {len(manifest_records)}")
    print(f"Манифест: {manifest_path}")
    if exact_selection:
        print("Точный список добавлен без удаления существующих записей и разметок.")
    print("Исходные изображения не копировались и не изменялись.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
