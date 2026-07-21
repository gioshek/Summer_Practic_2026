from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from common import (
    file_hash,
    iter_image_paths,
    load_yaml,
    normalize_to_uint8,
    project_input_path,
    project_path,
    read_image,
    resize_long_side,
    safe_slug,
    to_gray,
    write_csv,
    write_image,
    write_text,
)

HISTOGRAM_FIELDS = [f"hist_{index:02d}" for index in range(16)]
DATASET_FIELDS = [
    "image_id",
    "run_mode",
    "source_root",
    "relative_path",
    "source_group",
    "folder",
    "filename",
    "suffix",
    "file_size_bytes",
    "sha256",
    "width",
    "height",
    "channels",
    "dtype",
    "analysis_scale",
    "mean",
    "std",
    "min",
    "max",
    "p01",
    "p05",
    "p50",
    "p95",
    "p99",
    "dynamic_range_05_95",
    "entropy",
    "laplacian_variance",
    "edge_density",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "row_mean_std",
    "column_mean_std",
] + HISTOGRAM_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Аудит набора изображений Frontlight.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--mode", choices=["sample", "raw"], default="sample")
    parser.add_argument("--input", type=Path, help="Явный путь вместо пути из project.yaml")
    parser.add_argument("--output", type=Path, help="Явная папка вывода")
    parser.add_argument("--previews", action="store_true", help="Создать объединённые обзорные листы")
    parser.add_argument("--hashes", action="store_true", help="Посчитать SHA-256 и точные дубликаты")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def entropy_uint8(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return 0.0
    probabilities = histogram[histogram > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def image_features(image: np.ndarray, max_analysis_side: int) -> dict[str, float | int | str]:
    image_u8 = normalize_to_uint8(image)
    original_height, original_width = image_u8.shape[:2]
    channels = 1 if image_u8.ndim == 2 else int(image_u8.shape[2])

    analysis_image, scale = resize_long_side(image_u8, max_analysis_side)
    gray = to_gray(analysis_image)

    p01, p05, p50, p95, p99 = np.percentile(gray, [1, 5, 50, 95, 99])
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))
    row_mean = gray.mean(axis=1)
    column_mean = gray.mean(axis=0)
    histogram = cv2.calcHist([gray], [0], None, [16], [0, 256]).ravel()
    histogram = histogram / max(float(histogram.sum()), 1.0)

    result: dict[str, float | int | str] = {
        "width": int(original_width),
        "height": int(original_height),
        "channels": channels,
        "dtype": str(image.dtype),
        "analysis_scale": float(scale),
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "min": int(gray.min()),
        "max": int(gray.max()),
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "dynamic_range_05_95": float(p95 - p05),
        "entropy": entropy_uint8(gray),
        "laplacian_variance": laplacian_variance,
        "edge_density": edge_density,
        "dark_pixel_ratio": float(np.count_nonzero(gray <= 32) / max(gray.size, 1)),
        "bright_pixel_ratio": float(np.count_nonzero(gray >= 224) / max(gray.size, 1)),
        "row_mean_std": float(row_mean.std()),
        "column_mean_std": float(column_mean.std()),
    }
    for index, value in enumerate(histogram):
        result[f"hist_{index:02d}"] = float(value)
    return result


def source_group_for(relative_path: Path) -> str:
    return relative_path.parts[0] if relative_path.parts else "."


def to_thumbnail(image: np.ndarray, title: str, target_width: int) -> np.ndarray:
    image_u8 = normalize_to_uint8(image)
    if image_u8.ndim == 2:
        bgr = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    elif image_u8.shape[2] == 4:
        bgr = cv2.cvtColor(image_u8, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image_u8.copy()

    scale = target_width / max(bgr.shape[1], 1)
    target_height = max(1, int(round(bgr.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (target_width, target_height), interpolation=interpolation)
    header_height = 58
    canvas = np.full((target_height + header_height, target_width, 3), 245, dtype=np.uint8)
    canvas[header_height:] = resized
    short_title = title if len(title) <= 50 else "..." + title[-47:]
    cv2.putText(canvas, short_title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{image.shape[1]}x{image.shape[0]}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_contact_sheet(thumbnails: list[np.ndarray], columns: int) -> np.ndarray:
    if not thumbnails:
        return np.full((200, 600, 3), 255, dtype=np.uint8)
    cell_width = max(item.shape[1] for item in thumbnails)
    cell_height = max(item.shape[0] for item in thumbnails)
    rows = math.ceil(len(thumbnails) / columns)
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)
    for index, item in enumerate(thumbnails):
        row = index // columns
        column = index % columns
        y = row * cell_height
        x = column * cell_width
        sheet[y : y + item.shape[0], x : x + item.shape[1]] = item
    return sheet


def batched(items: list[np.ndarray], size: int) -> Iterable[list[np.ndarray]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def write_contact_sheets(
    grouped: dict[str, list[np.ndarray]],
    previews_root: Path,
    columns: int,
    page_size: int,
) -> list[Path]:
    previews_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    expected: set[Path] = set()
    for group_name, thumbnails in grouped.items():
        pages = list(batched(thumbnails, max(1, page_size)))
        for page_index, page in enumerate(pages, start=1):
            destination = previews_root / (
                f"dataset_contact_sheet_{safe_slug(group_name)}_"
                f"{page_index:03d}-of-{len(pages):03d}.jpg"
            )
            write_image(destination, make_contact_sheet(page, columns))
            written.append(destination)
            expected.add(destination.resolve())
    for old in previews_root.glob("dataset_contact_sheet_*.jpg"):
        if old.resolve() not in expected:
            old.unlink(missing_ok=True)
    return written


def duplicate_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        digest = str(record.get("sha256", ""))
        if digest:
            grouped[digest].append(record)
    rows: list[dict[str, object]] = []
    duplicate_group = 0
    for digest, group_records in sorted(grouped.items()):
        if len(group_records) < 2:
            continue
        duplicate_group += 1
        for record in group_records:
            rows.append(
                {
                    "duplicate_group": duplicate_group,
                    "sha256": digest,
                    "count_in_group": len(group_records),
                    "relative_path": record["relative_path"],
                    "source_group": record["source_group"],
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    audit_config = config.get("audit", {})
    if not isinstance(audit_config, dict):
        audit_config = {}

    input_root = (args.input or project_input_path(config, args.mode)).resolve()
    output_root = (args.output or project_path(config, "output", "output")).resolve()
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"Папка не найдена: {input_root}", file=sys.stderr)
        return 2

    paths = list(iter_image_paths(input_root, recursive=args.recursive))
    if not paths:
        print(f"Изображения не найдены: {input_root}", file=sys.stderr)
        return 3

    max_analysis_side = int(audit_config.get("max_analysis_side", 1200))
    thumb_width = int(audit_config.get("thumb_width", 320))
    sheet_columns = max(1, int(audit_config.get("sheet_columns", 3)))
    sheet_page_size = max(1, int(audit_config.get("sheet_page_size", 24)))

    records: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    thumbnails: dict[str, list[np.ndarray]] = defaultdict(list)
    if args.previews:
        thumbnails["all"] = []

    print(f"Режим: {args.mode}")
    print(f"Источник: {input_root}")
    print(f"Найдено изображений: {len(paths)}")
    print(f"SHA-256: {'да' if args.hashes else 'нет'}")
    print(f"Обзорные листы: {'да' if args.previews else 'нет'}")

    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(input_root)
        relative_text = relative.as_posix()
        group = source_group_for(relative)
        folder = relative.parent.as_posix() if relative.parent != Path(".") else "."
        try:
            image = read_image(path)
            features = image_features(image, max_analysis_side=max_analysis_side)
            record: dict[str, object] = {
                "image_id": index,
                "run_mode": args.mode,
                "source_root": str(input_root),
                "relative_path": relative_text,
                "source_group": group,
                "folder": folder,
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "file_size_bytes": path.stat().st_size,
                "sha256": file_hash(path) if args.hashes else "",
                **features,
            }
            records.append(record)
            if args.previews:
                thumb = to_thumbnail(image, relative_text, thumb_width)
                thumbnails["all"].append(thumb)
                thumbnails[group].append(thumb)
            print(f"[{index:03d}/{len(paths):03d}] OK: {relative_text}")
        except Exception as exc:
            errors.append({"relative_path": relative_text, "error": str(exc)})
            print(f"[{index:03d}/{len(paths):03d}] ОШИБКА: {relative_text}: {exc}")

    write_csv(reports_root / "dataset.csv", records, DATASET_FIELDS)
    write_csv(reports_root / "unreadable.csv", errors, ["relative_path", "error"])
    duplicates = duplicate_rows(records) if args.hashes else []
    write_csv(
        reports_root / "duplicates.csv",
        duplicates,
        ["duplicate_group", "sha256", "count_in_group", "relative_path", "source_group"],
    )

    preview_files: list[Path] = []
    if args.previews:
        preview_files = write_contact_sheets(
            thumbnails,
            output_root / "previews",
            sheet_columns,
            sheet_page_size,
        )

    folder_counter = Counter(str(record["source_group"]) for record in records)
    shape_counter = Counter(
        (record["width"], record["height"], record["channels"]) for record in records
    )
    means = np.array([float(record["mean"]) for record in records], dtype=np.float64)
    contrasts = np.array([float(record["std"]) for record in records], dtype=np.float64)

    lines = [
        "АУДИТ НАБОРА FRONTLIGHT",
        "",
        f"Режим: {args.mode}",
        f"Источник: {input_root}",
        f"Найдено файлов: {len(paths)}",
        f"Успешно прочитано: {len(records)}",
        f"Ошибок чтения: {len(errors)}",
        f"SHA-256 рассчитан: {'да' if args.hashes else 'нет'}",
        f"Обзорные листы созданы: {'да' if args.previews else 'нет'}",
        "",
        "По папкам:",
    ]
    for folder, count in sorted(folder_counter.items()):
        lines.append(f"- {folder}: {count}")
    lines.extend(["", "Форматы изображений:"])
    for shape, count in sorted(shape_counter.items()):
        width, height, channels = shape
        lines.append(f"- {width}x{height}, каналов {channels}: {count}")
    if len(means):
        lines.extend(
            [
                "",
                f"Средняя яркость: {means.mean():.2f}",
                f"Диапазон средней яркости: {means.min():.2f} .. {means.max():.2f}",
                f"Средний контраст: {contrasts.mean():.2f}",
                f"Диапазон контраста: {contrasts.min():.2f} .. {contrasts.max():.2f}",
            ]
        )
    lines.extend(
        [
            "",
            "Обновлены файлы:",
            "- output/reports/dataset.csv",
            "- output/reports/unreadable.csv",
            "- output/reports/duplicates.csv",
            "- output/reports/summary.txt",
        ]
    )
    if preview_files:
        lines.append(f"- output/previews: {len(preview_files)} обзорных листов")
    write_text(reports_root / "summary.txt", "\n".join(lines) + "\n")

    print("\nАудит завершён.")
    print(f"Таблица: {reports_root / 'dataset.csv'}")
    print(f"Сводка: {reports_root / 'summary.txt'}")
    if args.previews:
        print(f"Обзорные листы: {output_root / 'previews'}")
    return 0 if not errors else 4


if __name__ == "__main__":
    raise SystemExit(main())
