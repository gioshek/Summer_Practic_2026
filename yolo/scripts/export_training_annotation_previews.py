#!/usr/bin/env python3
"""Экспорт визуальных превью YOLO-разметки для D23 и D50."""

from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "wrapped", 1: "unwrapped"}
CLASS_COLORS = {0: (0, 190, 0), 1: (0, 0, 235)}
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("yolo/datasets/learning_curve"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/annotations"),
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--sheet-columns", type=int, default=3)
    parser.add_argument("--sheet-rows", type=int, default=3)
    return parser.parse_args()


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        return []

    objects = []
    for line_number, raw in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}, строка {line_number}: ожидалось 5 значений"
            )

        class_id = int(parts[0])
        xc, yc, width, height = map(float, parts[1:])

        if class_id not in CLASS_NAMES:
            raise ValueError(f"{label_path}: неизвестный класс {class_id}")

        objects.append((class_id, xc, yc, width, height))

    return objects


def draw_preview(
    image_path: Path,
    label_path: Path,
) -> tuple[np.ndarray, Counter]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Не удалось открыть {image_path}")

    height, width = image.shape[:2]
    counts: Counter = Counter()

    for class_id, xc, yc, box_width, box_height in read_labels(label_path):
        x1 = max(0, round((xc - box_width / 2) * width))
        y1 = max(0, round((yc - box_height / 2) * height))
        x2 = min(width - 1, round((xc + box_width / 2) * width))
        y2 = min(height - 1, round((yc + box_height / 2) * height))

        counts[class_id] += 1
        color = CLASS_COLORS[class_id]
        label = CLASS_NAMES[class_id]

        thickness = max(2, round(min(width, height) / 700))
        font_scale = max(0.55, min(width, height) / 1600)
        text_thickness = max(1, thickness - 1)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )

        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            image,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 4, max(text_height + 2, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    panel_height = min(100, max(60, height // 15))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (min(width, 700), panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    panel_text = (
        f"wrapped: {counts[0]} | unwrapped: {counts[1]} | "
        f"total: {counts[0] + counts[1]}"
    )
    cv2.putText(
        image,
        panel_text,
        (18, max(38, panel_height // 2 + 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.65, min(width, height) / 1500),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return image, counts


def make_contact_sheets(
    preview_paths: list[Path],
    output_dir: Path,
    columns: int,
    rows: int,
    jpeg_quality: int,
) -> None:
    per_sheet = columns * rows
    thumb_width = 600
    thumb_height = 380
    title_height = 55
    gap = 12

    for page_index in range(math.ceil(len(preview_paths) / per_sheet)):
        page_paths = preview_paths[
            page_index * per_sheet : (page_index + 1) * per_sheet
        ]

        canvas_width = columns * thumb_width + (columns + 1) * gap
        canvas_height = (
            rows * (thumb_height + title_height) + (rows + 1) * gap
        )
        canvas = np.full(
            (canvas_height, canvas_width, 3),
            245,
            dtype=np.uint8,
        )

        for index, path in enumerate(page_paths):
            image = cv2.imread(str(path))
            if image is None:
                continue

            row = index // columns
            column = index % columns
            x = gap + column * thumb_width
            y = gap + row * (thumb_height + title_height)

            scale = min(thumb_width / image.shape[1], thumb_height / image.shape[0])
            resized = cv2.resize(
                image,
                (
                    max(1, round(image.shape[1] * scale)),
                    max(1, round(image.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

            offset_x = x + (thumb_width - resized.shape[1]) // 2
            offset_y = y + (thumb_height - resized.shape[0]) // 2
            canvas[
                offset_y : offset_y + resized.shape[0],
                offset_x : offset_x + resized.shape[1],
            ] = resized

            title = path.stem
            if len(title) > 70:
                title = title[:67] + "..."

            cv2.putText(
                canvas,
                title,
                (x + 5, y + thumb_height + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        output_path = output_dir / f"contact_sheet_{page_index + 1:02d}.jpg"
        cv2.imwrite(
            str(output_path),
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )


def process_dataset(
    dataset_name: str,
    datasets_root: Path,
    output_root: Path,
    jpeg_quality: int,
    columns: int,
    rows: int,
) -> None:
    dataset_root = datasets_root / dataset_name
    images_root = dataset_root / "images" / "train"
    labels_root = dataset_root / "labels" / "train"

    if not images_root.is_dir():
        raise FileNotFoundError(images_root)
    if not labels_root.is_dir():
        raise FileNotFoundError(labels_root)

    output_dir = output_root / dataset_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    image_paths = sorted(
        path
        for path in images_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    total_counts: Counter = Counter()
    preview_paths = []

    for index, image_path in enumerate(image_paths, start=1):
        label_path = labels_root / f"{image_path.stem}.txt"
        preview, counts = draw_preview(image_path, label_path)
        total_counts.update(counts)

        output_path = output_dir / f"{index:03d}__{image_path.stem}.jpg"
        cv2.imwrite(
            str(output_path),
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        preview_paths.append(output_path)

    make_contact_sheets(
        preview_paths,
        output_dir,
        columns,
        rows,
        jpeg_quality,
    )

    summary = "\n".join(
        [
            f"Dataset: {dataset_name}",
            f"Images: {len(image_paths)}",
            f"Wrapped: {total_counts[0]}",
            f"Unwrapped: {total_counts[1]}",
            f"Total boxes: {total_counts[0] + total_counts[1]}",
            "",
        ]
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary, end="")


def main() -> int:
    args = parse_args()
    for dataset_name in ("d23", "d50"):
        process_dataset(
            dataset_name=dataset_name,
            datasets_root=args.datasets_root,
            output_root=args.output,
            jpeg_quality=args.jpeg_quality,
            columns=args.sheet_columns,
            rows=args.sheet_rows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
