#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAMES = {
    0: "wrapped",
    1: "unwrapped",
}

CLASS_COLORS = {
    0: (0, 200, 0),
    1: (255, 0, 255),
}

ALL_SLOTS = tuple(
    f"{number:02d}"
    for number in range(1, 24)
)

# Единственное выбранное изображение без конфет.
NEGATIVE_SLOTS = {"19"}

POSITIVE_SLOTS = set(ALL_SLOTS) - NEGATIVE_SLOTS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Полная проверка разметки 23 пилотных "
            "изображений."
        )
    )

    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="ZIP-архив экспорта CVAT.",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "yolo/annotation/qa_pilot23"
        ),
    )

    return parser.parse_args()


def normalize_names(value) -> dict[int, str]:
    if isinstance(value, list):
        return {
            index: str(name)
            for index, name in enumerate(value)
        }

    if isinstance(value, dict):
        return {
            int(index): str(name)
            for index, name in value.items()
        }

    return {}


def read_mapping(
    path: Path,
) -> dict[str, dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as source:
        rows = list(csv.DictReader(source))

    mapping = {}

    for row in rows:
        slot = row["slot"].strip()

        if slot in mapping:
            raise RuntimeError(
                f"Повторяющийся слот: {slot}"
            )

        mapping[slot] = row

    return mapping


def find_label(
    extracted_root: Path,
    stem: str,
) -> Path | None:
    matches = [
        path
        for path in extracted_root.rglob(
            f"{stem}.txt"
        )
        if "labels" in path.parts
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Несколько файлов меток для {stem}: "
            + ", ".join(str(path) for path in matches)
        )

    return matches[0]


def parse_boxes(
    label_path: Path,
    errors: list[str],
    warnings: list[str],
) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    seen_lines = set()

    with label_path.open(
        encoding="utf-8",
    ) as source:
        lines = source.readlines()

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        if line in seen_lines:
            warnings.append(
                f"{label_path.name}: возможная "
                f"дублирующаяся рамка в строке "
                f"{line_number}"
            )

        seen_lines.add(line)

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"{label_path.name}, строка {line_number}: "
                f"ожидалось 5 значений, получено "
                f"{len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])
            xc, yc, width, height = map(
                float,
                parts[1:],
            )
        except ValueError:
            errors.append(
                f"{label_path.name}, строка {line_number}: "
                "некорректные числа"
            )
            continue

        if class_id not in CLASS_NAMES:
            errors.append(
                f"{label_path.name}, строка {line_number}: "
                f"неизвестный класс {class_id}"
            )

        values = {
            "x_center": xc,
            "y_center": yc,
            "width": width,
            "height": height,
        }

        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                errors.append(
                    f"{label_path.name}, строка "
                    f"{line_number}: {name}={value} "
                    "вне диапазона [0, 1]"
                )

        if width <= 0 or height <= 0:
            errors.append(
                f"{label_path.name}, строка {line_number}: "
                "ширина и высота должны быть "
                "положительными"
            )

        tolerance = 1e-4

        left = xc - width / 2
        right = xc + width / 2
        top = yc - height / 2
        bottom = yc + height / 2

        if (
            left < -tolerance
            or right > 1.0 + tolerance
            or top < -tolerance
            or bottom > 1.0 + tolerance
        ):
            errors.append(
                f"{label_path.name}, строка "
                f"{line_number}: рамка выходит "
                "за пределы изображения"
            )

        boxes.append(
            (
                class_id,
                xc,
                yc,
                width,
                height,
            )
        )

    return boxes


def draw_boxes(
    image: np.ndarray,
    boxes: list[
        tuple[int, float, float, float, float]
    ],
) -> np.ndarray:
    result = image.copy()
    image_height, image_width = result.shape[:2]

    for number, (
        class_id,
        xc,
        yc,
        box_width,
        box_height,
    ) in enumerate(boxes, start=1):
        x1 = round(
            (xc - box_width / 2) * image_width
        )
        y1 = round(
            (yc - box_height / 2) * image_height
        )
        x2 = round(
            (xc + box_width / 2) * image_width
        )
        y2 = round(
            (yc + box_height / 2) * image_height
        )

        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))

        color = CLASS_COLORS.get(
            class_id,
            (255, 255, 255),
        )

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            4,
        )

        text = (
            f"{number}: "
            f"{CLASS_NAMES.get(class_id, class_id)}"
        )

        cv2.putText(
            result,
            text,
            (x1, max(35, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA,
        )

    return result


def add_header(
    image: np.ndarray,
    text: str,
) -> np.ndarray:
    header_height = 70

    result = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )

    cv2.putText(
        result,
        text,
        (20, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    return result


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
) -> None:
    columns = 2
    rows = 3
    cell_width = 900
    cell_height = 650

    sheet = np.zeros(
        (
            rows * cell_height,
            columns * cell_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        scale = min(
            cell_width / width,
            cell_height / height,
        )

        resized_width = max(
            1,
            round(width * scale),
        )
        resized_height = max(
            1,
            round(height * scale),
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        row = index // columns
        column = index % columns

        x_offset = (
            column * cell_width
            + (cell_width - resized_width) // 2
        )
        y_offset = (
            row * cell_height
            + (cell_height - resized_height) // 2
        )

        sheet[
            y_offset:y_offset + resized_height,
            x_offset:x_offset + resized_width,
        ] = resized

    cv2.imwrite(
        str(output_path),
        sheet,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            92,
        ],
    )


def main() -> int:
    args = parse_arguments()

    project_root = args.project_root.resolve()
    archive_path = args.archive.resolve()

    output_root = args.output

    if not output_root.is_absolute():
        output_root = project_root / output_root

    output_root = output_root.resolve()

    mapping_path = (
        project_root
        / "yolo"
        / "annotation"
        / "pilot_upload"
        / "pilot_mapping.csv"
    )

    images_root = (
        project_root
        / "yolo"
        / "annotation"
        / "pilot_upload"
        / "files"
    )

    if not archive_path.is_file():
        print(
            f"ОШИБКА: архив не найден: {archive_path}",
            file=sys.stderr,
        )
        return 1

    if not mapping_path.is_file():
        print(
            f"ОШИБКА: не найден {mapping_path}",
            file=sys.stderr,
        )
        return 1

    if output_root.exists():
        shutil.rmtree(output_root)

    extracted_root = output_root / "extracted"
    annotated_root = output_root / "annotated"
    sheets_root = output_root / "contact_sheets"

    extracted_root.mkdir(parents=True)
    annotated_root.mkdir(parents=True)
    sheets_root.mkdir(parents=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted_root)

    errors: list[str] = []
    warnings: list[str] = []

    mapping = read_mapping(mapping_path)

    if set(mapping) != set(ALL_SLOTS):
        errors.append(
            "pilot_mapping.csv не содержит "
            "ровно слоты 01–23"
        )

    data_yaml_files = list(
        extracted_root.rglob("data.yaml")
    )

    if len(data_yaml_files) != 1:
        errors.append(
            "Ожидался один data.yaml, найдено: "
            f"{len(data_yaml_files)}"
        )
    else:
        with data_yaml_files[0].open(
            encoding="utf-8",
        ) as source:
            config = yaml.safe_load(source)

        actual_names = normalize_names(
            config.get("names")
        )

        if actual_names != CLASS_NAMES:
            errors.append(
                "Неверные классы в data.yaml: "
                f"{actual_names}"
            )

    expected_filenames = {
        row["cvat_filename"].strip()
        for row in mapping.values()
    }

    train_files = list(
        extracted_root.rglob("train.txt")
    )

    if len(train_files) != 1:
        errors.append(
            "Ожидался один train.txt, найдено: "
            f"{len(train_files)}"
        )
    else:
        listed_filenames = {
            Path(line.strip()).name
            for line in train_files[0]
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }

        missing_in_train = (
            expected_filenames - listed_filenames
        )

        unexpected_in_train = (
            listed_filenames - expected_filenames
        )

        for filename in sorted(missing_in_train):
            errors.append(
                f"В train.txt отсутствует: {filename}"
            )

        for filename in sorted(unexpected_in_train):
            errors.append(
                f"Лишний файл в train.txt: {filename}"
            )

    stats_rows = []
    total_counts: Counter = Counter()
    annotated_paths = []

    for slot in ALL_SLOTS:
        row = mapping.get(slot)

        if row is None:
            continue

        filename = row["cvat_filename"].strip()
        image_path = images_root / filename

        if not image_path.is_file():
            errors.append(
                f"Слот {slot}: изображение не найдено: "
                f"{image_path}"
            )
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(
                f"Слот {slot}: OpenCV не смог открыть "
                f"{image_path}"
            )
            continue

        label_path = find_label(
            extracted_root,
            image_path.stem,
        )

        boxes = []

        if slot in POSITIVE_SLOTS:
            if label_path is None:
                errors.append(
                    f"Слот {slot}: отсутствует "
                    "файл меток"
                )
            else:
                boxes = parse_boxes(
                    label_path,
                    errors,
                    warnings,
                )

                if not boxes:
                    errors.append(
                        f"Слот {slot}: файл меток пуст"
                    )

        if slot in NEGATIVE_SLOTS:
            if label_path is not None:
                boxes = parse_boxes(
                    label_path,
                    errors,
                    warnings,
                )

                if boxes:
                    errors.append(
                        f"Слот {slot}: кадр только "
                        "со стикером содержит рамки"
                    )

        counts = Counter(
            class_id
            for class_id, *_ in boxes
        )

        total_counts.update(counts)

        annotated = draw_boxes(image, boxes)

        if slot in NEGATIVE_SLOTS and not boxes:
            cv2.putText(
                annotated,
                "NEGATIVE: sticker ignored",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )

        header = (
            f"Slot {slot} | "
            f"wrapped={counts[0]} | "
            f"unwrapped={counts[1]} | "
            f"total={len(boxes)}"
        )

        annotated = add_header(
            annotated,
            header,
        )

        output_path = (
            annotated_root
            / f"{slot}__review.jpg"
        )

        cv2.imwrite(
            str(output_path),
            annotated,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                92,
            ],
        )

        annotated_paths.append(output_path)

        stats_rows.append(
            {
                "slot": slot,
                "filename": filename,
                "wrapped_count": counts[0],
                "unwrapped_count": counts[1],
                "total_boxes": len(boxes),
                "negative_image": (
                    "yes"
                    if slot in NEGATIVE_SLOTS
                    else "no"
                ),
                "label_file": (
                    str(label_path)
                    if label_path is not None
                    else ""
                ),
            }
        )

    for sheet_number, start in enumerate(
        range(0, len(annotated_paths), 6),
        start=1,
    ):
        group = annotated_paths[start:start + 6]

        create_contact_sheet(
            group,
            sheets_root
            / f"contact_sheet_{sheet_number}.jpg",
        )

    stats_path = output_root / "annotation_stats.csv"

    with stats_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        fieldnames = [
            "slot",
            "filename",
            "wrapped_count",
            "unwrapped_count",
            "total_boxes",
            "negative_image",
            "label_file",
        ]

        writer = csv.DictWriter(
            target,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(stats_rows)

    summary_lines = [
        "Полная проверка candy_pilot_23",
        "==============================",
        "",
        f"Архив: {archive_path}",
        f"Проверено изображений: {len(stats_rows)}",
        f"Wrapped: {total_counts[0]}",
        f"Unwrapped: {total_counts[1]}",
        (
            "Всего рамок: "
            f"{total_counts[0] + total_counts[1]}"
        ),
        f"Ошибок: {len(errors)}",
        f"Предупреждений: {len(warnings)}",
        "",
    ]

    if errors:
        summary_lines.append("ОШИБКИ:")

        for error in errors:
            summary_lines.append(f"- {error}")

        summary_lines.append("")

    if warnings:
        summary_lines.append("ПРЕДУПРЕЖДЕНИЯ:")

        for warning in warnings:
            summary_lines.append(f"- {warning}")

        summary_lines.append("")

    if not errors:
        summary_lines.append(
            "Формальная проверка завершена успешно."
        )

    summary_path = output_root / "summary.txt"

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))

    print("Созданы:")
    print(f"- {summary_path}")
    print(f"- {stats_path}")
    print(f"- {annotated_root}")
    print(f"- {sheets_root}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
