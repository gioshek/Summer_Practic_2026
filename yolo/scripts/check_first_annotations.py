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
import yaml


CLASS_NAMES = {
    0: "wrapped",
    1: "unwrapped",
}

# Слоты, которые должны быть полностью проверены на этапе №5.
POSITIVE_SLOTS = {
    "01",
    "07",
    "09",
    "11",
    "18",
    "22",
}

# Кадр только со стикером.
NEGATIVE_SLOTS = {
    "19",
}

SELECTED_SLOTS = POSITIVE_SLOTS | NEGATIVE_SLOTS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Проверка первых аннотаций CVAT и создание "
            "изображений с нанесёнными YOLO-рамками."
        )
    )

    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="ZIP-архив экспорта CVAT.",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Корень проекта.",
    )

    return parser.parse_args()


def read_mapping(mapping_path: Path) -> dict[str, dict[str, str]]:
    with mapping_path.open(
        encoding="utf-8",
        newline="",
    ) as source:
        rows = list(csv.DictReader(source))

    return {
        row["slot"].strip(): row
        for row in rows
    }


def find_label_file(
    extracted_root: Path,
    image_stem: str,
) -> Path | None:
    matches = [
        path
        for path in extracted_root.rglob(
            f"{image_stem}.txt"
        )
        if "labels" in path.parts
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Найдено несколько меток для {image_stem}: "
            + ", ".join(str(path) for path in matches)
        )

    return matches[0]


def parse_label_file(
    label_path: Path,
    errors: list[str],
) -> list[tuple[int, float, float, float, float]]:
    boxes = []

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

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"{label_path}: строка {line_number} "
                f"содержит {len(parts)} значений вместо 5"
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(
                float,
                parts[1:],
            )
        except ValueError:
            errors.append(
                f"{label_path}: строка {line_number} "
                "содержит некорректные числа"
            )
            continue

        if class_id not in CLASS_NAMES:
            errors.append(
                f"{label_path}: неизвестный класс "
                f"{class_id}"
            )

        values = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }

        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                errors.append(
                    f"{label_path}: {name}={value} "
                    "находится вне диапазона [0, 1]"
                )

        if width <= 0.0 or height <= 0.0:
            errors.append(
                f"{label_path}: ширина и высота "
                "должны быть положительными"
            )

        left = x_center - width / 2
        right = x_center + width / 2
        top = y_center - height / 2
        bottom = y_center + height / 2

        tolerance = 1e-4

        if (
            left < -tolerance
            or right > 1.0 + tolerance
            or top < -tolerance
            or bottom > 1.0 + tolerance
        ):
            errors.append(
                f"{label_path}: рамка в строке "
                f"{line_number} выходит за изображение"
            )

        boxes.append(
            (
                class_id,
                x_center,
                y_center,
                width,
                height,
            )
        )

    return boxes


def draw_boxes(
    image,
    boxes: list[
        tuple[int, float, float, float, float]
    ],
):
    height, width = image.shape[:2]

    class_colors = {
        0: (0, 200, 0),
        1: (0, 0, 255),
    }

    for index, box in enumerate(boxes, start=1):
        class_id, xc, yc, bw, bh = box

        x1 = round((xc - bw / 2) * width)
        y1 = round((yc - bh / 2) * height)
        x2 = round((xc + bw / 2) * width)
        y2 = round((yc + bh / 2) * height)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        color = class_colors.get(
            class_id,
            (255, 255, 255),
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            4,
        )

        label = (
            f"{index}: "
            f"{CLASS_NAMES.get(class_id, class_id)}"
        )

        text_y = max(35, y1 - 10)

        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA,
        )

    return image


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
) -> None:
    if not image_paths:
        return

    columns = 2
    cell_width = 900
    cell_height = 600

    rows = (
        len(image_paths) + columns - 1
    ) // columns

    sheet = None

    import numpy as np

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

        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))

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

    output_root = (
        project_root
        / "yolo"
        / "annotation"
        / "qa_stage5"
    )

    extracted_root = output_root / "extracted"
    annotated_root = output_root / "annotated"
    report_path = output_root / "annotation_stats.csv"
    summary_path = output_root / "summary.txt"

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

    if extracted_root.exists():
        shutil.rmtree(extracted_root)

    if annotated_root.exists():
        shutil.rmtree(annotated_root)

    extracted_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    annotated_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted_root)

    errors: list[str] = []
    warnings: list[str] = []

    data_yaml_candidates = list(
        extracted_root.rglob("data.yaml")
    )

    if len(data_yaml_candidates) != 1:
        errors.append(
            "Ожидался ровно один data.yaml, найдено: "
            f"{len(data_yaml_candidates)}"
        )
    else:
        with data_yaml_candidates[0].open(
            encoding="utf-8",
        ) as source:
            configuration = yaml.safe_load(source)

        names = configuration.get("names")

        if isinstance(names, dict):
            actual_names = {
                int(key): str(value)
                for key, value in names.items()
            }
        elif isinstance(names, list):
            actual_names = {
                index: str(value)
                for index, value in enumerate(names)
            }
        else:
            actual_names = {}

        if actual_names != CLASS_NAMES:
            errors.append(
                "Неверные классы в data.yaml: "
                f"{actual_names}"
            )

    mapping = read_mapping(mapping_path)

    missing_slots = SELECTED_SLOTS - set(mapping)

    if missing_slots:
        errors.append(
            "В pilot_mapping.csv отсутствуют слоты: "
            + ", ".join(sorted(missing_slots))
        )

    report_rows = []
    total_classes = Counter()
    annotated_images = []

    for slot in sorted(SELECTED_SLOTS):
        if slot not in mapping:
            continue

        row = mapping[slot]
        cvat_filename = row["cvat_filename"].strip()

        image_path = images_root / cvat_filename

        if not image_path.is_file():
            errors.append(
                f"Слот {slot}: не найден файл "
                f"{image_path}"
            )
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(
                f"Слот {slot}: OpenCV не смог "
                f"открыть {image_path}"
            )
            continue

        image_stem = image_path.stem

        label_path = find_label_file(
            extracted_root,
            image_stem,
        )

        boxes = []

        if slot in POSITIVE_SLOTS:
            if label_path is None:
                errors.append(
                    f"Слот {slot}: отсутствует файл "
                    "меток для изображения с конфетами"
                )
            else:
                boxes = parse_label_file(
                    label_path,
                    errors,
                )

                if not boxes:
                    errors.append(
                        f"Слот {slot}: файл меток пуст"
                    )

        if slot in NEGATIVE_SLOTS:
            if label_path is not None:
                boxes = parse_label_file(
                    label_path,
                    errors,
                )

                if boxes:
                    errors.append(
                        f"Слот {slot}: отрицательный "
                        "кадр содержит рамки"
                    )

        counts = Counter(
            class_id
            for class_id, *_ in boxes
        )

        total_classes.update(counts)

        annotated = image.copy()

        if boxes:
            annotated = draw_boxes(
                annotated,
                boxes,
            )
        else:
            cv2.putText(
                annotated,
                "BACKGROUND: sticker ignored",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
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

        annotated_images.append(output_path)

        report_rows.append(
            {
                "slot": slot,
                "filename": cvat_filename,
                "expected_type": (
                    "positive"
                    if slot in POSITIVE_SLOTS
                    else "negative"
                ),
                "wrapped_count": counts[0],
                "unwrapped_count": counts[1],
                "total_boxes": len(boxes),
                "label_file": (
                    str(label_path)
                    if label_path is not None
                    else ""
                ),
            }
        )

    all_label_files = [
        path
        for path in extracted_root.rglob("*.txt")
        if "labels" in path.parts
    ]

    selected_stems = {
        Path(mapping[slot]["cvat_filename"]).stem
        for slot in SELECTED_SLOTS
        if slot in mapping
    }

    unexpected_labels = [
        path
        for path in all_label_files
        if path.stem not in selected_stems
    ]

    for path in unexpected_labels:
        warnings.append(
            "Есть метки на кадре вне текущей "
            f"проверочной выборки: {path.name}"
        )

    with report_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        fieldnames = [
            "slot",
            "filename",
            "expected_type",
            "wrapped_count",
            "unwrapped_count",
            "total_boxes",
            "label_file",
        ]

        writer = csv.DictWriter(
            target,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(report_rows)

    create_contact_sheet(
        annotated_images,
        output_root / "contact_sheet.jpg",
    )

    summary_lines = [
        "Проверка первых аннотаций",
        "==========================",
        "",
        f"Архив: {archive_path}",
        f"Проверено слотов: {len(report_rows)}",
        f"wrapped: {total_classes[0]}",
        f"unwrapped: {total_classes[1]}",
        (
            "Всего рамок: "
            f"{total_classes[0] + total_classes[1]}"
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

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))

    print("Созданы:")
    print(f"- {report_path}")
    print(f"- {summary_path}")
    print(
        f"- {output_root / 'contact_sheet.jpg'}"
    )
    print(f"- {annotated_root}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
