#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


Image.MAX_IMAGE_PIXELS = None

EXTENSIONS = {
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

EXPECTED_GROUPS = {
    "basler_120126",
    "basler_OldObj",
    "basler_tstRGB",
    "images",
}


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def calculate_dhash(image: Image.Image) -> str:
    small = image.convert("L").resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )

    pixels = np.asarray(small, dtype=np.uint8)
    differences = pixels[:, 1:] > pixels[:, :-1]

    value = 0

    for bit in differences.ravel():
        value = (value << 1) | int(bit)

    return f"{value:016x}"


def write_csv(
    path: Path,
    rows: list[dict],
    fields: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def scan_image(
    path: Path,
    root: Path,
    number: int,
    total: int,
) -> dict:
    relative_path = path.relative_to(root).as_posix()
    relative_parts = Path(relative_path).parts

    if len(relative_parts) > 1:
        group = relative_parts[0]
    else:
        group = "(root)"

    size_bytes = path.stat().st_size

    record = {
        "relative_path": relative_path,
        "group": group,
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024 / 1024, 4),
        "sha256": "",
        "width": "",
        "height": "",
        "megapixels": "",
        "mode": "",
        "channels": "",
        "mean_gray": "",
        "std_gray": "",
        "dynamic_range_p01_p99": "",
        "sharpness_laplacian": "",
        "dhash": "",
        "status": "unreadable",
        "error": "",
        "duplicate_group": "",
    }

    print(
        f"[{number:03d}/{total:03d}] "
        f"{relative_path}"
    )

    try:
        record["sha256"] = calculate_sha256(path)

        with Image.open(path) as opened:
            opened.load()

            image = ImageOps.exif_transpose(opened)

            width, height = image.size

            preview = image.convert("RGB")
            preview.thumbnail(
                (512, 512),
                Image.Resampling.LANCZOS,
            )

            gray = np.asarray(
                preview.convert("L"),
                dtype=np.uint8,
            )

            if gray.size == 0:
                raise ValueError(
                    "После чтения получено пустое изображение."
                )

            percentile_01 = float(
                np.percentile(gray, 1)
            )
            percentile_99 = float(
                np.percentile(gray, 99)
            )

            sharpness = float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F,
                ).var()
            )

            record.update(
                {
                    "width": width,
                    "height": height,
                    "megapixels": round(
                        width * height / 1_000_000,
                        4,
                    ),
                    "mode": image.mode,
                    "channels": len(image.getbands()),
                    "mean_gray": round(
                        float(gray.mean()),
                        4,
                    ),
                    "std_gray": round(
                        float(gray.std()),
                        4,
                    ),
                    "dynamic_range_p01_p99": round(
                        percentile_99 - percentile_01,
                        4,
                    ),
                    "sharpness_laplacian": round(
                        sharpness,
                        4,
                    ),
                    "dhash": calculate_dhash(preview),
                    "status": "ok",
                    "error": "",
                }
            )

    except Exception as error:
        record["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return record


def create_contact_sheets(
    root: Path,
    output_directory: Path,
    records: list[dict],
    columns: int = 5,
    rows: int = 4,
) -> int:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    font = ImageFont.load_default()

    cell_width = 300
    cell_height = 240

    thumbnail_width = 280
    thumbnail_height = 180

    images_per_page = columns * rows

    records_by_group = defaultdict(list)

    for record in records:
        records_by_group[record["group"]].append(
            record
        )

    page_count = 0

    for group, group_records in sorted(
        records_by_group.items()
    ):
        group_records.sort(
            key=lambda item: item["relative_path"]
        )

        for start in range(
            0,
            len(group_records),
            images_per_page,
        ):
            page_count += 1

            page_number = (
                start // images_per_page + 1
            )

            page = Image.new(
                "RGB",
                (
                    columns * cell_width,
                    rows * cell_height,
                ),
                "white",
            )

            draw = ImageDraw.Draw(page)

            page_records = group_records[
                start : start + images_per_page
            ]

            for position, record in enumerate(
                page_records
            ):
                column = position % columns
                row = position // columns

                x = column * cell_width
                y = row * cell_height

                source_path = (
                    root / record["relative_path"]
                )

                try:
                    with Image.open(
                        source_path
                    ) as opened:
                        opened.load()

                        image = ImageOps.exif_transpose(
                            opened
                        ).convert("RGB")

                        image.thumbnail(
                            (
                                thumbnail_width,
                                thumbnail_height,
                            ),
                            Image.Resampling.LANCZOS,
                        )

                        paste_x = (
                            x
                            + (
                                cell_width
                                - image.width
                            )
                            // 2
                        )

                        paste_y = (
                            y
                            + 8
                            + (
                                thumbnail_height
                                - image.height
                            )
                            // 2
                        )

                        page.paste(
                            image,
                            (paste_x, paste_y),
                        )

                except Exception:
                    draw.text(
                        (x + 10, y + 10),
                        "UNREADABLE",
                        fill="red",
                        font=font,
                    )

                filename = record["filename"]

                if len(filename) > 42:
                    filename = (
                        filename[:39] + "..."
                    )

                draw.text(
                    (x + 10, y + 194),
                    filename,
                    fill="black",
                    font=font,
                )

                draw.text(
                    (x + 10, y + 212),
                    (
                        f"{record['width']}x"
                        f"{record['height']} | "
                        f"{record['size_mb']} MB"
                    ),
                    fill="black",
                    font=font,
                )

            output_path = (
                output_directory
                / (
                    f"{group}_page_"
                    f"{page_number:03d}.jpg"
                )
            )

            page.save(
                output_path,
                format="JPEG",
                quality=88,
                optimize=True,
            )

    return page_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Технический аудит полного "
            "набора изображений."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=594,
    )

    parser.add_argument(
        "--near-distance",
        type=int,
        default=2,
    )

    arguments = parser.parse_args()

    root = arguments.root.resolve()
    output = arguments.output.resolve()

    if not root.is_dir():
        raise SystemExit(
            f"Не найдена папка: {root}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
    )

    image_paths = [
        path
        for path in all_files
        if path.suffix.lower() in EXTENSIONS
    ]

    non_image_paths = [
        path
        for path in all_files
        if path.suffix.lower()
        not in EXTENSIONS
    ]

    if not image_paths:
        raise SystemExit(
            "Изображения не найдены."
        )

    print(
        f"Найдено изображений: "
        f"{len(image_paths)}"
    )

    print(
        f"Посторонних файлов: "
        f"{len(non_image_paths)}"
    )

    print()

    records = []

    for number, path in enumerate(
        image_paths,
        start=1,
    ):
        records.append(
            scan_image(
                path=path,
                root=root,
                number=number,
                total=len(image_paths),
            )
        )

    records_by_hash = defaultdict(list)

    for record in records:
        if record["status"] == "ok":
            records_by_hash[
                record["sha256"]
            ].append(record)

    duplicate_rows = []
    duplicate_group_number = 0

    for duplicate_records in (
        records_by_hash.values()
    ):
        if len(duplicate_records) < 2:
            continue

        duplicate_group_number += 1

        duplicate_group_id = (
            f"DUP-"
            f"{duplicate_group_number:03d}"
        )

        for record in duplicate_records:
            record["duplicate_group"] = (
                duplicate_group_id
            )

            duplicate_rows.append(
                {
                    "duplicate_group":
                        duplicate_group_id,
                    "relative_path":
                        record["relative_path"],
                    "sha256":
                        record["sha256"],
                }
            )

    readable_records = [
        record
        for record in records
        if record["status"] == "ok"
    ]

    near_duplicate_rows = []

    for left_index, left in enumerate(
        readable_records
    ):
        for right in readable_records[
            left_index + 1 :
        ]:
            if (
                left["sha256"]
                == right["sha256"]
            ):
                continue

            if (
                left["width"],
                left["height"],
            ) != (
                right["width"],
                right["height"],
            ):
                continue

            left_hash = int(
                left["dhash"],
                16,
            )

            right_hash = int(
                right["dhash"],
                16,
            )

            distance = (
                left_hash ^ right_hash
            ).bit_count()

            if (
                distance
                <= arguments.near_distance
            ):
                near_duplicate_rows.append(
                    {
                        "hamming_distance":
                            distance,
                        "left_path":
                            left["relative_path"],
                        "right_path":
                            right["relative_path"],
                    }
                )

    near_duplicate_rows.sort(
        key=lambda item: (
            item["hamming_distance"],
            item["left_path"],
            item["right_path"],
        )
    )

    index_fields = list(
        records[0].keys()
    )

    write_csv(
        output / "dataset_index.csv",
        records,
        index_fields,
    )

    review_rows = []

    for record in records:
        review_rows.append(
            {
                "relative_path":
                    record["relative_path"],
                "group":
                    record["group"],
                "has_unwrapped": "",
                "has_sticker": "",
                "has_border_candy": "",
                "crowding": "",
                "review_status": "",
                "notes": "",
            }
        )

    write_csv(
        output / "review_manifest.csv",
        review_rows,
        list(review_rows[0].keys()),
    )

    write_csv(
        output / "exact_duplicates.csv",
        duplicate_rows,
        [
            "duplicate_group",
            "relative_path",
            "sha256",
        ],
    )

    write_csv(
        output / "near_duplicates.csv",
        near_duplicate_rows,
        [
            "hamming_distance",
            "left_path",
            "right_path",
        ],
    )

    unreadable_rows = [
        {
            "relative_path":
                record["relative_path"],
            "group":
                record["group"],
            "error":
                record["error"],
        }
        for record in records
        if record["status"] != "ok"
    ]

    write_csv(
        output / "unreadable_files.csv",
        unreadable_rows,
        [
            "relative_path",
            "group",
            "error",
        ],
    )

    non_image_rows = [
        {
            "relative_path":
                path.relative_to(
                    root
                ).as_posix(),
            "size_bytes":
                path.stat().st_size,
        }
        for path in non_image_paths
    ]

    write_csv(
        output / "non_image_files.csv",
        non_image_rows,
        [
            "relative_path",
            "size_bytes",
        ],
    )

    print()
    print("Создаю обзорные листы...")

    contact_sheet_pages = (
        create_contact_sheets(
            root=root,
            output_directory=(
                output / "contact_sheets"
            ),
            records=records,
        )
    )

    group_counts = Counter(
        record["group"]
        for record in records
    )

    dimension_counts = Counter(
        (
            f"{record['width']}x"
            f"{record['height']}"
        )
        for record in readable_records
    )

    actual_groups = set(
        group_counts.keys()
    )

    total_size_bytes = sum(
        record["size_bytes"]
        for record in records
    )

    summary = {
        "expected_count":
            arguments.expected_count,
        "actual_count":
            len(records),
        "count_matches":
            (
                len(records)
                == arguments.expected_count
            ),
        "readable_count":
            len(readable_records),
        "unreadable_count":
            len(unreadable_rows),
        "non_image_file_count":
            len(non_image_paths),
        "total_size_bytes":
            total_size_bytes,
        "total_size_gib":
            round(
                total_size_bytes / 1024**3,
                4,
            ),
        "group_counts":
            dict(
                sorted(
                    group_counts.items()
                )
            ),
        "dimension_counts":
            dict(
                sorted(
                    dimension_counts.items()
                )
            ),
        "missing_groups":
            sorted(
                EXPECTED_GROUPS
                - actual_groups
            ),
        "unexpected_groups":
            sorted(
                actual_groups
                - EXPECTED_GROUPS
            ),
        "exact_duplicate_group_count":
            duplicate_group_number,
        "exact_duplicate_file_count":
            len(duplicate_rows),
        "near_duplicate_pair_count":
            len(near_duplicate_rows),
        "contact_sheet_page_count":
            contact_sheet_pages,
    }

    summary_json_path = (
        output / "summary.json"
    )

    summary_json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "# Аудит датасета",
        "",
        (
            f"- Ожидалось изображений: "
            f"{summary['expected_count']}"
        ),
        (
            f"- Найдено изображений: "
            f"{summary['actual_count']}"
        ),
        (
            f"- Количество совпало: "
            f"{summary['count_matches']}"
        ),
        (
            f"- Читаемых: "
            f"{summary['readable_count']}"
        ),
        (
            f"- Нечитаемых: "
            f"{summary['unreadable_count']}"
        ),
        (
            f"- Общий размер: "
            f"{summary['total_size_gib']} GiB"
        ),
        (
            f"- Посторонних файлов: "
            f"{summary['non_image_file_count']}"
        ),
        (
            f"- Групп точных дубликатов: "
            f"{summary['exact_duplicate_group_count']}"
        ),
        (
            f"- Файлов в группах "
            f"точных дубликатов: "
            f"{summary['exact_duplicate_file_count']}"
        ),
        (
            f"- Пар кандидатов "
            f"в почти-дубликаты: "
            f"{summary['near_duplicate_pair_count']}"
        ),
        (
            f"- Обзорных листов: "
            f"{summary['contact_sheet_page_count']}"
        ),
        "",
        "## Количество по группам",
        "",
    ]

    for group, count in (
        summary["group_counts"].items()
    ):
        summary_lines.append(
            f"- `{group}`: {count}"
        )

    summary_lines.extend(
        [
            "",
            "## Размеры изображений",
            "",
        ]
    )

    for dimensions, count in (
        summary["dimension_counts"].items()
    ):
        summary_lines.append(
            f"- `{dimensions}`: {count}"
        )

    summary_lines.extend(
        [
            "",
            "## Проверка групп",
            "",
            (
                "- Отсутствующие: "
                + (
                    ", ".join(
                        summary[
                            "missing_groups"
                        ]
                    )
                    or "нет"
                )
            ),
            (
                "- Неожиданные: "
                + (
                    ", ".join(
                        summary[
                            "unexpected_groups"
                        ]
                    )
                    or "нет"
                )
            ),
            "",
            "## Созданные файлы",
            "",
            (
                "- `dataset_index.csv` — "
                "технический индекс."
            ),
            (
                "- `review_manifest.csv` — "
                "таблица ручной проверки."
            ),
            (
                "- `exact_duplicates.csv` — "
                "точные дубликаты."
            ),
            (
                "- `near_duplicates.csv` — "
                "кандидаты в почти-дубликаты."
            ),
            (
                "- `unreadable_files.csv` — "
                "нечитаемые изображения."
            ),
            (
                "- `non_image_files.csv` — "
                "посторонние файлы."
            ),
            (
                "- `contact_sheets/` — "
                "обзорные листы."
            ),
            "",
        ]
    )

    summary_markdown_path = (
        output / "summary.md"
    )

    summary_markdown_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("Аудит завершён.")

    print(
        f"Отчёт: "
        f"{summary_markdown_path}"
    )

    print(
        f"Индекс: "
        f"{output / 'dataset_index.csv'}"
    )


if __name__ == "__main__":
    main()
