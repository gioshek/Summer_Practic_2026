#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


GROUPS = (
    "basler_120126",
    "basler_OldObj",
    "images",
)

BATCH_QUOTAS = {
    "batch_to_50": {
        "basler_120126": 8,
        "basler_OldObj": 11,
        "images": 8,
    },
    "batch_to_75": {
        "basler_120126": 8,
        "basler_OldObj": 9,
        "images": 8,
    },
    "batch_to_100": {
        "basler_120126": 8,
        "basler_OldObj": 9,
        "images": 8,
    },
    "batch_to_125": {
        "basler_120126": 8,
        "basler_OldObj": 9,
        "images": 8,
    },
}

FIXED_VAL_SLOTS = ("05", "09", "11", "19", "22")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Фиксация holdout-выборки и создание "
            "инкрементальных партий изображений."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--holdout-size",
        type=int,
        default=119,
    )

    return parser.parse_args()


def allocate_proportionally(
    available_counts: dict[str, int],
    total: int,
) -> dict[str, int]:
    available_total = sum(available_counts.values())

    if total > available_total:
        raise ValueError(
            f"Запрошено {total}, доступно {available_total}"
        )

    raw = {
        group: total * count / available_total
        for group, count in available_counts.items()
    }

    quotas = {
        group: min(
            available_counts[group],
            math.floor(value),
        )
        for group, value in raw.items()
    }

    remaining = total - sum(quotas.values())

    order = sorted(
        available_counts,
        key=lambda group: (
            raw[group] - math.floor(raw[group]),
            available_counts[group],
        ),
        reverse=True,
    )

    while remaining > 0:
        changed = False

        for group in order:
            if quotas[group] < available_counts[group]:
                quotas[group] += 1
                remaining -= 1
                changed = True

                if remaining == 0:
                    break

        if not changed:
            raise RuntimeError(
                "Не удалось распределить квоты."
            )

    return quotas


def stratified_pick(
    paths: list[Path],
    count: int,
    rng: random.Random,
) -> list[Path]:
    if count == 0:
        return []

    if count > len(paths):
        raise ValueError(
            f"Нужно {count}, доступно {len(paths)}"
        )

    ordered = sorted(paths)
    selected = []

    for index in range(count):
        start = index * len(ordered) // count
        end = (index + 1) * len(ordered) // count

        segment = ordered[start:end]

        if not segment:
            raise RuntimeError(
                "Получился пустой временной сегмент."
            )

        selected.append(rng.choice(segment))

    return selected


def write_manifest(
    path: Path,
    set_name: str,
    selected: list[tuple[str, Path]],
    raw_root: Path,
) -> list[dict[str, str]]:
    rows = []

    for order, (group, source_path) in enumerate(
        selected,
        start=1,
    ):
        cvat_filename = (
            f"{order:03d}__{group}__{source_path.name}"
        )

        rows.append(
            {
                "set_name": set_name,
                "order": str(order),
                "group": group,
                "source_path": str(
                    source_path.relative_to(raw_root)
                ),
                "cvat_filename": cvat_filename,
            }
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "set_name",
                "order",
                "group",
                "source_path",
                "cvat_filename",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def create_contact_sheet(
    rows: list[dict[str, str]],
    raw_root: Path,
    output_path: Path,
) -> None:
    columns = 5
    cell_width = 360
    cell_height = 280

    row_count = math.ceil(len(rows) / columns)

    sheet = np.zeros(
        (
            row_count * cell_height,
            columns * cell_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, row in enumerate(rows):
        source_path = raw_root / row["source_path"]
        image = cv2.imread(str(source_path))

        if image is None:
            raise RuntimeError(
                f"Не удалось открыть {source_path}"
            )

        image_height, image_width = image.shape[:2]

        available_height = cell_height - 45

        scale = min(
            cell_width / image_width,
            available_height / image_height,
        )

        resized_width = max(
            1,
            round(image_width * scale),
        )
        resized_height = max(
            1,
            round(image_height * scale),
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        grid_row = index // columns
        grid_column = index % columns

        x0 = (
            grid_column * cell_width
            + (cell_width - resized_width) // 2
        )
        y0 = grid_row * cell_height + 40

        sheet[
            y0:y0 + resized_height,
            x0:x0 + resized_width,
        ] = resized

        label = (
            f"{row['order']} | {row['group']}"
        )

        cv2.putText(
            sheet,
            label,
            (
                grid_column * cell_width + 8,
                grid_row * cell_height + 28,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(output_path),
        sheet,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            92,
        ],
    )


def create_upload_archive(
    rows: list[dict[str, str]],
    raw_root: Path,
    upload_root: Path,
    set_name: str,
) -> None:
    batch_root = upload_root / set_name
    files_root = batch_root / "files"
    archive_path = upload_root / f"{set_name}.zip"

    files_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for row in rows:
        source_path = raw_root / row["source_path"]
        destination_path = (
            files_root / row["cvat_filename"]
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as archive:
        for image_path in sorted(
            files_root.glob("*.bmp")
        ):
            archive.write(
                image_path,
                arcname=image_path.name,
            )


def main() -> int:
    args = parse_arguments()
    rng = random.Random(args.seed)

    project_root = Path.cwd()
    raw_root = (
        project_root / "data" / "raw" / "frontlight"
    )

    pilot_mapping_path = (
        project_root
        / "yolo"
        / "annotation"
        / "pilot_upload"
        / "pilot_mapping.csv"
    )

    selection_root = (
        project_root
        / "yolo"
        / "selection"
        / "incremental_v1"
    )

    manifests_root = selection_root / "manifests"
    sheets_root = selection_root / "contact_sheets"

    upload_root = (
        project_root
        / "yolo"
        / "annotation"
        / "incremental_upload"
    )

    if not pilot_mapping_path.is_file():
        raise SystemExit(
            f"Не найден {pilot_mapping_path}"
        )

    if selection_root.exists():
        shutil.rmtree(selection_root)

    if upload_root.exists():
        shutil.rmtree(upload_root)

    manifests_root.mkdir(parents=True)
    sheets_root.mkdir(parents=True)
    upload_root.mkdir(parents=True)

    with pilot_mapping_path.open(
        encoding="utf-8",
        newline="",
    ) as source:
        pilot_rows = list(csv.DictReader(source))

    if len(pilot_rows) != 23:
        raise SystemExit(
            f"Ожидалось 23 пилотных строки, "
            f"найдено {len(pilot_rows)}"
        )

    pilot_paths = set()

    for row in pilot_rows:
        relative_path = (
            row.get("original_path")
            or row.get("image_path")
            or ""
        ).strip()

        if not relative_path:
            raise SystemExit(
                "В pilot_mapping.csv отсутствует "
                "столбец original_path или image_path."
            )

        pilot_paths.add(Path(relative_path))

    available: dict[str, list[Path]] = {}

    for group in GROUPS:
        group_root = raw_root / group

        files = sorted(group_root.glob("*.bmp"))

        available[group] = [
            path
            for path in files
            if path.relative_to(raw_root)
            not in pilot_paths
        ]

    holdout_quotas = allocate_proportionally(
        {
            group: len(paths)
            for group, paths in available.items()
        },
        args.holdout_size,
    )

    holdout_selected: list[tuple[str, Path]] = []

    for group in GROUPS:
        chosen = stratified_pick(
            available[group],
            holdout_quotas[group],
            rng,
        )

        chosen_set = set(chosen)

        available[group] = [
            path
            for path in available[group]
            if path not in chosen_set
        ]

        holdout_selected.extend(
            (group, path)
            for path in chosen
        )

    rng.shuffle(holdout_selected)

    holdout_rows = write_manifest(
        manifests_root / "final_holdout_119.csv",
        "final_holdout_119",
        holdout_selected,
        raw_root,
    )

    all_batch_rows: dict[
        str,
        list[dict[str, str]]
    ] = {}

    for set_name, quotas in BATCH_QUOTAS.items():
        selected: list[tuple[str, Path]] = []

        for group in GROUPS:
            requested = quotas[group]

            chosen = stratified_pick(
                available[group],
                requested,
                rng,
            )

            chosen_set = set(chosen)

            available[group] = [
                path
                for path in available[group]
                if path not in chosen_set
            ]

            selected.extend(
                (group, path)
                for path in chosen
            )

        rng.shuffle(selected)

        rows = write_manifest(
            manifests_root / f"{set_name}.csv",
            set_name,
            selected,
            raw_root,
        )

        all_batch_rows[set_name] = rows

        create_contact_sheet(
            rows,
            raw_root,
            sheets_root / f"{set_name}.jpg",
        )

        create_upload_archive(
            rows,
            raw_root,
            upload_root,
            set_name,
        )

    fixed_val_path = (
        selection_root / "fixed_validation_slots.txt"
    )

    fixed_val_path.write_text(
        "\n".join(FIXED_VAL_SLOTS) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "Инкрементальный план выборок",
        "============================",
        "",
        f"Seed: {args.seed}",
        f"Пилотных изображений: {len(pilot_rows)}",
        f"Fixed validation: {', '.join(FIXED_VAL_SLOTS)}",
        (
            "Final holdout: "
            f"{len(holdout_rows)} изображений"
        ),
        "",
        "Holdout по группам:",
    ]

    holdout_counts = Counter(
        row["group"]
        for row in holdout_rows
    )

    for group in GROUPS:
        summary_lines.append(
            f"- {group}: {holdout_counts[group]}"
        )

    summary_lines.append("")
    summary_lines.append("Обучающие партии:")

    for set_name, rows in all_batch_rows.items():
        counts = Counter(
            row["group"]
            for row in rows
        )

        summary_lines.append(
            f"- {set_name}: {len(rows)} "
            f"({', '.join(f'{g}={counts[g]}' for g in GROUPS)})"
        )

    summary_path = selection_root / "selection_summary.txt"

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))

    print("Созданы архивы CVAT:")

    for set_name in BATCH_QUOTAS:
        print(
            f"- {upload_root / (set_name + '.zip')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
