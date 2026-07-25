#!/usr/bin/env python3

from __future__ import annotations

import csv
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import yaml


CLASS_NAMES = {
    0: "wrapped",
    1: "unwrapped",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as source:
        return list(csv.DictReader(source))


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


def extract_export(
    archive_path: Path,
    output_path: Path,
) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Архив не найден: {archive_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_path)

    yaml_files = list(output_path.rglob("data.yaml"))

    if len(yaml_files) != 1:
        raise RuntimeError(
            f"В {archive_path.name} ожидался один "
            f"data.yaml, найдено: {len(yaml_files)}"
        )

    with yaml_files[0].open(
        encoding="utf-8",
    ) as source:
        config = yaml.safe_load(source)

    names = normalize_names(config.get("names"))

    if names != CLASS_NAMES:
        raise RuntimeError(
            f"Неверный порядок классов в "
            f"{archive_path.name}: {names}"
        )


def find_label(
    export_root: Path,
    image_stem: str,
) -> Path | None:
    matches = [
        path
        for path in export_root.rglob(
            f"{image_stem}.txt"
        )
        if "labels" in path.parts
    ]

    if not matches:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Для {image_stem} найдено несколько "
            f"файлов меток: {matches}"
        )

    return matches[0]


def validate_label(
    label_path: Path,
) -> Counter:
    counts: Counter = Counter()

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
            raise ValueError(
                f"{label_path}, строка {line_number}: "
                f"ожидалось 5 значений, "
                f"получено {len(parts)}"
            )

        try:
            class_id = int(parts[0])
            xc, yc, width, height = map(
                float,
                parts[1:],
            )
        except ValueError as error:
            raise ValueError(
                f"{label_path}, строка {line_number}: "
                "некорректное число"
            ) from error

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"{label_path}, строка {line_number}: "
                f"неизвестный класс {class_id}"
            )

        for name, value in (
            ("xc", xc),
            ("yc", yc),
            ("width", width),
            ("height", height),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{label_path}, строка {line_number}: "
                    f"{name}={value} вне [0, 1]"
                )

        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"{label_path}, строка {line_number}: "
                "неположительный размер рамки"
            )

        counts[class_id] += 1

    return counts


def prepare_dataset_root(
    dataset_root: Path,
) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    (dataset_root / "images" / "train").mkdir(
        parents=True
    )

    (dataset_root / "labels" / "train").mkdir(
        parents=True
    )


def add_rows(
    *,
    rows: list[dict[str, str]],
    source_column: str,
    raw_root: Path,
    export_root: Path,
    dataset_root: Path,
    source_set: str,
    manifest_rows: list[dict[str, str]],
    total_counts: Counter,
) -> None:
    for row in rows:
        relative_source = Path(
            row[source_column].strip()
        )

        cvat_filename = row[
            "cvat_filename"
        ].strip()

        source_image = raw_root / relative_source

        if not source_image.is_file():
            raise FileNotFoundError(
                f"Исходное изображение не найдено: "
                f"{source_image}"
            )

        destination_image = (
            dataset_root
            / "images"
            / "train"
            / cvat_filename
        )

        shutil.copy2(
            source_image,
            destination_image,
        )

        label_path = find_label(
            export_root,
            Path(cvat_filename).stem,
        )

        counts: Counter = Counter()

        if label_path is not None:
            counts = validate_label(label_path)

            destination_label = (
                dataset_root
                / "labels"
                / "train"
                / f"{Path(cvat_filename).stem}.txt"
            )

            shutil.copy2(
                label_path,
                destination_label,
            )

        total_counts.update(counts)

        manifest_rows.append(
            {
                "source_set": source_set,
                "source_path": str(relative_source),
                "filename": cvat_filename,
                "wrapped_count": counts[0],
                "unwrapped_count": counts[1],
                "total_boxes": (
                    counts[0] + counts[1]
                ),
                "has_label_file": (
                    "yes"
                    if label_path is not None
                    else "no"
                ),
            }
        )


def write_dataset_files(
    dataset_root: Path,
    manifest_rows: list[dict[str, str]],
) -> None:
    dataset_yaml = dataset_root / "dataset.yaml"

    # val указывает на train только для совместимости
    # структуры YAML. В обучении используется val=False.
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/train",
                "",
                "names:",
                "  0: wrapped",
                "  1: unwrapped",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = dataset_root / "manifest.csv"

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        fieldnames = [
            "source_set",
            "source_path",
            "filename",
            "wrapped_count",
            "unwrapped_count",
            "total_boxes",
            "has_label_file",
        ]

        writer = csv.DictWriter(
            target,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> int:
    project_root = Path.cwd().resolve()

    raw_root = (
        project_root
        / "data"
        / "raw"
        / "frontlight"
    )

    pilot_mapping_path = (
        project_root
        / "yolo"
        / "annotation"
        / "pilot_upload"
        / "pilot_mapping.csv"
    )

    batch_mapping_path = (
        project_root
        / "yolo"
        / "selection"
        / "incremental_v1"
        / "manifests"
        / "batch_to_50.csv"
    )

    pilot_archive = (
        project_root
        / "yolo"
        / "annotation"
        / "cvat_exports"
        / "candy_pilot_23_complete_v1.zip"
    )

    batch_archive = (
        project_root
        / "yolo"
        / "annotation"
        / "cvat_exports"
        / "candy_train_to_50_complete_v1.zip"
    )

    output_root = (
        project_root
        / "yolo"
        / "datasets"
        / "learning_curve"
    )

    temporary_root = (
        output_root / "_extracted"
    )

    required_paths = [
        raw_root,
        pilot_mapping_path,
        batch_mapping_path,
        pilot_archive,
        batch_archive,
    ]

    for path in required_paths:
        if not path.exists():
            print(
                f"ОШИБКА: не найдено: {path}",
                file=sys.stderr,
            )
            return 1

    if output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)

    pilot_export_root = (
        temporary_root / "pilot23"
    )

    batch_export_root = (
        temporary_root / "batch27"
    )

    extract_export(
        pilot_archive,
        pilot_export_root,
    )

    extract_export(
        batch_archive,
        batch_export_root,
    )

    pilot_rows = read_csv(pilot_mapping_path)
    batch_rows = read_csv(batch_mapping_path)

    if len(pilot_rows) != 23:
        raise RuntimeError(
            f"Ожидалось 23 пилотных изображения, "
            f"получено {len(pilot_rows)}"
        )

    if len(batch_rows) != 27:
        raise RuntimeError(
            f"Ожидалось 27 новых изображений, "
            f"получено {len(batch_rows)}"
        )

    configurations = {
        "d23": False,
        "d50": True,
    }

    for dataset_name, include_batch in (
        configurations.items()
    ):
        dataset_root = output_root / dataset_name

        prepare_dataset_root(dataset_root)

        manifest_rows = []
        total_counts: Counter = Counter()

        add_rows(
            rows=pilot_rows,
            source_column="original_path",
            raw_root=raw_root,
            export_root=pilot_export_root,
            dataset_root=dataset_root,
            source_set="pilot23",
            manifest_rows=manifest_rows,
            total_counts=total_counts,
        )

        if include_batch:
            add_rows(
                rows=batch_rows,
                source_column="source_path",
                raw_root=raw_root,
                export_root=batch_export_root,
                dataset_root=dataset_root,
                source_set="batch_to_50",
                manifest_rows=manifest_rows,
                total_counts=total_counts,
            )

        write_dataset_files(
            dataset_root,
            manifest_rows,
        )

        image_count = len(
            list(
                (
                    dataset_root
                    / "images"
                    / "train"
                ).glob("*.bmp")
            )
        )

        label_count = len(
            list(
                (
                    dataset_root
                    / "labels"
                    / "train"
                ).glob("*.txt")
            )
        )

        print()
        print(dataset_name.upper())
        print("=" * len(dataset_name))
        print(f"Изображений: {image_count}")
        print(f"Файлов меток: {label_count}")
        print(f"Wrapped: {total_counts[0]}")
        print(f"Unwrapped: {total_counts[1]}")
        print(
            "Всего рамок: "
            f"{total_counts[0] + total_counts[1]}"
        )
        print(
            f"YAML: {dataset_root / 'dataset.yaml'}"
        )

        expected_count = (
            50 if include_batch else 23
        )

        if image_count != expected_count:
            raise RuntimeError(
                f"{dataset_name}: ожидалось "
                f"{expected_count} изображений, "
                f"получено {image_count}"
            )

    shutil.rmtree(temporary_root)

    print()
    print("Датасеты D23 и D50 созданы успешно.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
