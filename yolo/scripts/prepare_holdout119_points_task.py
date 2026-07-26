#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path

import cv2


def main() -> int:
    project_root = Path.cwd().resolve()
    manifest_path = (
        project_root
        / "yolo/selection/incremental_v1/manifests/final_holdout_119.csv"
    )
    raw_root = project_root / "data/raw/frontlight"
    output_root = project_root / "yolo/annotation/holdout119_points_upload"
    files_root = output_root / "files"
    mapping_path = output_root / "holdout_mapping.csv"
    archive_path = output_root / "holdout119_points_task.zip"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)

    with manifest_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    if len(rows) != 119:
        raise RuntimeError(f"Expected 119 rows, got {len(rows)}")

    if output_root.exists():
        shutil.rmtree(output_root)
    files_root.mkdir(parents=True)

    mapping_rows: list[dict[str, str]] = []

    for order, row in enumerate(rows, start=1):
        source_relative = Path(row["source_path"].strip())
        group = row["group"].strip()
        source_path = raw_root / source_relative

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"OpenCV could not read {source_path}")

        cvat_filename = f"{order:03d}__{group}__{source_path.stem}.jpg"
        destination_path = files_root / cvat_filename

        if not cv2.imwrite(
            str(destination_path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        ):
            raise RuntimeError(f"Could not save {destination_path}")

        mapping_rows.append(
            {
                "order": str(order),
                "group": group,
                "source_path": source_relative.as_posix(),
                "cvat_filename": cvat_filename,
                "width": str(image.shape[1]),
                "height": str(image.shape[0]),
            }
        )

        print(f"[{order:03d}/119] {source_relative}")

    with mapping_path.open("w", encoding="utf-8", newline="") as target:
        fieldnames = [
            "order",
            "group",
            "source_path",
            "cvat_filename",
            "width",
            "height",
        ]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as archive:
        for image_path in sorted(files_root.glob("*.jpg")):
            archive.write(image_path, arcname=image_path.name)

    print("\nHoldout point-annotation task prepared.")
    print(f"Archive: {archive_path}")
    print(f"Mapping: {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
