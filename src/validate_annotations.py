from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from annotation_io import (
    CLASS_COLORS_BGR,
    count_pixels,
    load_label_mask,
    load_manifest,
    preview_path_for,
    read_metadata,
)
from common import load_yaml, project_input_path, project_path, read_image, to_bgr_uint8, write_csv, write_image

REPORT_FIELDS = [
    "source_relative_path",
    "source_group",
    "available_in_mode",
    "label_exists",
    "status",
    "pixels_labeled",
    "pixels_background",
    "pixels_candy_core",
    "pixels_sticker",
    "width",
    "height",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка ручной разметки.")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage2.yaml"))
    parser.add_argument("--mode", choices=["sample", "raw"], default="sample")
    parser.add_argument("--input", type=Path, help="Явный путь к изображениям")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--previews", action="store_true", help="Создать предпросмотры разметки")
    return parser.parse_args()


def make_preview(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    bgr = to_bgr_uint8(image)
    overlay = bgr.copy()
    for class_id, color in CLASS_COLORS_BGR.items():
        overlay[mask == class_id] = color
    labeled = mask > 0
    blended = cv2.addWeighted(bgr, 1.0 - alpha, overlay, alpha, 0.0)
    bgr[labeled] = blended[labeled]
    return bgr


def main() -> int:
    args = parse_args()
    project = load_yaml(args.project_config)
    ui = load_yaml(args.config)
    input_root = (args.input or project_input_path(project, args.mode)).resolve()
    annotation_root = project_path(project, "annotations", "annotations").resolve()
    manifest_path = annotation_root / "manifest.csv"
    report_path = annotation_root / "annotation_summary.csv"

    if not input_root.exists():
        print(f"Папка изображений не найдена: {input_root}", file=sys.stderr)
        return 2
    records = load_manifest(manifest_path)
    if not records:
        print(f"Манифест пуст или не найден: {manifest_path}", file=sys.stderr)
        return 3

    min_labeled = int(ui.get("minimum_labeled_pixels", 200))
    alpha = float(ui.get("overlay_alpha", 0.45))

    rows: list[dict[str, object]] = []
    available_count = 0
    complete_count = 0
    errors = 0

    for index, record in enumerate(records, start=1):
        relative = record.get("source_relative_path", "")
        source_group = record.get("source_group", "")
        image_path = input_root / relative
        available = image_path.exists()
        label_exists = False
        error_text = ""
        status = "not_started"
        width = int(float(record.get("width", 0) or 0))
        height = int(float(record.get("height", 0) or 0))
        counts = {"background": 0, "candy_core": 0, "sticker": 0}

        if available:
            available_count += 1
            try:
                image = read_image(image_path)
                height, width = image.shape[:2]
                mask_path = annotation_root / record.get("mask_relative_path", "")
                label_exists = bool(record.get("mask_relative_path")) and mask_path.exists()
                mask = load_label_mask(annotation_root, relative, (height, width))
                metadata = read_metadata(annotation_root, relative)
                status = str(metadata.get("status", record.get("label_status", "not_started")))
                counts = count_pixels(mask)

                if status == "complete":
                    labeled_pixels = (
                        counts["background"]
                        + counts["candy_core"]
                        + counts["sticker"]
                    )

                    if labeled_pixels < min_labeled:
                        error_text = (
                            f"слишком мало размеченных пикселей: "
                            f"{labeled_pixels} < {min_labeled}"
                        )
                    else:
                        complete_count += 1

                if args.previews and label_exists:
                    destination = preview_path_for(annotation_root, relative)
                    write_image(destination, make_preview(image, mask, alpha))
            except (OSError, ValueError) as exc:
                error_text = str(exc)
        else:
            error_text = "недоступно в выбранном режиме"

        if available and error_text:
            errors += 1

        labeled_total = counts["background"] + counts["candy_core"] + counts["sticker"]
        rows.append(
            {
                "source_relative_path": relative,
                "source_group": source_group,
                "available_in_mode": "yes" if available else "no",
                "label_exists": "yes" if label_exists else "no",
                "status": status,
                "pixels_labeled": labeled_total,
                "pixels_background": counts["background"],
                "pixels_candy_core": counts["candy_core"],
                "pixels_sticker": counts["sticker"],
                "width": width,
                "height": height,
                "error": error_text,
            }
        )
        print(
            f"[{index:03d}/{len(records):03d}] {relative}: "
            f"available={'yes' if available else 'no'}, status={status}, "
            f"bg={counts['background']}, core={counts['candy_core']}, "
            f"sticker={counts['sticker']}, error={error_text or '-'}"
        )

    write_csv(report_path, rows, REPORT_FIELDS)

    print("\nПроверка завершена.")
    print(f"Режим: {args.mode}")
    print(f"Доступных изображений: {available_count}")
    print(f"Завершённых без ошибок: {complete_count}")
    print(f"Ошибок в доступной разметке: {errors}")
    print(f"CSV: {report_path}")
    if args.previews:
        print(f"Предпросмотры: {annotation_root / 'previews'}")

    if errors:
        return 4
    if args.require_all and complete_count != available_count:
        print("Не все доступные изображения имеют корректный статус complete.", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
