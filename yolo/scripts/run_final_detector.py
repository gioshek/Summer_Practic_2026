#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CLASS_NAMES = {0: "wrapped", 1: "unwrapped"}
CLASS_COLORS = {0: (0, 210, 0), 1: (0, 0, 255)}  # BGR
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detection, classification and counting of wrapped/unwrapped candies."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo/runs/learning_curve/d50_e25/weights/last.pt"),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path, help="Image or directory with images.")
    group.add_argument("--manifest", type=Path, help="CSV with source_path column.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/frontlight"),
        help="Root for paths from --manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo/final_output/d50"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def normalize_names(value) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    return {}


def collect_images(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """Return pairs: absolute image path, relative output path."""
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        raw_root = args.raw_root.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if not raw_root.is_dir():
            raise FileNotFoundError(raw_root)

        rows: list[dict[str, str]]
        with manifest_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))

        if not rows or "source_path" not in rows[0]:
            raise RuntimeError("Manifest must contain source_path column.")

        result = []
        for row in rows:
            relative = Path(row["source_path"].strip())
            image_path = raw_root / relative
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            result.append((image_path, relative))
        return result

    source_path = args.source.resolve()
    if source_path.is_file():
        if source_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise RuntimeError(f"Unsupported image type: {source_path}")
        return [(source_path, Path(source_path.name))]

    if not source_path.is_dir():
        raise FileNotFoundError(source_path)

    images = [
        path
        for path in source_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    images.sort()
    if not images:
        raise RuntimeError(f"No images found in {source_path}")
    return [(path, path.relative_to(source_path)) for path in images]


def draw_panel(image: np.ndarray, counts: Counter, mean_conf: float) -> np.ndarray:
    result = image.copy()
    total = counts[0] + counts[1]
    defect_percent = 100.0 * counts[1] / total if total else 0.0

    panel_width = min(650, result.shape[1])
    panel_height = min(245, result.shape[0])
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, result, 0.22, 0, result)

    lines = [
        f"TOTAL OBJECTS: {total}",
        f"NON-DEFECTIVE (wrapped): {counts[0]}",
        f"DEFECTIVE (unwrapped): {counts[1]}",
        f"DEFECTIVE PERCENT: {defect_percent:.2f}%",
        f"MEAN CONFIDENCE: {mean_conf:.3f}",
    ]

    for index, text in enumerate(lines):
        color = (255, 255, 255)
        if text.startswith("DEFECTIVE (unwrapped)") or text.startswith("DEFECTIVE PERCENT"):
            color = (0, 0, 255)
        cv2.putText(
            result,
            text,
            (20, 42 + index * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            color,
            2,
            cv2.LINE_AA,
        )
    return result


def main() -> int:
    args = parse_arguments()
    model_path = args.model.resolve()
    output_root = args.output.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    images = collect_images(args)
    annotated_root = output_root / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = YOLO(str(model_path))
    actual_names = normalize_names(model.names)
    if actual_names != CLASS_NAMES:
        raise RuntimeError(f"Unexpected classes: {actual_names}")

    summary_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    overall: Counter = Counter()
    images_with_defect = 0

    for index, (image_path, relative_path) in enumerate(images, start=1):
        print(f"[{index:03d}/{len(images):03d}] {relative_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device="cpu",
            verbose=False,
        )[0]

        counts: Counter = Counter()
        confidences: list[float] = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            for object_index, (box, class_id, score) in enumerate(
                zip(boxes, classes, scores), start=1
            ):
                if class_id not in CLASS_NAMES:
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, min(image.shape[1] - 1, x1))
                y1 = max(0, min(image.shape[0] - 1, y1))
                x2 = max(0, min(image.shape[1] - 1, x2))
                y2 = max(0, min(image.shape[0] - 1, y2))

                counts[class_id] += 1
                confidences.append(float(score))
                color = CLASS_COLORS[class_id]

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                label = f"{CLASS_NAMES[class_id]} {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                label_top = max(0, y1 - text_height - baseline - 8)
                cv2.rectangle(
                    image,
                    (x1, label_top),
                    (min(image.shape[1] - 1, x1 + text_width + 8), y1),
                    color,
                    -1,
                )
                cv2.putText(
                    image,
                    label,
                    (x1 + 4, max(text_height + 2, y1 - baseline - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                detection_rows.append(
                    {
                        "image": relative_path.as_posix(),
                        "object_index": object_index,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": f"{float(score):.6f}",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        total = counts[0] + counts[1]
        defect_percent = 100.0 * counts[1] / total if total else 0.0

        overall.update(counts)
        if counts[1] > 0:
            images_with_defect += 1

        annotated = draw_panel(image, counts, mean_conf)
        output_path = (annotated_root / relative_path).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(output_path),
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        ):
            raise RuntimeError(f"Could not save {output_path}")

        summary_rows.append(
            {
                "image": relative_path.as_posix(),
                "total_objects": total,
                "wrapped": counts[0],
                "unwrapped": counts[1],
                "defective_percent": f"{defect_percent:.4f}",
                "defect_present": "yes" if counts[1] > 0 else "no",
                "mean_confidence": f"{mean_conf:.6f}",
                "annotated_image": str(output_path.relative_to(output_root)),
            }
        )

    summary_csv = output_root / "images_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as target:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    detections_csv = output_root / "detections.csv"
    with detections_csv.open("w", encoding="utf-8", newline="") as target:
        fieldnames = [
            "image",
            "object_index",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detection_rows)

    overall_total = overall[0] + overall[1]
    overall_percent = 100.0 * overall[1] / overall_total if overall_total else 0.0
    summary_text = "\n".join(
        [
            "FINAL CANDY DETECTION SUMMARY",
            "=============================",
            f"Model: {model_path}",
            f"Images processed: {len(images)}",
            f"Total objects: {overall_total}",
            f"Non-defective (wrapped): {overall[0]}",
            f"Defective (unwrapped): {overall[1]}",
            f"Defective percent: {overall_percent:.2f}%",
            f"Images with detected defect: {images_with_defect}",
            f"Confidence threshold: {args.conf}",
            f"IoU threshold: {args.iou}",
            "",
        ]
    )
    (output_root / "summary.txt").write_text(summary_text, encoding="utf-8")

    print("\nCompleted.")
    print(f"Annotated images: {annotated_root}")
    print(f"Per-image summary: {summary_csv}")
    print(f"Per-object detections: {detections_csv}")
    print(f"Overall summary: {output_root / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
