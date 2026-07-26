#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CLASS_NAMES = {0: "wrapped", 1: "unwrapped"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate unwrapped-candy detection using CVAT point ground truth."
    )
    parser.add_argument("--cvat-export", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("yolo/annotation/holdout119_points_upload/holdout_mapping.csv"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/frontlight"),
    )
    parser.add_argument(
        "--model-d23",
        type=Path,
        default=Path("yolo/runs/learning_curve/d23_e25/weights/last.pt"),
    )
    parser.add_argument(
        "--model-d50",
        type=Path,
        default=Path("yolo/runs/learning_curve/d50_e25/weights/last.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo/evaluation/holdout119_point_metrics"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def normalize_names(value) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    return {}


def parse_point_string(value: str) -> list[tuple[float, float]]:
    result = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        x_text, y_text = item.split(",", 1)
        result.append((float(x_text), float(y_text)))
    return result


def read_ground_truth(export_path: Path) -> dict[str, list[tuple[float, float]]]:
    if not export_path.is_file():
        raise FileNotFoundError(export_path)

    with zipfile.ZipFile(export_path) as archive:
        xml_candidates = [name for name in archive.namelist() if name.endswith("annotations.xml")]
        if len(xml_candidates) != 1:
            raise RuntimeError(
                f"Expected one annotations.xml, found {len(xml_candidates)}"
            )
        xml_data = archive.read(xml_candidates[0])

    root = ET.fromstring(xml_data)
    points_by_image: dict[str, list[tuple[float, float]]] = {}

    for image_node in root.findall(".//image"):
        filename = Path(image_node.attrib["name"]).name
        points: list[tuple[float, float]] = []
        for point_node in image_node.findall("points"):
            if point_node.attrib.get("label") != "unwrapped_gt":
                continue
            points.extend(parse_point_string(point_node.attrib.get("points", "")))
        points_by_image[filename] = points

    return points_by_image


def predict_unwrapped_boxes(
    model: YOLO,
    image_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
) -> list[tuple[float, float, float, float, float]]:
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=500,
        device="cpu",
        verbose=False,
    )[0]

    boxes: list[tuple[float, float, float, float, float]] = []
    if result.boxes is None:
        return boxes

    xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()

    for box, class_id, score in zip(xyxy, classes, scores):
        if class_id != 1:
            continue
        x1, y1, x2, y2 = map(float, box.tolist())
        boxes.append((x1, y1, x2, y2, float(score)))

    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes


def match_boxes_to_points(
    boxes: list[tuple[float, float, float, float, float]],
    points: list[tuple[float, float]],
) -> tuple[int, int, int, set[int], set[int]]:
    unmatched_points = set(range(len(points)))
    matched_boxes: set[int] = set()
    matched_points: set[int] = set()

    for box_index, (x1, y1, x2, y2, _score) in enumerate(boxes):
        candidates = [
            point_index
            for point_index in unmatched_points
            if x1 <= points[point_index][0] <= x2
            and y1 <= points[point_index][1] <= y2
        ]
        if not candidates:
            continue

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        best_point = min(
            candidates,
            key=lambda point_index: (
                points[point_index][0] - center_x
            ) ** 2
            + (points[point_index][1] - center_y) ** 2,
        )
        unmatched_points.remove(best_point)
        matched_boxes.add(box_index)
        matched_points.add(best_point)

    tp = len(matched_points)
    fp = len(boxes) - tp
    fn = len(points) - tp
    return tp, fp, fn, matched_boxes, matched_points


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def draw_evaluation(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float, float]],
    points: list[tuple[float, float]],
    matched_boxes: set[int],
    matched_points: set[int],
    model_name: str,
) -> np.ndarray:
    result = image.copy()

    for box_index, (x1, y1, x2, y2, score) in enumerate(boxes):
        color = (0, 0, 255) if box_index in matched_boxes else (0, 165, 255)
        label = f"TP {score:.2f}" if box_index in matched_boxes else f"FP {score:.2f}"
        cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(
            result,
            label,
            (int(x1), max(25, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    for point_index, (x, y) in enumerate(points):
        color = (255, 255, 0) if point_index in matched_points else (255, 0, 255)
        label = "GT" if point_index in matched_points else "FN"
        cv2.circle(result, (round(x), round(y)), 10, color, -1)
        cv2.putText(
            result,
            label,
            (round(x) + 12, round(y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        result,
        model_name,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return result


def main() -> int:
    args = parse_arguments()

    mapping_path = args.mapping.resolve()
    raw_root = args.raw_root.resolve()
    output_root = args.output.resolve()
    export_path = args.cvat_export.resolve()
    model_paths = {
        "D23": args.model_d23.resolve(),
        "D50": args.model_d50.resolve(),
    }

    for path in [mapping_path, raw_root, export_path, *model_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    with mapping_path.open(encoding="utf-8", newline="") as source:
        mapping_rows = list(csv.DictReader(source))

    if len(mapping_rows) != 119:
        raise RuntimeError(f"Expected 119 mapping rows, got {len(mapping_rows)}")

    ground_truth = read_ground_truth(export_path)

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    models: dict[str, YOLO] = {}
    for model_name, model_path in model_paths.items():
        model = YOLO(str(model_path))
        if normalize_names(model.names) != CLASS_NAMES:
            raise RuntimeError(f"{model_name}: unexpected classes {model.names}")
        models[model_name] = model
        (output_root / model_name.lower()).mkdir()

    totals = {"D23": Counter(), "D50": Counter()}
    image_totals = {"D23": Counter(), "D50": Counter()}
    report_rows: list[dict[str, object]] = []

    for image_index, row in enumerate(mapping_rows, start=1):
        cvat_filename = row["cvat_filename"].strip()
        source_relative = Path(row["source_path"].strip())
        image_path = raw_root / source_relative
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        points = ground_truth.get(cvat_filename, [])
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")

        row_result: dict[str, object] = {
            "order": image_index,
            "source_path": source_relative.as_posix(),
            "ground_truth_unwrapped": len(points),
        }

        print(f"[{image_index:03d}/119] {source_relative} | GT={len(points)}")

        for model_name, model in models.items():
            boxes = predict_unwrapped_boxes(
                model,
                image_path,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
            )
            tp, fp, fn, matched_boxes, matched_points = match_boxes_to_points(boxes, points)

            totals[model_name].update({"tp": tp, "fp": fp, "fn": fn})

            gt_defect = len(points) > 0
            pred_defect = len(boxes) > 0
            if gt_defect and pred_defect:
                image_totals[model_name]["tp"] += 1
            elif not gt_defect and pred_defect:
                image_totals[model_name]["fp"] += 1
            elif gt_defect and not pred_defect:
                image_totals[model_name]["fn"] += 1
            else:
                image_totals[model_name]["tn"] += 1

            prefix = model_name.lower()
            row_result[f"{prefix}_predicted_unwrapped"] = len(boxes)
            row_result[f"{prefix}_tp"] = tp
            row_result[f"{prefix}_fp"] = fp
            row_result[f"{prefix}_fn"] = fn

            annotated = draw_evaluation(
                image,
                boxes,
                points,
                matched_boxes,
                matched_points,
                model_name,
            )
            output_path = output_root / prefix / f"{image_index:03d}.jpg"
            cv2.imwrite(
                str(output_path),
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )

        report_rows.append(row_result)

    report_path = output_root / "per_image_metrics.csv"
    with report_path.open("w", encoding="utf-8", newline="") as target:
        fieldnames = list(report_rows[0].keys())
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    summary_lines = [
        "UNWRAPPED-CANDY POINT METRICS",
        "==============================",
        f"Images: {len(mapping_rows)}",
        f"Confidence threshold: {args.conf}",
        "",
    ]

    for model_name in ("D23", "D50"):
        tp = totals[model_name]["tp"]
        fp = totals[model_name]["fp"]
        fn = totals[model_name]["fn"]
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        image_tp = image_totals[model_name]["tp"]
        image_fp = image_totals[model_name]["fp"]
        image_fn = image_totals[model_name]["fn"]
        image_tn = image_totals[model_name]["tn"]
        image_accuracy = safe_divide(
            image_tp + image_tn,
            image_tp + image_fp + image_fn + image_tn,
        )

        summary_lines.extend(
            [
                f"{model_name}:",
                f"- TP defective candies: {tp}",
                f"- FN missed defective candies: {fn}",
                f"- FP false defect detections: {fp}",
                f"- defect recall: {recall * 100:.2f}%",
                f"- defect precision: {precision * 100:.2f}%",
                f"- defect F1: {f1 * 100:.2f}%",
                f"- image-level accuracy: {image_accuracy * 100:.2f}%",
                "",
            ]
        )

    summary_path = output_root / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + summary_path.read_text(encoding="utf-8"))
    print(f"Per-image metrics: {report_path}")
    print(f"Visual checks: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
