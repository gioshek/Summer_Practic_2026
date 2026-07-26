#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


CLASS_NAMES = {
    0: "wrapped",
    1: "unwrapped",
}

CLASS_COLORS = {
    0: (0, 210, 0),      # зелёный
    1: (0, 0, 255),      # красный
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-d23",
        type=Path,
        default=Path(
            "yolo/runs/learning_curve/"
            "d23_e25/weights/last.pt"
        ),
    )

    parser.add_argument(
        "--model-d50",
        type=Path,
        default=Path(
            "yolo/runs/learning_curve/"
            "d50_e25/weights/last.pt"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "yolo/selection/incremental_v1/"
            "manifests/final_holdout_119.csv"
        ),
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/raw/frontlight"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "yolo/evaluation/"
            "holdout119_d23_vs_d50"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
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


def check_model_classes(
    model: YOLO,
    model_name: str,
) -> None:
    names = normalize_names(model.names)

    if names != CLASS_NAMES:
        raise RuntimeError(
            f"{model_name}: неверные классы: {names}"
        )


def add_transparent_panel(
    image: np.ndarray,
    model_name: str,
    counts: Counter,
    mean_confidence: float,
) -> np.ndarray:
    result = image.copy()

    panel_width = min(
        620,
        result.shape[1],
    )

    panel_height = min(
        235,
        result.shape[0],
    )

    overlay = result.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width, panel_height),
        (20, 20, 20),
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.78,
        result,
        0.22,
        0,
        result,
    )

    defect = counts[1] > 0

    lines = [
        f"MODEL: {model_name}",
        f"wrapped: {counts[0]}",
        f"unwrapped: {counts[1]}",
        f"total: {counts[0] + counts[1]}",
        (
            "DEFECT: YES"
            if defect
            else "DEFECT: NO"
        ),
        f"mean confidence: {mean_confidence:.3f}",
    ]

    for index, text in enumerate(lines):
        color = (255, 255, 255)

        if text.startswith("DEFECT"):
            color = (
                (0, 0, 255)
                if defect
                else (0, 220, 0)
            )

        cv2.putText(
            result,
            text,
            (20, 38 + index * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            color,
            2,
            cv2.LINE_AA,
        )

    return result


def predict_and_draw(
    *,
    model: YOLO,
    model_name: str,
    image_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
) -> tuple[np.ndarray, Counter, float]:
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Не удалось открыть {image_path}"
        )

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=500,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    counts: Counter = Counter()
    confidences = []

    if result.boxes is not None:
        xyxy = result.boxes.xyxy.cpu().numpy()
        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )
        scores = result.boxes.conf.cpu().numpy()

        for box, class_id, score in zip(
            xyxy,
            classes,
            scores,
        ):
            if class_id not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.tolist(),
            )

            counts[class_id] += 1
            confidences.append(float(score))

            color = CLASS_COLORS[class_id]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            label = (
                f"{CLASS_NAMES[class_id]} "
                f"{score:.2f}"
            )

            text_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                2,
            )

            text_width, text_height = text_size

            label_top = max(
                0,
                y1 - text_height - baseline - 8,
            )

            cv2.rectangle(
                image,
                (x1, label_top),
                (
                    min(
                        image.shape[1] - 1,
                        x1 + text_width + 8,
                    ),
                    y1,
                ),
                color,
                thickness=-1,
            )

            cv2.putText(
                image,
                label,
                (
                    x1 + 4,
                    max(
                        text_height + 2,
                        y1 - baseline - 4,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    mean_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    image = add_transparent_panel(
        image,
        model_name,
        counts,
        mean_confidence,
    )

    return image, counts, mean_confidence


def resize_for_comparison(
    image: np.ndarray,
    max_width: int = 1400,
) -> np.ndarray:
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width

    return cv2.resize(
        image,
        (
            max_width,
            round(height * scale),
        ),
        interpolation=cv2.INTER_AREA,
    )


def make_comparison(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left = resize_for_comparison(left)
    right = resize_for_comparison(right)

    target_height = min(
        left.shape[0],
        right.shape[0],
    )

    def resize_height(
        image: np.ndarray,
        height: int,
    ) -> np.ndarray:
        scale = height / image.shape[0]

        return cv2.resize(
            image,
            (
                round(image.shape[1] * scale),
                height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    left = resize_height(left, target_height)
    right = resize_height(right, target_height)

    divider = np.full(
        (
            target_height,
            12,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    return np.hstack(
        [left, divider, right]
    )


def main() -> int:
    args = parse_arguments()

    model_d23_path = args.model_d23.resolve()
    model_d50_path = args.model_d50.resolve()
    manifest_path = args.manifest.resolve()
    raw_root = args.raw_root.resolve()
    output_root = args.output.resolve()

    for path in (
        model_d23_path,
        model_d50_path,
        manifest_path,
        raw_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if output_root.exists():
        shutil.rmtree(output_root)

    d23_root = output_root / "d23"
    d50_root = output_root / "d50"
    comparisons_root = (
        output_root / "comparisons"
    )

    d23_root.mkdir(parents=True)
    d50_root.mkdir(parents=True)
    comparisons_root.mkdir(parents=True)

    torch.set_num_threads(args.threads)

    print("Загрузка D23...")
    model_d23 = YOLO(str(model_d23_path))

    print("Загрузка D50...")
    model_d50 = YOLO(str(model_d50_path))

    check_model_classes(model_d23, "D23")
    check_model_classes(model_d50, "D50")

    with manifest_path.open(
        encoding="utf-8",
        newline="",
    ) as source:
        rows = list(csv.DictReader(source))

    if len(rows) != 119:
        raise RuntimeError(
            f"Ожидалось 119 holdout-изображений, "
            f"получено {len(rows)}"
        )

    report_rows = []
    total_d23: Counter = Counter()
    total_d50: Counter = Counter()
    html_entries = []

    for index, row in enumerate(rows, start=1):
        relative_path = Path(
            row["source_path"].strip()
        )

        image_path = raw_root / relative_path

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        group = row["group"].strip()

        safe_filename = (
            f"{index:03d}__{group}__"
            f"{image_path.stem}.jpg"
        )

        print(
            f"[{index:03d}/119] "
            f"{relative_path}"
        )

        image_d23, counts_d23, confidence_d23 = (
            predict_and_draw(
                model=model_d23,
                model_name="D23",
                image_path=image_path,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
            )
        )

        image_d50, counts_d50, confidence_d50 = (
            predict_and_draw(
                model=model_d50,
                model_name="D50",
                image_path=image_path,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
            )
        )

        total_d23.update(counts_d23)
        total_d50.update(counts_d50)

        d23_path = d23_root / safe_filename
        d50_path = d50_root / safe_filename

        comparison_path = (
            comparisons_root / safe_filename
        )

        cv2.imwrite(
            str(d23_path),
            image_d23,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                92,
            ],
        )

        cv2.imwrite(
            str(d50_path),
            image_d50,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                92,
            ],
        )

        comparison = make_comparison(
            image_d23,
            image_d50,
        )

        cv2.imwrite(
            str(comparison_path),
            comparison,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                90,
            ],
        )

        report_rows.append(
            {
                "order": index,
                "group": group,
                "source_path": str(relative_path),
                "d23_wrapped": counts_d23[0],
                "d23_unwrapped": counts_d23[1],
                "d23_total": (
                    counts_d23[0] + counts_d23[1]
                ),
                "d23_defect": (
                    "yes"
                    if counts_d23[1] > 0
                    else "no"
                ),
                "d23_mean_confidence": (
                    f"{confidence_d23:.6f}"
                ),
                "d50_wrapped": counts_d50[0],
                "d50_unwrapped": counts_d50[1],
                "d50_total": (
                    counts_d50[0] + counts_d50[1]
                ),
                "d50_defect": (
                    "yes"
                    if counts_d50[1] > 0
                    else "no"
                ),
                "d50_mean_confidence": (
                    f"{confidence_d50:.6f}"
                ),
                "manual_better_model": "",
                "manual_comment": "",
            }
        )

        html_entries.append(
            (
                index,
                str(relative_path),
                safe_filename,
                counts_d23.copy(),
                counts_d50.copy(),
            )
        )

    csv_path = output_root / "predictions.csv"

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        fieldnames = list(
            report_rows[0].keys()
        )

        writer = csv.DictWriter(
            target,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(report_rows)

    summary = "\n".join(
        [
            "Визуальное сравнение D23 и D50",
            "================================",
            "",
            "Важно: эталонной разметки holdout нет.",
            "Это качественная, а не количественная оценка.",
            "",
            f"Изображений: {len(rows)}",
            f"Confidence threshold: {args.conf}",
            f"IoU threshold: {args.iou}",
            "",
            "D23:",
            f"- wrapped: {total_d23[0]}",
            f"- unwrapped: {total_d23[1]}",
            (
                f"- total: "
                f"{total_d23[0] + total_d23[1]}"
            ),
            "",
            "D50:",
            f"- wrapped: {total_d50[0]}",
            f"- unwrapped: {total_d50[1]}",
            (
                f"- total: "
                f"{total_d50[0] + total_d50[1]}"
            ),
            "",
        ]
    )

    summary_path = output_root / "summary.txt"

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    html_parts = [
        "<!doctype html>",
        "<html lang='ru'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>D23 против D50</title>",
        "<style>",
        "body{font-family:sans-serif;margin:24px;"
        "background:#f4f4f4;color:#111}",
        ".item{background:white;padding:18px;"
        "margin-bottom:24px;border-radius:10px}",
        "img{max-width:100%;height:auto;"
        "border:1px solid #bbb}",
        "code{font-size:14px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>D23 против D50 на 119 изображениях</h1>",
        (
            "<p><b>Слева D23, справа D50.</b> "
            "Зелёный — wrapped, красный — unwrapped.</p>"
        ),
        (
            "<p>Отсутствие рамки у видимой конфеты "
            "означает визуальный пропуск модели.</p>"
        ),
    ]

    for (
        index,
        source_path,
        safe_filename,
        counts_d23,
        counts_d50,
    ) in html_entries:
        html_parts.extend(
            [
                "<div class='item'>",
                (
                    f"<h2>{index:03d}. "
                    f"{html.escape(source_path)}</h2>"
                ),
                (
                    "<p>"
                    f"D23: wrapped={counts_d23[0]}, "
                    f"unwrapped={counts_d23[1]}; "
                    f"D50: wrapped={counts_d50[0]}, "
                    f"unwrapped={counts_d50[1]}"
                    "</p>"
                ),
                (
                    f"<img src='comparisons/"
                    f"{html.escape(safe_filename)}'>"
                ),
                "</div>",
            ]
        )

    html_parts.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    html_path = output_root / "index.html"

    html_path.write_text(
        "\n".join(html_parts),
        encoding="utf-8",
    )

    print()
    print("Готово.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"HTML: {html_path}")
    print(f"D23 images: {d23_root}")
    print(f"D50 images: {d50_root}")
    print(f"Comparisons: {comparisons_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
