#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO


DATASETS = {
    "d23": Path(
        "yolo/datasets/learning_curve/"
        "d23/dataset.yaml"
    ),
    "d50": Path(
        "yolo/datasets/learning_curve/"
        "d50/dataset.yaml"
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only",
        choices=("d23", "d50", "both"),
        default="both",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )

    return parser.parse_args()


def train_model(
    dataset_name: str,
    data_path: Path,
    epochs: int,
    imgsz: int,
    batch: int,
) -> None:
    weights_path = Path(
        "yolo/weights/yolov8n.pt"
    ).resolve()

    data_path = data_path.resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    run_name = f"{dataset_name}_e{epochs}"

    print()
    print("=" * 70)
    print(f"Обучение модели {dataset_name.upper()}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch: {batch}")
    print(f"Run name: {run_name}")
    print("=" * 70)

    started = time.time()

    # Каждый запуск начинает обучение заново
    # с одного и того же yolov8n.pt.
    model = YOLO(str(weights_path))

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=0,
        cache=False,
        val=False,
        patience=0,
        seed=42,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        amp=False,
        save=True,
        save_period=5,
        plots=True,
        project=str(
            Path(
                "yolo/runs/learning_curve"
            ).resolve()
        ),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - started

    last_weights = (
        Path("yolo/runs/learning_curve")
        / run_name
        / "weights"
        / "last.pt"
    )

    if not last_weights.is_file():
        raise RuntimeError(
            f"Не создан файл {last_weights}"
        )

    print()
    print(
        f"{dataset_name.upper()} завершена "
        f"за {elapsed / 60:.1f} мин."
    )
    print(f"Веса: {last_weights}")


def main() -> int:
    args = parse_arguments()

    os.environ["OMP_NUM_THREADS"] = str(
        args.threads
    )

    os.environ["MKL_NUM_THREADS"] = str(
        args.threads
    )

    os.environ["OPENBLAS_NUM_THREADS"] = str(
        args.threads
    )

    torch.set_num_threads(args.threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if args.only == "both":
        selected = ("d23", "d50")
    else:
        selected = (args.only,)

    for dataset_name in selected:
        train_model(
            dataset_name=dataset_name,
            data_path=DATASETS[dataset_name],
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )

    print()
    print("Все выбранные обучения завершены.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
