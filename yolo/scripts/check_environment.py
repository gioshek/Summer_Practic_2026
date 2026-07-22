from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "yolo" / "weights" / "yolov8n.pt"
SAMPLE_ROOT = PROJECT_ROOT / "data" / "sample" / "frontlight"


def find_sample_image() -> Path:
    extensions = {".bmp", ".png", ".jpg", ".jpeg"}

    images = sorted(
        path
        for path in SAMPLE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )

    if not images:
        raise FileNotFoundError(
            f"В папке нет тестовых изображений: {SAMPLE_ROOT}"
        )

    return images[0]


def main() -> None:
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Weights: {WEIGHTS_PATH}")

    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Не найдены веса YOLOv8n: {WEIGHTS_PATH}"
        )

    sample_image = find_sample_image()
    print(f"Sample image: {sample_image}")

    model = YOLO(str(WEIGHTS_PATH))

    results = model.predict(
        source=str(sample_image),
        device="cpu",
        imgsz=640,
        verbose=False,
    )

    if not results:
        raise RuntimeError("YOLO не вернула результат.")

    detections = len(results[0].boxes)

    print(f"Обнаружено объектов COCO-моделью: {detections}")
    print("Окружение YOLOv8 настроено корректно.")


if __name__ == "__main__":
    main()
