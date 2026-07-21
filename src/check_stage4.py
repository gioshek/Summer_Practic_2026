from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import load_yaml, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка результатов этапа №4.3.")
    parser.add_argument("--project-config", default="configs/project.yaml")
    return parser.parse_args()


def count_files(path: Path, suffixes: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.project_config))
    output_root = project_path(config, "output", "output").resolve()
    report_root = output_root / "reports"

    images_csv = report_root / "stage4_images.csv"
    candidates_csv = report_root / "stage4_candidates.csv"
    summary_path = report_root / "stage4_summary.txt"

    rows: list[dict[str, str]] = []
    if images_csv.exists():
        with images_csv.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    error_rows = [row for row in rows if row.get("error")]
    candidate_count = sum(int(row.get("candidate_count") or 0) for row in rows if not row.get("error"))
    marker_count = sum(int(row.get("marker_count") or 0) for row in rows if not row.get("error"))
    removed_sensor_border_pixels = sum(
        int(row.get("removed_sensor_border_pixels") or 0)
        for row in rows
        if not row.get("error")
    )
    sticker_count = sum(row.get("valid_sticker") == "yes" for row in rows)

    print(f"Отчёт по изображениям: {images_csv}")
    print(f"Существует: {'да' if images_csv.exists() else 'нет'}")
    print(f"Отчёт по кандидатам: {candidates_csv}")
    print(f"Существует: {'да' if candidates_csv.exists() else 'нет'}")
    print(f"Сводка: {summary_path}")
    print(f"Существует: {'да' if summary_path.exists() else 'нет'}")
    print(f"Строк изображений: {len(rows)}")
    print(f"Ошибок: {len(error_rows)}")
    print(f"Подтверждённых стикеров: {sticker_count}")
    print(f"Радиально-симметричных маркеров: {marker_count}")
    print(f"Центров-кандидатов: {candidate_count}")
    print(f"Удалено пикселей тёмных пограничных полос: {removed_sensor_border_pixels}")
    print(f"Очищенных масок: {count_files(output_root / 'cleaned_masks', {'.png'})}")
    print(f"Масок стикера: {count_files(output_root / 'sticker_masks', {'.png'})}")
    print(f"Масок маркеров: {count_files(output_root / 'marker_masks', {'.png'})}")
    print(f"Просмотровых маркеров: {count_files(output_root / 'marker_previews', {'.png'})}")
    print(f"Масок экземпляров: {count_files(output_root / 'instance_labels', {'.png'})}")
    print(f"Аннотированных изображений: {count_files(output_root / 'annotated', {'.jpg', '.jpeg', '.png'})}")

    required = [images_csv, candidates_csv, summary_path]
    return 0 if all(path.exists() for path in required) and not error_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
