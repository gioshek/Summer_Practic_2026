#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "feature/yolov8" ]]; then
    echo "ОШИБКА: текущая ветка — $BRANCH, ожидалась feature/yolov8"
    exit 1
fi

echo "Подготовка материалов для Git..."

mkdir -p \
    docs/images/examples \
    docs/images/annotations \
    yolo/release/models \
    yolo/release/annotations \
    yolo/release/training/d23_e25 \
    yolo/release/training/d50_e25

python yolo/scripts/export_training_annotation_previews.py

cp -f \
    yolo/runs/learning_curve/d23_e25/weights/last.pt \
    yolo/release/models/d23_e25_last.pt

cp -f \
    yolo/runs/learning_curve/d50_e25/weights/last.pt \
    yolo/release/models/d50_e25_last.pt

for archive in \
    candy_pilot_23_complete_v1.zip \
    candy_train_to_50_complete_v1.zip \
    holdout119_unwrapped_gt_v1.zip
do
    cp -f \
        "yolo/annotation/cvat_exports/$archive" \
        "yolo/release/annotations/$archive"
done

TRAIN_FILES=(
    args.yaml
    results.csv
    results.png
    labels.jpg
    confusion_matrix.png
    confusion_matrix_normalized.png
    BoxF1_curve.png
    BoxP_curve.png
    BoxPR_curve.png
    BoxR_curve.png
)

for dataset in d23 d50
do
    source_dir="yolo/runs/learning_curve/${dataset}_e25"
    target_dir="yolo/release/training/${dataset}_e25"

    for filename in "${TRAIN_FILES[@]}"
    do
        cp -f \
            "$source_dir/$filename" \
            "$target_dir/$filename"
    done
done

copy_example() {
    local source_pattern="$1"
    local destination="$2"
    local source_file

    source_file="$(find yolo/evaluation/holdout119_d23_vs_d50/comparisons \
        -maxdepth 1 -type f -name "$source_pattern" -print -quit)"

    if [[ -z "$source_file" ]]; then
        echo "ПРЕДУПРЕЖДЕНИЕ: не найден пример $source_pattern"
        return
    fi

    cp -f "$source_file" "docs/images/examples/$destination"
}

copy_example "005__*.jpg" "comparison_005_dark.jpg"
copy_example "035__*.jpg" "comparison_035_dense.jpg"
copy_example "068__*.jpg" "comparison_068_many.jpg"
copy_example "080__*.jpg" "comparison_080_mixed.jpg"

D23_RESULT_COUNT="$(
    find yolo/evaluation/holdout119_d23_vs_d50/d23 \
        -maxdepth 1 -type f -name '*.jpg' | wc -l
)"

D50_RESULT_COUNT="$(
    find yolo/evaluation/holdout119_d23_vs_d50/d50 \
        -maxdepth 1 -type f -name '*.jpg' | wc -l
)"

D23_PREVIEW_COUNT="$(
    find docs/images/annotations/d23 \
        -maxdepth 1 -type f -name '[0-9][0-9][0-9]__*.jpg' | wc -l
)"

D50_PREVIEW_COUNT="$(
    find docs/images/annotations/d50 \
        -maxdepth 1 -type f -name '[0-9][0-9][0-9]__*.jpg' | wc -l
)"

echo
echo "Проверка количества файлов:"
echo "- разметка D23: $D23_PREVIEW_COUNT"
echo "- разметка D50: $D50_PREVIEW_COUNT"
echo "- holdout D23: $D23_RESULT_COUNT"
echo "- holdout D50: $D50_RESULT_COUNT"

[[ "$D23_PREVIEW_COUNT" -eq 23 ]]
[[ "$D50_PREVIEW_COUNT" -eq 50 ]]
[[ "$D23_RESULT_COUNT" -eq 119 ]]
[[ "$D50_RESULT_COUNT" -eq 119 ]]

echo
echo "Размеры подготовленных материалов:"
du -sh \
    docs/images \
    yolo/release \
    yolo/evaluation/holdout119_d23_vs_d50/d23 \
    yolo/evaluation/holdout119_d23_vs_d50/d50

echo
echo "Подготовка завершена успешно."
