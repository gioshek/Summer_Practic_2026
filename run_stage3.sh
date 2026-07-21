#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat <<'EOF'
Использование:
  ./run_stage3.sh add-extra
  ./run_stage3.sh train [raw|sample] [дополнительные аргументы]
  ./run_stage3.sh diagnostic [raw|sample] [--probabilities]
  ./run_stage3.sh predict [sample|raw] [--limit N] [--probabilities]
  ./run_stage3.sh check

Рекомендуемая последовательность этапа 3.1:
  ./run_stage3.sh add-extra
  ./run_stage2.sh check raw
  ./run_stage2.sh annotate raw --start 20260116_155328_028
  ./run_stage2.sh validate raw --require-all
  ./run_stage3.sh train raw
  ./run_stage3.sh check
  ./run_stage3.sh diagnostic raw

После визуальной проверки диагностики:
  ./run_stage3.sh predict sample
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 0
fi

ACTION="$1"
shift
MODE=""
if [[ $# -gt 0 && ( "$1" == "sample" || "$1" == "raw" ) ]]; then
    MODE="$1"
    shift
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f "venv/bin/activate" ]]; then
        source "venv/bin/activate"
        echo "Виртуальное окружение активировано автоматически: $VIRTUAL_ENV"
    else
        echo "Ошибка: venv не активно и venv/bin/activate не найден." >&2
        exit 1
    fi
else
    echo "Виртуальное окружение уже активно: $VIRTUAL_ENV"
fi

mkdir -p models output/reports output/masks output/annotated annotations

case "$ACTION" in
    add-extra)
        INPUT_DIR="data/raw/frontlight"
        if [[ ! -d "$INPUT_DIR" ]]; then
            echo "Ошибка: не найдена папка $INPUT_DIR" >&2
            exit 2
        fi
        python3 src/prepare_annotation_set.py \
            --project-config configs/project.yaml \
            --mode raw \
            --include-list configs/stage3_1_extra_images.txt \
            --skip-missing \
            "$@"
        ;;
    train)
        MODE="${MODE:-raw}"
        INPUT_DIR="data/${MODE}/frontlight"
        if [[ ! -d "$INPUT_DIR" ]]; then
            echo "Ошибка: не найдена папка $INPUT_DIR" >&2
            exit 2
        fi
        python3 src/train_pixel_segmenter.py \
            --project-config configs/project.yaml \
            --config configs/stage3.yaml \
            --mode "$MODE" \
            "$@"
        ;;
    diagnostic)
        MODE="${MODE:-raw}"
        INPUT_DIR="data/${MODE}/frontlight"
        if [[ ! -d "$INPUT_DIR" ]]; then
            echo "Ошибка: не найдена папка $INPUT_DIR" >&2
            exit 2
        fi
        python3 src/predict_pixel_segmenter.py \
            --project-config configs/project.yaml \
            --config configs/stage3.yaml \
            --mode "$MODE" \
            --include-list configs/stage3_1_diagnostic.txt \
            --skip-missing \
            "$@"
        ;;
    predict)
        MODE="${MODE:-sample}"
        INPUT_DIR="data/${MODE}/frontlight"
        if [[ ! -d "$INPUT_DIR" ]]; then
            echo "Ошибка: не найдена папка $INPUT_DIR" >&2
            exit 2
        fi
        python3 src/predict_pixel_segmenter.py \
            --project-config configs/project.yaml \
            --config configs/stage3.yaml \
            --mode "$MODE" \
            "$@"
        ;;
    check)
        python3 src/check_pixel_model.py --project-config configs/project.yaml "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Неизвестное действие: $ACTION" >&2
        usage
        exit 3
        ;;
esac
