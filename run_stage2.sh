#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat <<'EOF'
Использование:
  ./run_stage2.sh prepare  [sample|raw] [дополнительные аргументы]
  ./run_stage2.sh check    [sample|raw]
  ./run_stage2.sh annotate [sample|raw] [дополнительные аргументы]
  ./run_stage2.sh validate [sample|raw] [--require-all] [--previews]

Рекомендуемый первый запуск:
  ./run_stage2.sh prepare sample
  ./run_stage2.sh check sample
  ./run_stage2.sh annotate sample
  ./run_stage2.sh validate sample --require-all

Примеры:
  ./run_stage2.sh prepare sample --count 12
  ./run_stage2.sh annotate sample --start RGB_green
  ./run_stage2.sh validate sample --previews
  ./run_stage2.sh prepare raw --count 20
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 0
fi

ACTION="$1"
shift
MODE="sample"
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

if [[ "$MODE" == "sample" ]]; then
    INPUT_DIR="data/sample/frontlight"
else
    INPUT_DIR="data/raw/frontlight"
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Ошибка: не найдена папка $INPUT_DIR" >&2
    exit 2
fi

mkdir -p annotations/pixel_masks annotations/metadata models output/reports tests

COMMON=(--project-config configs/project.yaml --mode "$MODE")

case "$ACTION" in
    prepare)
        python3 src/prepare_annotation_set.py "${COMMON[@]}" "$@"
        ;;
    check)
        python3 src/annotate.py "${COMMON[@]}" --config configs/stage2.yaml --check "$@"
        ;;
    annotate)
        python3 src/annotate.py "${COMMON[@]}" --config configs/stage2.yaml "$@"
        ;;
    validate)
        python3 src/validate_annotations.py "${COMMON[@]}" --config configs/stage2.yaml "$@"
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
