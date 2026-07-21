#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat <<'EOF'
Использование:
  ./run_stage4.sh process [sample|raw] [--limit N] [--debug]
  ./run_stage4.sh diagnostic [sample|raw] [--debug]
  ./run_stage4.sh check
  ./run_stage4.sh smoke

Основная проверка этапа №4:
  ./run_stage3.sh predict sample
  ./run_stage4.sh process sample --debug
  ./run_stage4.sh check
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

mkdir -p output/reports

case "$ACTION" in
    process)
        MODE="${MODE:-sample}"
        python3 src/postprocess_candidates.py \
            --project-config configs/project.yaml \
            --config configs/stage4.yaml \
            --mode "$MODE" \
            "$@"
        ;;
    diagnostic)
        MODE="${MODE:-raw}"
        python3 src/postprocess_candidates.py \
            --project-config configs/project.yaml \
            --config configs/stage4.yaml \
            --mode "$MODE" \
            --include-list configs/stage3_1_diagnostic.txt \
            --skip-missing \
            "$@"
        ;;
    check)
        python3 src/check_stage4.py --project-config configs/project.yaml "$@"
        ;;
    smoke)
        python3 tests/smoke_test_stage4.py
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
