#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

MODE="sample"
PREVIEWS=0
HASHES=0

for argument in "$@"; do
    case "$argument" in
        sample|raw)
            MODE="$argument"
            ;;
        --previews)
            PREVIEWS=1
            ;;
        --hashes)
            HASHES=1
            ;;
        -h|--help)
            cat <<'EOF'
Использование:
  ./run_stage1.sh [sample|raw] [--previews] [--hashes]

Примеры:
  ./run_stage1.sh
  ./run_stage1.sh sample
  ./run_stage1.sh raw
  ./run_stage1.sh sample --previews
  ./run_stage1.sh raw --hashes --previews
EOF
            exit 0
            ;;
        *)
            echo "Неизвестный аргумент: $argument" >&2
            exit 2
            ;;
    esac
done

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
    exit 3
fi

mkdir -p output/reports annotations models tests

ARGS=(--config configs/project.yaml --mode "$MODE")
if [[ "$PREVIEWS" -eq 1 ]]; then
    ARGS+=(--previews)
fi
if [[ "$HASHES" -eq 1 ]]; then
    ARGS+=(--hashes)
fi

echo
echo "Проверка окружения"
python3 src/check_environment.py

echo
echo "Аудит режима $MODE: $INPUT_DIR"
python3 src/audit_dataset.py "${ARGS[@]}"

echo
echo "Этап 1 завершён."
echo "Таблица: output/reports/dataset.csv"
echo "Сводка: output/reports/summary.txt"
if [[ "$PREVIEWS" -eq 1 ]]; then
    echo "Обзорные листы: output/previews"
fi
