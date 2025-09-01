#!/bin/bash

# CTRvision Notebook Runner Script
# Запускает эксперименты из блокнота CTRvision
# Этот скрипт находится по пути, указанному в требованиях: src/notebook/ctr-vision.sh

echo "🎯 CTRvision: Запуск экспериментов по анализу эффективности продуктовых изображений"
echo "============================================================================="

# Установка переменных окружения
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NOTEBOOK_PATH="$PROJECT_ROOT/src/notebooks/ctr-vision.ipynb"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

echo "📁 Корневая директория проекта: $PROJECT_ROOT"
echo "📁 Блокнот расположен: $NOTEBOOK_PATH"

# Переход в корневую директорию проекта
cd "$PROJECT_ROOT"

echo ""
echo "🚀 Запуск блокнота с экспериментами..."
echo ""

# Проверяем существование блокнота
if [[ ! -f "$NOTEBOOK_PATH" ]]; then
    echo "❌ Блокнот не найден: $NOTEBOOK_PATH"
    exit 1
fi

# Опции запуска:
# 1. Запустить Jupyter notebook
# 2. Использовать существующий notebook.sh скрипт из корня проекта

echo "📋 Доступные варианты запуска:"
echo "  1. Запустить Jupyter notebook напрямую"
echo "  2. Использовать проектный скрипт notebook.sh"
echo "  3. Конвертировать в Python и запустить"
echo ""
echo "Выберите вариант (1-3) или нажмите Enter для варианта 1:"
read -r choice

case $choice in
    2)
        echo "🔧 Используем проектный скрипт..."
        if [[ -f "$PROJECT_ROOT/notebook.sh" ]]; then
            bash "$PROJECT_ROOT/notebook.sh"
        else
            echo "❌ Файл notebook.sh не найден в корне проекта"
        fi
        ;;
    3)
        echo "📝 Конвертируем блокнот в Python скрипт..."
        if command -v jupyter &> /dev/null; then
            jupyter nbconvert --to python "$NOTEBOOK_PATH" --output "/tmp/ctr-vision-experiments.py"
            echo "✅ Конвертированный скрипт: /tmp/ctr-vision-experiments.py"
            echo "🏃 Запускаем..."
            python "/tmp/ctr-vision-experiments.py"
        else
            echo "❌ Jupyter не найден для конвертации"
        fi
        ;;
    *)
        echo "📓 Запускаем Jupyter notebook..."
        if command -v jupyter &> /dev/null; then
            echo "🌐 Блокнот будет доступен в браузере"
            echo "📂 Автоматически откроется: $NOTEBOOK_PATH"
            jupyter notebook "$NOTEBOOK_PATH" --allow-root --ip=0.0.0.0 --port=8888 --no-browser
        else
            echo "❌ Jupyter не найден. Установите его:"
            echo "   pip install jupyter"
            echo ""
            echo "📋 Альтернативно, запустите эксперименты через основной скрипт:"
            echo "   python src/scripts/main.py --step download"
            echo "   python src/scripts/main.py --step train"  
            echo "   python src/scripts/main.py --step plot"
        fi
        ;;
esac

echo ""
echo "🎊 Готово! Работа с CTRvision завершена."