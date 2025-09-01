#!/bin/bash

# CTRvision Notebook Runner Script
# Запускает эксперименты из блокнота CTRvision

echo "🎯 CTRvision: Запуск экспериментов по анализу эффективности продуктовых изображений"
echo "============================================================================="

# Установка переменных окружения
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

echo "📁 Корневая директория проекта: $PROJECT_ROOT"
echo "📁 Директория скриптов: $SCRIPT_DIR"

# Переход в корневую директорию проекта
cd "$PROJECT_ROOT"

echo ""
echo "🚀 Запуск блокнота с экспериментами..."
echo ""

# Опции запуска:
# 1. Запустить Jupyter notebook
# 2. Конвертировать и запустить как Python скрипт

if command -v jupyter &> /dev/null; then
    echo "📓 Найден Jupyter, запускаем блокнот..."
    echo "🌐 Откройте браузер и перейдите по адресу, который будет показан ниже"
    echo ""
    jupyter notebook "$SCRIPT_DIR/ctr-vision.ipynb" --allow-root --ip=0.0.0.0 --port=8888
else
    echo "❌ Jupyter не найден. Попробуйте установить его:"
    echo "   pip install jupyter"
    echo ""
    echo "🔄 Альтернативно, можно конвертировать блокнот в Python скрипт:"
    
    if command -v nbconvert &> /dev/null; then
        echo "📝 Конвертируем блокнот в Python скрипт..."
        jupyter nbconvert --to python "$SCRIPT_DIR/ctr-vision.ipynb" --output "$SCRIPT_DIR/ctr-vision-converted.py"
        echo "✅ Конвертированный скрипт: $SCRIPT_DIR/ctr-vision-converted.py"
        echo ""
        echo "🏃 Запустить конвертированный скрипт? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            python "$SCRIPT_DIR/ctr-vision-converted.py"
        fi
    else
        echo "❌ nbconvert не найден. Установите: pip install nbconvert"
        echo ""
        echo "📋 Для запуска экспериментов вручную:"
        echo "   1. cd $PROJECT_ROOT"
        echo "   2. python src/scripts/main.py --step download"
        echo "   3. python src/scripts/main.py --step train"  
        echo "   4. python src/scripts/main.py --step plot"
    fi
fi

echo ""
echo "🎊 Готово! Эксперименты CTRvision завершены."