#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; }

is_docker() {
    if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

if [ -z "${INSTALL_TYPE}" ]; then
    if is_docker; then
        INSTALL_TYPE="minimal"
    else
        INSTALL_TYPE="full"
    fi
fi

log_info "Тип установки: $INSTALL_TYPE"

run_cmd() {
    if is_docker; then
        "$@"
    else
        if command -v sudo >/dev/null 2>&1; then
            sudo "$@"
        else
            "$@"
        fi
    fi
}

apt_install() {
    if is_docker; then
        apt-get update && apt-get install -y --no-install-recommends "$@"
    else
        run_cmd apt update && run_cmd apt install -y "$@"
    fi
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        return 0
    else
        return 1
    fi
}

check_env() {
    if command -v conda &> /dev/null && conda info --envs | grep -q "$1"; then
        return 0
    else
        return 1
    fi
}

if [ "$INSTALL_TYPE" = "full" ]; then
    log_info "Проверка и установка общих зависимостей"

    if ! check_command python3; then
        apt_install python3-venv build-essential libssl-dev libffi-dev python3-dev || log_error "Ошибка установки Python зависимостей"
    else
        log_skip "Python зависимости уже установлены"
    fi
fi

if [ "$INSTALL_TYPE" = "full" ]; then
    log_info "Проверка и установка pyenv"
    if ! check_dir "$HOME/.pyenv"; then
        curl -fsSL https://pyenv.run | bash || log_error "Ошибка установки pyenv"
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    else
        log_skip "pyenv уже установлен"
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi
fi

if [ "$INSTALL_TYPE" = "full" ]; then
    log_info "Установка зависимостей pyenv"
    apt_install make zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev liblzma-dev || log_warning "Ошибка установки некоторых зависимостей pyenv"
fi

if [ "$INSTALL_TYPE" = "full" ]; then
    log_info "Проверка и установка Python 3.10.6"
    if ! pyenv versions | grep -q "3.10.6"; then
        pyenv install 3.10.6 || log_error "Ошибка установки Python"
    else
        log_skip "Python 3.10.6 уже установлен"
    fi

    pyenv global 3.10.6 || log_error "Ошибка настройки Python"
    pyenv shell 3.10.6 || log_error "Ошибка настройки Python"
fi

if [ "$INSTALL_TYPE" = "full" ]; then
    log_info "Установка среды проекта"
    if ! check_command uv; then
        pip install uv || log_error "Ошибка установки uv"
    fi

    if [ ! -d ".venv" ]; then
        uv venv || log_error "Ошибка создания виртуального окружения"
    else
        log_skip "Виртуальное окружение уже существует"
    fi

    if [ -f "pyproject.toml" ] || [ -f "requirements.txt" ]; then
        uv sync --index-strategy unsafe-best-match || log_error "Ошибка синхронизации зависимостей"
    else
        log_warning "Файлы зависимостей не найдены"
    fi
fi

mkdir -p './data'

log_success "Установка завершена успешно!"