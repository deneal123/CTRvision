#!/bin/bash

set -e 

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1" >&2; }
info() { echo -e "${GREEN}[INFO]${NC} $1"; }

if [ $# -eq 0 ]; then
    error "Usage: $0 <path_to_script.py> [script_arguments...]"
fi

SCRIPT_PATH="$1"
shift

if [ ! -f "$SCRIPT_PATH" ]; then
    error "Script file '$SCRIPT_PATH' not found!"
fi

if [[ ! "$SCRIPT_PATH" =~ \.py$ ]]; then
    warning "File '$SCRIPT_PATH' doesn't have .py extension. Continuing anyway."
fi

export PROJECT_PATH="$(pwd)/src"
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"

VENV_PATH="./.venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    info "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    warning "Virtual environment $VENV_PATH not found. Running without activation."
fi

TORCH_LIB_PATH="$VENV_PATH/lib/python3.10/site-packages/torch/lib"
if [ -d "$TORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TORCH_LIB_PATH"
fi

info "Running script: $SCRIPT_PATH"
info "With arguments: $@"
uv run --project "$PROJECT_PATH" --active "$SCRIPT_PATH" "$@"