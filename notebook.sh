#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PROJECT_ROOT}/.venv/lib/python3.10/site-packages/torch/lib"
cd "$PROJECT_ROOT"
uv run --project "$PROJECT_ROOT" jupyter notebook --notebook-dir="$PROJECT_ROOT/src/notebooks" \
    --allow-root --IdentityProvider.token='' \
    --ServerApp.password='' --ServerApp.allow_origin='*'
