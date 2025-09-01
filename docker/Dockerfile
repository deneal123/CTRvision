ARG APP_TAG=${APP_TAG}
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    INSTALL_TYPE="minimal" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    CUDA_HOME=/usr/local/cuda

WORKDIR $APP_HOME

# Установка Python 3.10 и системных зависимостей
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-venv \
        build-essential \
        curl \
        git \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && pip3 install --no-cache-dir --upgrade pip

RUN mkdir -p /app/logs /app/.cache \
    && chmod -R 777 /app/logs /app/.cache

RUN pip install --no-cache-dir uv \
    && uv --version

COPY pyproject.toml ./

RUN uv python pin 3.10.6
RUN uv venv
RUN uv pip install setuptools wheel --upgrade pip
RUN uv sync --no-build-isolation --index-strategy unsafe-best-match

COPY . .

COPY install.sh /usr/local/bin/install.sh
RUN chmod +x /usr/local/bin/install.sh

ENTRYPOINT ["/usr/local/bin/install.sh"]