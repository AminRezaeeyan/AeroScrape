FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/airflow \
    AEROSCRAPE_HOME=/opt/airflow \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# libgomp1 is required by LightGBM, curl by the compose health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --disable-pip-version-check uv==0.12.3

WORKDIR /opt/airflow

# Install locked dependencies first so code changes don't invalidate this layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --all-extras --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --all-extras --no-dev
