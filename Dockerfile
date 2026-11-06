# syntax=docker/dockerfile:1.7
# One Dockerfile, one slim image per role: each service passes the extras it needs (EXTRAS),
# so the API image carries no Airflow and the demo carries no Airflow or SHAP.
#
#   docker build --build-arg EXTRAS="--extra api" -t aeroscrape/api .
#
# Without network access to PyPI, tools/build_offline.sh builds the same images from the
# local uv cache (target runtime-prebuilt).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ---------------------------------------------------------------------------- base
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    AEROSCRAPE_HOME=/opt/aeroscrape \
    AIRFLOW_HOME=/opt/airflow
# libgomp1: LightGBM's OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# A non-root user whose IDs match the host user, so files written to bind mounts stay yours
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} app \
    && useradd --uid ${UID} --gid ${GID} --create-home app \
    && mkdir -p /opt/aeroscrape /opt/airflow \
    && chown app:app /opt/aeroscrape /opt/airflow

# ---------------------------------------------------------------------------- builder
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300
ARG EXTRAS=""
WORKDIR /build
# Dependencies first, so editing code doesn't reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS}
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS}

# ---------------------------------------------------------------------------- runtime
FROM base AS runtime
COPY --from=builder /opt/venv /opt/venv
WORKDIR /opt/aeroscrape
# The configuration is part of the image; compose mounts the working copy over it
COPY --chown=app:app config.yaml ./
COPY --chown=app:app .streamlit ./.streamlit
USER app

# ---------------------------------------------------------------------------- offline variant
# The same runtime with a virtualenv built outside `docker build` (tools/build_offline.sh
# passes it as --build-context prebuilt-venv=<dir>).
FROM scratch AS prebuilt-venv

FROM base AS runtime-prebuilt
COPY --from=prebuilt-venv / /opt/venv
WORKDIR /opt/aeroscrape
COPY --chown=app:app config.yaml ./
COPY --chown=app:app .streamlit ./.streamlit
USER app
