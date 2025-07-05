FROM python:3.11-slim-buster

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/airflow # Default AIRFLOW_HOME inside the container

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .

RUN pip install --no-cache-dir --compile --disable-pip-version-check --default-timeout=100 -r requirements.txt

COPY . .