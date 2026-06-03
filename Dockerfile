FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/airflow
RUN echo "deb http://mirror-linux.runflare.com/debian bookworm main" > /etc/apt/sources.list.d/runflare.list && \
    echo "deb http://mirror-linux.runflare.com/debian-security bookworm-security main" >> /etc/apt/sources.list.d/runflare.list && \
    rm -f /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        wget \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/airflow

COPY requirements.txt .

RUN pip install --no-cache-dir --compile --disable-pip-version-check --default-timeout=120 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Create project folders
RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/data /opt/airflow/models /opt/airflow/scripts /opt/airflow/utils

COPY . .
