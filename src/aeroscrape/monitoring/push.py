"""Push the metrics of batch jobs to the Prometheus Pushgateway.

A batch job exits before Prometheus could scrape it, so it pushes its numbers instead. Each
push replaces the job's previous metrics, so a label that disappears (an airport with no
rows this time) doesn't linger. Monitoring must never break the job it watches: a failed
push is logged and ignored.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from aeroscrape.config import get_app_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaugeValue:
    name: str
    help: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)


def pushgateway_address(config: dict[str, Any] | None = None) -> str:
    """PUSHGATEWAY_URL, else monitoring.pushgateway in config.yaml; empty disables pushing."""
    if "PUSHGATEWAY_URL" in os.environ:
        return os.environ["PUSHGATEWAY_URL"]
    config = config if config is not None else get_app_config()
    return config.get("monitoring", {}).get("pushgateway") or ""


def build_registry(values: list[GaugeValue]) -> CollectorRegistry:
    registry = CollectorRegistry()
    gauges: dict[str, Gauge] = {}
    for v in values:
        if v.name not in gauges:
            gauges[v.name] = Gauge(v.name, v.help, sorted(v.labels), registry=registry)
        gauge = gauges[v.name]
        (gauge.labels(**v.labels) if v.labels else gauge).set(v.value)
    return registry


def push_gauges(job: str, values: list[GaugeValue], config: dict[str, Any] | None = None) -> bool:
    """Replace the job's metrics on the Pushgateway. Returns whether the push succeeded."""
    address = pushgateway_address(config)
    if not address or not values:
        return False
    try:
        push_to_gateway(address, job=job, registry=build_registry(values), timeout=10)
        return True
    except Exception as e:
        logger.warning("Could not push %s metrics to %s: %s", job, address, e)
        return False
