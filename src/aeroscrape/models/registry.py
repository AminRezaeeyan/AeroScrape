"""Registry operations: which model version serves, and moving it.

python -m aeroscrape.models.registry status
python -m aeroscrape.models.registry rollback 7    # make version 7 the production model
"""

import argparse
import logging
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from aeroscrape.config import get_app_config

logger = logging.getLogger(__name__)

CHAMPION = "production"
CHALLENGER = "challenger"


def client_and_name(config: dict[str, Any] | None = None) -> tuple[MlflowClient, str]:
    config = config or get_app_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    return MlflowClient(), config["pipeline"]["model_registry_name"]


def alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        # Versions come back as int or str depending on the tracking store
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def promote(client: MlflowClient, name: str, version: str, reason: str) -> None:
    """Point the production alias at `version`, recording the previous version on it."""
    previous = alias_version(client, name, CHAMPION)
    client.set_registered_model_alias(name, CHAMPION, version)
    client.set_model_version_tag(name, version, "promoted_reason", reason)
    if previous and previous != version:
        client.set_model_version_tag(name, version, "replaced_version", previous)
    logger.info("%s: production %s -> %s (%s)", name, previous, version, reason)


def status(client: MlflowClient, name: str) -> str:
    lines = [f"{name}:"]
    for alias in (CHAMPION, CHALLENGER):
        version = alias_version(client, name, alias)
        if version is None:
            lines.append(f"  {alias:<11} -")
            continue
        tags = client.get_model_version(name, version).tags
        lines.append(
            f"  {alias:<11} version {version}  commit {tags.get('git_commit', '?')[:8]}  "
            f"gate {tags.get('gate_decision', '-')}"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Inspect or move the production model alias.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Show the production and challenger versions")
    rollback = sub.add_parser("rollback", help="Make an earlier version the production model")
    rollback.add_argument("version", help="Model version to serve")
    rollback.add_argument("--reason", default="manual rollback", help="Recorded on the model version")
    args = parser.parse_args()

    client, name = client_and_name()
    if args.command == "rollback":
        client.get_model_version(name, args.version)  # fails clearly if the version doesn't exist
        promote(client, name, args.version, reason=args.reason)
    print(status(client, name))


if __name__ == "__main__":
    main()
