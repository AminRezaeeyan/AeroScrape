#!/usr/bin/env bash
# Build the Docker images without network access, from the local uv cache.
#
#   tools/build_offline.sh               # all images
#   tools/build_offline.sh api demo      # some of them
#
# For each image, a throwaway container of the runtime base image bind-mounts the uv cache
# (no copy) and installs the locked dependencies offline into a virtualenv on the host; the
# Dockerfile's runtime-prebuilt target then copies that virtualenv into the image. Needs a
# uv cache that holds every locked package, which `make setup` produces.
set -euo pipefail

declare -A EXTRAS=(
  [core]=""
  [api]="--extra api"
  [demo]="--extra ui"
  [airflow]="--extra airflow --extra api"
)
BASE_IMAGE=python:3.12-slim-bookworm
UV_BIN=${UV_BIN:-$(command -v uv)}
UV_CACHE=${UV_CACHE:-$HOME/.cache/uv}
images=("$@")
[ ${#images[@]} -eq 0 ] && images=(core api demo airflow)

cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

for image in "${images[@]}"; do
  [ -n "${EXTRAS[$image]+set}" ] || { echo "unknown image: $image" >&2; exit 1; }
  echo "== aeroscrape/$image"
  mkdir -p "$work/$image/src" "$work/$image/venv"
  # A writable copy of the project: building the package writes next to its sources
  cp -r pyproject.toml uv.lock README.md src "$work/$image/src/"
  docker run --rm --user "$(id -u):$(id -g)" \
    -v "$UV_BIN:/usr/local/bin/uv:ro" \
    -v "$UV_CACHE:/uv-cache" \
    -v "$work/$image/src:/build" \
    -v "$work/$image/venv:/opt/venv" \
    -e UV_CACHE_DIR=/uv-cache -e UV_PROJECT_ENVIRONMENT=/opt/venv -e UV_OFFLINE=1 \
    -e UV_COMPILE_BYTECODE=1 -e UV_LINK_MODE=copy -e HOME=/tmp \
    -w /build "$BASE_IMAGE" \
    uv sync --frozen --no-dev --no-editable ${EXTRAS[$image]}
  docker build --build-context prebuilt-venv="$work/$image/venv" --target runtime-prebuilt \
    -t "aeroscrape/$image:latest" .
  rm -rf -- "$work/$image"
done
