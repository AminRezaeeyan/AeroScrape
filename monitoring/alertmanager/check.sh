#!/bin/sh
# Render the config the way entrypoint.sh does, with the test credentials passed in, and validate it.
# Used by `make check-monitoring` and CI.
set -eu
sed -e "s|\${TELEGRAM_BOT_TOKEN}|${TELEGRAM_BOT_TOKEN}|" -e "s|\${TELEGRAM_CHAT_ID}|${TELEGRAM_CHAT_ID}|" \
  /etc/alertmanager/alertmanager.yml.tmpl > /tmp/alertmanager.yml
amtool check-config /tmp/alertmanager.yml
