#!/bin/sh
# Render the Alertmanager config from the template and start Alertmanager.
# Without a real bot token and chat id (.env still has the placeholders), alerts are kept in
# Alertmanager's UI (http://localhost:9093) but sent nowhere, so the stack runs without a bot.
set -eu
config=/tmp/alertmanager.yml
case "${TELEGRAM_BOT_TOKEN:-}${TELEGRAM_CHAT_ID:-}" in
  "" | *SET_YOUR*)
    echo "entrypoint: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; alerts are shown in the UI only" >&2
    cat > "$config" <<'YAML'
route: {receiver: ui-only, group_by: [alertname, airport]}
receivers: [{name: ui-only}]
YAML
    ;;
  *)
    sed -e "s|\${TELEGRAM_BOT_TOKEN}|${TELEGRAM_BOT_TOKEN}|" -e "s|\${TELEGRAM_CHAT_ID}|${TELEGRAM_CHAT_ID}|" \
      /etc/alertmanager/alertmanager.yml.tmpl > "$config"
    ;;
esac
exec /bin/alertmanager --config.file="$config" --storage.path=/alertmanager "$@"
