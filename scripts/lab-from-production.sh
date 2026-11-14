#!/usr/bin/env bash
# Run an experiment against a throwaway copy of the production state (#383).
#
# Production is never touched: the copy lives under SUZKA_LAB_BASE, the lab
# API listens on its own loopback port, and the copy is deleted afterwards.
# The lab runs the deployed code and model, so it needs the same GPU memory
# again (about 7.7 GB for gemma-4-12B-it in 4 bit); keep experiments short.
#
# Usage (on the deploy host, as a user with sudo):
#   scripts/lab-from-production.sh [--port N] [--timeout SECONDS] [--keep] -- CMD...
#
# CMD runs as the invoking user with SUZKA_LAB_URL (the lab API base URL) and
# SUZKA_LAB_DIR set. It is killed after --timeout seconds (default 3600).
# The copy is taken while production runs; if it caught a write midway, the
# lab may refuse to start, which never affects production.
set -euo pipefail

PROD_ROOT="${SUZKA_PROD_ROOT:-/opt/project-suzka}"
PROD_CONFIG="${SUZKA_PROD_CONFIG:-/etc/project-suzka/config.yaml}"
PROD_ENV="${SUZKA_PROD_ENV:-/etc/project-suzka/backend.env}"
SERVICE="${SUZKA_SERVICE:-suzka-api}"
SERVICE_USER="${SUZKA_SERVICE_USER:-suzka}"
LAB_BASE="${SUZKA_LAB_BASE:-/var/lib/suzka/lab}"
PORT=8101
TIMEOUT=3600
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --) shift; break ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ $# -eq 0 ]]; then
  printf 'usage: %s [--port N] [--timeout S] [--keep] -- CMD...\n' "$0" >&2
  exit 2
fi
PROD_PORT="$(sudo sed -n 's/^  port: *\([0-9]*\).*/\1/p' "${PROD_CONFIG}" | head -1)"
if [[ "${PORT}" == "${PROD_PORT}" ]]; then
  printf 'lab port %s is the production port\n' "${PORT}" >&2
  exit 2
fi
if curl -s -o /dev/null "http://127.0.0.1:${PORT}/health"; then
  printf 'port %s is already in use\n' "${PORT}" >&2
  exit 2
fi
# Every state path must be relative, so it resolves inside the lab copy.
if sudo grep -nE "${PROD_ROOT}|: +/" "${PROD_CONFIG}"; then
  printf 'config has absolute paths; the lab could write production state\n' >&2
  exit 2
fi

sudo install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 700 "${LAB_BASE}"
LAB="$(sudo -u "${SERVICE_USER}" mktemp -d "${LAB_BASE}/lab-$(date +%Y%m%d-%H%M%S)-XXXX")"
SERVER_PID=""

# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    sudo kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do sudo kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    sudo kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ "${KEEP}" == 1 ]]; then
    printf 'lab kept at %s\n' "${LAB}" >&2
  else
    sudo rm -rf --one-file-system "${LAB}"
  fi
}
trap cleanup EXIT

sudo cp -a "${PROD_ROOT}/.suzka" "${LAB}/.suzka"
sudo rm -f "${LAB}/.suzka/.event_journal.jsonl.lock"
sudo sed -E "s/^(  port:) *[0-9]+/\1 ${PORT}/" "${PROD_CONFIG}" | sudo tee "${LAB}/config.yaml" >/dev/null
sudo install -m 600 "${PROD_ENV}" "${LAB}/backend.env"
sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" "${LAB}"

# The service's own environment (library path, uv/python, HOME for the model cache).
mapfile -t SERVICE_ENV < <(systemctl show "${SERVICE}" -p Environment --value | tr ' ' '\n' | grep -v '^$')
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/GPU before lab: /' >&2 || true

# SERVER_PID is sudo's; sudo relays the cleanup's SIGTERM to the server.
sudo -u "${SERVICE_USER}" env -i "${SERVICE_ENV[@]}" \
  "$(command -v bash)" -c 'set -a; source "$1/backend.env"; set +a; cd "$1" && exec env SUZKA_CONFIG_PATH="$1/config.yaml" PYTHONPATH="$2" "$2/.venv/bin/python" -m suzka.api.server >"$1/server.log" 2>&1' \
  _ "${LAB}" "${PROD_ROOT}" &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" | grep -q 200; then
    break
  fi
  if ! sudo kill -0 "${SERVER_PID}" 2>/dev/null; then
    printf 'lab server exited during startup\n' >&2
    sudo tail -20 "${LAB}/server.log" >&2 || true
    exit 1
  fi
  sleep 5
done
printf 'lab API at http://127.0.0.1:%s (state %s)\n' "${PORT}" "${LAB}" >&2

status=0
SUZKA_LAB_URL="http://127.0.0.1:${PORT}" SUZKA_LAB_DIR="${LAB}" \
  timeout "${TIMEOUT}" "$@" || status=$?
exit "${status}"
