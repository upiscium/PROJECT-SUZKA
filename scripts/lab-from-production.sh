#!/usr/bin/env bash
# Run an experiment against a throwaway copy of the production state (#383).
#
# Production is never touched: the copy lives under SUZKA_LAB_BASE, the lab
# API runs as its own transient systemd unit on its own loopback port, and
# both are removed afterwards. The lab runs the deployed code and model, so it
# needs the same GPU memory again (about 7.7 GB for gemma-4-12B-it in 4 bit);
# keep experiments short.
#
# Usage (on the deploy host, as a user with sudo):
#   scripts/lab-from-production.sh [--port N] [--timeout SECONDS] [--keep] -- CMD...
#
# CMD runs as the invoking user with SUZKA_LAB_URL (the lab API base URL) and
# SUZKA_LAB_DIR set. It is killed after --timeout seconds (default 3600).
# The copy is taken while production runs; if it caught a write midway, the
# lab may refuse to start, which never affects production. If this script is
# killed with SIGKILL, stop the lab with `systemctl stop 'suzka-lab-*'`.
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
if ! [[ "${PORT}" =~ ^[1-9][0-9]{0,4}$ ]] || (( PORT > 65535 )); then
  printf 'invalid port: %s\n' "${PORT}" >&2
  exit 2
fi
if ! [[ "${TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'invalid timeout: %s\n' "${TIMEOUT}" >&2
  exit 2
fi
if [[ -n "$(ss -ltnH "sport = :${PORT}")" ]]; then
  printf 'port %s is already in use\n' "${PORT}" >&2
  exit 2
fi

sudo install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 700 "${LAB_BASE}"
LAB="$(sudo -u "${SERVICE_USER}" mktemp -d "${LAB_BASE}/lab-$(date +%Y%m%d-%H%M%S)-XXXX")"
UNIT="suzka-lab-$(basename "${LAB}")"
STARTED=0

# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  if [[ "${STARTED}" == 1 ]]; then
    # Stops the whole cgroup: the server and anything it started.
    sudo systemctl stop "${UNIT}" 2>/dev/null || true
    sudo systemctl reset-failed "${UNIT}" 2>/dev/null || true
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
# A symlink in the copy could lead lab writes back out of it.
if [[ -n "$(sudo find "${LAB}/.suzka" -type l -print -quit)" ]]; then
  printf 'the state copy contains symlinks; refusing\n' >&2
  exit 1
fi
sudo cp "${PROD_CONFIG}" "${LAB}/prod-config.yaml"
sudo cp "${PROD_ENV}" "${LAB}/backend.env"
# systemd's EnvironmentFile= overrides Environment=, and a later file wins over
# an earlier one, so these lab settings beat anything in backend.env.
printf '%s\n' "SUZKA_CONFIG_PATH=${LAB}/config.yaml" "PYTHONPATH=${PROD_ROOT}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 | sudo tee "${LAB}/lab.env" >/dev/null
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
sudo cp "${SCRIPTS}/lab_prepare.py" "${LAB}/lab_prepare.py"
sudo chown -R "${SERVICE_USER}:${SERVICE_USER}" "${LAB}"
sudo chmod 600 "${LAB}/backend.env" "${LAB}/lab.env" "${LAB}/prod-config.yaml" "${LAB}/lab_prepare.py"

# The service's own environment (library path, uv/python, HOME for the model cache).
SERVICE_ENV=()
while IFS= read -r assignment; do
  [[ -n "${assignment}" ]] && SERVICE_ENV+=("${assignment}")
done < <(systemctl show "${SERVICE}" -p Environment --value | tr ' ' '\n')
PYTHON="${PROD_ROOT}/.venv/bin/python"

# Writes LAB/config.yaml and checks the parsed settings: every Path stays in
# LAB, and the lab port is not the production port.
sudo -u "${SERVICE_USER}" env -i "${SERVICE_ENV[@]}" PYTHONPATH="${PROD_ROOT}" \
  "${PYTHON}" "${LAB}/lab_prepare.py" "${LAB}/prod-config.yaml" "${LAB}" "${PORT}" >/dev/null

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/GPU before lab: /' >&2 || true

SETENV=()
for assignment in "${SERVICE_ENV[@]}"; do
  SETENV+=("--setenv=${assignment}")
done
STARTED=1
sudo systemd-run --quiet --unit="${UNIT}" --collect \
  -p User="${SERVICE_USER}" -p Group="${SERVICE_USER}" \
  -p WorkingDirectory="${LAB}" \
  -p EnvironmentFile="${LAB}/backend.env" -p EnvironmentFile="${LAB}/lab.env" \
  -p ReadOnlyPaths="${PROD_ROOT}" -p ReadOnlyPaths="$(dirname "${PROD_CONFIG}")" \
  -p StandardOutput="file:${LAB}/server.log" -p StandardError="file:${LAB}/server.log" \
  "${SETENV[@]}" \
  "${PYTHON}" -m suzka.api.server

ready=0
for _ in $(seq 1 120); do
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health")" == 200 ]]; then
    ready=1
    break
  fi
  if ! systemctl is-active --quiet "${UNIT}"; then
    break
  fi
  sleep 5
done
if [[ "${ready}" != 1 ]]; then
  printf 'lab did not become healthy\n' >&2
  sudo tail -20 "${LAB}/server.log" >&2 || true
  exit 1
fi
MAIN_PID="$(systemctl show -p MainPID --value "${UNIT}")"
if ! sudo cat "/proc/${MAIN_PID}/environ" | tr '\0' '\n' | grep -qx "SUZKA_CONFIG_PATH=${LAB}/config.yaml"; then
  printf 'lab server is not using the lab config; refusing\n' >&2
  exit 1
fi
printf 'lab API at http://127.0.0.1:%s (unit %s, state %s)\n' "${PORT}" "${UNIT}" "${LAB}" >&2

status=0
SUZKA_LAB_URL="http://127.0.0.1:${PORT}" SUZKA_LAB_DIR="${LAB}" \
  timeout "${TIMEOUT}" "$@" || status=$?
exit "${status}"
