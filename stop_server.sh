#!/usr/bin/env bash
# Detiene el servidor Node del proyecto BikeRental.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-3000}"
PID_FILE="$ROOT_DIR/outputs/node_server.pid"

stop_pid() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "Proceso $pid detenido."
    return 0
  fi
  return 1
}

stopped=0

if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && stop_pid "$pid"; then
    stopped=1
  fi
  rm -f "$PID_FILE"
fi

# Por si el servidor se inició sin PID file o el puerto quedó ocupado
pids="$(lsof -ti ":$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$pids" ]; then
  for pid in $pids; do
    if stop_pid "$pid"; then
      stopped=1
    fi
  done
fi

if [ "$stopped" -eq 1 ]; then
  echo "Servidor detenido (puerto $PORT libre)."
else
  echo "No hay servidor escuchando en el puerto $PORT."
fi
