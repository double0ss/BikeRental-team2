#!/usr/bin/env bash
# Arranque del proyecto BikeRental (venv + Node). macOS / Linux.
# Uso: ./start_project.sh   o   bash start_project.sh
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Ejecuta con bash:  bash start_project.sh"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
MODEL_PATH="$ROOT_DIR/outputs/model_cnt.pkl"
PORT="${PORT:-3000}"
PID_FILE="$ROOT_DIR/outputs/node_server.pid"
LOG_FILE="$ROOT_DIR/outputs/node_server.log"
HOST="${HOST:-0.0.0.0}"

resolve_python_cmd() {
  if [ -x "$VENV_DIR/bin/python" ]; then
    echo "$VENV_DIR/bin/python"
    return 0
  fi
  if [ -x "$VENV_DIR/bin/python3" ]; then
    echo "$VENV_DIR/bin/python3"
    return 0
  fi
  for candidate in \
    "/usr/local/bin/python3.10" \
    "/usr/local/bin/python3.9" \
    "/opt/homebrew/bin/python3.10" \
    "/opt/homebrew/bin/python3.9" \
    "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3" \
    "/Library/Frameworks/Python.framework/Versions/3.9/bin/python3" \
    "/usr/local/bin/python3" \
    "/opt/homebrew/bin/python3" \
    "/usr/bin/python3"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  command -v python3 2>/dev/null || return 1
}

resolve_node_cmd() {
  # NVM (login shells / Mac con nvm instalado)
  if [ -s "$HOME/.nvm/nvm.sh" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.nvm/nvm.sh" 2>/dev/null || true
  fi

  if command -v node >/dev/null 2>&1; then
    command -v node
    return 0
  fi

  for candidate in \
    "/usr/local/bin/node" \
    "/opt/homebrew/bin/node" \
    "$HOME/.nvm/versions/node/"*/bin/node; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_CMD="$(resolve_python_cmd)"; then
  echo "Error: python3 no encontrado. Instala Python 3.9+."
  exit 1
fi

if ! NODE_CMD="$(resolve_node_cmd)"; then
  echo "Error: node no encontrado. Instala Node.js (brew, nvm o https://nodejs.org)."
  exit 1
fi

export PATH="$(dirname "$NODE_CMD"):$PATH"

mkdir -p "$ROOT_DIR/outputs"

# Crear venv si falta o está roto
if [ ! -x "$VENV_DIR/bin/python" ] && [ ! -x "$VENV_DIR/bin/python3" ]; then
  echo "Creando entorno virtual en venv/..."
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

if [ -x "$VENV_DIR/bin/python" ]; then
  PY_BIN="$VENV_DIR/bin/python"
elif [ -x "$VENV_DIR/bin/python3" ]; then
  PY_BIN="$VENV_DIR/bin/python3"
else
  echo "Error: no se pudo crear el intérprete del entorno virtual."
  exit 1
fi

echo "Python: $("$PY_BIN" -c 'import sys; print(sys.executable)')"
echo "Node:   $NODE_CMD ($("$NODE_CMD" --version))"

echo "Instalando dependencias Python..."
cd "$ROOT_DIR"
PIP_OPTS="--trusted-host pypi.org --trusted-host files.pythonhosted.org"
"$PY_BIN" -m pip install --upgrade pip -q $PIP_OPTS || \
  "$PY_BIN" -m pip install --upgrade pip -q $PIP_OPTS --index-url https://pypi.org/simple
"$PY_BIN" -m pip install -r "$ROOT_DIR/requirements.txt" -q $PIP_OPTS || \
  "$PY_BIN" -m pip install -r "$ROOT_DIR/requirements.txt" -q $PIP_OPTS --index-url https://pypi.org/simple

"$PY_BIN" -c "import sklearn, pandas, joblib; print('Dependencias Python OK')" || {
  echo "Error: falló la verificación de dependencias Python."
  exit 1
}

if [ ! -f "$MODEL_PATH" ]; then
  echo "No hay modelo — ejecutando refine_model.py (puede tardar varios minutos)..."
  "$PY_BIN" "$ROOT_DIR/refine_model.py"
else
  echo "Modelo encontrado: outputs/model_cnt.pkl"
fi

echo "Instalando dependencias Node..."
cd "$ROOT_DIR/node_server"
npm install --silent

if lsof -i ":$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo ""
  echo "El puerto $PORT ya está en uso."
  echo "  Web: http://localhost:$PORT"
  echo "  Para detener: ./stop_server.sh"
  exit 0
fi

export PYTHON_BIN="$PY_BIN"
export PORT="$PORT"
export HOST="$HOST"

echo ""
echo "Iniciando servidor en http://localhost:$PORT (también http://$(hostname):$PORT en la red local)"
echo "Log: $LOG_FILE"
echo "Detener: ./stop_server.sh"
echo ""

# Segundo plano con PID para stop_server.sh
nohup "$NODE_CMD" server.js >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
sleep 1

if kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "Servidor iniciado (PID $SERVER_PID)."
  if curl -sf "http://127.0.0.1:$PORT/models" >/dev/null 2>&1; then
    echo "API respondiendo correctamente."
  else
    echo "Servidor en marcha; espera unos segundos y abre http://localhost:$PORT"
  fi
else
  echo "Error: el servidor no arrancó. Revisa $LOG_FILE"
  tail -20 "$LOG_FILE" 2>/dev/null || true
  exit 1
fi
