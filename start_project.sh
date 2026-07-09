#!/usr/bin/env bash
# Arranque rápido del proyecto (venv + Node). No requiere conda.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
PY_BIN="$VENV_DIR/bin/python"
MODEL_PATH="$ROOT_DIR/outputs/model_cnt.pkl"
PORT="${PORT:-3000}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 no está instalado o no está en el PATH."
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Error: node no está instalado o no está en el PATH."
  exit 1
fi

mkdir -p "$ROOT_DIR/outputs"

# Crear venv si no existe
if [ ! -x "$PY_BIN" ]; then
  echo "Creando entorno virtual en venv/..."
  python3 -m venv "$VENV_DIR"
fi

echo "Instalando dependencias Python..."
"$PY_BIN" -m pip install --upgrade pip -q
"$PY_BIN" -m pip install -r "$ROOT_DIR/requirements.txt" -q

"$PY_BIN" -c "import sklearn, pandas, joblib" || {
  echo "Error: falló la verificación de dependencias Python."
  exit 1
}

# Entrenar solo si no hay modelo guardado
if [ ! -f "$MODEL_PATH" ]; then
  echo "No se encontró outputs/model_cnt.pkl — ejecutando refine_model.py (puede tardar)..."
  "$PY_BIN" "$ROOT_DIR/refine_model.py"
else
  echo "Modelo encontrado: outputs/model_cnt.pkl (omitiendo entrenamiento)."
fi

echo "Instalando dependencias Node..."
cd "$ROOT_DIR/node_server"
npm install --silent

# Evitar duplicar servidor en el mismo puerto
if lsof -i ":$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Ya hay un proceso escuchando en el puerto $PORT."
  echo "Abre http://localhost:$PORT o detén el proceso y vuelve a ejecutar este script."
  exit 0
fi

export PYTHON_BIN="$PY_BIN"
export PORT="$PORT"

echo "Iniciando servidor en http://localhost:$PORT ..."
exec node server.js
