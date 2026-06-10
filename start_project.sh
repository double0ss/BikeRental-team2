#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BIN="$HOME/miniconda/bin/conda"
ENV_NAME="bike-env"
PY_BIN="$HOME/miniconda/envs/${ENV_NAME}/bin/python"

# Accept TOS and create env if missing
$CONDA_BIN tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
$CONDA_BIN tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
if ! $CONDA_BIN env list | grep -q "${ENV_NAME}"; then
  echo "Creating conda env ${ENV_NAME} with Python 3.11"
  $CONDA_BIN create -n ${ENV_NAME} python=3.11 -y
fi

# Install python deps via pip in the env
echo "Installing Python requirements into ${ENV_NAME}..."
$CONDA_BIN run -n ${ENV_NAME} python -m pip install --upgrade pip
$CONDA_BIN run -n ${ENV_NAME} python -m pip install -r "$ROOT_DIR/requirements.txt" --prefer-binary

# Re-run training to ensure model and outputs exist
echo "Running training script..."
$PY_BIN "$ROOT_DIR/train_predictor.py"

# Install Node dependencies
echo "Installing Node server dependencies..."
cd "$ROOT_DIR/node_server"
if [ -f package.json ]; then
  npm install
fi

# Start Node server (detached)
export CONDA_PYTHON="$PY_BIN"
echo "Starting Node server on port 3000 (background)..."
nohup node server.js > "$ROOT_DIR/outputs/node_server.log" 2>&1 &

echo "All done. API available at http://localhost:3000/predict"
