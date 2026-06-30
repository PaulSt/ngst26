#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PORT="${PORT:-8888}"
TOKEN="${TOKEN:-myst-local}"

cd "$ROOT"

rm -rf public webgui_scenes
mkdir -p webgui_scenes

export WEBGUI_SCENE_DIR="$ROOT/webgui_scenes"
export WEBGUI_BASE="${WEBGUI_BASE:-}"
export PYTHONPATH="$ROOT:$ROOT/chapters${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"$PYTHON" -m jupyter book clean --execute --html --yes

"$PYTHON" -m jupyter server \
  --no-browser \
  --ip=127.0.0.1 \
  --port="$PORT" \
  --ServerApp.root_dir="$ROOT" \
  --IdentityProvider.token="$TOKEN" \
  --ServerApp.disable_check_xsrf=True \
  > jupyterlog.txt 2>&1 &
JPID=$!

cleanup() {
  kill "$JPID" 2>/dev/null || true
  wait "$JPID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
  if "$PYTHON" -c "import socket; socket.create_connection(('127.0.0.1', $PORT), 2).close()"; then
    break
  fi
  sleep 1
done

export JUPYTER_BASE_URL="http://127.0.0.1:$PORT"
export JUPYTER_TOKEN="$TOKEN"

"$PYTHON" -m jupyter book build --execute --execute-parallel 1 --html --ci

mkdir -p public/webgui_scenes
cp -a _build/html/. public/
cp -a webgui_scenes/. public/webgui_scenes/
touch public/.nojekyll

echo "Built local test site in: $ROOT/public"
echo "Serve it with: $PYTHON -m http.server 8000 --directory public"
