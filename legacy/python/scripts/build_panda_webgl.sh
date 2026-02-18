#!/usr/bin/env bash
set -euo pipefail

EMSDK_DIR="${EMSDK_DIR:-$HOME/opt/emsdk}"
PANDA_DIR="${PANDA_DIR:-$HOME/opt/panda3d-webgl}"

if [[ ! -d "$EMSDK_DIR" ]]; then
  echo "Emsdk not found at $EMSDK_DIR. Run scripts/setup_arch_webgl.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$EMSDK_DIR/emsdk_env.sh"

if command -v python3.12 >/dev/null 2>&1; then
  PY312=python3.12
elif [[ -x "$HOME/opt/python312/bin/python3.12" ]]; then
  PY312="$HOME/opt/python312/bin/python3.12"
else
  echo "Python 3.12 not found. Install python3.12 or build it at $HOME/opt/python312/bin/python3.12." >&2
  exit 1
fi

echo "[1/4] Cloning Panda3D..."
mkdir -p "$HOME/opt"
if [[ ! -d "$PANDA_DIR/.git" ]]; then
  git clone https://github.com/panda3d/panda3d.git "$PANDA_DIR"
fi

cd "$PANDA_DIR"

echo "[2/4] Downloading precompiled webgl deps..."
if [[ ! -f webgl-editor-and-dependencies.zip ]]; then
  wget https://rdb.name/webgl-editor-and-dependencies.zip
fi
unzip -o webgl-editor-and-dependencies.zip

echo "[3/4] Building Panda3D for Emscripten (this can take a while)..."
"$PY312" makepanda/makepanda.py \
  --nothing \
  --use-python \
  --use-vorbis \
  --use-bullet \
  --use-zlib \
  --use-freetype \
  --use-harfbuzz \
  --use-openal \
  --no-png \
  --use-direct \
  --use-gles2 \
  --optimize 4 \
  --static \
  --target emscripten \
  --threads 4

echo "[4/4] Done."
echo "Now adapt/create freezify.py for this project (see docs/webgl-arch-wsl2.md)."
