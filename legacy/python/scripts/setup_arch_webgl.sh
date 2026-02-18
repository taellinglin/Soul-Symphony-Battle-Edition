#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Installing system packages..."
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm --needed base-devel git wget unzip python python-pip nodejs npm

if command -v python3.12 >/dev/null 2>&1; then
  PY312=python3.12
elif command -v python3 >/dev/null 2>&1; then
  PY312=python3
else
  echo "Python not found" >&2
  exit 1
fi

echo "[2/5] Preparing Emscripten SDK..."
EMSDK_DIR="$HOME/opt/emsdk"
mkdir -p "$HOME/opt"
if [[ ! -d "$EMSDK_DIR" ]]; then
  git clone https://github.com/emscripten-core/emsdk.git "$EMSDK_DIR"
fi

cd "$EMSDK_DIR"

echo "[3/5] Installing latest Emscripten..."
./emsdk install latest
./emsdk activate latest

echo "[4/5] Verifying toolchain..."
# shellcheck disable=SC1091
source ./emsdk_env.sh
emcc -v || true

cat <<EOF

[5/5] Done.

Next:
  bash scripts/build_panda_webgl.sh

EOF
