#!/usr/bin/env python3
"""
Project-specific WebGL freezify starter for Soul Symphony 2.

This is a *starter scaffold* for Panda3D Web builds on Arch WSL2.
It prepares:
- Web PRC defaults
- preload manifest (asset list)
- HTML shell

Then it prints the exact places where you hook into Panda's official
WebGL freezify flow from your Panda checkout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "webbuild"
DEFAULT_PANDA_WEBGL_DIR = Path.home() / "opt" / "panda3d-webgl"

# Keep this aligned with your project layout.
PRELOAD_PATHS = [
    "main.py",
    "accel_math.py",
    "ball_visuals.py",
    "camera.py",
    "level.py",
    "player.py",
    "soundfx.py",
    "weapon_system.py",
    "world.py",
    "requirements.txt",
    "graphics",
    "soundfx",
    "bgm",
]

WEB_PRC = """\
default-model-extension
vfs-implicit-pz false
model-path .
window-title Soul Symphony 2 (Web)
notify-level-glgsg warning
audio-library-name p3openal_audio
"""

HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <link rel=\"icon\" href=\"data:,\" />
  <title>Soul Symphony 2 (WebGL)</title>
  <style>
    html,body{margin:0;height:100%;background:#000;color:#fff;font-family:system-ui,sans-serif}
    #app{display:grid;place-items:center;height:100%}
    #overlay{position:fixed;inset:0;display:grid;place-items:center;background:rgba(0,0,0,.82);z-index:9999}
    #overlay-inner{display:flex;flex-direction:column;gap:12px;align-items:center}
    #status-text{font-size:14px;opacity:.9}
    #start-game{display:none;padding:10px 16px;border:1px solid #666;background:#111;color:#fff;cursor:pointer}
  </style>
</head>
<body>
  <div id=\"app\">Loading Soul Symphony 2…</div>
  <div id=\"overlay\">
    <div id=\"overlay-inner\">
      <div id=\"status-text\">Loading Soul Symphony 2…</div>
      <button id=\"start-game\" type=\"button\">Start Game</button>
    </div>
  </div>
  <script>
    const statusEl = document.getElementById('app');
    const statusTextEl = document.getElementById('status-text');
    const overlayEl = document.getElementById('overlay');
    const startButtonEl = document.getElementById('start-game');

    window.Module = window.Module || {};
    window.Module.setStatus = function (text) {
      const value = text || 'Loading Soul Symphony 2…';
      if (statusEl) {
        statusEl.textContent = value;
      }
      if (statusTextEl) {
        statusTextEl.textContent = value;
      }
      if ((value === 'Ready to start' || value === 'Done!') && startButtonEl) {
        startButtonEl.style.display = 'inline-block';
      }
    };

    if (startButtonEl) {
      startButtonEl.addEventListener('click', function () {
        if (typeof window.__soul_start === 'function') {
          window.__soul_start();
        } else if (window.Module && typeof window.Module._loadPython === 'function') {
          window.Module.setStatus('Starting Python...');
          window.Module._loadPython();
        }
        if (overlayEl) {
          overlayEl.style.display = 'none';
        }
      });
    }
  </script>
  <script src=\"app.js?v=20260214b\"></script>
</body>
</html>
"""


def _collect_existing_preload_paths(root: Path) -> list[str]:
    found: list[str] = []
    for rel in PRELOAD_PATHS:
        p = root / rel
        if p.exists():
            found.append(rel)
    return found


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Soul Symphony 2 WebGL freezify starter artifacts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for web starter artifacts.")
    parser.add_argument("--panda-webgl-dir", type=Path, default=DEFAULT_PANDA_WEBGL_DIR, help="Path to panda3d-webgl checkout.")
    args = parser.parse_args()

    root = PROJECT_ROOT
    out_dir = args.out_dir.resolve()
    panda_webgl_dir = args.panda_webgl_dir.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _collect_existing_preload_paths(root)
    manifest_path = out_dir / "preload_manifest.txt"
    _write_text(manifest_path, "\n".join(existing) + "\n")

    prc_path = out_dir / "app.prc"
    _write_text(prc_path, WEB_PRC)

    html_path = out_dir / "index.html"
    _write_text(html_path, HTML_TEMPLATE)

    print("[ok] Generated starter artifacts:")
    print(f"  - {manifest_path}")
    print(f"  - {prc_path}")
    print(f"  - {html_path}")
    print()

    print("[next] Use Panda WebGL freezify flow and plug these files in:")
    print("  1) Ensure Panda WebGL build exists:")
    print("     bash scripts/build_panda_webgl.sh")
    print()
    print("  2) Open Panda sample freezify for reference:")
    print(f"     {panda_webgl_dir / 'editor' / 'freezify.py'}")
    print()
    print("  3) In your project-specific freezify implementation, include:")
    print("     - app entry: main.py")
    print(f"     - preload list: {manifest_path}")
    print(f"     - web PRC: {prc_path}")
    print()
    print("  4) Host output with:")
    print("     http-server")
    print()
    print("[note] This script intentionally does not hardcode emcc/freezer internals,")
    print("       since Panda WebGL freezify details can vary by Panda revision.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
