# GPU / CUDA 設定

このプロジェクトは起動時に CUDA とマルチGPUの可視化を自動判定します（`torch` または `cupy` が入っている場合）。

- `SOULSYM_ENABLE_CUDA=1` で CUDA 判定を有効化（既定: `1`）
- `SOULSYM_ENABLE_MULTI_GPU=1` で複数GPUを可視化（既定: `1`）
- `SOULSYM_CUDA_DEVICES=0,1` のように指定すると使用GPUを固定

PowerShell 例:

```powershell
$env:SOULSYM_ENABLE_CUDA="1"
$env:SOULSYM_ENABLE_MULTI_GPU="1"
$env:SOULSYM_CUDA_DEVICES="0,1"
python main.py
```

注意:
- Panda3Dのレンダリング自体は通常1GPUで動作します。
- この設定は CUDA の利用可否と可視GPUを制御するための実行時設定です。

# Panda3D Procedural Geometric Dungeon

This prototype uses a **BSP-first hybrid generator** optimized for geometric rooms and architecture-heavy features:

- Rectilinear rooms and hallways via BSP partitioning
- Connectivity via MST + a few extra loops
- Architectural modules: stairs, ramps, pillars, dome accents, wall decorations

## Why this approach
For geometric dungeons, BSP provides clean, structured spaces and predictable corridors. It is easier to compose with modular architecture than cave-focused methods like cellular automata.

## Setup

```bash
python -m pip install -r requirements.txt
python main.py
```

## Controls
- `WASD`: move
- Mouse: look
- `Esc`: toggle mouse lock

## Notes
- This is intentionally lightweight and code-first.
- Meshes are built from Panda3D primitive models and transforms for fast iteration.
