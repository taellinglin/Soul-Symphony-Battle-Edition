# WebGL Build (Arch WSL2)

このプロジェクトを Arch WSL2 上で Panda3D WebGL ビルドするための最短手順です。

## 0) 前提

- Windows 側で `wsl -l -v` に `Arch` が `VERSION 2` で表示される
- Arch WSL2 のシェルを開ける
- このリポジトリは Linux 側でもアクセスできる場所にある

## 1) 依存セットアップ（Arch）

Arch シェルで以下を実行:

```bash
cd /path/to/Soul\ Symphony\ 2
bash scripts/setup_arch_webgl.sh
```

このスクリプトで以下を設定します:

- `base-devel`, `git`, `wget`, `unzip`
- `python`, `python-pip`（必要に応じて `python312`）
- `nodejs`, `npm`
- Emscripten SDK (`~/opt/emsdk`)

## 2) Panda3D を Emscripten ターゲットでビルド

```bash
cd /path/to/Soul\ Symphony\ 2
bash scripts/build_panda_webgl.sh
```

実行内容:

- `panda3d` を `~/opt/panda3d-webgl` に clone
- `webgl-editor-and-dependencies.zip` を展開
- `makepanda.py --target emscripten` 実行

## 3) アプリ側フリーズ（freezify.py）

このプロジェクトにはまだ `freezify.py` を置いていないため、次のいずれかが必要です:

- Panda3D WebGL サンプル (`editor` / `roaming-ralph`) の `freezify.py` をコピーして本プロジェクト用に調整
- もしくは本プロジェクト専用 `freezify.py` を新規作成

このリポジトリにはスターターとして `scripts/freezify.py` を追加済みです。

```bash
python scripts/freezify.py
```

生成物:

- `webbuild/preload_manifest.txt`
- `webbuild/app.prc`
- `webbuild/index.html`

これらを Panda 側の WebGL freezify 実装に組み込んでください。

## 4) ローカル配信

ビルド成果物ディレクトリで:

```bash
http-server
```

`python -m http.server` より `http-server` を推奨（WASM MIME / 圧縮対応）。

## 5) Web向けPRC推奨

少なくとも以下を入れて不要HTTPリクエストを抑制:

```prc
default-model-extension
vfs-implicit-pz false
model-path .
```

## Troubleshooting

- デバッグ時は最適化レベルを落とす（`--optimize 3` など）
- `stack overflow` が出る場合は `freezify.py` 側で `-sSTACK_SIZE=<bytes>` を増やす
- ブラウザコンソール（F12）で Panda3D/Emscripten エラーを確認

---

必要なら次に、このリポジトリ向けの `freezify.py` 雛形（`main.py` 起動・assets preload）まで作成できます。
