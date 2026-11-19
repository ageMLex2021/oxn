# OXH — OX-Neuro Hyper

> **Disclaimer: use at your own risk.** This is experimental research
> software. The authors accept no liability for damaged, corrupted or lost
> images, hardware failures (including GPU damage), or any other damages
> arising from its use. Always keep backups of your originals — a lossy
> neural codec is not an archive format. See `LICENSE` (MIT) for the full
> no-warranty text.
>
> **Использование на ваш страх и риск.** Это экспериментальный исследовательский
> код. Авторы не несут ответственности за повреждённые или утерянные фотографии,
> выход из строя оборудования (включая GPU) и любой иной ущерб. Всегда
> сохраняйте оригиналы — lossy-нейрокодек не является архивным форматом.

Neural image compressor: transformer autoencoder + variable-rate scaler +
hyperprior and deterministic integer OXN8 context entropy models (rANS). Beats
JPEG by up to +2.9 dB PSNR at matched bitrates below ~0.3 bpp (see
`ARCHITECTURE.md`). New files use adaptive OXN8 by default or flat OXN8 with
`--flat`; decompression supports OXN1–OXN9 and selects the format automatically
from its four-byte magic header.

## Setup (Linux / macOS)

```bash
cd oxn
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For GPU (CUDA) install the matching torch build from https://pytorch.org,
e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
Without CUDA everything runs on CPU (slower, same output).

## Setup (Windows)

1. Install Python 3.12+ from https://www.python.org/downloads/ —
   during setup tick **"Add python.exe to PATH"**.
2. In PowerShell or `cmd`:

```bat
cd oxn
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. GPU on Windows (NVIDIA): same torch CUDA build —
   `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
   Without a GPU everything runs on CPU (slower, same output).
4. The C accelerator `ox` is optional on Windows: the codec detects its
   absence and falls back to pure-Python rANS automatically. If you want it,
   install [MinGW-w64](https://www.mingw-w64.org/) or use WSL and run
   `gcc -O3 -o ox.exe ox.c` (the CUDA part needs the NVIDIA CUDA toolkit and
   `nvcc`; without it just skip).
5. Run:

```bat
python ox_ui.py
python oxh_viewer.py path\to\photo.oxh
```

To open `.oxh` by double-click on Windows: right-click an `.oxh` file →
"Open with" → "Choose another app" → browse to a small wrapper, e.g. create
`oxh_viewer.bat` next to the scripts:

```bat
@echo off
call "%~dp0venv\Scripts\activate.bat"
python "%~dp0oxh_viewer.py" %*
```

and select that `.bat` as the default app for `.oxh`.

## Model checkpoints

The models are in git — no training needed:

- `ox_neuro.pt` (~67 MB) — the codec itself (autoencoder + hyperprior);
- `ox_ctx7.pt` (~1.1 MB) — the channel-group context model required for OXN8
  and legacy OXN7 decompression;
- `oxn8_canon.npz` (~16 KB) — immutable OXN8 GELU, bucket-boundary and rANS
  tables; do not regenerate it on different machines;
- `ox_ctx.pt` (~1.1 MB) — the checkerboard context model required only for
  legacy OXN5/OXN6 decompression;
- `ox_boost.pt` (~0.8 MB) — the optional small conv predictor for faster
  adaptive OXN8 compression;
- `ox_neuro_tiny.pt` + `ox_ctx9_tiny.pt` (~0.7 MB total) — local OXN9 experiment.

To retrain the codec from scratch: put training images into `data/train/`
and run

```bash
python3 ox_neuro.py train --steps 40000 --batch 8 --data data/train \
    --dim 320 --depth 6 --latent 24 --refine --hyper --window 8 \
    --w_mse 300 --w_msssim 100 --w_hp 100 --out ox_neuro.pt
```

## Optional: fast entropy stage (recommended)

`ox` is a small C accelerator for the parametric rANS used by the codec.
Rebuild the native executable and `libox.so` for the current architecture:

```bash
make            # needs gcc; `make gpu` additionally uses nvcc if present
```

The UI and viewer reuse the loaded models between files. If the native library
and binary are missing or fail, the codec transparently falls back to subprocess
or pure-Python rANS (everything still works, just slower). A binary built for
another CPU architecture also takes the fallback path; rebuild it with `make`.

## Run the compressor UI

```bash
./run_ui.sh                             # starts the server if needed and opens
                                        # http://127.0.0.1:8000 in the browser
python3 ox_ui.py                        # same, without opening a browser
python3 ox_ui.py --host 0.0.0.0       # LAN mode: other devices (Windows PC,
                                      # Android phone) use it via browser at
                                      # http://<your-ip>:8000 — no porting needed
```

Drag&drop images or whole folders onto the drop zone. Two actions only:
**Compress** (images -> `.oxh`) and **Decompress** (OXN1–OXN9 `.oxh` -> `.jpg`,
selected automatically by magic header). Quality slider 1..100 (100 = best).
**adaptive OXN8** is on by default; turn it off for flat OXN8. Both modes use
the same deterministic integer entropy model. **boost speed for adaptive** uses
`ox_boost.pt` to choose the per-patch rate faster at the cost of roughly 0.5 dB
PSNR. Each queued file has an `x` button to remove it before starting.
**skip existing** (on by default) skips files whose result already exists in
the chosen save folder; uncheck it to overwrite. **verify** (off by default)
makes the encoder decode the entropy stream back and compare — a safety net
against unreadable files, slow on huge images. The **choose save dir**
button picks the output folder once (Chrome/Edge); without it — or if the
browser lacks the folder picker, like Firefox — results are saved as
regular downloads (skip-existing works only with a chosen folder).
Unsupported formats are skipped silently.
GPU is used automatically when CUDA is present; on VRAM exhaustion the codec
retries on CPU automatically.

## OXH viewer

```bash
python3 oxh_viewer.py path/to/photo.oxh
```

Opens a local viewer page: mouse wheel zoom, drag to pan, arrow keys / buttons
to move between `.oxh` files in the same folder. Non-`.oxh` files are skipped
silently.

To open `.oxh` files by double-click in the file manager (Linux):

```bash
cp oxh-viewer.desktop ~/.local/share/applications/
# fix the Exec= path inside if the folder differs
xdg-mime default oxh-viewer.desktop application/x-oxh
```

(You may also register the MIME type: create
`~/.local/share/mime/packages/oxh.xml` with
`<mime-type type="application/x-oxh"><glob pattern="*.oxh"/></mime-type>`
and run `update-mime-database ~/.local/share/mime`.)

## CLI

```bash
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 [--verify]  # adaptive OXN8
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --flat      # flat OXN8
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --boost     # faster OXN8
python3 ox_neuro.py d photo.oxh photo.jpg                   # OXN1–OXN9 by header
python3 ox_neuro.py bench photo.jpg        # vs JPEG and earlier OXN formats
```

On the local `oxn9-tiny` branch the UI also offers **OXN9 tiny**: an experimental
111×-smaller model intended for speed/quality experiments. It uses separate
`ox_neuro_tiny.pt` and `ox_ctx9_tiny.pt` files, writes the `OXN9` magic and never
replaces OXN8. Decompression selects the tiny model automatically from the
header. CLI equivalent: add `--tiny` to compression; decompression needs no flag.

OXN8 compression requires `ox_neuro.pt`, `ox_ctx7.pt` and `oxn8_canon.npz`.
To retrain the context branch, run `python3 ox_neuro.py trainctx --steps 40000
--groups 6,6,6,6 --data <images>`.

## Files

| File | Purpose |
|---|---|
| `ox_neuro.py` | codec core: adaptive/flat OXN8 compression, OXN1–OXN9 decompression |
| `oxn8_canon.npz` | immutable integer OXN8 GELU/bucket/rANS tables |
| `ox.py` | static rANS entropy engine (reference) |
| `ox.c` / `ox_cuda.cu` / `Makefile` / `ox` / `libox.so` | native C/CUDA accelerator |
| `ox_ui.py` | compressor web UI |
| `oxh_viewer.py` | .oxh viewer |
| `ox_neuro.pt` | codec checkpoint |
| `ox_boost.pt` | conv predictor for faster adaptive OXN8 compression |
| `ox_ctx.pt` | checkerboard context model for legacy OXN5/OXN6 |
| `ox_ctx7.pt` | channel-group context model for OXN8 and legacy OXN7 |
| `TASK_ARM64.md` | handoff: перенос обучения на arm64 |
