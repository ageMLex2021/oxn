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
per-patch adaptive rate + hyperprior entropy model (rANS). Beats JPEG by up to
+2.9 dB PSNR at matched bitrates below ~0.3 bpp (see `ARCHITECTURE.md`).

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
5. Copy the model checkpoint `ox_neuro.pt` into the folder (see
   "Model checkpoint" below), then:

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

## Model checkpoint

The file `ox_neuro.pt` (~67 MB) is NOT in git (see `.gitignore`). Either:

- copy a trained checkpoint into this folder as `ox_neuro.pt`, or
- train your own: put training images into `data/train/` and run

```bash
python3 ox_neuro.py train --steps 40000 --batch 8 --data data/train \
    --dim 320 --depth 6 --latent 24 --refine --hyper --window 8 \
    --w_mse 300 --w_msssim 100 --w_hp 100 --out ox_neuro.pt
```

## Optional: fast entropy stage (recommended)

`ox` is a small C accelerator for the parametric rANS used by the codec.
A prebuilt Linux binary is included; to rebuild:

```bash
make            # needs gcc; `make gpu` additionally uses nvcc if present
```

If the binary is missing or fails, the codec transparently falls back to a
pure-Python rANS (everything still works, just slower).

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
**Compress** (images -> `.oxh`) and **Decompress** (`.oxh` -> `.jpg`).
Quality slider 1..100 (100 = best). The **adaptive** checkbox (on by
default) enables per-patch adaptive rate: noticeably smaller files, but
much slower on huge images (the decoder runs once per rate candidate).
**boost speed for adaptive** (on by default) replaces 5 of the 7 decoder
runs with a small conv predictor (`ox_boost.pt`): ~2.3x faster adaptive at
the cost of roughly 0.5 dB PSNR. Needs `ox_boost.pt` next to `ox_neuro.pt`.
Each queued file has an `x` button to remove it before starting.
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
python3 ox_neuro.py c photo.jpg photo.oxh --q 100 --adaptive
python3 ox_neuro.py d photo.oxh photo.jpg
python3 ox_neuro.py bench photo.jpg        # vs JPEG table
```

## Files

| File | Purpose |
|---|---|
| `ox_neuro.py` | codec core: model, train/compress/decompress, containers OXN3/OXN4 |
| `ox.py` | static rANS entropy engine (reference) |
| `ox.c` / `ox_cuda.cu` / `Makefile` / `ox` | C/CUDA accelerator (`prans`) |
| `ox_ui.py` | compressor web UI |
| `oxh_viewer.py` | .oxh viewer |
| `ox_neuro.pt` | model checkpoint |
