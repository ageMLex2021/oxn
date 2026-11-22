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
`--flat`; decompression supports OXN1–OXN17 and OXNT and selects the format automatically
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
- `ox_neuro_rec.pt` — recurrent perceptual decoder used by OXN17;
- `oxn_pico.pt` — compact checkpoint for the standalone PICO reimplementation;
- `oxn_pico_v4_soup50.pt` — balanced full-corpus perceptual PICO v4;
- `oxn_pico_v5_sparse_frozen_7k.pt` — selective sparse low-rate PICO v5;
- `ox_tiny_v4.pt`, `ox_tiny_v4_ctx.pt`, `ox_tiny_v4_boost.pt` — perceptual Tiny v4,
  its entropy context and two-decode scale predictor;
- `ox_ctx7.pt` (~1.1 MB) — the channel-group context model required for OXN8
  and legacy OXN7 decompression;
- `oxn8_canon.npz` (~16 KB) — immutable OXN8 GELU, bucket-boundary and rANS
  tables; do not regenerate it on different machines;
- `oxn14_canon.npz` — empirical Gaussian rANS tables for OXN14;
- `oxn15_canon.npz` — channel-group empirical tables for OXN15/16;
- `ox_ctx.pt` (~1.1 MB) — the checkerboard context model required only for
  legacy OXN5/OXN6 decompression;
- `ox_boost.pt` (~0.8 MB) — the optional small conv predictor for faster
  adaptive OXN8 compression;
- `ox_neuro_tiny.pt` + `ox_ctx9_tiny.pt` (~0.8 MB total) — local tiny OXN9/OXNT experiment;
- `ox_ctx10.pt` (~1.1 MB) — context model for the experimental OXN10 bitstream;
- `ox_ctx11.pt` (~1.8 MB) — context and spatial one-shot sigma for OXN11;
- `ox_ctx12.pt` (~1.8 MB) — table-aware sigma and 2×2 grid context for OXN12;
- `ox_ctx13.pt` (~1.8 MB) — OXN13 2×2 context with two 12-channel groups.

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
**Compress** (images -> `.oxh`) and **Decompress** (OXN1–OXN17 and OXNT `.oxh` -> `.jpg`,
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
retries on CPU automatically. When their total size fits the available-memory
budget, all selectable compression models (OXN8, OXN17, tiny and PICO versions),
including their contexts and boost, are preloaded and kept resident before the HTTP server starts.
Otherwise models use exclusive loading. Decompression activates the runtime
required by the stream header; legacy decode-only contexts are unloaded on the
next activation, and an OOM fallback disables resident mode.

## OXH viewer

The viewer switches the active runtime when adjacent files require different
models. It reuses the compressor UI's resident cache when enabled; legacy
decode-only contexts are still loaded and released on demand.

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
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --oxn10    # experimental OXN10
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --oxn11    # spatial-sigma OXN11
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --oxn12    # 2×2-grid OXN12
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --oxn13    # 2×2 + 12+12 channels
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --oxn14    # empirical rANS tables
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --oxn16    # smallest OXN14/15
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --oxn17    # recurrent + best-of
python3 ox_neuro.py c photo.jpg photo.oxh --q 75 --oxn17_fast  # single encode
python3 ox_neuro.py d photo.oxh photo.jpg                   # OXN1–OXN17 and OXNT by header
python3 ox_neuro.py bench photo.jpg        # vs JPEG and earlier OXN formats
```

The UI model selector includes **OXN17 fast** (one encode), **OXN17 best-of**
(two encodes, slightly smaller output), legacy **PICO reimplementation**, and
**PICO auto · low/high best profile**, **PICO v4 perceptual · OXP6**,
**PICO v5 sparse low-rate · OXP7**, **PICO MoE 6 rate experts · OXP9**, and
**Tiny v4 perceptual · OXNV + scale boost**. Auto selects v5 for target ≤0.25 bpp
(or q≤10) and v4 otherwise. MoE shares analysis/entropy and activates one synthesis expert
per rate, with an anchor-based spatial base layer for ultra-low rates.
OXP7 stores a compressed nonzero bitmap and
rANS-codes only nonzero residuals, choosing plain coding whenever that is smaller.
On ARM/CUDA, PICO uses GPU scale inference at all sizes, overlaps independent entropy
streams with GPU work and returns GPU-quantized uint8 RGB. This reduced the measured
12-MP OXP9 roundtrip to 1.366 s encode / 0.946 s decode; see `BENCHMARKS.md`.
OXP6 records the fast/FP64 mode and always selects the v4 checkpoint; OXNV similarly
selects Tiny v4, preventing accidental decoding with legacy weights. Tiny v4 boost
uses a checkpoint-specific predictor and two exact decoder passes instead of seven.
OXP1–OXP9 and OXNT/OXNV decompression is supported by UI and viewer (OXP5 and fast OXP6/7/8/9 require CUDA).
The standalone PICO architecture reimplementation can be run with:

```bash
python3 oxn_pico.py c input.jpg output.oxh --q 75 --verify
python3 oxn_pico.py d output.oxh restored.png
python3 oxn_pico.py eval --data data/kodak --limit 4 --q 75 \
    --out data/eval/pico_results.json
```

For matched PICO/tiny experiment benchmarks, use `bench_research.py`, for example:

```bash
python3 bench_research.py data/kodak/kodim01.png --model tiny \
    --q 75 100 --runs 3 --lpips
python3 bench_research.py data/kodak/kodim01.png --model pico \
    --q 75 100 --runs 3 --lpips
python3 bench_research.py data/kodak/kodim01.png --model pico \
    --target-bpp 0.5 --runs 1 --lpips
```

Training accepts comma-separated, recursively scanned corpus roots and `--exclude`
filename substrings for holdout images. `--w_tal`, `--w_color`, `--w_seam` and `--w_text` enable the
multi-resolution, low-frequency color, patch-continuity and gradient-saliency
fidelity losses. New checkpoints record these arguments and corpus size. In the
UI, **target bpp** above zero selects a q-search for tiny/PICO instead of the
usual quality level; it takes several encoding passes.

It requires `oxn_pico.pt` and remains a research baseline, not Apple weights or a
claim of PICO-level quality. Images that do not fit the memory-derived working
budget are encoded as independent tiles. `--tile 0` selects the core size from
free RAM/VRAM with a GPU-throughput cap (the default); a positive multiple of 64
forces a fixed size.
A frame that fits is processed whole. OXP4 tiles include an automatic halo of up
to 64 pixels, which is included in the memory budget and cropped during assembly;
`--overlap 0` selects OXP3 without a halo. Large frames can use two CUDA streams:
`--workers 1` disables them and `--workers 2` forces two within the same total
memory budget. On ARM with CUDA, frames within the throughput-optimal tile area
use the separate OXP5 experiment: TF32 channels-last convolution and GPU FP32
scale inference without the CPU FP64 canonical copy. Larger frames retain the
measured-faster FP64 path. Use `--no-fast` or `--fast` to override this choice. Legacy OXP1/OXP2 decoding remains supported, and training loads
only the current random crops instead of caching the full dataset in RAM.

The UI also offers **OXNT tiny**: an experimental 88×-smaller model intended for
speed/quality experiments. It uses separate `ox_neuro_tiny.pt` and
`ox_ctx9_tiny.pt` files and writes the `OXNT` magic so new training candidates do
not get mixed with legacy OXN9 streams. Decompression selects the tiny model
automatically from the header. CLI equivalent: add `--tiny` to compression;
decompression needs no flag.

**OXN10** is a CLI-only experiment with the OXN8 autoencoder and `OXNA` header.
It decodes all rANS streams before context reconstruction, uses hyperprior-derived
Gaussian widths, and predicts residual centers on the GPU. Compression requires
`ox_neuro.pt`, `ox_ctx10.pt` and `oxn8_canon.npz`; UI and viewer decode OXN10
automatically. To train or resume its context model:

```bash
python3 ox_neuro.py trainctx --steps 40000 --groups 6,6,6,6 \
    --data data/train --ckpt ox_neuro.pt --init ox_ctx7.pt --one_shot
# after an interrupted run, for example from step 28000:
python3 ox_neuro.py trainctx --steps 40000 --resume_step 28000 \
    --groups 6,6,6,6 --data data/train --ckpt ox_neuro.pt \
    --init ox_ctx10.pt.part --one_shot
```

**OXN11** adds a spatial one-shot sigma network with deterministic integer
inference and writes `OXNB`. It preserves the OXN10 GPU context and old OXNA
compatibility while reducing the bitrate overhead. To train this variant:

```bash
python3 ox_neuro.py trainctx --steps 4000 --groups 6,6,6,6 \
    --data data/train --ckpt ox_neuro.pt --init ox_ctx10.pt \
    --one_shot --sigma_only --scale_net --ctx_out ox_ctx11.pt
```

**OXN12** (`OXNC`) combines table-aware sigma with four 2×2 spatial phases and
one 24-channel group, reducing context passes from seven to three. After
table-aware calibration, train its context with:

```bash
python3 ox_neuro.py trainctx --steps 8000 --groups 24 --lr 0.0001 \
    --data data/train --ckpt ox_neuro.pt --init ox_ctx11_table.pt \
    --one_shot --scale_net --table_loss --grid4 --ctx_only \
    --ctx_out ox_ctx12.pt
```

**OXN13** (`OXND`) uses the same spatial sigma and 2×2 phases with two
12-channel groups. Below q100 it also refines the adaptive scale map using exact
canonical rANS table costs. It improves low-q bitrate while keeping seven context
passes:

```bash
python3 ox_neuro.py trainctx --steps 8000 --groups 12,12 --lr 0.0001 \
    --data data/train --ckpt ox_neuro.pt --init ox_ctx12.pt \
    --one_shot --scale_net --table_loss --grid4 --ctx_only \
    --ctx_out ox_ctx13.pt
```

**OXN14** (`OXNE`) reuses the OXN13 model and allocation but replaces the Gaussian
rANS tables with empirical distributions measured on the training corpus. It
requires `oxn14_canon.npz`; older formats continue to use `oxn8_canon.npz`.
**OXN15** (`OXNG`) splits these tables by channel group; **OXN16** tries OXN14 and
OXN15 and stores the smaller stream, so it never increases the selected file size.
A separate recurrent checkpoint can be selected with `--ckpt`; `--recurrent N` and
`--w_lpips` are used only when training that experimental decoder. **OXN17**
combines the Rec1 decoder with OXN16 best-of entropy coding and writes `OXNH` or
`OXNI`, so CLI/UI/viewer automatically load `ox_neuro_rec.pt` during decoding.
For the optional matched-bpp external benchmark, install `lpips torchvision`,
ensure FFmpeg provides `libjxl` and `libaom-av1`, then run `python3
bench_external.py`.

OXN8 compression requires `ox_neuro.pt`, `ox_ctx7.pt` and `oxn8_canon.npz`.
To retrain the context branch, run `python3 ox_neuro.py trainctx --steps 40000
--groups 6,6,6,6 --data <images>`.

## Files

| File | Purpose |
|---|---|
| `ox_neuro.py` | codec core: adaptive/flat OXN8 compression, OXN1–OXN17 and OXNT decompression |
| `oxn8_canon.npz` | immutable integer OXN8 GELU/bucket/rANS tables |
| `oxn14_canon.npz` | empirical OXN14 Gaussian rANS tables |
| `oxn15_canon.npz` | channel-group empirical OXN15/16 rANS tables |
| `oxn_mr.py` / `oxn_wave.py` | rejected multiresolution and Haar research prototypes |
| `oxn_pico.py` | PICO CVPR 2026 architecture reimplementation and 6-expert MoE (OXP1–OXP9) |
| `bench_external.py` | matched-bpp OXN14/JPEG XL/AVIF benchmark (optional LPIPS deps) |
| `bench_research.py` | matched PICO/tiny RD, color, boundary, runtime and memory benchmark |
| `ox_data.py` | lazy EXIF-aware multi-root training crops with holdout exclusions |
| `ox_scale_booster.py` | checkpoint-specific Tiny scale-distortion predictor trainer |
| `bench_pico.py` | OXN14 against official PICO article reconstructions/bpp |
| `oxn_booster.py` | rejected OXNF spatial/wavelet residual-booster prototype |
| `ox.py` | static rANS entropy engine (reference) |
| `ox.c` / `ox_cuda.cu` / `Makefile` / `ox` / `libox.so` | native C/CUDA accelerator |
| `ox_ui.py` | compressor web UI |
| `oxh_viewer.py` | .oxh viewer |
| `ox_neuro.pt` | codec checkpoint |
| `ox_neuro_rec.pt` | recurrent perceptual decoder for OXN17 |
| `ox_boost.pt` | conv predictor for faster adaptive OXN8 compression |
| `ox_ctx.pt` | checkerboard context model for legacy OXN5/OXN6 |
| `ox_ctx7.pt` | channel-group context model for OXN8 and legacy OXN7 |
| `TASK_ARM64.md` | handoff: перенос обучения на arm64 |
