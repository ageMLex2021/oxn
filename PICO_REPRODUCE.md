# Reproducing the OXP9 PICO MoE Experiment

Branch: [`pico`](https://github.com/ageMLex2021/oxn/tree/pico)

This package contains the current independent PICO implementation and the final six-expert
OXP9 checkpoint. It does not include historical PICO checkpoints or new heavy weights for
Tiny/OXN models. The final checkpoint is sufficient for compression, decompression, evaluation,
and further training initialization.

## Included Files

- `oxn_pico.py` — model architecture, training, OXP1–OXP9 codec, tiling, and runtime.
- `oxn_pico_triton.py` — optional FP16-input/FP32-accumulation analysis backend with fallback.
- `oxn_pico_v8_moe6_final.pt` — final 6-expert OXP9 model, 48,282,977 bytes.
- `ox_neuro.py`, `ox.py`, `ox.c`, `ox_cuda.cu` — shared metrics, canonical entropy utilities,
  and native rANS implementation.
- `ox_data.py`, `oxn_mr.py` — training data and differentiable rate-loss dependencies.
- `oxn8_canon.npz`, `oxn14_canon.npz` — immutable canonical entropy tables.
- `Makefile` — builds the native CPU/CUDA utilities.
- `bench_research.py` — real roundtrip, RD, runtime, memory, and boundary benchmark.
- `bench_pico_data.py` — downloads the official Apple evaluation assets.
- `bench_pico_reimpl.py` — compares OXP9 with official Apple PICO reconstructions.
- `PICO_SPEED_BRIEF_EN.md` — architecture, hardware, metrics, bottlenecks, and rejected ideas.
- `PICO_SPEED_BRIEF.md` — Russian version of the brief.
- `MOE_RESEARCH_STATE.md`, `RESEARCH_IDEAS.md` — model lineage and research log.

The implementation is independent of Apple and does not contain Apple model weights or codec
source code.

## Setup

```bash
git clone --branch pico https://github.com/ageMLex2021/oxn.git
cd oxn
python3 -m pip install -r requirements.txt
make
```

`make` builds `libox.so` for native rANS. If no native library is available, the codec has
slower fallback paths. For metric evaluation, additionally install LPIPS:

```bash
python3 -m pip install lpips
```

PyTorch must be installed with CUDA support to use the ARM/CUDA fast path. CPU FP64 remains
available for compatible non-fast streams and encoding with `--no-fast`. On ARM/CUDA, the
Triton analysis backend is enabled when available; set `OXP9_TRITON=0` to use the original
PyTorch convolution path. Reproducible experiment reports and commands are in
`speed_experiments/`.

## Compress and Decompress

The standalone CLI defaults to `oxn_pico_v8_moe6_final.pt` and OXP9:

```bash
python3 oxn_pico.py c input.jpg restored.oxh --q 20 --verify
python3 oxn_pico.py d restored.oxh restored.png
```

Explicit checkpoint and expert selection:

```bash
python3 oxn_pico.py c input.jpg restored.oxh \
  --ckpt oxn_pico_v8_moe6_final.pt --stream-version 9 --expert 2 --q 20
python3 oxn_pico.py d restored.oxh restored.png \
  --ckpt oxn_pico_v8_moe6_final.pt
```

For large images, auto tiling is derived from available memory. `--tile`, `--workers`, and
`--overlap` can override it. A frame that fits the budget is processed as a whole.

## Benchmark an Image or Directory

Single-image runtime/RD benchmark with warmup and repeated measurements:

```bash
python3 bench_research.py input.jpg --model pico \
  --ckpt oxn_pico_v8_moe6_final.pt --stream-version 9 \
  --q 20 --warmup 2 --runs 5 --lpips --out result.json
```

Directory evaluation through the standalone CLI:

```bash
python3 oxn_pico.py eval --data data/kodak --limit 4 --q 20 \
  --ckpt oxn_pico_v8_moe6_final.pt --stream-version 9 --out result.json
```

## Reproduce the Apple Scene Comparison

Download the official public reference and Apple reconstruction images, their reported bpp,
and Apple's data license:

```bash
python3 bench_pico_data.py
python3 -m pip install lpips
python3 bench_pico_reimpl.py \
  --ckpt oxn_pico_v8_moe6_final.pt --stream-version 9 \
  --out apple_comparison.json
```

The downloader uses Apple's public `ml-pico` repository. Evaluation assets are subject to
Apple's `LICENSE_DATA`, downloaded alongside them. The benchmark searches q independently for
each scene, writes actual OXP9 streams and reconstructions, and computes PSNR, MS-SSIM, and
LPIPS against the common reference images.

Expected current average at the fixed-q comparison recorded in the brief:

| Codec | bpp | PSNR | MS-SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Apple PICO | 0.29000 | 29.548 dB | 0.96973 | 0.04118 |
| OXP9 | 0.31932 | 26.924 dB | 0.91442 | 0.27475 |

The q-search benchmark may choose slightly different operating points. On some scenes OXP9
does not reach exactly Apple's bpp, so this is an actual-operating-point comparison, not a
perfectly matched BD-rate claim.

## Expected Runtime on the Development Machine

Development hardware: ARM64, 20 CPU cores, NVIDIA GB10 with 48 SMs, CUDA 13.0, PyTorch
2.14.1+cu130. Timings on other hardware will differ.

| Input | Encode | Decode | bpp |
|---|---:|---:|---:|
| 4000×3000, q20 | 1.082 s | 0.925 s | 0.134005 |
| 12288×16384, q10, auto tile 1280 | 16.372 s | 14.992 s | 0.0957699 |

Apple reports 0.230 s encode and 0.150 s decode for 12 MP on iPhone 17 Pro Max. This is not a
same-hardware comparison. OXP9 does not currently match Apple speed or perceptual RD.

## Training and Further Fine-Tuning

Initialize a new run from the published final model:

```bash
python3 oxn_pico.py train --data /path/to/images \
  --init oxn_pico_v8_moe6_final.pt --moe 6 --channel_mask \
  --out oxn_pico_finetuned.pt --steps 4000 --crop 256
```

`MOE_RESEARCH_STATE.md` records the historical multi-stage recipe and expert sources. Those
historical intermediate checkpoints are intentionally not published; exact reproduction of
the full training lineage would require them. They are not needed to reproduce inference,
roundtrips, benchmarks, or to start fine-tuning from the final model.

## Documentation Links

- [Speed and architecture brief](PICO_SPEED_BRIEF_EN.md)
- [Russian speed and architecture brief](PICO_SPEED_BRIEF.md)
- [Model state and training lineage](MOE_RESEARCH_STATE.md)
- [Research ideas and rejected experiments](RESEARCH_IDEAS.md)
- [Original Apple PICO paper](https://arxiv.org/html/2605.05148v1)
- [Official Apple PICO reconstructions](https://github.com/apple/ml-pico)
