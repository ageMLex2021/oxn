# OX-Neuro Hyper — Compressor Architecture

**Author: Alexey Kruglov, Kazakhstan — a7017420338@gmail.com**

## Idea

OX-Neuro is a lossy neural image codec. Unlike generative approaches
(diffusion), it never hallucinates detail: a transformer learns to predict
patch content from its neighbours, and an entropy coder squeezes the
remainder to the Shannon limit.

## Pipeline

```
image
  -> 16x16 patches -> linear projection -> 6 transformer blocks
     (windowed 8x8 attention with half-window shift, Swin-style)
  -> latent: 24 numbers per patch (~32x byte bottleneck)
  -> scaled quantization: round(z*s)/s — a single model covers every
     compression ratio (q 1..100 maps to s on a log scale)
  -> ADAPTIVE rate: every patch picks its own scale from a ladder
     0.35..8.0 minimizing "distortion + lambda * bits" (bit allocation)
  -> HYPERPRIOR: a small conv network reads the latent and predicts
     (mu, b) of a Laplace distribution per token; rANS codes only the
     residual q - mu with a per-token table
  -> the scale map and the hyper-latent are rANS-coded too
```

The decoder mirrors this; a small conv refinement head smooths patch seams,
and a Sobel (high-pass) training loss preserves edge sharpness.

## Measured performance (Kodak, honest holdout — kodim21–24 never trained on)

| Bitrate | OX-Neuro Hyper | JPEG |
|---|---|---|
| 0.207 bpp | **25.93 dB / MS-SSIM 0.923** | 23.03 dB / 0.785 |
| 0.245 bpp | **27.03 dB / 0.944** | 25.04 dB / 0.844 |
| 0.278 bpp | **27.24 dB / 0.951** | 25.98 dB / 0.871 |

- Below ~0.3 bpp OX-Neuro Hyper **beats JPEG by 1.3–2.9 dB PSNR** and by a
  wide MS-SSIM margin; no 8x8 blocking artifacts at all.
- At ~1% of source size JPEG collapses into color noise while OX-Neuro keeps
  content intact (faces, tones).
- Adaptive vs flat: −37…60% size at almost the same PSNR.
- Optional boost (`--boost`, `ox_boost.pt`): the per-patch scale is chosen
  by a small conv net fed with the latent, per-scale quant error, bitrate
  estimates, patch pixel stats and the true distortions of only 2 of the 7
  ladder scales — ~2.3x faster adaptive encode for ≈0.5 dB PSNR. Trained on
  the ladder's own outputs (no external labels).
- Real 201 MP photo (14142×14235): 0.426 bpp, roundtrip PSNR 32.6 dB,
  ~20 s encode (GPU), ~20 s decode to JPEG; CPU works too.

## Engineering properties

- One model for all compression ratios (no per-bitrate weights).
- Any image size: windowed attention + chunking driven by *free* memory
  (VRAM/RAM), correct on unified-memory systems (DGX Spark, Jetson);
  automatic CPU retry on VRAM exhaustion.
- EXIF orientation honored on load.
- Entropy stage accelerated by a C module (bit-identical to the Python
  reference; automatic fallback). The C decoder also reports how many stream
  bytes were consumed: a mismatch with the stream length means the hyperprior
  tables did not match the encoder.
- Bit-exact entropy: the hyperprior (mu/buckets) runs on CPU in float64 with
  oneDNN disabled at both ends — stock conv results differ at ulp level
  between processes (oneDNN picks kernels by process state: thread count,
  buffer alignment), which used to desync the rANS stream on rare occasions.
  The encoder can optionally self-check a file by decoding the latent stream
  back (`--verify` / the verify checkbox; off by default — slow on huge
  images). Legacy files (GPU-era and a short
  fp32-CPU era) are rescued at decode by a cascade of numerics candidates
  verified by the stream-length check.

## Honest limitations

- Above ~0.6 bpp JPEG still wins on PSNR; the reserve is more data and
  longer training.
- Screenshots / text / flat graphics belong to lossless (PNG) — a neural
  codec is the wrong tool there.
- Trained on BSDS500 + Kodak + DIV2K-valid; production use wants a larger
  corpus.

## Roadmap

1. More data (full DIV2K, CLIC) and longer training.
2. Diffusion/generative decoder mode for ultra-low bitrates (plausibility
   over fidelity).
3. Full CUDA port of the entropy stage.
