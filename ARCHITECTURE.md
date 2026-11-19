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
  -> RATE: adaptive per-patch ladder by default, or one flat scale with --flat
     (both are OXN8; scales are stored as Q8.8 integers)
  -> INTEGER HYPERPRIOR: quantized convolutions predict the anchor distribution
  -> INTEGER CONTEXT: four channel groups (6,6,6,6), each coded in two
     checkerboard passes; all activations, GELU and bucket decisions are canonical
  -> rANS codes residuals, hyper-latent and the adaptive scale map using
     frozen frequency tables
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
- OXN8 bit-exact entropy: hyperprior/context weights and activations use Q20.12
  integers. Convolution kernels process integer-valued float64 tensors whose
  products and partial sums stay below 2^53, making accumulation exact regardless
  of order. Integer round/shift, frozen GELU/bucket/rANS tables and Q8.8 scales
  remove platform-dependent transcendental functions from the entropy path.
  The pixel autoencoder may still use GPU floating point after entropy decode.
- Legacy OXN3–OXN7 entropy: hyperprior/context run on CPU in float64 with oneDNN
  disabled. This reduces same-machine ulp variation but cannot guarantee that
  old x86 files decode on ARM. Legacy numerics candidates are verified by the
  stream-length check where available.
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

## Legacy OXN5/OXN6/OXN7 — context entropy models

Ideas from NLAIC (arXiv:1910.06244) and the checkerboard context model
(He et al., 2021), applied on top of the frozen OXN3/OXN4 transform: the
latent probability depends not only on the hyperprior but also on
already-decoded neighbours. Even-parity tokens ("anchors") are rANS-coded
first with the usual hyperprior Laplace; odd-parity tokens are coded with a
Gaussian whose (mu, sigma) come from a small masked-conv network (~460K
params, `ox_ctx.pt`) that sees only the anchors plus the per-token log s
map — decode is two parallel passes, not pixel-by-pixel autoregression.
Only the context branch is trained (`python3 ox_neuro.py trainctx
--steps 40000 --data <images>`); the codec weights are untouched, and
OXN1–OXN4 files decode exactly as before.

- **OXN5** = flat rate + context (legacy decode).
- **OXN6** = adaptive per-patch ladder (OXN4) + context (legacy decode). The
  rate ladder is stored as float16 and the ctx inputs (anchor latent, log s
  map) are derived from exactly those float16 values.
- **OXN7** = ELIC-style channel groups (6,6,6,6) + checkerboard inside
  each group, one shared ctx net (`ox_ctx7.pt`, unknown channels/positions
  are zeroed — the zero pattern is deterministic on both ends; legacy decode).

Measured on the Kodak holdout (kodim21–24, never trained on), identical
PSNR: **OXN5 is 5.7–18.6% smaller than OXN3** (avg −11.9%, gain grows with
quality); **OXN6 is 1.5–14.5% smaller than OXN4** (avg −9.5%, gain grows
towards low bitrates); after 40K context-training steps **OXN7 is a further
4.46–15.92% smaller than OXN5** on average at q15/q50/q92 (up to −31% vs OXN3
at q92). Decode is automatic by magic bytes for OXN1–OXN8. Legacy bit-exactness
rules keeping the rANS stream in sync: the rate scaler is quantized (float32 in
OXN5/OXN7, float16 ladder in OXN6) before use, the context net is fed the
*decoded* (clipped to ±256) latent, and ctx runs CPU fp64 with mkldnn off,
like the hyperprior canon.


## OXN8 — deterministic integer entropy canon

OXN8 is the only format produced by new compression. Its mode byte selects an
adaptive per-patch scale map (default) or one flat scale (`--flat`). Both modes
reuse the trained `ox_ctx7.pt` channel-group network; no context retraining was
required for integer conversion.

- Activations and weights use Q20.12, with round-to-nearest-even and explicit
  signed shifts/divisions. Scales use Q8.8 and their Q20.12 logarithms are stored
  in the file, not recomputed by the decoder.
- Integer-valued float64 convolution is exact by construction: clamped inputs,
  quantized weights and kernel sizes keep every product and partial sum below
  2^53. Independent int64 reference loops are used to verify Conv2d and
  ConvTranspose2d behavior.
- GELU interpolation, Laplace/Gaussian bucket boundaries and all 48 rANS
  cumulative tables come from `oxn8_canon.npz` (version 1). This file is part of
  the format and must remain immutable. OXN8 uses bucket IDs 48–95; legacy table
  generation is unchanged.
- The encoder-side adaptive rate choice may use floating point because its
  decision is stored in the file. Every decoder decision that selects an rANS
  table or residual center follows the integer canon.
- C and Python parametric rANS produce identical OXN8 bytes for identical
  latents. GPU-encoded files pass fresh-process GPU and CPU-only decodes with
  `--verify`. Actual x86↔ARM exchange is still required as a final cross-machine
  acceptance test.

At q50 on kodim21–24, with `--verify`, OXN8 adaptive totals **34,883 bytes vs
37,920 for OXN6** (−8.0% total) at essentially equal PSNR (per-image changes
within +0.03 dB). OXN8 flat totals **57,889 bytes vs 57,854 for OXN7** (+0.06%),
with PSNR changes within ±0.003 dB.
