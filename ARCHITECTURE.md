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
- Entropy stage accelerated by a native C library (bit-identical to the Python
  reference; automatic subprocess/Python fallback). Canonical tables and the
  slot-to-symbol LUT are cached once per process; legacy tables load lazily.
  Native Q20.12 requantization and GELU preserve the integer canon. The C decoder
  also reports how many stream bytes were consumed: a mismatch with the stream
  length means the hyperprior tables did not match the encoder.
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
at q92). Decode is automatic by magic bytes for OXN1–OXN9. Legacy bit-exactness
rules keeping the rANS stream in sync: the rate scaler is quantized (float32 in
OXN5/OXN7, float16 ladder in OXN6) before use, the context net is fed the
*decoded* (clipped to ±256) latent, and ctx runs CPU fp64 with mkldnn off,
like the hyperprior canon.


## OXN8 — deterministic integer entropy canon

OXN8 is the stable base format; newer experimental formats are selected explicitly.
Its mode byte selects an
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
  `--verify`; the user has also confirmed actual x86↔ARM exchange.
- The UI and viewer keep the codec loaded between files. Adaptive encoding
  batches decoder-ladder candidates within the memory budget and uses a GPU
  floating-point estimate only for encoder-side rate selection; the final
  entropy path remains integer Q20.12. Reconstruction decoding uses BF16 on
  CUDA, while encoder latents stay FP32 at ordinary sizes to avoid a bitrate
  regression.

At q50 on kodim21–24, with `--verify`, OXN8 adaptive totals **34,883 bytes vs
37,920 for OXN6** (−8.0% total) at essentially equal PSNR (per-image changes
within +0.03 dB). OXN8 flat totals **57,889 bytes vs 57,854 for OXN7** (+0.06%),
with PSNR changes within ±0.003 dB.

## OXN9 tiny — local speed experiment

The `oxn9-tiny` branch keeps OXN8 intact and adds an auto-detected OXN9 stream.
Its architecture uses dim=48, one encoder/decoder block, latent=8, hyper width
and channels=8, an 8-wide context network and a convolutional refinement head:
195,043 parameters in total versus 17,123,051 for OXN8 (87.8× fewer). After the
initial 4,000-step run it was fine-tuned for 40,000 steps on Kodak01–20 plus
available local photographs, jointly with the context model and MS-SSIM/Sobel
losses. Kodak21–24 remained an honest holdout. At q50 the retrained model improves
mean PSNR from 21.46 to 22.65 dB and MS-SSIM from 0.731 to 0.849; at q75 PSNR
improves from 21.63 to 22.73 dB while mean bitrate falls slightly. See
`BENCHMARKS.md` for the rate-quality table. New tiny training candidates use the
`OXNT` magic to keep their context versions distinct from legacy OXN9 streams.
`ox_data.py` provides lazy multi-root crops with holdout exclusions; tiny and PICO
share multi-resolution TAL, per-patch color, seam-continuity and gradient-saliency
fidelity losses. `bench_research.py` records RD metrics, color shift, boundary
error, runtime and memory for both experiment families.

## OXN10 — one-shot entropy experiment

OXN10 keeps the OXN8 autoencoder, latent representation and canonical integer
hyperprior, but writes a separate `OXNA` stream. Gaussian bucket IDs are selected
once from hyperprior `logb`; all residual streams are rANS-decoded before context
reconstruction. The context network then predicts only residual-center offsets,
in seven group/phase passes on the GPU (or FP32 CPU when CUDA is unavailable),
instead of predicting both center and width through the integer CPU path. The
reconstruction decoder and therefore PSNR are unchanged at a given scale map.
This is a latency/bitrate experiment, not a replacement for OXN8's fully integer
context canon. `trainctx --one_shot` trains the center offsets with fixed
hyperprior widths; `--resume_step` resumes the original cosine schedule after an
interrupted run. A 40,000-step candidate increased the actual bucketed bitrate,
so the initial OXN7-initialized context is retained. See `BENCHMARKS.md` for the
speed/bitrate tradeoff.

## OXN11 — spatial one-shot sigma

OXN11 (`OXNB`) preserves OXN10 compatibility while adding a three-layer scale
network. It receives only hyperprior `mu`, `logb` and the stored scale map, then
predicts spatially varying log-sigma in one pass before rANS decoding. Its Q20.12
integer CPU inference avoids GPU/CPU bucket desynchronization; the seven context
passes for residual centers still run on GPU. Per-channel-only calibration and
longer/joint training were rejected after actual holdout bitrate measurements.
The selected 4,000-step scale network reduces OXN10 overhead from 6–16% to 3–7%
on Kodak21–24 and from 32% to 5% on the 12 MP q100 benchmark.

## OXN12 — table-aware training and 2×2 grid

A straight-through training loss evaluates the exact canonical rANS table cost
using rounded residual centers and Q20.12 bucket selection, while retaining the
continuous Gaussian gradient. This reduces OXN11 bitrate by another 0.4–0.6%.
OXN12 (`OXNC`) then replaces channel-group checkerboard with four 2×2 spatial
phases over one 24-channel group. Only three phases require context inference.
The scale network remains frozen during grid training. Grid context trades a
small low-q bitrate regression for high-q gains and slightly faster decoding;
both OXN11 and OXN12 remain available.

## OXN13 — grouped 2×2 context

OXN13 (`OXND`) keeps the table-aware scale network and 2×2 phases but splits the
latent into two 12-channel groups. This restores causal cross-channel information
with seven context passes, matching OXN11's pass count. An encoder-side rate
refinement then evaluates every scale candidate against actual canonical table
costs at the current context centers and widths, and rebuilds the scale map once.
It is enabled below q100; q100 keeps the original allocation. The result is
+1.9% bpp at q25, −0.9% at q50 and +1.2% at q75 over OXN8. Quantization-aware
scale fine-tuning was tested but rejected after a small consistent regression.
Further work should target the one-shot scale predictor rather than adding more
sequential context passes.

## OXN14 — empirical canonical rANS tables

OXN14 (`OXNE`) keeps the OXN13 model and allocation but replaces the Gaussian
frequency tables with empirical residual distributions measured on Kodak01–20 at
five quality levels. Tables use a 100,000-count Gaussian prior, nonzero frequency
for every symbol and exact RT=4096 normalization. New bucket IDs 96–119 select
these tables; OXN8–13 retain the original canon. Native, subprocess and pure-Python
rANS share the same tables. Rate refinement naturally uses the new table costs.
On the honest Kodak21–24 holdout OXN14 is smaller than OXN8 from q25 through q75,
with only +0.9% overhead at q92.

## OXN15/16 — channel-group tables and best-of selection

OXN15 (`OXNG`) adds 48 empirical Gaussian tables: one 24-bucket set per 12-channel
group, using bucket IDs 120–167. Statistics come only from Kodak01–20. Since the
standalone gain is mixed, OXN16 is not a new bitstream: it encodes OXN14 and OXN15
and returns the smaller candidate. The existing magic header selects decoding.
This guarantees no per-image size regression, at the cost of two encodes.
OXN17 applies this best-of encoder to the Rec1 checkpoint and changes the magic to
`OXNH` (shared tables) or `OXNI` (channel-group tables). These headers make the
decoder select `ox_neuro_rec.pt` automatically; OXNE/OXNG continue to select the
base checkpoint. OXN17 fast writes only `OXNH`, performs one encode and uses eight
exact-RD scale-map refinement passes; sixteen passes showed no additional gain.

## Recurrent joint decoder experiment

`--recurrent N` adds a zero-initialized convolutional refinement block shared by
N sequential reconstruction iterations. With `N=3` the initial checkpoint is an
exact OXN14 identity. `--freeze_enc` now includes both refinement families in the
trainable decoder set while freezing encoder and entropy parameters. Optional
`--w_lpips` adds differentiable AlexNet LPIPS to MSE/MS-SSIM training. A recurrent
checkpoint remains stream-compatible with OXN13–16; its decoder can also improve
the encoder-side scale allocation. The perceptual candidate is stored separately
and is not the default model. See `BENCHMARKS.md` for measured trade-offs.

`--flow N` is a separate conditional rectified-flow experiment. A small time-
conditioned convolutional velocity head starts from the learned reconstruction,
runs N Euler steps with two fixed noise seeds and applies an exponential agreement
gate to their residuals. `--w_flow` adds the flow-matching target during joint
decoder training. The encoder and transmitted latent remain unchanged; no
free-running semantic generation is used.

## OXN-MR and OXN-W research prototypes

`oxn_mr.py` contains jointly trained YCoCg `/16→/8→/4` models, a shared synthesis
block, hyperprior and exact table-aware discrete rate loss. OXM1 uses all levels;
OXM2 gates levels by scale; OXM3 additionally stores per-image float16 channel
scales optimized by encoder-side gradient RD. None of MR1–MR4 beat OXN14 on the
holdout, so they are not integrated into the UI or default pipeline.

`oxn_wave.py` is a separate three-level YCoCg/Haar sparse-pyramid experiment with
adaptive or canonical rANS per subband (`OXW1`). It also trails OXN14 and remains
a standalone prototype.

## PICO paper reimplementation

`oxn_pico.py` reproduces the public CVPR 2026 architecture diagrams: ConvScale311
backbones with learned input/output and activation scales, Haar resampling, four
outer-analysis downsamplings and four synthesis upsamplings, a `/64` hyper-latent,
separate scale and context decoders, and a four-phase 2×2 one-shot context with
elementwise residual centers and quantization widths. Quality uses interpolated
8-dimensional level conditioning and level-dependent latent gains.

OXP1 used legacy Gaussian rANS tables. OXP2 switches to the empirical OXN14
canon; scale inference remains CPU FP64 while context inference follows PICO and
runs on the accelerator. Training uses exact empirical-table cost for the loss
value and a smooth Gaussian straight-through gradient, avoiding both RT optimism
and unstable discrete-table gradients. Perceptual training additionally supports
LPIPS, multi-resolution tiling loss and a warm-started PatchGAN discriminator.
OXP3 stores independently coded tiles to bound inference memory on large images.
OXP4 adds a spatial halo (up to 64 pixels by default): the expanded tile is
coded independently, then the halo is cropped during reconstruction to reduce
seams. By default the encoder derives the core size from free RAM/VRAM, worker
count, halo and a conservative per-pixel working-set estimate. The throughput cap
is `2048 * sqrt((SM_count * clock) / (48 * 2.418 GHz))`, clamped to 1024–4096:
the GB10 calibration is scaled to the current GPU. A frame is processed whole
whenever it fits; a fixed multiple-of-64 size remains available for
experiments. Tiled inference can overlap two CUDA streams, and reconstruction is
assembled in stream order. OXP5 is a separate ARM/CUDA experiment: TF32
channels-last analysis convolutions and GPU FP32 scale inference replace the CPU FP64
canonical path. On ARM/CUDA this is now the default at every frame size; synthesis and
RGB output use contiguous layout because that is measurably faster on GB10. The four
spatial phases remain causally ordered, but use strided slices rather than CPU mask
transfers. Hyper entropy overlaps GPU inference, independent spatial entropy streams use
a five-thread CPU pool, and final RGB quantization transfers uint8 instead of float32.
OXP5 streams are not claimed to be cross-device
deterministic. OXP6 uses the tiled container with an explicit fast/FP64 flag and selects
the full-corpus v4 checkpoint in UI/viewer, so legacy OXP1–OXP5 and v4 weights cannot be
mixed accidentally. OXP7 adds sparse residual coding: a packed/zlib nonzero bitmap and
plain rANS only for nonzero values, with per-stream plain/sparse selection. Its selective
training freezes analysis, predictor and entropy while synthesis and level/channel masks
adapt. OXP9 stores an explicit expert ID and selects one of six synthesis experts by
target bitrate; analysis and entropy are shared. The ultra-low expert also uses a 2x2
anchor spatial base layer, while higher-rate experts use the full latent grid.
OXNV applies the same checkpoint separation to Tiny v4. Its
checkpoint-specific scale predictor uses latent, quantization, rate and pixel features
to estimate all seven distortion levels; only two levels require exact decoding.
OXP1/OXP2
remain readable. ConvScale input/output scales are folded into convolution weights once
at runtime initialization. Training reads random crops on demand rather than
retaining the complete dataset as float32 tensors; experiment arguments and
corpus size are stored in new checkpoints. This is an independent implementation
from the paper, not Apple weights or code.
