# PICO/OXP9: Speed Optimization Brief

State date: 2026-11-22.

Reproduction instructions and publication: [`PICO_REPRODUCE.md`](PICO_REPRODUCE.md).

This document describes the current working PICO MoE implementation, target hardware,
measured performance, and bottlenecks. It is intended to help generate new optimization
ideas. This is an independent implementation of the architecture from the Apple PICO paper,
not Apple code or weights.

## 1. Goals and Constraints

The current primary objective is to reduce encode/decode latency without degrading actual
bpp, PSNR, MS-SSIM, or LPIPS. The priority target is ARM64 + NVIDIA GB10. Backward
compatibility is not required for new experimental streams, but existing OXP6–OXP9 streams
must remain readable. Tiny and other codecs are outside the scope of this task.

Current accepted model: `oxn_pico_v8_moe6_final.pt`.
Implementation: `oxn_pico.py`.
Format: OXP9, sparse residual coding, explicit synthesis expert ID.
Changing weights or architecture is allowed if a real matched-bpp benchmark demonstrates an
improvement. A speed increase at the cost of a significant bpp increase or quality reduction
is not considered a success.

## 2. Target Hardware and Software Environment

- CPU: ARM64, 20 physical cores:
  - 10 × Cortex-X925, up to 3.9 GHz;
  - 10 × Cortex-A725, up to 2.808 GHz.
- GPU: NVIDIA GB10, compute capability 12.1.
- 48 streaming multiprocessors, 24 MiB L2.
- GPU memory: approximately 124.6 GB; the system reports 121 GiB RAM, and memory is
  effectively shared between CPU and GPU on this platform.
- CUDA 13.0, PyTorch 2.14.1+cu130.
- OS: Linux aarch64.
- CPU fallback should exploit parallelism across independent tasks, but must not introduce
  expensive thread-per-symbol or thread-per-tiny-stream overhead.

Apple PICO is optimized for Apple's own Apple/ARM/ANE environment. Its figures are a reference
point, not a guaranteed like-for-like hardware comparison with GB10.

## 3. Current Model Architecture

### 3.1. Overall Pipeline

The analysis network converts RGB input into a latent `y` with 192 channels and a spatial
resolution of approximately H/16 × W/16. An additional hyper branch creates a representation
at H/64 × W/64. The quantized hyper-latent is used to reconstruct:

- `sigma` for the entropy model through the scale decoder;
- `prior` for the predictor through the context decoder.

The predictor receives `prior`, previously reconstructed spatial phases, and an 8-dimensional
level vector. It produces a residual center `mu` and a quantization width for each latent
element.

Reconstruction synthesis restores RGB from the dequantized latent. Only one of six synthesis
experts is executed; the other experts are not evaluated at that moment.

### 3.2. Analysis

- HaarDown 3 → 64.
- Level conditioning: 8 channels are added to the features.
- ConvScale311 blocks with GELU and learned scales.
- Successive HaarDown and feature blocks: 64 → 96 → 96 → latent 192.
- Hyper branch: 96 → 128 → 192 → 192.
- ConvScale is reparameterized at runtime: input/output scales are folded into the weights
  in advance.
- Some 3×3 ConvScale layers are grouped convolutions, typically with 2–4 groups.

### 3.3. Spatial Context

There are four 2×2 checkerboard phases:

1. `(even y, even x)`;
2. `(even y, odd x)`;
3. `(odd y, even x)`;
4. `(odd y, odd x)`.

Each subsequent phase uses previously reconstructed phases. Therefore, the four predictor
passes are causally sequential; they cannot simply be run simultaneously without changing
the entropy model and retraining. These are four large spatial passes, not pixel-by-pixel
autoregression.

In the current codec path, masks have been replaced with strided slices `a::2, b::2`. This
eliminated mask transfers to the CPU and full-resolution `where` operations.

### 3.4. MoE Rate Control

- Shared analysis, predictor, scale decoder, context decoder, and entropy tables.
- Six synthesis experts, with only one active at a time.
- The expert is selected explicitly by target bpp:
  - 0: ultra-low, `<0.075 bpp`;
  - 1: low, `0.075–0.15`;
  - 2: mid-low, `0.15–0.3`;
  - 3: mid, `0.3–0.75`;
  - 4: high, `0.75–1.5`;
  - 5: ultra-high, `>1.5`.
- The ultra-low expert additionally uses a 2×2 spatial anchor base layer: residual symbols
  are stored only at anchor positions, while the remaining positions are reconstructed by
  the predictor.
- MoE increases the number of stored parameters but does not execute six synthesis networks
  simultaneously.

### 3.5. Entropy and Sparse Coding

- Parametric rANS, using a native C implementation with frozen canonical tables.
- OXP9 additionally stores a packed nonzero residual bitmap compressed with zlib level 9.
- rANS encodes only nonzero residuals.
- For a nonempty stream, both sparse and plain variants are computed and the smaller one is
  selected.
- For a large all-zero stream, plain rANS is no longer computed.
- Hyper entropy runs on the CPU concurrently with GPU scale/predictor work.
- Four independent spatial entropy encode jobs run in parallel after the GPU work.
- Decode-side rANS remains sequential within each stream; this is a property of the current
  bitstream, not an incidental Python limitation.

### 3.6. Tiling and Memory

- If an image fits within the calculated memory budget, it is processed as a whole.
- For large images, tile size is calculated from available RAM/VRAM, worker count, and an
  estimated working set.
- The maximum tile size is scaled according to GPU SM count and clock relative to GB10.
- The default halo is up to 64 pixels; the halo is cropped during assembly.
- Multi-tile inference can use two CUDA streams.
- Practical result: 201-MP images use 2048 tiles, but forcing a 12-MP image into small tiles
  proved disadvantageous.

## 4. Current Runtime Path

On ARM/CUDA, fast mode is now used for every image size:

- analysis uses TF32 channels-last convolutions;
- the scale decoder runs in GPU FP32 instead of CPU FP64;
- synthesis uses contiguous layout because it is faster than channels-last on GB10;
- spatial phases use slices without CPU mask transfers;
- GPU/CPU copies for spatial symbols use pinned buffers and a single synchronization;
- independent entropy encode jobs overlap GPU work or run concurrently with one another;
- final RGB clip/quantization runs on the GPU;
- uint8 RGB rather than float32 is copied to the CPU: 36 MB instead of 144 MB for 12 MP.

The old FP64 path is preserved and tested. Fast OXP streams do not claim cross-device
determinism.

## 5. Current Measured Results

### 5.1. Speed Before and After the Latest Optimization

4000×3000, q20, OXP9, five measured runs after warmup:

| Metric | Before | Current |
|---|---:|---:|
| Encode | 2.07 s | 1.366 s |
| Decode | 1.61 s | 0.946 s |
| Bitrate | 0.134012 bpp | 0.134012 bpp |

201 MP, 12288×16384, q10, OXP9:

| Metric | Before | Current |
|---|---:|---:|
| Encode | 27.30 s | 22.313 s |
| Decode | 21.08 s | 14.169 s |
| Bitrate | 0.0879519 bpp | 0.0879519 bpp |
| PSNR | 20.459 dB | 20.459 dB |
| Peak process RSS | approximately 3.17 GiB | approximately 3.33 GiB |

For the 12-MP benchmark: peak CUDA reserved memory is approximately 10.85 GB, and process
peak RSS is approximately 2.24 GB. For 201 MP: peak CUDA reserved memory is approximately
12.47 GB.

The Apple PICO reference point for a similar 12-MP scenario is approximately 0.23 s encode
and 0.15 s decode. The current gap is still approximately 5.9× for encode and 6.3× for decode.
Apple's speed has not been matched.

### 5.2. Accepted Model Quality

Apple scenes, average:

- 0.31932 bpp;
- PSNR 26.924 dB;
- MS-SSIM 0.91442;
- LPIPS 0.27475.

Screenshots, average:

- 0.19990 bpp;
- PSNR 27.409 dB;
- MS-SSIM 0.94073;
- LPIPS 0.29265.

After runtime optimization, all four Apple-scene OXP9 streams remained byte-identical.
PSNR/MS-SSIM/LPIPS changes appear only in the last displayed decimal places. All six experts
passed real roundtrips. Existing OXP6/OXP7/OXP8/OXP9 streams decode successfully.

### 5.3. Current Comparison with Original Apple PICO

Sources of the original results: the [Apple/CVPR 2026 paper](https://arxiv.org/html/2605.05148v1)
and the [official reconstruction images/bpp](https://github.com/apple/ml-pico). Apple does not
publish an encoder/decoder or model weights, so a local roundtrip of original PICO is not
possible. OXP9 quality metrics are computed against the same reference images; Apple metrics
are computed from its official reconstruction PNGs. Speed is measured on different hardware
and is a reference point, not a strict like-for-like benchmark.

| Parameter | Apple PICO | Current OXP9 | Difference / note |
|---|---:|---:|---|
| 12-MP encode | 0.230 s, iPhone 17 Pro Max | 1.366 s, GB10 | OXP9 is 5.94× slower |
| 12-MP decode | 0.150 s, iPhone 17 Pro Max | 0.946 s, GB10 | OXP9 is 6.31× slower |
| 12-MP encode + decode, sum | 0.380 s, derived sum | 2.312 s, derived sum | OXP9 is 6.08× slower |
| Total parameters | not published | 11,969,456 | Shared pool + 6 experts |
| Active parameters during reconstruction | not published | 6,809,396 | Shared pool + 1 expert |
| Checkpoint size | not published | 48,282,977 bytes, 46.05 MiB | `oxn_pico_v8_moe6_final.pt` |
| 12-MP peak process RSS | not published | approximately 2.24 GB | Local benchmark |
| 12-MP peak CUDA reserved | not applicable to iPhone | approximately 10.85 GB | Not equal to actual memory in use |
| Rate control | fine-grained, according to the paper | 8-dimensional level + 6 target-bpp experts | One OXP9 expert is active |
| Cross-platform entropy | bit-exact scale decoder claimed | FP64 legacy path; fast mode does not guarantee cross-device determinism | Different deployment guarantees |
| Training corpus | exact size not used in this comparison | 120,799 images | Holdout images excluded |
| Subjective BD-rate study | 2.3–3× savings over AV1/AV2/VVC/ECM/JPEG-AI | not conducted | The conclusion cannot be transferred to OXP9 |

Quality and bitrate on the official scenes. Lower LPIPS is better; higher values are better
for the other quality metrics. OXP9 does not reach exactly the same bpp on every scene, so
this table shows actual operating points rather than perfectly matched-bpp BD-rate.

| Scene | bpp Apple | bpp OXP9 | PSNR Apple | PSNR OXP9 | MS-SSIM Apple | MS-SSIM OXP9 | LPIPS Apple | LPIPS OXP9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| philipp-reiner-207 | 0.341 | 0.35946 | 25.266 | 24.198 | 0.95424 | 0.88563 | 0.05968 | 0.29185 |
| b1400e99ef413c8fee7d5c0be41d7d1e | 0.095 | 0.11439 | 34.886 | 32.322 | 0.99003 | 0.97161 | 0.01624 | 0.25799 |
| alexander-shustov-73 | 0.507 | 0.50708 | 25.468 | 22.530 | 0.96119 | 0.88350 | 0.06107 | 0.29315 |
| kodim23 | 0.217 | 0.29635 | 32.572 | 28.647 | 0.97348 | 0.91693 | 0.02774 | 0.25599 |
| **Average** | **0.29000** | **0.31932** | **29.548** | **26.924** | **0.96973** | **0.91442** | **0.04118** | **0.27475** |

Average gap: OXP9 uses 10.11% more bits, has 2.624 dB lower PSNR, 0.05531 lower MS-SSIM,
and 0.23357 higher LPIPS. Apple PICO currently leads in both speed and perceptual RD. These
local results must not be presented as a victory over Apple.

### 5.4. Triton Analysis Backend After the Original Comparison

The separate `oxp9-speed` branch adds a specialized direct grouped 3×3+GELU analysis kernel.
It uses FP16 inputs/weights, FP32 accumulation/GELU, and NHWC implicit GEMM on Triton/tensor
cores. Synthesis, scale/context/predictor, and the decoder are unchanged. Use
`OXP9_TRITON=0` or the original PyTorch error fallback to disable/bypass it.

| Scenario | Baseline encode/decode | Triton encode/decode | bpp |
|---|---:|---:|---:|
| 4000×3000, q20, 5–7 runs | 1.420 / 0.977 s | **1.082 / 0.925 s** | 0.134012 → 0.134005 |
| 201 MP, q10, auto tiling | 22.374 / 15.759 s, tile 960 | 16.372 / 14.992 s, tile 1280 | different auto-tile operating points |

The 201-MP row validates memory/stability but is not a controlled same-tile speed or
matched-bpp comparison. The Apple-scene average after the backend is 0.319315 bpp, 26.92439
dB PSNR, 0.914458 MS-SSIM, and 0.274703 LPIPS. All experts and existing streams passed.
Cold first small-image encode was 1.498 s, followed by a cached 0.0297 s repeat. Details and
reports are in `speed_experiments/`.

## 6. Bottlenecks in Decreasing Order of Importance

### 6.1. Analysis Encoder: Grouped FP32/TF32 ConvScale — the Main Problem

4000×3000 profile before the latest minor improvements:

- complete analysis: approximately 0.85–0.90 s;
- `e12`: approximately 0.206 s;
- `e22`: approximately 0.243 s;
- `e11`: approximately 0.108 s;
- `e21`: approximately 0.081 s;
- HaarDown and the remaining analysis stages are significantly cheaper.

Thus, analysis accounts for roughly two-thirds of current encode time. The most expensive
operations are grouped 3×3 convolutions. The generic PyTorch/cuDNN grouped path makes poor
use of GB10/tensor-core capabilities. The largest expected gain is a specialized native/AOT
grouped-convolution pipeline with sensible fusion and controlled numerics.

Simply converting grouped convolution into dense convolution with zero inter-group weights
has already been tested and provided almost no benefit: additional FLOPs offset the dense
kernel advantage.

### 6.2. Synthesis Decoder — the Main Decode Problem

Before layout optimization, synthesis took approximately 0.51 s in the FP64/default path and
approximately 0.75 s in the unsuccessful channels-last fast path. After switching synthesis
to contiguous layout, decode is approximately 0.90–0.95 s in a stable warm benchmark.

Synthesis contains four HaarUp levels and ConvScale311 blocks. It is the second most
important candidate for native fused kernels. To accelerate both encode and decode, the
optimized convolution pipeline should cover both analysis and synthesis.

### 6.3. Sparse Bitmap and Encode-Side Entropy

The old sequential sparse/plain encode took approximately 0.44 s. Within it, native rANS
accounted for approximately 0.16 s of CPU work; the rest consisted of bitmap processing,
zlib, nonzero selection, and related operations. Moving the hyper job and running four
spatial jobs in parallel now overlaps part of this work with the GPU or other CPU jobs.

CPU work from multiple parallel jobs must not be added together and reported as wall-clock
time. Nevertheless, bitmap/zlib and the need to compute a plain candidate for nonempty
streams remain the next opportunity. Wall-clock critical-path measurements are required,
rather than optimization based on the sum of thread times.

Decode-side entropy is much smaller: approximately 0.05 s for sparse decode in the 12-MP
profile. Therefore, GPU rANS or complex parallel rANS decoding is not the first priority.

### 6.4. Remaining CPU↔GPU Transfers and Synchronizations

Previously there were eight separate spatial transfers plus checkerboard mask transfers.
Masks are no longer transferred, and pinned buffers plus a single synchronization are now
used. This produced a gain, but the following transfers are still required by the current
format:

- quantized symbols and sigma to the CPU for native rANS encoding;
- decoded residuals back to the GPU between the four causal phases during decode;
- sigma to the CPU for rANS table selection during decode.

Completely removing these transitions would require a GPU entropy implementation or a
different entropy/context design. Because decode rANS accounts for a small share of runtime,
it must first be demonstrated that a new GPU path will not lose to native C through launch
and synchronization overhead.

### 6.5. Scale/Context Decoders and Predictor

In the 12-MP fast profile:

- scale decoder: approximately 0.02 s;
- context decoder: approximately 0.02 s;
- four predictor passes combined: approximately 0.035 s.

These are no longer the main bottlenecks. Optimizing them cannot deliver the sixfold
improvement needed to approach Apple.

### 6.6. Tile Assembly

There are no unnecessary tiles in the whole-image 12-MP path. For 201 MP, assembly remains
part of runtime, but most time is spent in repeated analysis/synthesis/entropy work across
tiles. Increasing the number of CUDA streams or reducing tile size does not guarantee a
parallel speedup and increases bpp because of duplicated context and halo.

## 7. What Has Already Been Tried and Why It Was Not Accepted

### Successful Changes Already Included in the Current Baseline

- GPU FP32 scale decoder instead of CPU FP64 on ARM/CUDA: removes approximately 0.5 s at
  12 MP.
- Fast mode at every image size: 201-MP decode improved from 21.08 to 14.169 s.
- Contiguous synthesis: faster than channels-last on GB10.
- Strided spatial slices: identical bitstream without mask transfers or full-resolution
  `where` operations.
- Parallel independent entropy encode jobs: overlap between CPU and GPU work.
- Empty sparse streams without full plain rANS: removes three expensive all-zero passes in
  the typical ultra-low case.
- GPU uint8 quantization/output: smaller final transfer and less CPU postprocessing.
- Pinned spatial copies with one synchronization.
- These changes preserved Apple-scene streams byte-for-byte and passed the 201-MP roundtrip.

### Rejected or Currently Unintegrated Ideas

1. **Dense convolution instead of grouped convolution with zero inter-group weights.**
   Mathematically equivalent, but additional FLOPs almost completely offset the gain. One
   variant showed only approximately 0.04 s improvement; numerics and the bitstream changed.

2. **HaarDown fusion into a single 2×2 convolution.**
   A correct mathematical factorization was implemented as an experiment but did not produce
   a stable speedup. Numerics and several stream bytes changed. Reverted.

3. **HaarUp through ConvTranspose2d or fused 1×1 + pixel shuffle.**
   Did not produce a stable decode speedup and changed reconstruction numerics. Applying
   HaarUp fusion to scale/context cannot be used with an old stream: changing sigma causes
   entropy desynchronization.

4. **cuDNN benchmark/autotune.**
   The first encode took approximately 42.5 s and the first decode approximately 14.9 s due
   to algorithm search. There was no warm-speed gain; the stream and reconstruction numerics
   changed. Not accepted.

5. **`torch.compile` for analysis/synthesis/scale.**
   In some runs, encode improved to approximately 1.28 s, and synthesis occasionally produced
   a small gain. However, initial compilation took approximately 43–72 s, shape/expert
   specialization issues appeared, and numerics and the bitstream changed. This was not
   accepted for an interactive codec. AOT with prebuilt kernels remains a separate possible
   direction.

6. **`torch.compile` for the predictor.**
   Previously changed the bitstream and did not deliver an acceptable result. The predictor
   takes only approximately 0.035 s, so revisiting it is not worthwhile.

7. **Thread-per-stream asynchronous rANS during prediction.**
   Was previously approximately three times slower. The currently accepted design differs:
   the hyper job overlaps GPU work, and four spatial jobs start after GPU causal work has
   finished.

8. **Forced 12-MP tiling and two CUDA streams.**
   2048/1536/1024 tiles did not consistently accelerate the whole-image path. Bitrate
   increased from 0.134012 to approximately 0.144–0.159 bpp because of halo and duplicated
   context. Not accepted.

9. **Channels-last synthesis.**
   Was slower than contiguous on GB10; an expensive noncontiguous HWC transfer also increased
   decode time. Channels-last is now retained primarily for analysis.

10. **BF16 PICO.**
    Previously caused a severe quality regression: PSNR fell from approximately 21.37 to
    12.9–15.9 dB. Do not use without QAT or other compensating training.

11. **Simply parallelizing causal phases.**
    Incompatible with current entropy dependencies: the next phase predictor depends on
    decoded previous phases. This requires a new model/bitstream and retraining, not merely
    a different scheduler.

12. **Further MoE/retraining as a solution to the current runtime gap.**
    Only one expert is active, so six experts are not the cause of a sixfold slowdown.
    Convolution runtime is currently more important than adding experts.

## 8. Most Promising Directions for New Ideas

1. Specialized GB10 kernels for grouped 3×3 ConvScale, especially `e12` and `e22`.
2. An AOT/native pipeline with preselected algorithms, avoiding 40–70-second runtime
   compilation.
3. ConvScale + GELU + residual/Haar kernel fusion where it demonstrably reduces memory
   traffic.
4. Integer/QAT tensor-core inference with matched-bpp RD validation; ordinary BF16 has
   already been rejected.
5. Power/clock and warmup profiling: short isolated runs on GB10 are sometimes significantly
   slower than a stable warm state, so medians across multiple runs are required.
6. Cheaper sparse bitmap/plain-candidate selection only after measuring the actual encode
   critical path.
7. A new parallel-friendly context architecture makes sense only as a separate retraining
   experiment; the current four-phase predictor itself is no longer the main bottleneck.

Every candidate requires a real OXP9 roundtrip, all six experts, entropy consistency checks,
matched-bpp PSNR/MS-SSIM/LPIPS, 12-MP encode/decode medians, a 201-MP memory/tiling regression
check, and validation of existing streams. Success must not be judged only by imports,
compilation, or the timing of one isolated CUDA call.
