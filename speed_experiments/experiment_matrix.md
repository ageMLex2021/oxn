# OXP9 Speed Experiment Matrix

All timings are measured locally on ARM64 + NVIDIA GB10. Microbenchmarks use CUDA events;
full codec benchmarks use wall-clock `time.perf_counter()` around complete operations.

## Environment and Baseline

- Baseline commit: `ae604e75faff17c9a02a35586b7b16d0947c1785`.
- PyTorch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24.0, Triton 3.8.0.
- TensorRT, ONNX, and ONNX Runtime are not installed. TensorRT POC: **REJECT / unavailable**
  without changing the working software stack. No driver/system update was performed.
- Current 12-MP baseline (`baseline_12mp.json`, five runs after two warmups):
  encode 1.420312 s, decode 0.976711 s, 201018 bytes / 0.134012 bpp,
  PSNR 17.693015, MS-SSIM 0.696739, LPIPS 0.436864.
- Current 201-MP baseline (`baseline_201mp.json`, five runs after two warmups):
  encode 22.373806 s, decode 15.759252 s. Auto tiling selected 960 in this memory state;
  this must not be mixed with older 2048-tile results without noting the different bpp.

## TensorRT e12

- Hypothesis: AOT graph compilation could improve grouped convolution.
- Result: TensorRT/ONNX/ONNX Runtime unavailable in the actual environment.
- Decision: **REJECT for this checkout**. No claim about TensorRT performance is made.

## Winograd F(2×2,3×3), e12

- Implementation: `kernels/triton_winograd.py`; grouped NHWC F(2×2), pretransformed weights,
  fused bias/GELU, specialized 32 input channels/group and 32-output blocks.
- Numerical result at real shape: max abs 0.002154, mean abs 0.0000866.
- Real-shape timing: baseline median 0.132213 s, candidate median 0.132823 s.
- Full codec: not integrated because the isolated candidate was not faster.
- Decision: **REJECT**. Transform/multiply overhead offset the theoretical FLOP reduction.
  F(4×4) was not expanded after this stop-rule result.

## Direct Triton Grouped Conv3×3 + GELU

- Implementation: `kernels/triton_direct.py`; nine TF32/FP16 `tl.dot` implicit-GEMM steps,
  NHWC, no dense zero expansion, pretransformed grouped weights, fused bias/GELU.
- e12 TF32 block 64: baseline 0.126696 s, candidate 0.035086 s.
- e22 TF32 block 32: baseline 0.047959 s, candidate 0.019179 s.
- e12 FP16 block 64: baseline 0.123623 s, candidate 0.033299 s.
- FP16 uses FP32 accumulation/GELU. It was more accurate than TF32 on this workload:
  mean abs 0.000000535 versus approximately 0.000267 for TF32.
- Full synthesis trial: bitstream unchanged, max RGB difference 1, but decode median 1.007 s
  versus the local 0.977 s baseline; layout/conversion overhead removed the kernel gain.
- Decision: **ACCEPT analysis FP16; REJECT current synthesis trial.**

## Full CS311 Pointwise Fusion

- Implementation: `kernels/triton_cs311.py`; grouped 3×3+GELU, 1×1+GELU,
  1×1+residual as three Triton kernels.
- e12 full-block baseline median 0.205203 s, candidate 0.122305 s.
- This did not improve the already faster direct grouped kernel + existing PyTorch pointwise
  path sufficiently to justify additional numerical drift/complexity.
- Decision: **REJECT for production; retained as POC.**

## Production Triton Analysis, FP32 Accumulation

- Production code: `oxn_pico_triton.py`; enabled on ARM/CUDA fast path, `OXP9_TRITON=0`
  disables it. Original convolution fallback is retained for errors and unsupported shapes.
- Twelve-MP final candidate (`reports/triton_production_fp16_12mp.json`, seven runs after
  two warmups): encode **1.081851 s**, decode **0.924915 s**, 201008 bytes / 0.13400533 bpp,
  PSNR 17.693300, MS-SSIM 0.696788, LPIPS 0.436955, CUDA reserved 8.523 GB.
- Against the fresh baseline: encode −23.83%, decode −5.30%, 10 fewer bytes; quality metrics
  are effectively unchanged at this operating point. Different processes can have small
  fast-synthesis numerical variation, so final acceptance uses the complete RD suite.
- Cold genuinely empty Triton cache: runtime setup 0.591 s, first 768×512 encode 1.498 s,
  second cached encode 0.0297 s.
- All six target-bpp experts passed verified roundtrips. Sixteen old OXP6–OXP9 fixtures decoded.
- Apple-scene candidate average: 0.31931476 bpp, 26.92439 dB PSNR, 0.91445766 MS-SSIM,
  0.27470330 LPIPS; previous OXP9 average was 0.31932173 / 26.92419 / 0.91442063 /
  0.27475078. No RD regression was observed at the shown precision.
- 201-MP candidate passed five runs after two warmups: encode 16.372 s, decode 14.992 s,
  PSNR 20.45939, MS-SSIM 0.924189, RSS 4.77 GB. Auto tiling chose 1280 versus baseline 960,
  so these timings/bpp are a stability result, not a controlled same-tile comparison.
- Decision: **ACCEPT**. The candidate is not presented as matching Apple PICO speed.
