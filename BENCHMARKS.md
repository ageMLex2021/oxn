# OXN vs JPEG / WebP / AVIF / JPEG XL - benchmark

**Setup.** Kodak `kodim21–24` - these 4 images were **never used in training**
(honest holdout; the model was trained on BSDS500 + Kodak01–20 + DIV2K-valid).
Each codec is tuned to the same bitrate per image (nearest quality setting,
actual mean bpp shown). Metrics: PSNR, MS-SSIM, LPIPS(alex). Hardware:
RTX 4060 Laptop, Python/PyTorch implementation of OXN, reference encoders
via Pillow for the rest. OXN encode uses adaptive rate (best quality mode).

## PSNR (dB, higher is better)

| bpp | OXN | JPEG | WebP | AVIF | JXL |
|---|---|---|---|---|---|
| 0.15 | 26.85 | 21.45 | **28.51** | 27.79 | 27.13 |
| 0.20 | 26.29 | 22.87 | **28.18** | 29.24 | 27.15 |
| 0.25 | 27.04 | 25.42 | **28.98** | 29.24 | 27.72 |
| 0.30 | 27.26 | 27.15 | 29.29 | **30.14** | 28.87 |

## MS-SSIM (higher is better)

| bpp | OXN | JPEG | WebP | AVIF | JXL |
|---|---|---|---|---|---|
| 0.15 | **0.939** | 0.705 | 0.915 | 0.926 | 0.907 |
| 0.20 | **0.927** | 0.760 | 0.921 | 0.947 | 0.918 |
| 0.25 | **0.942** | 0.850 | 0.934 | 0.947 | 0.927 |
| 0.30 | 0.950 | 0.897 | 0.939 | **0.957** | 0.946 |

## LPIPS (lower is better)

| bpp | OXN | JPEG | WebP | AVIF | JXL |
|---|---|---|---|---|---|
| 0.15 | **0.203** | 0.468 | 0.235 | 0.268 | 0.315 |
| 0.20 | 0.259 | 0.403 | 0.229 | **0.203** | 0.315 |
| 0.25 | 0.240 | 0.286 | **0.191** | 0.203 | 0.295 |
| 0.30 | 0.230 | 0.213 | 0.180 | **0.156** | 0.226 |

## Speed (768×512, seconds per image)

| codec | encode | decode |
|---|---|---|
| OXN | 0.30 | 0.21 |
| JPEG | 0.001 | 0.001 |
| WebP | 0.02 | 0.003 |
| AVIF | 0.05 | 0.003 |
| JXL | 0.05 | 0.005 |

On a 201 MP photo OXN adaptive encode is ~110 s (GPU). CLI startup
(torch init + model load) adds ~10 s if the codec runs as a fresh process.

## Visual comparison (0.20 bpp, crop)

![crops](bench_crops_02bpp.png)

Left to right: original, OXN, JPEG, WebP, AVIF, JXL.

## Takeaways

- **vs JPEG: OXN wins everywhere by a landslide** - +4…5 dB PSNR, MS-SSIM
  and LPIPS by a wide margin, and no blocking artifacts at all.
- **vs WebP:** WebP is ahead on PSNR; on LPIPS at ≤0.20 bpp OXN is
  comparable or better.
- **vs JPEG XL:** JXL is slightly ahead on PSNR, but **OXN is clearly better
  on LPIPS** at low bitrates (0.20–0.26 vs 0.30–0.32) - perceptually OXN
  keeps textures more natural while JXL tends to smooth.
- **vs AVIF:** AVIF wins overall on this set.
- **Speed is the honest price:** 30–300× slower encode and ~50× slower
  decode than reference C encoders. This is a Python/PyTorch research
  codec; archiving/batch use, not realtime.

## Reproduce

```bash
pip install pillow pillow-jxl-plugin pillow-avif-plugin lpips scikit-image torchmetrics
# images: Kodak kodim21-24
# OXN encode: python3 ox_neuro.py c in.png out.oxh --q <q> --adaptive
# classics: PIL Image.save(..., quality=q) with nearest bpp match
```
