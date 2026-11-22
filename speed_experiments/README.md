# OXP9 Speed Experiments

Development baseline commit: `ae604e75faff17c9a02a35586b7b16d0947c1785`.
Experiment branch: `oxp9-speed`.
Model: `oxn_pico_v8_moe6_final.pt`; OXP9; q20 for 12 MP and q10 for 201 MP.

## Baseline

```bash
python3 bench_research.py /tmp/oxn_pico_12mp.png --model pico \
  --ckpt oxn_pico_v8_moe6_final.pt --stream-version 9 --q 20 \
  --warmup 2 --runs 5 --lpips --out speed_experiments/baseline_12mp.json

python3 bench_research.py /home/agemlex/Downloads/IMG_20261015_111154.jpg \
  --model pico --ckpt oxn_pico_v8_moe6_final.pt --stream-version 9 --q 10 \
  --warmup 2 --runs 5 --out speed_experiments/baseline_201mp.json
```

## Isolated Kernels

```bash
python3 speed_experiments/kernels/triton_direct.py \
  --layer e12 --height 1500 --width 2000 --block 64 \
  --precision fp16 --warmup 3 --runs 10 \
  --out speed_experiments/reports/triton_direct_e12_fp16.json

python3 speed_experiments/kernels/triton_winograd.py \
  --layer e12 --height 1500 --width 2000 --warmup 3 --runs 10 \
  --out speed_experiments/reports/triton_winograd_e12.json

python3 speed_experiments/kernels/triton_cs311.py \
  --layer e12 --height 1500 --width 2000 --block 64 \
  --warmup 3 --runs 10 --out speed_experiments/reports/triton_cs311_e12.json
```

## Full Production Candidate

The optional production backend is in `oxn_pico_triton.py`. It replaces only compatible
analysis grouped 3×3+GELU operations. Inputs/weights use FP16, accumulation and GELU use FP32.
It is enabled by default on ARM/CUDA fast mode when Triton is installed. Disable it with:

```bash
OXP9_TRITON=0 python3 bench_research.py input.png --model pico \
  --ckpt oxn_pico_v8_moe6_final.pt --stream-version 9 --q 20 \
  --warmup 2 --runs 5 --lpips
```

Enable explicitly and save the candidate report:

```bash
OXP9_TRITON=1 python3 bench_research.py input.png --model pico \
  --ckpt oxn_pico_v8_moe6_final.pt --stream-version 9 --q 20 \
  --warmup 2 --runs 5 --lpips \
  --out speed_experiments/reports/triton_production_fp16_12mp.json
```

See `experiment_matrix.md` for accept/reject decisions. Numerical differences in analysis can
change encoder symbols; actual bitstream size and decoded RD must therefore be checked. The
backend does not modify scale/context/predictor, synthesis, entropy tables, or the decoder.
