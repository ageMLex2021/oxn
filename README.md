# OXP9 PICO MoE

This branch contains only the independent PICO/OXP9 experiment: the current six-expert model,
its final weights, codec/training source, native entropy dependencies, benchmarks, and research
reports. It intentionally contains no Tiny/OXN model weights or multi-codec UI.

## Start Here

- [Reproduction guide](PICO_REPRODUCE.md)
- [Architecture, speed, quality, and bottlenecks](PICO_SPEED_BRIEF_EN.md)
- [Russian brief](PICO_SPEED_BRIEF.md)
- [Experiment reports](speed_experiments/README.md)

## Quick Roundtrip

```bash
python3 -m pip install -r requirements.txt
make
python3 oxn_pico.py c input.jpg restored.oxh --q 20 --verify
python3 oxn_pico.py d restored.oxh restored.png
```

The CLI defaults to `oxn_pico_v8_moe6_final.pt` and OXP9. On ARM/CUDA, the Triton analysis
backend is used when available. Set `OXP9_TRITON=0` to restore the original PyTorch path.

This is an independent implementation, not Apple code or weights. Apple PICO remains ahead
in the documented speed and perceptual-RD comparison; no victory is claimed.

License: see `LICENSE`. Downloaded Apple evaluation assets are separately covered by the
`LICENSE_DATA` fetched by `bench_pico_data.py`.
