#!/usr/bin/env python3
"""Full OXP9 benchmark for optional Triton analysis Conv3x3+GELU replacement."""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'speed_experiments', 'kernels'))

import ox_neuro as on
import oxn_pico as pico
from triton_direct import DirectGrouped3x3


class FusedConvGelu(nn.Module):
    def __init__(self, conv, block):
        super().__init__()
        self.kernel = DirectGrouped3x3(conv, gelu=True, block=block)

    def forward(self, x):
        out = self.kernel(x[0].permute(1, 2, 0).contiguous())
        return out.permute(2, 0, 1)[None]


def patch(model, variant, block):
    selected = []
    prefixes = {'e12': ('e12.',), 'e12e22': ('e12.', 'e22.'),
                'analysis': ('e11.', 'e12.', 'e21.', 'e22.', 'e31.', 'e32.'),
                'all': ('e11.', 'e12.', 'e21.', 'e22.', 'e31.', 'e32.'),
                'synthesis': ()}[variant]
    with torch.no_grad():
        for name, module in model.named_modules():
            if not name.startswith(prefixes) or not isinstance(module, pico.CS311):
                continue
            conv = module.net[0]
            module.net[0] = FusedConvGelu(conv, block)
            module.net[1] = nn.Identity()
            selected.append(name)
        if variant in ('synthesis', 'all'):
            for name, module in model.named_modules():
                if not name.startswith('synthesis_experts.') or not isinstance(module, pico.CS311):
                    continue
                conv = module.net[0]
                if conv.groups == 1:
                    continue
                module.net[0] = FusedConvGelu(conv, block)
                module.net[1] = nn.Identity()
                selected.append(name)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--variant', choices=('e12', 'e12e22', 'analysis', 'synthesis', 'all'),
                    default='e12e22')
    ap.add_argument('--block', type=int, default=32, choices=(16, 32, 64, 128))
    ap.add_argument('--q', type=float, default=20)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    src = on.load_rgb(args.image)
    ckpt = os.path.join(ROOT, 'oxn_pico_v8_moe6_final.pt')
    baseline_blob = pico.compress(src, args.q, ckpt, stream_version=9)
    baseline_rec = pico.decompress(baseline_blob, ckpt)
    model, canon, hyper, dev = pico._runtime(ckpt, True)
    selected = patch(model, args.variant, args.block)
    started = time.perf_counter()
    blob = pico.compress(src, args.q, ckpt, stream_version=9)
    cold_s = time.perf_counter() - started
    rec = pico.decompress(blob, ckpt)
    times = []
    decode_times = []
    blobs = []
    for _ in range(args.warmup):
        pico.compress(src, args.q, ckpt, stream_version=9)
    for _ in range(args.runs):
        started = time.perf_counter()
        blob = pico.compress(src, args.q, ckpt, stream_version=9)
        times.append(time.perf_counter() - started)
        started = time.perf_counter()
        rec = pico.decompress(blob, ckpt)
        decode_times.append(time.perf_counter() - started)
        blobs.append(blob)
    delta = rec.astype(np.int16) - baseline_rec.astype(np.int16)
    result = {
        'image': args.image, 'variant': args.variant, 'block': args.block,
        'selected_modules': selected, 'q': args.q, 'stream_version': 9,
        'cold_first_encode_s': cold_s, 'encode_s': times, 'decode_s': decode_times,
        'encode_median_s': statistics.median(times),
        'decode_median_s': statistics.median(decode_times),
        'bytes': len(blob), 'bpp': len(blob) * 8 / (src.shape[0] * src.shape[1]),
        'baseline_bytes': len(baseline_blob),
        'baseline_bpp': len(baseline_blob) * 8 / (src.shape[0] * src.shape[1]),
        'bitstream_identical': blob == baseline_blob,
        'stable_bitstream': all(x == blob for x in blobs),
        'blob_sha256': hashlib.sha256(blob).hexdigest(),
        'baseline_blob_sha256': hashlib.sha256(baseline_blob).hexdigest(),
        'psnr': on.psnr(src, rec), 'baseline_psnr': on.psnr(src, baseline_rec),
        'max_abs_rgb': int(np.abs(delta).max()), 'mean_abs_rgb': float(np.abs(delta).mean()),
        'changed_pixel_fraction': float(np.any(delta != 0, axis=2).mean()),
        'rec_sha256': hashlib.sha256(rec.tobytes()).hexdigest(),
        'baseline_rec_sha256': hashlib.sha256(baseline_rec.tobytes()).hexdigest(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    Path(args.out).with_suffix('.oxh').write_bytes(blob)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
