#!/usr/bin/env python3
"""Full CS311 POC: direct grouped 3x3 plus fused NHWC pointwise operations."""

import argparse
import json
import os
import statistics
import sys

import torch
import triton
import triton.language as tl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triton_direct import DirectGrouped3x3


@triton.jit
def _pointwise_kernel(xp, wp, bp, rp, op, pixels, cin, cout,
                      M: tl.constexpr, GELU: tl.constexpr, RESIDUAL: tl.constexpr):
    pid = tl.program_id(0)
    out_block = tl.program_id(1)
    pos = pid * M + tl.arange(0, M)
    out = out_block * 32 + tl.arange(0, 32)
    acc = tl.zeros((M, 32), dtype=tl.float32)
    for ks in tl.range(0, cin, 32):
        k = ks + tl.arange(0, 32)
        xv = tl.load(xp + pos[:, None] * cin + k[None, :],
                     mask=(pos < pixels)[:, None], other=0.0)
        wv = tl.load(wp + k[:, None] * cout + out[None, :])
        acc = tl.dot(xv, wv, acc, input_precision='tf32')
    acc += tl.load(bp + out)[None, :]
    if GELU:
        acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))
    if RESIDUAL:
        acc += tl.load(rp + pos[:, None] * cout + out[None, :],
                       mask=(pos < pixels)[:, None], other=0.0)
    tl.store(op + pos[:, None] * cout + out[None, :], acc,
             mask=(pos < pixels)[:, None])


class Pointwise:
    def __init__(self, conv, gelu, residual=False, block=64):
        self.weight = conv.weight.detach().reshape(conv.out_channels, conv.in_channels).t().contiguous()
        self.bias = conv.bias.detach()
        self.cout = conv.out_channels
        self.cin = conv.in_channels
        self.gelu = gelu
        self.residual = residual
        self.block = block

    def __call__(self, x, residual=None):
        pixels, cin = x.shape
        out = torch.empty((pixels, self.cout), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(pixels, self.block), triton.cdiv(self.cout, 32))
        _pointwise_kernel[grid](x, self.weight, self.bias, residual if residual is not None else x,
                                out, pixels, cin, self.cout, M=self.block,
                                GELU=self.gelu, RESIDUAL=self.residual, num_warps=4)
        return out


class FusedCS311:
    def __init__(self, block, block_size=64):
        self.conv3 = DirectGrouped3x3(block.net[0], gelu=True, block=block_size)
        self.point1 = Pointwise(block.net[2], gelu=True, block=block_size)
        self.point2 = Pointwise(block.net[4], gelu=False, residual=True, block=block_size)

    def __call__(self, x):
        h, w, cin = x.shape
        flat = x.reshape(h * w, cin)
        h1 = self.conv3(x)
        h2 = self.point1(h1.reshape(h * w, -1))
        out = self.point2(h2, flat)
        return out.reshape(h, w, -1)


def benchmark(layer, h, w, runs, warmup, block_size):
    import oxn_pico as pico
    model, _ = pico._load('oxn_pico_v8_moe6_final.pt')
    model.reparameterize()
    block = getattr(model, layer)[0].cuda().to(memory_format=torch.channels_last)
    x = torch.randn(1, block.net[0].in_channels, h, w, device='cuda').to(memory_format=torch.channels_last)
    xn = x[0].permute(1, 2, 0).contiguous()
    candidate = FusedCS311(block, block_size)
    with torch.no_grad():
        ref = block(x)[0].permute(1, 2, 0).contiguous()
        got = candidate(xn)
    delta = (ref - got).abs()
    relative_l2 = ((ref - got).norm() / ref.norm()).item()
    baseline = []
    candidate_times = []
    for _ in range(warmup):
        with torch.no_grad():
            block(x)
        candidate(xn)
    torch.cuda.synchronize()
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            block(x)
        end.record(); end.synchronize(); baseline.append(start.elapsed_time(end) / 1000)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); candidate(xn); end.record(); end.synchronize()
        candidate_times.append(start.elapsed_time(end) / 1000)
    return {'layer': layer, 'shape': [h, w], 'block': block_size,
            'baseline_s': baseline, 'candidate_s': candidate_times,
            'baseline_median_s': statistics.median(baseline),
            'candidate_median_s': statistics.median(candidate_times),
            'max_abs': delta.max().item(), 'mean_abs': delta.mean().item(),
            'relative_l2': relative_l2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', default='e12', choices=('e11', 'e12', 'e21', 'e22'))
    ap.add_argument('--height', type=int, default=1500)
    ap.add_argument('--width', type=int, default=2000)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--block', type=int, default=64, choices=(32, 64, 128))
    ap.add_argument('--out')
    args = ap.parse_args()
    result = benchmark(args.layer, args.height, args.width, args.runs, args.warmup, args.block)
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
