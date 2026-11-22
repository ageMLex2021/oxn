#!/usr/bin/env python3
"""Direct grouped NHWC 3x3 implicit-GEMM POC with fused bias/GELU."""

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@triton.jit
def _direct_kernel(xp, wp, bp, op, h, w, cin, cout, groups,
                   pixels, M: tl.constexpr, GELU: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    out_block = tl.program_id(1)
    pos = pid * M + tl.arange(0, M)
    yy = pos // w
    xx = pos % w
    out_start = out_block * 32
    cout_group = cout // groups
    group = out_start // cout_group
    out_local = out_start % cout_group + tl.arange(0, 32)
    out = out_start + tl.arange(0, 32)
    k = tl.arange(0, 32)
    acc = tl.zeros((M, 32), dtype=tl.float32)
    wbase = wp + group * (9 * 32 * cout_group) + out_local[None, :]
    for r in tl.static_range(9):
        ky = r // 3
        kx = r % 3
        iy = yy + ky - 1
        ix = xx + kx - 1
        valid = (pos < pixels) & (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
        xv = tl.load(xp + iy[:, None] * (w * cin) + ix[:, None] * cin +
                     group * 32 + k[None, :], mask=valid[:, None], other=0.0)
        wv = tl.load(wbase + r * (32 * cout_group) + k[:, None] * cout_group)
        if MODE == 'fp16':
            acc = tl.dot(xv.to(tl.float16), wv, acc)
        elif MODE == 'bf16':
            acc = tl.dot(xv.to(tl.bfloat16), wv, acc)
        else:
            acc = tl.dot(xv, wv, acc, input_precision='tf32')
    acc += tl.load(bp + out)[None, :]
    if GELU:
        acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))
    tl.store(op + yy[:, None] * (w * cout) + xx[:, None] * cout + out[None, :],
             acc, mask=(pos < pixels)[:, None])


def transform_weight(weight, groups):
    cout, cin_group = weight.shape[:2]
    cout_group = cout // groups
    return weight.reshape(groups, cout_group, cin_group, 9).permute(0, 3, 2, 1).contiguous()


class DirectGrouped3x3:
    def __init__(self, conv, gelu=True, block=32, precision='tf32'):
        self.groups = conv.groups
        self.cout = conv.out_channels
        self.precision = precision
        dtype = torch.float16 if precision == 'fp16' else torch.bfloat16 if precision == 'bf16' else torch.float32
        self.w = transform_weight(conv.weight.detach(), self.groups).to(dtype)
        self.bias = conv.bias.detach()
        self.gelu = gelu
        self.block = block

    def __call__(self, x):
        h, w, cin = x.shape
        assert cin // self.groups == 32
        out = torch.empty((h, w, self.cout), device=x.device, dtype=x.dtype)
        pixels = h * w
        grid = (triton.cdiv(pixels, self.block), triton.cdiv(self.cout, 32))
        _direct_kernel[grid](x, self.w, self.bias, out, h, w, cin, self.cout,
                             self.groups, pixels, M=self.block, GELU=self.gelu,
                             MODE=self.precision, num_warps=4)
        return out


def benchmark(layer, h, w, runs, warmup, block, precision):
    import oxn_pico as pico
    model, _ = pico._load('oxn_pico_v8_moe6_final.pt')
    model.reparameterize()
    conv = getattr(model, layer)[0].net[0].cuda()
    x = torch.randn(1, conv.in_channels, h, w, device='cuda').to(memory_format=torch.channels_last)
    xn = x[0].permute(1, 2, 0).contiguous()
    candidate = DirectGrouped3x3(conv, block=block, precision=precision)
    with torch.no_grad():
        ref = F.gelu(conv(x))[0].permute(1, 2, 0).contiguous()
        got = candidate(xn)
    delta = (ref - got).abs()
    relative_l2 = ((ref - got).norm() / ref.norm()).item()
    torch.cuda.synchronize()
    baseline = []
    candidate_times = []
    for _ in range(warmup):
        with torch.no_grad():
            F.gelu(conv(x))
        candidate(xn)
    torch.cuda.synchronize()
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            F.gelu(conv(x))
        end.record(); end.synchronize()
        baseline.append(start.elapsed_time(end) / 1000)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); candidate(xn); end.record(); end.synchronize()
        candidate_times.append(start.elapsed_time(end) / 1000)
    return {'layer': layer, 'shape': [h, w, conv.in_channels, conv.out_channels],
            'groups': conv.groups, 'block': block, 'precision': precision, 'baseline_s': baseline,
            'candidate_s': candidate_times, 'baseline_median_s': statistics.median(baseline),
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
    ap.add_argument('--block', type=int, default=32, choices=(16, 32, 64, 128))
    ap.add_argument('--precision', choices=('tf32', 'fp16', 'bf16'), default='tf32')
    ap.add_argument('--out')
    args = ap.parse_args()
    result = benchmark(args.layer, args.height, args.width, args.runs, args.warmup,
                       args.block, args.precision)
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
