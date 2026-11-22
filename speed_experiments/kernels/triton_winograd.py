#!/usr/bin/env python3
"""Isolated F(2x2, 3x3) grouped NHWC convolution POC for PICO CS311."""

import argparse
import json
import statistics
import time
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@triton.jit
def _load_d(xp, y, x, h, w, stride_h, stride_w, base):
    pos = y * stride_h + x * stride_w + base + tl.arange(0, 32)
    return tl.load(xp + pos, mask=(y >= 0) & (y < h) & (x >= 0) & (x < w), other=0.0)


@triton.jit
def _mul(ubase, ustep, pos, v):
    u = tl.load(ubase + pos * ustep)
    return tl.sum(u * v[None, :], axis=1)


@triton.jit
def _activate(v, bias, GELU: tl.constexpr):
    v = v + bias
    if GELU:
        v = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
    return v


@triton.jit
def _winograd_kernel(xp, up, bp, op, h, w, cin, cout, groups,
                     tiles_x, tiles_y, GELU: tl.constexpr):
    tile = tl.program_id(0)
    out_block = tl.program_id(1)
    ty = tile // tiles_x
    tx = tile % tiles_x
    y0 = ty * 2
    x0 = tx * 2
    out_start = out_block * 32
    cout_group = cout // groups
    group = out_start // cout_group
    out_local = out_start % cout_group + tl.arange(0, 32)
    out = out_start + tl.arange(0, 32)
    k = tl.arange(0, 32)
    base = group * 32
    sh = w * cin
    sw = cin
    d00 = _load_d(xp, y0 - 1, x0 - 1, h, w, sh, sw, base)
    d01 = _load_d(xp, y0 - 1, x0, h, w, sh, sw, base)
    d02 = _load_d(xp, y0 - 1, x0 + 1, h, w, sh, sw, base)
    d03 = _load_d(xp, y0 - 1, x0 + 2, h, w, sh, sw, base)
    d10 = _load_d(xp, y0, x0 - 1, h, w, sh, sw, base)
    d11 = _load_d(xp, y0, x0, h, w, sh, sw, base)
    d12 = _load_d(xp, y0, x0 + 1, h, w, sh, sw, base)
    d13 = _load_d(xp, y0, x0 + 2, h, w, sh, sw, base)
    d20 = _load_d(xp, y0 + 1, x0 - 1, h, w, sh, sw, base)
    d21 = _load_d(xp, y0 + 1, x0, h, w, sh, sw, base)
    d22 = _load_d(xp, y0 + 1, x0 + 1, h, w, sh, sw, base)
    d23 = _load_d(xp, y0 + 1, x0 + 2, h, w, sh, sw, base)
    d30 = _load_d(xp, y0 + 2, x0 - 1, h, w, sh, sw, base)
    d31 = _load_d(xp, y0 + 2, x0, h, w, sh, sw, base)
    d32 = _load_d(xp, y0 + 2, x0 + 1, h, w, sh, sw, base)
    d33 = _load_d(xp, y0 + 2, x0 + 2, h, w, sh, sw, base)

    t00 = d00 - d20; t01 = d01 - d21; t02 = d02 - d22; t03 = d03 - d23
    v00 = t00 - t02; v01 = t01 + t02; v02 = t02 - t01; v03 = t01 - t03
    t00 = d10 + d20; t01 = d11 + d21; t02 = d12 + d22; t03 = d13 + d23
    v10 = t00 - t02; v11 = t01 + t02; v12 = t02 - t01; v13 = t01 - t03
    t00 = d20 - d10; t01 = d21 - d11; t02 = d22 - d12; t03 = d23 - d13
    v20 = t00 - t02; v21 = t01 + t02; v22 = t02 - t01; v23 = t01 - t03
    t00 = d10 - d30; t01 = d11 - d31; t02 = d12 - d32; t03 = d13 - d33
    v30 = t00 - t02; v31 = t01 + t02; v32 = t02 - t01; v33 = t01 - t03

    ubase = up + group * cout_group * 32 + out_local[:, None] * 32 + k[None, :]
    ustep = groups * cout_group * 32

    m00 = _mul(ubase, ustep, 0, v00); m01 = _mul(ubase, ustep, 1, v01)
    m02 = _mul(ubase, ustep, 2, v02); m03 = _mul(ubase, ustep, 3, v03)
    m10 = _mul(ubase, ustep, 4, v10); m11 = _mul(ubase, ustep, 5, v11)
    m12 = _mul(ubase, ustep, 6, v12); m13 = _mul(ubase, ustep, 7, v13)
    m20 = _mul(ubase, ustep, 8, v20); m21 = _mul(ubase, ustep, 9, v21)
    m22 = _mul(ubase, ustep, 10, v22); m23 = _mul(ubase, ustep, 11, v23)
    m30 = _mul(ubase, ustep, 12, v30); m31 = _mul(ubase, ustep, 13, v31)
    m32 = _mul(ubase, ustep, 14, v32); m33 = _mul(ubase, ustep, 15, v33)

    t00 = m00 + m10 + m20; t01 = m01 + m11 + m21; t02 = m02 + m12 + m22; t03 = m03 + m13 + m23
    t10 = m10 - m20 - m30; t11 = m11 - m21 - m31; t12 = m12 - m22 - m32; t13 = m13 - m23 - m33
    bias = tl.load(bp + out)
    o00 = _activate(t00 + t01 + t02, bias, GELU)
    o01 = _activate(t01 - t02 - t03, bias, GELU)
    o10 = _activate(t10 + t11 + t12, bias, GELU)
    o11 = _activate(t11 - t12 - t13, bias, GELU)
    mask_out = out < cout
    osh = w * cout
    osw = cout
    tl.store(op + y0 * osh + x0 * osw + out, o00, mask=mask_out)
    tl.store(op + y0 * osh + (x0 + 1) * osw + out, o01, mask=mask_out & (x0 + 1 < w))
    tl.store(op + (y0 + 1) * osh + x0 * osw + out, o10, mask=mask_out & (y0 + 1 < h))
    tl.store(op + (y0 + 1) * osh + (x0 + 1) * osw + out, o11,
             mask=mask_out & (y0 + 1 < h) & (x0 + 1 < w))


def transform_weight(weight, groups):
    cout, cin_group = weight.shape[:2]
    g = weight.new_tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5],
                           [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]])
    u = torch.einsum('ar,okrs,bs->okab', g, weight, g)
    cout_group = cout // groups
    u = u.reshape(groups, cout_group, cin_group, 4, 4)
    u = u.permute(3, 4, 0, 1, 2).contiguous()
    return u.reshape(16, groups, cout_group, cin_group)


class WinogradGrouped3x3:
    def __init__(self, conv, gelu=True):
        self.groups = conv.groups
        self.cout = conv.out_channels
        self.u = transform_weight(conv.weight.detach(), self.groups)
        self.bias = conv.bias.detach()
        self.gelu = gelu

    def __call__(self, x):
        h, w, cin = x.shape
        assert cin // self.groups == 32
        out = torch.empty((h, w, self.cout), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(h, 2) * triton.cdiv(w, 2), triton.cdiv(self.cout, 32))
        _winograd_kernel[grid](x, self.u, self.bias, out, h, w, cin, self.cout,
                               self.groups, triton.cdiv(w, 2), triton.cdiv(h, 2),
                               GELU=self.gelu, num_warps=4)
        return out


def benchmark(layer, h, w, runs, warmup):
    import oxn_pico as pico
    model, _ = pico._load('oxn_pico_v8_moe6_final.pt')
    model.reparameterize()
    conv = getattr(model, layer)[0].net[0].cuda()
    x = torch.randn(1, conv.in_channels, h, w, device='cuda').to(memory_format=torch.channels_last)
    xn = x[0].permute(1, 2, 0).contiguous()
    candidate = WinogradGrouped3x3(conv)
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
            ref = F.gelu(conv(x))
        end.record(); end.synchronize()
        baseline.append(start.elapsed_time(end) / 1000)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); got = candidate(xn); end.record(); end.synchronize()
        candidate_times.append(start.elapsed_time(end) / 1000)
    return {'layer': layer, 'shape': [h, w, conv.in_channels, conv.out_channels],
            'groups': conv.groups, 'baseline_s': baseline, 'candidate_s': candidate_times,
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
    ap.add_argument('--out')
    args = ap.parse_args()
    result = benchmark(args.layer, args.height, args.width, args.runs, args.warmup)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
