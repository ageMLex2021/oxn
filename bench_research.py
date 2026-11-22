#!/usr/bin/env python3
"""Matched rate/distortion/runtime benchmark for the PICO and tiny experiments."""

import argparse
import json
import math
import os
import resource
import statistics
import struct
import time

import numpy as np
import torch
from PIL import Image

import ox_neuro as ox
import oxn_pico as pico


def _compress(src, model, q, ckpt, ctx, verify, target_bpp=None, candidate_count=0,
              stream_version=5):
    if model == 'pico':
        if target_bpp:
            return pico.compress_target_bpp(src, target_bpp, ckpt, verify,
                                            stream_version=stream_version)
        return pico.compress(src, q, ckpt, verify, stream_version=stream_version)
    ox.TINY_CKPT = ckpt
    ox.TINY_CTX7 = ctx
    ox.CTX7_FILE = ctx
    ox._BOOST_ON = False
    if target_bpp:
        return ox.neuro_compress_tiny_v2_target_bpp(src, target_bpp, ckpt, ctx)
    return ox.neuro_compress_tiny_v2(src, q, ckpt=ckpt, ctx_file=ctx,
                                     candidate_count=candidate_count)


def _roundtrip(src, model, q, ckpt, ctx, verify, target_bpp=None, candidate_count=0,
               stream_version=5):
    blob = _compress(src, model, q, ckpt, ctx, verify, target_bpp, candidate_count,
                     stream_version)
    return blob, pico.decompress(blob, ckpt) if model == 'pico' else ox.neuro_decompress(blob)


def _image_metrics(src, rec, metric_size, lp, gk, dev):
    h, w = src.shape[:2]
    mse = np.zeros(3, np.float64)
    delta = np.zeros(3, np.float64)
    for y in range(0, h, 256):
        a = src[y:y + 256].astype(np.float32)
        b = rec[y:y + 256].astype(np.float32)
        d = b - a
        mse += (d * d).sum((0, 1))
        delta += d.sum((0, 1))
    mse = float(mse.sum() / (h * w * 3))
    result = {
        'psnr': 99.0 if mse == 0 else -10 * math.log10(mse / (255.0 * 255.0)),
        'delta_rgb': (delta / (h * w)).tolist(),
    }
    if max(h, w) > metric_size:
        scale = metric_size / max(h, w)
        size = max(1, round(w * scale)), max(1, round(h * scale))
        src = np.asarray(Image.fromarray(src).resize(size, Image.Resampling.LANCZOS))
        rec = np.asarray(Image.fromarray(rec).resize(size, Image.Resampling.LANCZOS))
    a = torch.from_numpy(src.copy()).permute(2, 0, 1)[None].to(dev).float() / 255
    b = torch.from_numpy(rec.copy()).permute(2, 0, 1)[None].to(dev).float() / 255
    with torch.no_grad():
        result['ms_ssim'] = ox.ms_ssim(a, b, gk).item()
        if lp is not None:
            result['lpips'] = lp(a * 2 - 1, b * 2 - 1).item()
    return result


def _boundary_mse(src, rec, blob, model):
    if model == 'tiny':
        tile = 16
        pos = 4
    elif blob[:4] in (b'OXP3', b'OXP4', b'OXP5', b'OXP6', b'OXP7', b'OXP8', b'OXP9'):
        tile = struct.unpack('<I', blob[20:24])[0]
        pos = 24 if blob[:4] == b'OXP3' else 30 if blob[:4] == b'OXP9' else \
            29 if blob[:4] in (b'OXP6', b'OXP7', b'OXP8') else 28
    else:
        return None
    h, w = src.shape[:2]
    mask = np.zeros((h, w), bool)
    for y in range(tile, h, tile):
        mask[y:y + 1] = True
    for x in range(tile, w, tile):
        mask[:, x:x + 1] = True
    if not mask.any():
        return None
    d = (src.astype(np.float32) - rec.astype(np.float32)) ** 2
    return {'tile': tile, 'header_bytes': pos,
            'boundary_mse': float(d[mask].mean()),
            'interior_mse': float(d[~mask].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('images', nargs='+')
    ap.add_argument('--model', choices=('pico', 'tiny'), required=True)
    ap.add_argument('--q', type=float, nargs='+', default=[100])
    ap.add_argument('--target-bpp', type=float)
    ap.add_argument('--scale-candidates', type=int, default=0)
    ap.add_argument('--stream-version', type=int, choices=(5, 6, 7, 8, 9), default=5)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--ckpt')
    ap.add_argument('--ctx')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--lpips', action='store_true')
    ap.add_argument('--metrics-size', type=int, default=2048)
    ap.add_argument('--out')
    args = ap.parse_args()
    ckpt = args.ckpt or (pico.CKPT if args.model == 'pico' else ox.TINY_CKPT)
    ctx = args.ctx or ox.TINY_CTX7
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lp = None
    if args.lpips:
        import lpips
        lp = lpips.LPIPS(net='alex').to(dev).eval()
    gk = ox._gauss_kernel().to(dev)
    rows = []
    for filename in args.images:
        src = ox.load_rgb(filename)
        npix = src.shape[0] * src.shape[1]
        for q in args.q:
            for _ in range(args.warmup):
                _roundtrip(src, args.model, q, ckpt, ctx, args.verify, args.target_bpp,
                           args.scale_candidates, args.stream_version)
            if dev == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            enc, dec = [], []
            blob = rec = None
            for _ in range(args.runs):
                started = time.time()
                blob = _compress(src, args.model, q, ckpt, ctx, args.verify,
                                 args.target_bpp, args.scale_candidates, args.stream_version)
                enc.append(time.time() - started)
                started = time.time()
                rec = pico.decompress(blob, ckpt) if args.model == 'pico' else ox.neuro_decompress(blob)
                dec.append(time.time() - started)
            encoded_q = struct.unpack('<f', blob[12:16])[0] \
                if args.target_bpp and args.model == 'pico' else q
            row = {
                'image': filename, 'model': args.model, 'q': encoded_q,
                'target_bpp': args.target_bpp,
                'scale_candidates': args.scale_candidates,
                'stream_version': args.stream_version,
                'width': src.shape[1], 'height': src.shape[0],
                'magic': blob[:4].decode(), 'bytes': len(blob),
                'bpp': len(blob) * 8 / npix,
                'encode_s': statistics.median(enc),
                'decode_s': statistics.median(dec),
                'encode_runs_s': enc, 'decode_runs_s': dec,
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if dev == 'cuda' else 0,
                'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'boundary': _boundary_mse(src, rec, blob, args.model),
                **_image_metrics(src, rec, args.metrics_size, lp, gk, dev),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del rec, blob
            if dev == 'cuda':
                torch.cuda.empty_cache()
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
