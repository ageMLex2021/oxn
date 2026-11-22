#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import lpips

import ox_neuro as ox
import oxn_pico as pico


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', default='oxn_pico_reimpl')
    ap.add_argument('--stream-version', type=int, choices=(5, 6, 7, 8, 9), default=5)
    args = ap.parse_args()
    root = Path('data/eval/pico')
    rates = json.loads((root / 'bpp_data.json').read_text())
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lp = lpips.LPIPS(net='alex').to(dev).eval()
    gk = ox._gauss_kernel().to(dev)

    def metrics(src, rec):
        a = torch.from_numpy(src.copy()).permute(2, 0, 1)[None].to(dev).float() / 255
        b = torch.from_numpy(rec.copy()).permute(2, 0, 1)[None].to(dev).float() / 255
        with torch.no_grad():
            return {'psnr': ox.psnr(src, rec), 'ms_ssim': ox.ms_ssim(a, b, gk).item(),
                    'lpips': lp(a * 2 - 1, b * 2 - 1).item()}

    results = {}
    for scene, rr in rates.items():
        folder = root / scene
        src = ox.load_rgb(folder / 'reference.png')
        ref = ox.load_rgb(folder / 'pico.png')
        target = float(rr['pico'])
        best = None
        lo, hi = 1.0, 100.0
        for _ in range(12):
            q = (lo + hi) / 2
            blob = pico.compress(src, q, args.ckpt, stream_version=args.stream_version)
            bpp = len(blob) * 8 / (src.shape[0] * src.shape[1])
            error = abs(bpp - target)
            if best is None or error < best[0]:
                best = error, q, blob, bpp
            if bpp < target:
                lo = q
            else:
                hi = q
        _, q, blob, bpp = best
        rec = pico.decompress(blob, args.ckpt)
        results[scene] = {
            'PICO': {'bpp': target, **metrics(src, ref)},
            'reimplementation': {'q': q, 'bpp': bpp, 'bytes': len(blob), **metrics(src, rec)},
        }
        print(scene, json.dumps(results[scene]), flush=True)
        (folder / f'{args.prefix}.oxh').write_bytes(blob)
        from PIL import Image
        Image.fromarray(rec).save(folder / f'{args.prefix}.png')
        Path(args.out).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
