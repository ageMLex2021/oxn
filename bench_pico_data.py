#!/usr/bin/env python3
"""Download the public Apple PICO evaluation assets used by bench_pico_reimpl.py."""

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = 'https://raw.githubusercontent.com/apple-aiml-research/ml-pico/main'
SCENES = ('philipp-reiner-207', 'b1400e99ef413c8fee7d5c0be41d7d1e',
          'alexander-shustov-73', 'kodim23')


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    path.write_bytes(data)
    print(f'{path} ({len(data)} bytes)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('data/eval/pico'))
    args = ap.parse_args()
    download(f'{ROOT}/LICENSE_DATA', args.out / 'LICENSE_DATA')
    download(f'{ROOT}/assets/bpp_data.json', args.out / 'bpp_data.json')
    with (args.out / 'bpp_data.json').open() as f:
        rates = json.load(f)
    for scene in SCENES:
        if scene not in rates:
            raise RuntimeError(f'missing bpp for {scene}')
        for name in ('reference.png', 'pico.png'):
            download(f'{ROOT}/assets/scenes/{scene}/{name}', args.out / scene / name)


if __name__ == '__main__':
    main()
