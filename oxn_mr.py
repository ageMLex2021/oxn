#!/usr/bin/env python3
"""OXN-MR: jointly trained coarse-to-fine neural image codec prototype."""

import argparse
import math
import os
import struct
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ox_neuro import (N_BUCK, RES_RANGE, _bucket_idx, load_rgb, q2s,
                      rans_dec_param, rans_enc_param)

DEFAULT_CKPT = os.path.join(HERE, 'oxn_mr.pt')


def _gauss_bits(q, mu, sigma):
    import torch
    sq2 = math.sqrt(2.0)
    edges = torch.stack((q + 0.5, q - 0.5))
    cdf = 0.5 * (1 + torch.erf((edges - mu) / (sigma * sq2)))
    return -torch.log2((cdf[0] - cdf[1]).clamp_min(1e-9))


def _table_sigma(sig):
    from ox_neuro import LOG_B_MIN, LOG_B_STEP
    idx = torch.round((torch.log2(sig) - LOG_B_MIN) / LOG_B_STEP).clamp(0, N_BUCK - 1)
    center = torch.exp2(LOG_B_MIN + (idx + 0.5) * LOG_B_STEP)
    return sig + (center - sig).detach()


def _color(x):
    import torch
    r, g, b = x.chunk(3, 1)
    return torch.cat((0.25 * r + 0.5 * g + 0.25 * b,
                      0.5 * r - 0.5 * b,
                      -0.25 * r + 0.5 * g - 0.25 * b), 1)


def _uncolor(x):
    import torch
    y, co, cg = x.chunk(3, 1)
    r = y + co - cg
    g = y + cg
    b = y - co - cg
    return torch.cat((r, g, b), 1)


class MRModel(__import__('torch').nn.Module):
    def __init__(self, width=96, latent0=4, latent1=8, latent2=16, hyper_c=8,
                 scalable=False, threshold1=1.2, threshold2=4.5):
        import torch.nn as nn
        super().__init__()
        self.width = width
        self.channels = (latent0, latent1, latent2)
        self.hyper_c = hyper_c
        self.scalable = scalable
        self.threshold1 = threshold1
        self.threshold2 = threshold2

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(width, width, 3, padding=1)
                self.c2 = nn.Conv2d(width, width, 3, padding=1)

            def forward(self, x):
                h = torch.nn.functional.gelu(self.c1(x))
                return x + self.c2(h)

        self.stem = nn.Sequential(nn.Conv2d(3, width, 5, padding=2), nn.GELU(), Block())
        self.down = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 4, 2, 1), nn.GELU(), Block())
            for _ in range(4))
        self.heads = nn.ModuleList(nn.Conv2d(width, c, 3, padding=1)
                                   for c in self.channels)
        self.hyper_enc = nn.Sequential(
            nn.Conv2d(latent2, width, 3, 2, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, 2, 1), nn.GELU(),
            nn.Conv2d(width, hyper_c, 3, padding=1))
        self.hyper_dec = nn.Sequential(
            nn.ConvTranspose2d(hyper_c, width, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(width, width, 4, 2, 1), nn.GELU(),
            nn.Conv2d(width, latent2 * 2, 3, padding=1))
        self.ctx1 = nn.Sequential(
            nn.Conv2d(latent2 + hyper_c, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 1), nn.GELU(),
            nn.Conv2d(width, latent1 * 2, 1))
        self.ctx0 = nn.Sequential(
            nn.Conv2d(latent2 + latent1 + hyper_c, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 1), nn.GELU(),
            nn.Conv2d(width, latent0 * 2, 1))
        self.hyper_logsig = nn.Parameter(torch.full((hyper_c,), -0.5))
        self.from2 = nn.Conv2d(latent2, width, 3, padding=1)
        self.merge1 = nn.Conv2d(latent1, width, 1)
        self.merge0 = nn.Conv2d(latent0, width, 1)
        self.shared = Block()
        self.up = nn.ModuleList(nn.ConvTranspose2d(width, width, 4, 2, 1)
                                for _ in range(2))
        self.to_color = nn.Conv2d(width, 48, 3, padding=1)

    def analysis(self, x):
        h = self.stem(_color(x))
        h = self.down[0](h)
        h = self.down[1](h)
        z0 = self.heads[0](h)
        h = self.down[2](h)
        z1 = self.heads[1](h)
        h = self.down[3](h)
        z2 = self.heads[2](h)
        return z0, z1, z2

    def entropy2(self, z2):
        w = self.hyper_enc(z2)
        wq = w + (torch.round(w * 2) / 2 - w).detach()
        mb = self.hyper_dec(wq)
        mu, ls = mb.chunk(2, 1)
        return w, wq, mu, torch.nn.functional.softplus(ls) + 0.03

    def entropy1(self, z2q, wq):
        import torch.nn.functional as F
        size = (z2q.shape[2] * 2, z2q.shape[3] * 2)
        hp = F.interpolate(wq, size=size, mode='bilinear', align_corners=False)
        coarse = F.interpolate(z2q, size=size, mode='bilinear', align_corners=False)
        mu, ls = self.ctx1(torch.cat((coarse, hp), 1)).chunk(2, 1)
        return mu, torch.nn.functional.softplus(ls) + 0.03

    def entropy0(self, z1q, z2q, wq):
        import torch.nn.functional as F
        size = z1q.shape[2:]
        size = (size[0] * 2, size[1] * 2)
        hp = F.interpolate(wq, size=size, mode='bilinear', align_corners=False)
        coarse = F.interpolate(z2q, size=size, mode='bilinear', align_corners=False)
        fine = F.interpolate(z1q, size=size, mode='bilinear', align_corners=False)
        mu, ls = self.ctx0(torch.cat((coarse, fine, hp), 1)).chunk(2, 1)
        return mu, torch.nn.functional.softplus(ls) + 0.03

    def synthesis(self, z0, z1, z2):
        import torch.nn.functional as F
        h = self.shared(self.from2(z2))
        h = self.shared(self.up[0](h) + self.merge1(z1))
        h = self.shared(self.up[1](h) + self.merge0(z0))
        return _uncolor(F.pixel_shuffle(self.to_color(h), 4))

    @staticmethod
    def _ste(z, scale):
        zs = z * scale
        return zs + (torch.round(zs) - zs).detach()

    def forward(self, x, scale):
        z0, z1, z2 = self.analysis(x)
        if isinstance(scale, (tuple, list)):
            s0, s1, s2 = scale
            base_scale = float(s2.detach().mean())
        else:
            s0 = s1 = s2 = scale
            base_scale = scale
        w, wq, mu2, sig2 = self.entropy2(z2)
        q2s = self._ste(z2, s2)
        z2q = q2s / s2
        mu1, sig1 = self.entropy1(z2q, wq)
        q1s = self._ste(z1, s1)
        z1q = q1s / s1
        mu0, sig0 = self.entropy0(z1q, z2q, wq)
        q0s = self._ste(z0, s0)
        use1 = not self.scalable or base_scale >= self.threshold1
        use0 = not self.scalable or base_scale >= self.threshold2
        if not use1:
            q1s = q1s * 0
            z1q = q1s
        if not use0:
            q0s = q0s * 0
        xr = self.synthesis(q0s / s0, z1q, z2q)
        sigh = torch.nn.functional.softplus(self.hyper_logsig) + 0.03
        sigh = sigh[None, :, None, None]
        bits = _gauss_bits(wq * 2, torch.zeros_like(wq), _table_sigma(sigh)).sum(dim=(1, 2, 3))
        bpp = bits / (x.shape[2] * x.shape[3])
        for level, (q, mu, sig) in enumerate(((q2s, mu2, sig2),
                                              (q1s, mu1, sig1),
                                              (q0s, mu0, sig0))):
            if level == 1 and not use1 or level == 2 and not use0:
                continue
            mu_r = mu + (torch.round(mu) - mu).detach()
            bits = _gauss_bits(q, mu_r, _table_sigma(sig)).sum(dim=(1, 2, 3))
            bpp = bpp + bits / (x.shape[2] * x.shape[3])
        return xr, bpp, (z0, z1, z2)


def _checkpoint(path):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = MRModel(**ck.get('arch', {}))
    model.load_state_dict(ck['model'])
    return model.eval(), ck.get('arch', {})


def _np_params(*values):
    return [v[0].cpu().numpy().astype(np.float64) for v in values]


def _encode_stream(q, mu, sig):
    mu_r = np.rint(mu).astype(np.int64)
    residual = np.clip(q - mu_r, -RES_RANGE, RES_RANGE - 1) + RES_RANGE
    bids = N_BUCK + _bucket_idx(np.log(sig))
    bids = np.broadcast_to(bids, q.shape)
    blob = rans_enc_param(residual.reshape(-1).astype(np.int32),
                          bids.reshape(-1).astype(np.int32))
    return blob, residual - RES_RANGE + mu_r


def _decode_stream(blob, mu, sig, shape):
    bids = N_BUCK + _bucket_idx(np.log(sig))
    bids = np.broadcast_to(bids, shape)
    residual, used = rans_dec_param(blob, bids.reshape(-1).astype(np.int32))
    if used != len(blob):
        raise RuntimeError(f'entropy stream desync: {used} != {len(blob)}')
    q = residual.reshape(shape).astype(np.int64) - RES_RANGE + np.rint(mu).astype(np.int64)
    return q


def _pad(rgb):
    h, w = rgb.shape[:2]
    hp, wp = (h + 63) // 64 * 64, (w + 63) // 64 * 64
    py, px = hp - h, wp - w
    if py or px:
        rgb = np.pad(rgb, ((0, py), (0, px), (0, 0)), mode='reflect')
    return rgb.astype(np.float32) / 255.0, hp, wp


def compress(rgb, q=75, ckpt=DEFAULT_CKPT, verify=False, channel_scales=None):
    import torch
    model, arch = _checkpoint(ckpt)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    canon, _ = _checkpoint(ckpt)
    canon.to(torch.float64)
    scale = float(np.float32(q2s(q)))
    if channel_scales is None:
        scales = [np.full(c, scale, dtype=np.float32) for c in model.channels]
    else:
        scales = [np.asarray(s, dtype=np.float16).astype(np.float32)
                  for s in channel_scales]
        if [len(s) for s in scales] != list(model.channels):
            raise ValueError('channel scale count does not match model')
    h, w = rgb.shape[:2]
    canvas, hp, wp = _pad(rgb)
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        z0, z1, z2 = model.analysis(x)
        wh = model.hyper_enc(z2)
        arrays = [np.rint(z[0].cpu().float().numpy() * s[:, None, None]).astype(np.int64)
                  for z, s in zip((z0, z1, z2), scales)]
        qw = np.rint(wh[0].cpu().float().numpy() * 2).astype(np.int64)
    q0, q1, q2 = arrays
    use1 = not model.scalable or scale >= model.threshold1
    use0 = not model.scalable or scale >= model.threshold2
    if not use1:
        q1.fill(0)
    if not use0:
        q0.fill(0)
    z2t = torch.from_numpy(q2.astype(np.float64))[None].div(
        torch.from_numpy(scales[2].astype(np.float64))[:, None, None])
    wqt = torch.from_numpy(qw.astype(np.float64))[None] / 2
    with torch.no_grad():
        mu2, ls2 = canon.hyper_dec(wqt).chunk(2, 1)
        sig2 = torch.nn.functional.softplus(ls2) + 0.03
        mu1, sig1 = canon.entropy1(z2t, wqt)
        z1t = torch.from_numpy(q1.astype(np.float64))[None].div(
        torch.from_numpy(scales[1].astype(np.float64))[:, None, None])
        mu0, sig0 = canon.entropy0(z1t, z2t, wqt)
    mu2, sig2, mu1, sig1, mu0, sig0 = _np_params(mu2, sig2, mu1, sig1, mu0, sig0)
    sigh = torch.nn.functional.softplus(canon.hyper_logsig) + 0.03
    sigh = sigh.detach().cpu().numpy().astype(np.float64)[:, None, None]
    mu_h = np.zeros_like(sigh)
    streams = []
    for level, (zi, mui, sigi) in enumerate(((qw, mu_h, sigh), (q2, mu2, sig2),
                                             (q1, mu1, sig1), (q0, mu0, sig0))):
        if level == 2 and not use1 or level == 3 and not use0:
            streams.append(b'')
            continue
        blob, qback = _encode_stream(zi, mui, sigi)
        if verify:
            qtest = _decode_stream(blob, mui, sigi, zi.shape)
            if not np.array_equal(qtest, qback):
                raise RuntimeError('internal entropy verification failed')
        streams.append(blob)
    magic = b'OXM3' if channel_scales is not None else \
        (b'OXM2' if model.scalable else b'OXM1')
    out = bytearray(magic)
    out += struct.pack('<IIf', h, w, scale)
    if model.scalable:
        out += struct.pack('<B', int(use1) | int(use0) * 2)
    if channel_scales is not None:
        for s in scales:
            out += s.astype(np.float16).tobytes()
    for blob in streams:
        out += struct.pack('<I', len(blob)) + blob
    return bytes(out)


def optimize_channel_scales(rgb, q=75, ckpt=DEFAULT_CKPT, steps=60):
    model, _ = _checkpoint(ckpt)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    h, w = rgb.shape[:2]
    canvas, _, _ = _pad(rgb)
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None].to(dev)
    initial = math.log(float(np.float32(q2s(q))))
    logs = [torch.nn.Parameter(torch.full((1, c, 1, 1), initial, device=dev))
            for c in model.channels]
    opt = torch.optim.Adam(logs, lr=0.04)
    best = None
    for step in range(steps):
        scales = [v.exp() for v in logs]
        xr, bpp, _ = model(x, scales)
        mse = ((xr[:, :, :h, :w] - x[:, :, :h, :w]) ** 2).mean()
        loss = 1000 * mse + (25 / math.exp(2 * initial)) * bpp.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(logs, 1)
        opt.step()
        with torch.no_grad():
            for v in logs:
                v.clamp_(math.log(0.25), math.log(12))
        value = float(loss.detach())
        if best is None or value < best[0]:
            best = (value, [s.detach().cpu().numpy()[0, :, 0, 0] for s in scales])
        if step == 0 or (step + 1) % 20 == 0:
            print(f'RD optimize {step + 1}/{steps}: loss {value:.4f}, '
                  f'PSNR~{-10 * math.log10(max(float(mse.detach()), 1e-12)):.3f}, '
                  f'bpp~{float(bpp.mean().detach()):.4f}', flush=True)
    return best[1]


def decompress(blob, ckpt=DEFAULT_CKPT):
    import torch
    if blob[:4] not in (b'OXM1', b'OXM2', b'OXM3'):
        raise ValueError('not an OXN-MR stream')
    h, w, scale = struct.unpack('<IIf', blob[4:16])
    flags = blob[16] if blob[:4] in (b'OXM2', b'OXM3') else 3
    model, _ = _checkpoint(ckpt)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    canon, _ = _checkpoint(ckpt)
    canon.to(torch.float64)
    hp, wp = (h + 63) // 64 * 64, (w + 63) // 64 * 64
    shapes = ((model.hyper_c, hp // 64, wp // 64),
              (model.channels[2], hp // 16, wp // 16),
              (model.channels[1], hp // 8, wp // 8),
              (model.channels[0], hp // 4, wp // 4))
    pos = 17 if blob[:4] in (b'OXM2', b'OXM3') else 16
    if blob[:4] == b'OXM3':
        scales = []
        for c in model.channels:
            n = c * 2
            scales.append(np.frombuffer(blob[pos:pos + n], dtype=np.float16)
                          .astype(np.float32))
            pos += n
    else:
        scales = [np.full(c, scale, dtype=np.float32) for c in model.channels]
    streams = []
    for _ in range(4):
        n, = struct.unpack('<I', blob[pos:pos + 4])
        pos += 4
        streams.append(blob[pos:pos + n])
        pos += n
    if pos != len(blob):
        raise ValueError('trailing bytes in OXM1 stream')
    sigh = torch.nn.functional.softplus(canon.hyper_logsig) + 0.03
    sigh = sigh.detach().cpu().numpy().astype(np.float64)[:, None, None]
    qw = _decode_stream(streams[0], np.zeros_like(sigh), sigh, shapes[0])
    wqt = torch.from_numpy(qw.astype(np.float64))[None] / 2
    with torch.no_grad():
        mu2, ls2 = canon.hyper_dec(wqt).chunk(2, 1)
        sig2 = torch.nn.functional.softplus(ls2) + 0.03
    q2 = _decode_stream(streams[1], *_np_params(mu2, sig2), shapes[1])
    z2t = torch.from_numpy(q2.astype(np.float64))[None].div(
        torch.from_numpy(scales[2].astype(np.float64))[:, None, None])
    with torch.no_grad():
        mu1, sig1 = canon.entropy1(z2t, wqt)
    if flags & 1:
        q1 = _decode_stream(streams[2], *_np_params(mu1, sig1), shapes[2])
    else:
        q1 = np.zeros(shapes[2], dtype=np.int64)
    z1t = torch.from_numpy(q1.astype(np.float64))[None].div(
        torch.from_numpy(scales[1].astype(np.float64))[:, None, None])
    with torch.no_grad():
        mu0, sig0 = canon.entropy0(z1t, z2t, wqt)
    if flags & 2:
        q0 = _decode_stream(streams[3], *_np_params(mu0, sig0), shapes[3])
    else:
        q0 = np.zeros(shapes[3], dtype=np.int64)
    tensors = [torch.from_numpy(q.astype(np.float32) / s[:, None, None])[None].to(dev)
               for q, s in zip((q0, q1, q2), scales)]
    with torch.no_grad():
        xr = model.synthesis(tensors[0], tensors[1], tensors[2])
    rec = xr[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip(rec[:h, :w] * 255 + 0.5, 0, 255).astype(np.uint8)


def _images(path, crop):
    import torch
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    result = []
    for name in files:
        a = load_rgb(os.path.join(path, name)).astype(np.float32) / 255
        if a.shape[0] >= crop and a.shape[1] >= crop:
            result.append(torch.from_numpy(a).permute(2, 0, 1))
    if not result:
        raise RuntimeError(f'no training images of at least {crop}x{crop} in {path}')
    return result


def train(args):
    import torch
    from ox_neuro import _gauss_kernel, ms_ssim
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch = {'width': args.width, 'latent0': args.latent0,
            'latent1': args.latent1, 'latent2': args.latent2, 'hyper_c': args.hyper_c,
            'scalable': args.scalable, 'threshold1': args.threshold1,
            'threshold2': args.threshold2}
    model = MRModel(**arch).to(dev)
    start = 0
    if args.init:
        ck = torch.load(args.init, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'])
        start = int(ck.get('step', 0))
        print(f'init {args.init}, step {start}')
    imgs = _images(args.data, args.crop)
    print(f'train: {len(imgs)} images, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {dev}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 20)
    gk = _gauss_kernel().to(dev)
    t0 = time.time()
    for local_step in range(1, args.steps + 1):
        step = start + local_step
        crops = []
        for _ in range(args.batch):
            im = imgs[np.random.randint(len(imgs))]
            _, h, w = im.shape
            y, x = np.random.randint(h - args.crop + 1), np.random.randint(w - args.crop + 1)
            c = im[:, y:y + args.crop, x:x + args.crop]
            if np.random.rand() < 0.5:
                c = torch.flip(c, (2,))
            crops.append(c)
        xb = torch.stack(crops).to(dev)
        s = math.exp(np.random.uniform(math.log(args.smin), math.log(args.smax)))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            xr, bpp, _ = model(xb, s)
            mse = ((xr - xb) ** 2).mean()
            loss = args.w_mse * mse + (args.rate_weight / (s * s)) * bpp.mean()
            if args.w_msssim:
                with torch.autocast('cuda', enabled=False):
                    loss = loss + args.w_msssim * (1 - ms_ssim(
                        xr.float().clamp(0, 1), xb.float(), gk))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        sched.step()
        if local_step == 1 or local_step % 100 == 0:
            psnr = -10 * math.log10(max(mse.item(), 1e-9))
            print(f'step {step} loss {loss.item():.4f} mse {mse.item():.6f} '
                  f'psnr~{psnr:.2f} bpp~{bpp.mean().item():.4f} s={s:.2f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
        if local_step % 1000 == 0:
            torch.save({'model': model.state_dict(), 'arch': arch, 'step': step}, args.out + '.part')
    torch.save({'model': model.state_dict(), 'arch': arch, 'step': start + args.steps}, args.out)
    print(f'saved {args.out} ({time.time() - t0:.0f}s)')


def evaluate(args):
    import torch
    from ox_neuro import _gauss_kernel, ms_ssim
    files = sorted(f for f in os.listdir(args.data) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if args.limit:
        files = files[-args.limit:]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gk = _gauss_kernel().to(dev)
    try:
        import lpips
        lp = lpips.LPIPS(net='alex').to(dev).eval()
    except ImportError:
        lp = None
    rows = []
    for name in files:
        rgb = load_rgb(os.path.join(args.data, name))
        t0 = time.time()
        scales = optimize_channel_scales(rgb, args.q, args.ckpt, args.optimize) \
            if args.optimize else None
        blob = compress(rgb, args.q, args.ckpt, args.verify, scales)
        enc = time.time() - t0
        t0 = time.time()
        rec = decompress(blob, args.ckpt)
        dec = time.time() - t0
        a = rgb.astype(np.float32) / 255
        b = rec.astype(np.float32) / 255
        mse = float(np.mean((a - b) ** 2))
        psnr = -10 * math.log10(max(mse, 1e-12))
        dev = gk.device
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            ms = ms_ssim(ta, tb, gk).item()
            lv = lp(ta * 2 - 1, tb * 2 - 1).item() if lp is not None else None
        row = {'name': name, 'bpp': len(blob) * 8 / rgb.size * 3,
               'psnr': psnr, 'ms_ssim': ms, 'lpips': lv,
               'encode_s': enc, 'decode_s': dec}
        rows.append(row)
        ltext = f'{lv:.5f}' if lv is not None else 'n/a'
        print(f'{name}: {row["bpp"]:.4f} bpp, {psnr:.3f} dB, MS-SSIM {ms:.5f}, '
              f'LPIPS {ltext}, {enc:.2f}/{dec:.2f}s', flush=True)
    import json
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    keys = ['bpp', 'psnr', 'ms_ssim', 'encode_s', 'decode_s']
    if lp is not None:
        keys.append('lpips')
    print('mean:', {k: float(np.mean([r[k] for r in rows])) for k in keys})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=('train', 'c', 'd', 'eval'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'train'))
    ap.add_argument('--out', default=DEFAULT_CKPT)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--init')
    ap.add_argument('--q', type=int, default=75)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--optimize', type=int, default=0,
                    help='per-image channel-wise RD optimization steps')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--latent0', type=int, default=4)
    ap.add_argument('--latent1', type=int, default=8)
    ap.add_argument('--latent2', type=int, default=16)
    ap.add_argument('--hyper_c', type=int, default=8)
    ap.add_argument('--scalable', action='store_true')
    ap.add_argument('--threshold1', type=float, default=1.2)
    ap.add_argument('--threshold2', type=float, default=4.5)
    ap.add_argument('--smin', type=float, default=0.5)
    ap.add_argument('--smax', type=float, default=6.0)
    ap.add_argument('--w_mse', type=float, default=1000.0)
    ap.add_argument('--rate_weight', type=float, default=25.0)
    ap.add_argument('--w_msssim', type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a)
    elif a.cmd == 'c':
        if len(a.args) != 2:
            ap.error('c requires input image and output .oxh')
        rgb = load_rgb(a.args[0])
        t0 = time.time()
        scales = optimize_channel_scales(rgb, a.q, a.ckpt, a.optimize) \
            if a.optimize else None
        blob = compress(rgb, a.q, a.ckpt, a.verify, scales)
        with open(a.args[1], 'wb') as f:
            f.write(blob)
        print(f'{a.args[0]} -> {a.args[1]}: {len(blob)} bytes, '
              f'{len(blob) * 8 / (rgb.shape[0] * rgb.shape[1]):.4f} bpp, '
              f'{time.time() - t0:.2f}s')
    elif a.cmd == 'd':
        if len(a.args) != 2:
            ap.error('d requires input .oxh and output image')
        blob = open(a.args[0], 'rb').read()
        t0 = time.time()
        rec = decompress(blob, a.ckpt)
        Image.fromarray(rec).save(a.args[1])
        print(f'{a.args[0]} -> {a.args[1]} ({time.time() - t0:.2f}s)')
    else:
        evaluate(a)


if __name__ == '__main__':
    main()
