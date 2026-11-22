#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OX-Neuro — learned image codec (lossy): a transformer autoencoder trained
to reconstruct itself; the compression ratio is set by a latent SCALER s:
z -> round(z*s)/s. One model covers the whole quality range.

Entropy coding: adaptive OXN8 by default or flat OXN8 with --flat. Its integer
hyperprior/context path, frozen tables and canonical rANS avoid cross-architecture
entropy desynchronization. Experimental OXN10/OXN11 move context inference to
the GPU; OXN11 adds spatial one-shot sigma, OXN12/OXN13 use a 2x2 spatial grid,
and OXN14 adds empirical canonical rANS tables. Decompression auto-detects formats
through OXN14 by magic header.

Usage:
  python3 ox_neuro.py train [--steps 40000] [--data data/train]
  python3 ox_neuro.py c in.png out.oxh [--q 100] [--flat] [--boost] [--verify]
  python3 ox_neuro.py d in.oxh out.jpg
  python3 ox_neuro.py bench in.png
"""

import argparse
import io
import math
import os
import struct
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np

from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None  # локальный инструмент: свои файлы, лимит PIL не нужен


def load_rgb(path):
    """Загрузка изображения с применением EXIF-ориентации (повороты телефона)."""
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'ox_neuro.pt')
REC_CKPT = os.path.join(HERE, 'ox_neuro_rec.pt')
PATCH = 16
DIM = 256
DEPTH = 4
LATENT = 16
HEADS = 8
S_MIN, S_MAX = 0.35, 8.0          # диапазон скейлера (степени сжатия)

sys.path.insert(0, HERE)
from ox import rans_encode, rans_decode      # энтропийный движок OX
import ox_data


def q2s(q):
    """качество 1..100 -> скейлер s (лог-шкала)"""
    q = min(100, max(1, q))
    t = (q - 1) / 99.0
    return math.exp(math.log(S_MIN) + t * (math.log(S_MAX) - math.log(S_MIN)))


def s2q(s):
    t = (math.log(s) - math.log(S_MIN)) / (math.log(S_MAX) - math.log(S_MIN))
    return round(1 + 99 * min(1, max(0, t)))


def _sobel_kernel(ch=3):
    """Фиксированные фильтры Собеля dx/dy (нормированы), (2*ch, 1, 3, 3)."""
    import torch
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
    ky = kx.t()
    k = torch.stack([kx, ky]).unsqueeze(1)          # 2 x 1 x 3 x 3
    return k.repeat(ch, 1, 1, 1).contiguous()       # 2*ch x 1 x 3 x 3


def highpass_mse(xr, x, sk):
    """MSE по градиентам Собеля — штраф за потерю границ/резкости."""
    import torch.nn.functional as F
    ch = xr.shape[1]
    gx_r = F.conv2d(xr, sk, padding=1, groups=ch)
    gx = F.conv2d(x, sk, padding=1, groups=ch)
    return ((gx_r - gx) ** 2).mean()


def tiling_artifact_loss(xr, x, sizes=(1, 2, 4, 8)):
    import torch.nn.functional as F
    xr = xr.float().clamp(0, 1)
    x = x.float()
    losses = []
    for size in sizes:
        if size > 1:
            a, b = F.avg_pool2d(xr, size), F.avg_pool2d(x, size)
        else:
            a, b = xr, x
        losses.append((a - b).abs().mean())
    return sum(losses) / len(losses)


def color_balance_loss(xr, x, patch=16):
    import torch.nn.functional as F
    xr = xr.float().clamp(0, 1)
    return F.avg_pool2d(xr - x.float(), patch).abs().mean()


def patch_seam_loss(xr, x, patch=16):
    xr = xr.float().clamp(0, 1)
    x = x.float()
    rx = xr[:, :, :, patch - 1:-1:patch] - xr[:, :, :, patch::patch]
    xx = x[:, :, :, patch - 1:-1:patch] - x[:, :, :, patch::patch]
    ry = xr[:, :, patch - 1:-1:patch] - xr[:, :, patch::patch]
    xy = x[:, :, patch - 1:-1:patch] - x[:, :, patch::patch]
    return ((rx - xx) ** 2).mean() + ((ry - xy) ** 2).mean()


def text_saliency(x):
    import torch.nn.functional as F
    gray = x.float().mean(1, keepdim=True)
    gx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0)).abs()
    gy = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 0, 0, 1)).abs()
    saliency = F.avg_pool2d(gx + gy, 5, 1, 2)
    return (saliency > 0.08).float().detach()


def text_fidelity_loss(xr, x):
    xr = xr.float().clamp(0, 1)
    x = x.float()
    mask = text_saliency(x)
    error = (xr - x).abs().mean(1, keepdim=True)
    return (error * mask).sum() / mask.sum().clamp_min(1)


# ------------------------------------------------------------ модель -------
def _gauss_kernel(ch=3, ks=11, sigma=1.5):
    import torch
    g = torch.exp(-((torch.arange(ks, dtype=torch.float32) - ks // 2) ** 2) /
                  (2 * sigma * sigma))
    g = g / g.sum()
    k = g[:, None] @ g[None, :]
    return k.expand(ch, 1, ks, ks).contiguous()


def ssim(x, y, k):
    """SSIM для тензоров Bx3xHxW в [0,1]; возвращает (ssim, contrast*structure)."""
    import torch
    import torch.nn.functional as F
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch = x.shape[1]
    mu_x = F.conv2d(x, k, groups=ch)
    mu_y = F.conv2d(y, k, groups=ch)
    xx = F.conv2d(x * x, k, groups=ch) - mu_x ** 2
    yy = F.conv2d(y * y, k, groups=ch) - mu_y ** 2
    xy = F.conv2d(x * y, k, groups=ch) - mu_x * mu_y
    cs = (2 * xy + C2) / (xx + yy + C2)
    s = cs * (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    return s.mean(dim=(1, 2, 3)), cs.mean(dim=(1, 2, 3))


def ms_ssim(x, y, k, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    """MS-SSIM (Wang et al.), x/y: Bx3xHxW в [0,1] -> скаляр."""
    import torch
    import torch.nn.functional as F
    mcs, mss = [], None
    for i in range(5):
        s, c = ssim(x, y, k)
        mcs.append(c.clamp(min=0))
        mss = s.clamp(min=0)
        if i < 4:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    w = torch.tensor(weights, device=x.device)
    v = torch.stack(mcs[:-1]).pow(w[:4, None]).prod(0) * mss.pow(w[4])
    return v.mean()
def build_model(patch=PATCH, dim=DIM, depth=DEPTH, latent=LATENT, heads=HEADS,
                shared=False, moe=0, refine=False, hyper=False, window=0,
                se=False, hyper_width=64, hyper_c=32, pos_tokens=4096,
                recurrent=0, flow=0):
    import torch
    import torch.nn as nn

    HYPER_C = hyper_c
    HYPER_W = hyper_width

    def _linc(layer, h):
        """Linear по токенам с чанкованием — размер чанка от свободной памяти."""
        n = _tok_chunk(h.is_cuda, dim)
        if h.shape[1] <= n:
            return layer(h)
        return torch.cat([layer(c_) for c_ in h.split(n, 1)], 1)

    class HyperEnc(nn.Module):
        """Латентная сетка C x ht x wt -> гипер-латент HYPER_C x ht/4 x wt/4."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(latent, HYPER_W, 3, padding=1), nn.GELU(),
                nn.Conv2d(HYPER_W, HYPER_W, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(HYPER_W, HYPER_C, 3, stride=2, padding=1))

        def forward(self, g):
            return self.net(g)

    class HyperDec(nn.Module):
        """Гипер-латент -> (mu, log b) распределения Лапласа на токен×канал."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(HYPER_C, HYPER_W, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(HYPER_W, HYPER_W, 4, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(HYPER_W, latent * 2, 3, padding=1))

        def forward(self, w):
            return self.net(w)

    class SEGate(nn.Module):
        """Attention-маска на латенте (идея NLAM из NLAIC): сеть сама
        решает, каким каналам какого токена дать вес — неявное распределение
        бит по важности. Zero-init -> в точности тождественное отображение."""
        def __init__(self, ch):
            super().__init__()
            h = max(8, ch * 2)
            self.f1 = nn.Linear(ch, h)
            self.f2 = nn.Linear(h, ch)
            nn.init.zeros_(self.f2.weight)
            nn.init.zeros_(self.f2.bias)

        def forward(self, z):
            return z * torch.sigmoid(self.f2(
                torch.nn.functional.gelu(self.f1(z)))) * 2.0

    class RefineHead(nn.Module):
        """Свёрточная голова поверх unpatchify: сглаживает швы между
        патчами 16x16 (работает поперёк границ). Zero-init на выходе —
        в точности тождественное отображение при инициализации."""
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 64, 3, padding=1)
            self.c2 = nn.Conv2d(64, 64, 3, padding=1)
            self.c3 = nn.Conv2d(64, 3, 3, padding=1)
            nn.init.zeros_(self.c3.weight)
            nn.init.zeros_(self.c3.bias)

        def forward(self, x):
            B, C, H, W = x.shape
            if H * W > 4_194_304:   # >4 Мпикс: refine по тайлам с перекрытием
                bud = _mem_budget('cuda' if x.is_cuda else 'cpu')
                tile = int(np.clip(int(math.sqrt(bud // 800)), 256, 2048))
                ov = 8
                out = x.clone()
                for yy in range(0, H, tile):
                    for xx in range(0, W, tile):
                        y0, y1 = max(yy - ov, 0), min(yy + tile + ov, H)
                        x0, x1 = max(xx - ov, 0), min(xx + tile + ov, W)
                        t = x[:, :, y0:y1, x0:x1]
                        h1 = torch.nn.functional.gelu(self.c1(t))
                        r = t + self.c3(torch.nn.functional.gelu(self.c2(h1)))
                        ye, xe = min(yy + tile, H), min(xx + tile, W)
                        out[:, :, yy:ye, xx:xe] = \
                            r[:, :, yy - y0:ye - y0, xx - x0:xe - x0]
                return out
            h = torch.nn.functional.gelu(self.c1(x))
            h = torch.nn.functional.gelu(self.c2(h))
            return x + self.c3(h)

    class FlowHead(nn.Module):
        def __init__(self, steps):
            super().__init__()
            self.steps = steps
            self.c1 = nn.Conv2d(7, 64, 3, padding=1)
            self.c2 = nn.Conv2d(64, 64, 3, padding=1)
            self.c3 = nn.Conv2d(64, 3, 3, padding=1)
            nn.init.zeros_(self.c3.weight)
            nn.init.zeros_(self.c3.bias)

        def velocity(self, x, cond, t):
            if torch.is_tensor(t):
                tt = t.to(device=x.device, dtype=x.dtype).expand(
                    -1, -1, x.shape[2], x.shape[3])
            else:
                tt = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), t,
                                device=x.device, dtype=x.dtype)
            h = torch.cat((x, cond, tt), 1)
            h = torch.nn.functional.gelu(self.c1(h))
            h = torch.nn.functional.gelu(self.c2(h))
            return self.c3(h)

        def sample(self, cond, seed):
            gen = torch.Generator(device=cond.device)
            gen.manual_seed(seed)
            noise = torch.randn(cond.shape, generator=gen, device=cond.device,
                                dtype=cond.dtype)
            x = cond + 0.03 * noise
            for i in range(self.steps):
                x = x + self.velocity(x, cond, (i + 0.5) / self.steps) / self.steps
            return x

        def forward(self, cond):
            r0 = self.sample(cond, 17) - cond
            r1 = self.sample(cond, 29) - cond
            agreement = torch.exp(-20 * (r0 - r1).abs())
            return cond + (r0 + r1) * 0.5 * agreement

    class MoEFFN(nn.Module):
        """FFN-смесь экспертов: роутер по [признаки токена, log s]."""
        def __init__(self, n_exp):
            super().__init__()
            self.experts = nn.ModuleList(
                nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                              nn.Linear(dim * 2, dim))
                for _ in range(n_exp))
            self.router = nn.Linear(dim + 1, n_exp)

        def forward(self, x, logs):
            n = _tok_chunk(x.is_cuda, dim) // 2
            if x.shape[1] > n:   # большие изображения: чанки по токенам
                xs = x.split(n, 1)
                ls = logs.split(n, 1) if logs is not None else [None] * len(xs)
                return torch.cat([self.forward(a, b) for a, b in zip(xs, ls)], 1)
            h = torch.cat([x, logs.unsqueeze(-1)], -1) if logs is not None \
                else torch.cat([x, torch.zeros_like(x[..., :1])], -1)
            g = torch.softmax(self.router(h.float()), -1).to(x.dtype)
            outs = torch.stack([e(x) for e in self.experts], -1)  # B,T,dim,E
            return (outs * g.unsqueeze(2)).sum(-1)

    class Block(nn.Module):
        def __init__(self, n_exp=0):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.ln2 = nn.LayerNorm(dim)
            if n_exp:
                self.mlp = MoEFFN(n_exp)
            else:
                self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                         nn.Linear(dim * 4, dim))
            self.n_exp = n_exp

        def forward(self, x, logs=None, ht=0, wt=0, shift=False):
            h = self.ln1(x)
            if window and ht and wt:
                # оконный attention (Swin): окна window x window токенов,
                # на нечётных слоях сдвиг на пол-окна через паддинг
                B, T, C = h.shape
                g = h.reshape(B, ht, wt, C)
                sh = window // 2 if shift else 0
                top = left = sh
                bottom = (-(ht + sh)) % window
                right = (-(wt + sh)) % window
                g = torch.nn.functional.pad(g, (0, 0, left, right, top, bottom),
                                            mode='replicate')
                H2, W2 = g.shape[1], g.shape[2]
                gw = g.reshape(B, H2 // window, window, W2 // window, window, C)
                gw = gw.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, C)
                nw_max = max(64, _tok_chunk(x.is_cuda, dim) // (window * window))
                if gw.shape[0] > nw_max:   # большие изображения: чанки по окнам
                    aw = torch.cat([self.attn(c_, c_, c_, need_weights=False)[0]
                                    for c_ in gw.split(nw_max, 0)], 0)
                else:
                    aw = self.attn(gw, gw, gw, need_weights=False)[0]
                g = aw.reshape(B, H2 // window, W2 // window, window, window, C)
                g = g.permute(0, 1, 3, 2, 4, 5).reshape(B, H2, W2, C)
                g = g[:, top:top + ht, left:left + wt]
                x = x + g.reshape(B, T, C)
            else:
                x = x + self.attn(h, h, h, need_weights=False)[0]
            h = self.ln2(x)
            if self.n_exp:
                x = x + self.mlp(h, logs)
            else:
                n = _tok_chunk(x.is_cuda, dim)
                if h.shape[1] > n:
                    x = x + torch.cat([self.mlp(c_) for c_ in h.split(n, 1)], 1)
                else:
                    x = x + self.mlp(h)
            return x

    class OxFormer(nn.Module):
        """Патчи patch×patch -> трансформер -> латент C на токен -> трансформер -> патчи."""
        def __init__(self):
            super().__init__()
            self.patch = patch
            self.embed = nn.Linear(patch * patch * 3, dim)
            self.pos = nn.Parameter(torch.zeros(1, pos_tokens, dim))
            self.depth = depth
            self.shared = shared
            if shared:
                # один блок, прокручиваемый depth раз (weight tying);
                # step-эмбеддинг отличает проходы, как в Universal Transformer
                self.enc_block = Block()
                self.dec_block = Block(moe)
                self.enc_step = nn.Parameter(torch.zeros(1, depth, dim))
                self.dec_step = nn.Parameter(torch.zeros(1, depth, dim))
                nn.init.trunc_normal_(self.enc_step, std=0.02)
                nn.init.trunc_normal_(self.dec_step, std=0.02)
            else:
                self.enc = nn.ModuleList(Block() for _ in range(depth))
                self.dec = nn.ModuleList(Block(moe) for _ in range(depth))
            self.to_lat = nn.Linear(dim, latent)
            self.from_lat = nn.Linear(latent, dim)
            self.to_pix = nn.Linear(dim, patch * patch * 3)
            self.se = se
            if se:
                self.se_enc = SEGate(latent)
                self.se_dec = SEGate(latent)
            self.refine = RefineHead() if refine else None
            self.recurrent = RefineHead() if recurrent else None
            self.recurrent_steps = recurrent
            self.flow = FlowHead(flow) if flow else None
            if hyper:
                self.hyper_enc = HyperEnc()
                self.hyper_dec = HyperDec()
                self.log_bw = nn.Parameter(torch.zeros(HYPER_C))
            else:
                self.hyper_enc = None
            self.latent = latent
            self.log_b = nn.Parameter(torch.zeros(latent))  # log-scale Laplace
            self.scale_emb = nn.Linear(1, dim)   # conditioning: log s токена
            nn.init.zeros_(self.scale_emb.weight)
            nn.init.zeros_(self.scale_emb.bias)
            nn.init.trunc_normal_(self.pos, std=0.02)

        def patchify(self, x):
            B, _, H, W = x.shape
            p = self.patch
            x = x.reshape(B, 3, H // p, p, W // p, p)
            x = x.permute(0, 2, 4, 1, 3, 5)
            return x.reshape(B, -1, p * p * 3)

        def unpatchify(self, x, H, W):
            B = x.shape[0]
            p = self.patch
            x = x.reshape(B, H // p, W // p, 3, p, p)
            x = x.permute(0, 3, 1, 4, 2, 5)
            return x.reshape(B, 3, H, W)

        def posemb(self, ht, wt):
            """Позиционные эмбеддинги для сетки ht x wt токенов.
            Обученная сетка 16x16 (кроп 256 при patch 16) билинейно
            интерполируется на любой размер — как в ViT."""
            src = self.pos[:, :256].reshape(1, 16, 16, -1).permute(0, 3, 1, 2)
            out = torch.nn.functional.interpolate(
                src.float(), size=(ht, wt), mode='bilinear',
                align_corners=False).to(src.dtype)
            return out.flatten(2).transpose(1, 2)          # 1, ht*wt, dim

        def encode(self, x):
            H, W = x.shape[2], x.shape[3]
            p = self.patch
            ht, wt = H // p, W // p
            h = _linc(self.embed, self.patchify(x))
            h = h + self.posemb(ht, wt).to(h.dtype)
            if self.shared:
                for i in range(self.depth):
                    h = self.enc_block(h + self.enc_step[:, i:i + 1],
                                       None, ht, wt, i % 2 == 1)
            else:
                for i, b in enumerate(self.enc):
                    h = b(h, None, ht, wt, i % 2 == 1)
            z = _linc(self.to_lat, h)
            if self.se:
                z = self.se_enc(z)
            return z

        def decode(self, z, H, W, logs=None):
            p = self.patch
            ht, wt = H // p, W // p
            if self.se:
                z = self.se_dec(z)
            h = _linc(self.from_lat, z)
            h = h + self.posemb(ht, wt).to(h.dtype)
            if logs is not None:
                h = h + self.scale_emb(logs.unsqueeze(-1))
            if self.shared:
                for i in range(self.depth):
                    h = self.dec_block(h + self.dec_step[:, i:i + 1],
                                       logs, ht, wt, i % 2 == 1)
            else:
                for i, b in enumerate(self.dec):
                    h = b(h, logs, ht, wt, i % 2 == 1)
            out = self.unpatchify(_linc(self.to_pix, h), H, W)
            if self.refine is not None:
                out = self.refine(out)
            if self.recurrent is not None:
                for _ in range(self.recurrent_steps):
                    out = self.recurrent(out)
            if self.flow is not None and not self.training:
                out = self.flow(out)
            return out

        def forward(self, x, s):
            """s: float или тензор (B, T) — степень сжатия по токенам."""
            B, _, H, W = x.shape
            z = self.encode(x)
            if not torch.is_tensor(s):
                s = torch.full((B, z.shape[1]), float(s), device=z.device)
            sm = s.unsqueeze(-1)                              # B x T x 1
            zs = z * sm
            # straight-through квантизатор
            qt = torch.round(zs)
            zq = zs + (qt - zs).detach()
            if self.hyper_enc is not None:
                # hyperprior: предсказываем (mu, b) латента из гипер-латента
                p = self.patch
                ht, wt = H // p, W // p
                grid = (zq / sm).transpose(1, 2).reshape(B, self.latent, ht, wt)
                ph, pw = (-ht) % 4, (-wt) % 4
                gpad = torch.nn.functional.pad(grid, (0, pw, 0, ph),
                                               mode='replicate')
                w = self.hyper_enc(gpad)
                wq = w + (torch.round(w) - w).detach()
                mb = self.hyper_dec(wq)[:, :, :ht, :wt]
                mb = mb.reshape(B, 2 * self.latent, -1).transpose(1, 2)
                mu, logb = mb.chunk(2, -1)                    # B x T x C
                b = torch.nn.functional.softplus(logb) + 1e-2
                t = zs + (torch.rand_like(zs) - 0.5)
                bits_z = ((t - mu).abs() / (b * math.log(2))
                          + torch.log2(2 * b)).sum(-1)        # B x T
                bw = torch.nn.functional.softplus(self.log_bw) + 1e-3
                bw = bw[None, :, None, None]                # 1 x HC x 1 x 1
                tw = w + (torch.rand_like(w) - 0.5)
                bits_w = (tw.abs() / (bw * math.log(2))
                          + torch.log2(2 * bw)).sum(dim=(1, 2, 3))
                bits = bits_z + (bits_w / (ht * wt)).unsqueeze(1)
                bpp = bits / (3 * p * p)                      # B x T
            else:
                # rate-прокси: шум вместо round
                b = torch.nn.functional.softplus(self.log_b) + 1e-3
                t = zs + (torch.rand_like(zs) - 0.5)
                bits = (t.abs() / (b * math.log(2)) + torch.log2(2 * b)).sum(-1)
                bpp = bits / (3 * self.patch * self.patch)    # B x T, бит/пиксель
            xr = self.decode(zq / sm, H, W, torch.log(s))
            if self.flow is not None and self.training:
                cond = xr
                noise = torch.randn_like(cond)
                x0 = cond + 0.03 * noise
                t = torch.rand(B, 1, 1, 1, device=x.device, dtype=x0.dtype)
                xt = (1 - t) * x0 + t * x
                v = self.flow.velocity(xt, cond, t)
                self._flow_loss = ((v - (x - x0)) ** 2).mean()
                xr = self.flow(cond)
            return xr, bpp

    return OxFormer()


# ---- совместный ретрен кодека с OXN7-энтропией (train --ctx7) -------------
def _forward_ctx7(model, ctx7, xb, smap, groups):
    """Дифференцируемый forward с полной OXN7-моделью битрейта: UNAQ-шум +
    STE, rate = Лаплас (группа 0, якоря) + Гаусс от ctx7 (остальные проходы)
    + гипер-латент. Возвращает (xr, bpp) в той же нормировке, что и
    OxFormer.forward."""
    import torch
    B, _, H, W = xb.shape
    p = model.patch
    g = H // p
    C = model.latent
    T_ = g * g
    z = model.encode(xb)
    sm = smap.unsqueeze(-1)
    zs = z * sm
    zq = zs + (torch.round(zs) - zs).detach()                 # STE
    grid = (zq / sm).transpose(1, 2).reshape(B, C, g, g)
    ph, pw = (-g) % 4, (-g) % 4
    gpad = torch.nn.functional.pad(grid, (0, pw, 0, ph), mode='replicate')
    w = model.hyper_enc(gpad)
    wq = w + (torch.round(w) - w).detach()
    mb = model.hyper_dec(wq)[:, :, :g, :g]
    mu_h, logb_h = mb.chunk(2, 1)
    b_h = torch.nn.functional.softplus(logb_h) + 1e-2
    anch = torch.from_numpy(_anchor_mask(g, g)).to(z.device)
    anch = anch.reshape(1, 1, g, g)
    nanch = ~anch
    t = (zs + (torch.rand_like(zs) - 0.5)).transpose(1, 2).reshape(B, C, g, g)
    bits = torch.zeros(B, g, g, device=z.device)
    hi0 = groups[0]
    # группа 0, якоря: Лаплас от hyperprior
    bits = bits + anch * (((t[:, :hi0] - mu_h[:, :hi0]).abs()
                           / (b_h[:, :hi0] * math.log(2)))
                          + torch.log2(2 * b_h[:, :hi0])).sum(1)
    qd = torch.zeros_like(grid)
    qd[:, :hi0] = grid[:, :hi0] * anch
    logs_ch = torch.log(smap).reshape(B, 1, g, g)
    sq2 = math.sqrt(2.0)
    for gi, (lo, hi) in enumerate(_group_cuts(groups)):
        for phase in (True, False):
            pm = anch if phase else nanch
            if gi == 0 and phase:
                continue
            feats = torch.cat([qd, mu_h, logb_h, logs_ch], 1)
            dmu, lsig = ctx7(feats).chunk(2, 1)
            mu = mu_h[:, lo:hi] + dmu[:, lo:hi]
            sig = torch.nn.functional.softplus(lsig[:, lo:hi]) + 1e-2
            tg = t[:, lo:hi]
            cdf = 0.5 * (1 + torch.erf(
                (torch.stack([tg + 0.5, tg - 0.5]) - mu) / (sig * sq2)))
            prob = (cdf[0] - cdf[1]).clamp_min(1e-9)
            bg = (-torch.log2(prob)).sum(1)                   # B x g x g
            bits = bits + torch.where(pm, bg, torch.zeros_like(bg))
            qd[:, lo:hi] = torch.where(pm, grid[:, lo:hi], qd[:, lo:hi])
    bw = torch.nn.functional.softplus(model.log_bw) + 1e-3
    bw = bw[None, :, None, None]
    tw = w + (torch.rand_like(w) - 0.5)
    bits_w = (tw.abs() / (bw * math.log(2))
              + torch.log2(2 * bw)).sum(dim=(1, 2, 3))
    bits = bits.reshape(B, T_) + (bits_w / T_).unsqueeze(1)
    bpp = bits / (3 * p * p)
    xr = model.decode(zq / sm, H, W, torch.log(smap))
    return xr, bpp


# ------------------------------------------------------------ обучение -----
def _training_payload(args, model, image_count):
    return {
        'model': model.state_dict(),
        'arch': {'patch': PATCH, 'dim': args.dim, 'depth': args.depth,
                 'latent': args.latent, 'heads': args.heads, 'shared': args.shared,
                 'moe': args.moe, 'refine': args.refine, 'hyper': args.hyper,
                 'window': args.window, 'se': args.se,
                 'hyper_width': args.hyper_width, 'hyper_c': args.hyper_c,
                 'pos_tokens': args.pos_tokens,
                 'recurrent': args.recurrent, 'flow': args.flow},
        'training': {'args': dict(vars(args)), 'image_count': image_count},
    }


def cmd_train(args):
    import torch

    crop_min = args.crop
    imgs = ox_data.train_files(args.data, crop_min, args.exclude)
    print(f'train: {len(imgs)} images from {args.data}')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(dim=args.dim, depth=args.depth, latent=args.latent,
                        heads=args.heads, shared=args.shared, moe=args.moe,
                        refine=args.refine, hyper=args.hyper,
                        window=args.window, se=args.se,
                        hyper_width=args.hyper_width, hyper_c=args.hyper_c,
                        pos_tokens=args.pos_tokens, recurrent=args.recurrent,
                        flow=args.flow).to(dev)
    if args.init:
        ck = torch.load(args.init, map_location='cpu', weights_only=True)
        current = model.state_dict()
        compatible = {k: v for k, v in ck['model'].items()
                      if k in current and current[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
        print(f'init from {args.init}: {len(compatible)}/{len(current)} tensors')
    if args.freeze_enc:
        n_frozen = 0
        for name, prm in model.named_parameters():
            if not (name.startswith(('dec', 'to_pix', 'from_lat', 'scale_emb',
                                     'refine', 'recurrent', 'flow'))):
                prm.requires_grad_(False)
                n_frozen += prm.numel()
        print(f'frozen encoder parameters: {n_frozen / 1e6:.2f}M')
    gk = _gauss_kernel().to(dev)
    sk = _sobel_kernel().to(dev)
    lp = None
    if args.w_lpips > 0:
        import lpips
        lp = lpips.LPIPS(net='alex').to(dev).eval()
        for prm in lp.parameters():
            prm.requires_grad_(False)
    ctx7 = None
    groups7 = None
    if args.ctx7:
        if not os.path.exists(args.ctx_file):
            sys.exit(f'--ctx7: нет {args.ctx_file} — сначала trainctx '
                     f'--groups ...')
        ck7 = torch.load(args.ctx_file, map_location='cpu', weights_only=True)
        groups7 = tuple(int(v) for v in ck7['groups'])
        if sum(groups7) != args.latent:
            sys.exit(f'--ctx7: groups {groups7} не сходятся с '
                     f'latent={args.latent}')
        ctx7 = _build_ctx7(args.latent,
                           int(ck7.get('hidden', CTX_HIDDEN))).to(dev)
        ctx7.load_state_dict(ck7['state'])
        print(f'ctx7 rate model: {args.ctx_file}, groups={groups7}')
    params = list(model.parameters())
    if ctx7 is not None:
        params += list(ctx7.parameters())
    n_params = sum(p.numel() for p in model.parameters())
    print(f'parameters: {n_params / 1e6:.2f}M, device: {dev}')
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr / 20)
    crop = args.crop
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # случайный батч кропов
        xs = []
        for _ in range(args.batch):
            c = ox_data.random_crop(imgs[np.random.randint(len(imgs))], crop)
            if np.random.rand() < 0.5:
                c = torch.flip(c, dims=[2])
            xs.append(c)
        xb = torch.stack(xs).to(dev, non_blocking=True)
        # поле степеней сжатия: 40% — равномерное, 60% — кусочно-постоянное
        T_ = (crop // PATCH) * (crop // PATCH)
        if np.random.rand() < 0.4:
            s0 = math.exp(np.random.uniform(math.log(S_MIN), math.log(S_MAX)))
            smap = torch.full((args.batch, T_), s0, device=dev)
        else:
            g = crop // PATCH
            bs = int(np.random.choice([v for v in (2, 4, 8, 16) if v <= g]))
            nb = g // bs
            fields = np.exp(np.random.uniform(math.log(S_MIN), math.log(S_MAX),
                                              size=(args.batch, nb, nb)))
            field = np.repeat(np.repeat(fields, bs, 1), bs, 2)
            smap = torch.from_numpy(
                field.reshape(args.batch, T_).astype(np.float32)).to(dev)
        lam = 25.0 / (smap * smap)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            if ctx7 is not None:
                xr, bpp = _forward_ctx7(model, ctx7, xb, smap, groups7)
            else:
                xr, bpp = model(xb, smap)
            mse = torch.mean((xr - xb) ** 2)
            loss = args.w_mse * mse + (lam * bpp).mean()
            if args.w_flow > 0:
                loss = loss + args.w_flow * model._flow_loss
            if args.w_msssim > 0:
                # MS-SSIM считаем строго в fp32: в bf16 дисперсии/ковариации
                # теряют точность и градиент лосса становится разрушительным
                with torch.autocast('cuda', enabled=False):
                    loss = loss + args.w_msssim * (
                        1 - ms_ssim(xr.float().clamp(0, 1), xb.float(), gk))
            if args.w_lpips > 0:
                with torch.autocast('cuda', enabled=False):
                    loss = loss + args.w_lpips * lp(
                        xr.float().clamp(0, 1) * 2 - 1, xb.float() * 2 - 1).mean()
            if args.w_hp > 0:
                loss = loss + args.w_hp * highpass_mse(
                    xr.float().clamp(0, 1), xb.float(), sk)
            with torch.autocast('cuda', enabled=False):
                if args.w_tal > 0:
                    loss = loss + args.w_tal * tiling_artifact_loss(
                        xr, xb, sizes=(4, 16, 64))
                if args.w_color > 0:
                    loss = loss + args.w_color * color_balance_loss(xr, xb, PATCH)
                if args.w_seam > 0:
                    loss = loss + args.w_seam * patch_seam_loss(xr, xb, PATCH)
                if args.w_text > 0:
                    loss = loss + args.w_text * text_fidelity_loss(xr, xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1:
            psnr = -10 * math.log10(max(mse.item(), 1e-9))
            print(f'  step {step:5d}/{args.steps}  loss {loss.item():.3f}  '
                  f'mse {mse.item():.5f}  psnr~{psnr:.2f}  '
                  f'bpp~{bpp.mean().item():.3f}  '
                  f's~{smap.mean().item():.2f}  ({time.time() - t0:.0f} с)')
        if step % 5000 == 0:
            torch.save(_training_payload(args, model, len(imgs)), args.out + '.part')
            if ctx7 is not None:
                torch.save({'state': ctx7.state_dict(), 'latent': args.latent,
                            'hidden': ctx7.net[0].out_channels,
                            'groups': groups7}, args.ctx_file + '.part')
    torch.save(_training_payload(args, model, len(imgs)), args.out)
    if ctx7 is not None:
        torch.save({'state': ctx7.state_dict(), 'latent': args.latent,
                    'hidden': ctx7.net[0].out_channels,
                    'groups': groups7}, args.ctx_file)
        print(f'saved: {args.ctx_file}')
    print(f'saved: {args.out}  ({time.time() - t0:.0f} s)')


# ------------------------------------------------------- сжатие/распаковка -

# ---- универсальное чанкование по доступной памяти --------------------------
_MEM_CACHE = {}
def _mem_budget(dev):
    """Бюджет памяти на один проход: ~1/2 СВОБОДНОЙ VRAM (GPU)
    или ~1/6 СВОБОДНОЙ RAM (CPU). На unified-memory устройствах
    (DGX Spark, Jetson) пул общий — nvidia-smi его не видит, поэтому
    берём минимум из двух показаний (оба драйверные, не nvidia-smi)."""
    if dev in _MEM_CACHE:
        return _MEM_CACHE[dev]
    ram = 1 << 30
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable'):
                ram = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    if dev == 'cuda':
        try:
            import torch
            free, _ = torch.cuda.mem_get_info()
            b = int(free * 0.5)
        except Exception:
            b = 1 << 30
        b = min(b, ram // 6)      # unified memory: не зажать CPU-пул
    else:
        b = ram // 6
    _MEM_CACHE[dev] = b
    return b


def _tok_chunk(is_cuda, dim):
    """Токенов за проход: из бюджета памяти (~48 байт на dim на токен)."""
    per_tok = dim * 48
    return int(np.clip(_mem_budget('cuda' if is_cuda else 'cpu') // per_tok,
                       8192, 262144))


def _maybe_autocast(dev, npix):
    """bf16 на больших изображениях (иначе 200+ Мпикс не влезает в GPU)."""
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16,
                          enabled=(dev == 'cuda' and npix > 16_000_000))


def _decode_autocast(dev):
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda'))


_FORCE_CPU = False     # после CUDA OOM переключаемся на CPU насовсем
_VERIFY = False        # самопроверка энтропийного потока после сжатия (--verify)
_RANS_STATS = None
_MODEL_CACHE = {}


def _load(path=None):
    import torch
    path = path or CKPT
    dev = 'cpu' if _FORCE_CPU or not torch.cuda.is_available() else 'cuda'
    key = (path, dev)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not os.path.exists(path):
        sys.exit(f'no checkpoint {path} — train first: python3 ox_neuro.py train')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    arch = ck.get('arch') or {'patch': PATCH, 'dim': DIM, 'depth': DEPTH,
                              'latent': LATENT, 'heads': HEADS}
    arch.setdefault('shared', False)
    arch.setdefault('moe', 0)
    arch.setdefault('refine', False)
    arch.setdefault('hyper', False)
    arch.setdefault('window', 0)
    arch.setdefault('se', False)
    arch.setdefault('hyper_width', 64)
    arch.setdefault('hyper_c', 32)
    arch.setdefault('pos_tokens', 4096)
    arch.setdefault('recurrent', 0)
    arch.setdefault('flow', 0)
    model = build_model(**arch)
    model.load_state_dict(ck['model'], strict=False)  # старые чекпоинты без scale_emb
    model.__dict__['_ckpt_path'] = path
    model = model.eval().to(dev)
    _MODEL_CACHE[key] = (model, dev)
    return model, dev


# ---- общее канальное rANS-кодирование квантованных латентов ---------------
def _encode_channels(zq):
    """zq: int32 T x C -> body bytes (rANS по каждому каналу)."""
    T_, C = zq.shape
    body = bytearray()
    for c in range(C):
        v = zq[:, c]
        vmin = int(v.min())
        u = (v - vmin).astype(np.int64)
        if u.max() < 256:
            blob = rans_encode(u.astype(np.uint8).tobytes())
            body += struct.pack('<Bi', 1, vmin) + struct.pack('<I', len(blob)) + blob
        else:
            lo = (u & 0xFF).astype(np.uint8).tobytes()
            hi = (u >> 8).astype(np.uint8).tobytes()
            b1, b2 = rans_encode(lo), rans_encode(hi)
            body += struct.pack('<Bi', 2, vmin)
            body += struct.pack('<I', len(b1)) + b1
            body += struct.pack('<I', len(b2)) + b2
    return bytes(body)


def _decode_channels(body, C, T_):
    pos = 0
    chans = []
    for c in range(C):
        kind, vmin = struct.unpack('<Bi', body[pos:pos + 5]); pos += 5
        (l1,) = struct.unpack('<I', body[pos:pos + 4]); pos += 4
        d1 = np.frombuffer(rans_decode(body[pos:pos + l1]), np.uint8).astype(np.int64)
        pos += l1
        u = d1
        if kind == 2:
            (l2,) = struct.unpack('<I', body[pos:pos + 4]); pos += 4
            d2 = np.frombuffer(rans_decode(body[pos:pos + l2]), np.uint8).astype(np.int64)
            pos += l2
            u = d1 + (d2 << 8)
        chans.append(u + vmin)
    zq = np.stack(chans, axis=1).astype(np.float32)   # T x C
    assert zq.shape[0] == T_
    return zq


# ---- параметрический rANS для hyperprior (формат OXN3) ---------------------
# Латент кодируется с ПОТОКЕННОЙ моделью: hyper-сеть предсказывает (mu, b)
# Лапласа; в rANS идёт остаток q - round(mu), b квантуется в бакеты.
RES_RANGE = 256                     # остаток зажат в [-256, 255]
LOG_B_MIN, LOG_B_STEP, N_BUCK = -4.5, 0.5, 24
CANON8_FILE = os.path.join(HERE, 'oxn8_canon.npz')
CANON14_FILE = os.path.join(HERE, 'oxn14_canon.npz')
CANON15_FILE = os.path.join(HERE, 'oxn15_canon.npz')
_CANON8 = None
_CANON14 = None
_CANON15 = None
RT, RSCALE, RL = 4096, 12, 1 << 23


def _canon8():
    global _CANON8
    if _CANON8 is None:
        if not os.path.exists(CANON8_FILE):
            raise RuntimeError(f'no OXN8 canonical tables: {CANON8_FILE}')
        with np.load(CANON8_FILE, allow_pickle=False) as ck:
            if int(ck['version'][0]) != 1:
                raise RuntimeError('unsupported OXN8 canonical table version')
            _CANON8 = {name: ck[name].copy() for name in ck.files
                       if name != 'version'}
        if _CANON8['tables'].shape != (2 * N_BUCK, 513):
            raise RuntimeError('invalid OXN8 canonical tables')
    return _CANON8


def _canon14():
    global _CANON14
    if _CANON14 is None:
        if not os.path.exists(CANON14_FILE):
            raise RuntimeError(f'no OXN14 canonical tables: {CANON14_FILE}')
        with np.load(CANON14_FILE, allow_pickle=False) as ck:
            if int(ck['version'][0]) != 1:
                raise RuntimeError('unsupported OXN14 canonical table version')
            tables = ck['tables'].copy()
        if tables.shape != (N_BUCK, 513) or np.any(tables[:, 0] != 0) \
                or np.any(tables[:, -1] != RT) or np.any(np.diff(tables) < 0):
            raise RuntimeError('invalid OXN14 canonical tables')
        _CANON14 = {'tables': tables}
    return _CANON14


def _canon15():
    global _CANON15
    if _CANON15 is None:
        if not os.path.exists(CANON15_FILE):
            raise RuntimeError(f'no OXN15 canonical tables: {CANON15_FILE}')
        with np.load(CANON15_FILE, allow_pickle=False) as ck:
            if int(ck['version'][0]) != 1:
                raise RuntimeError('unsupported OXN15 canonical table version')
            tables = ck['tables'].copy()
        if tables.shape != (2 * N_BUCK, 513) or np.any(tables[:, 0] != 0) \
                or np.any(tables[:, -1] != RT) or np.any(np.diff(tables) < 0):
            raise RuntimeError('invalid OXN15 canonical tables')
        _CANON15 = {'tables': tables}
    return _CANON15

# Ускорение: параметрический rANS через C-бинарник ox (с fallback на Python)
OX_BIN = os.path.join(HERE, 'ox')
_PRANS_TABLES = None
_PRANS_LEGACY = False
_PRANS_NATIVE = None
_PRANS_NATIVE_LEGACY = False


def _prans_tables(legacy=True):
    global _PRANS_TABLES, _PRANS_LEGACY
    if _PRANS_TABLES is None:
        ntabs = 7 * N_BUCK if os.path.exists(CANON15_FILE) else 5 * N_BUCK
        _PRANS_TABLES = np.zeros((ntabs, 513), dtype='<u4')
        _PRANS_TABLES[2 * N_BUCK:4 * N_BUCK] = _canon8()['tables']
        if os.path.exists(CANON14_FILE):
            _PRANS_TABLES[4 * N_BUCK:5 * N_BUCK] = _canon14()['tables']
        if os.path.exists(CANON15_FILE):
            _PRANS_TABLES[5 * N_BUCK:] = _canon15()['tables']
    if legacy and not _PRANS_LEGACY:
        _PRANS_TABLES[:2 * N_BUCK] = np.stack(
            [_lap_table(i) for i in range(N_BUCK)]
            + [_gauss_table(i) for i in range(N_BUCK)])
        _PRANS_LEGACY = True
    return _PRANS_TABLES


def _prans_native(legacy=False):
    global _PRANS_NATIVE, _PRANS_NATIVE_LEGACY
    if _PRANS_NATIVE is None:
        try:
            import ctypes
            lib = ctypes.CDLL(os.path.join(HERE, 'libox.so'))
            tables = _prans_tables(legacy)
            lut = np.empty((len(tables), RT), dtype='<u2')
            lib.ox_prans_make_lut.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                              ctypes.c_void_p]
            lib.ox_prans_make_lut.restype = None
            lib.ox_prans_make_lut(tables.ctypes.data, len(tables), lut.ctypes.data)
            _PRANS_NATIVE_LEGACY = legacy
            lib.ox_prans_encode.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
            lib.ox_prans_encode.restype = ctypes.c_int
            lib.ox_prans_decode.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32)]
            lib.ox_prans_decode.restype = ctypes.c_int
            lib.oxn8_requant.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_void_p]
            lib.oxn8_requant.restype = ctypes.c_int
            lib.oxn8_gelu.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_void_p, ctypes.c_size_t]
            lib.oxn8_gelu.restype = None
            _PRANS_NATIVE = (lib, tables, lut)
        except Exception:
            _PRANS_NATIVE = False
            print('OXN: libox.so unavailable; using subprocess/Python rANS',
                  file=sys.stderr)
    if _PRANS_NATIVE and legacy and not _PRANS_NATIVE_LEGACY:
        lib, tables, lut = _PRANS_NATIVE
        _prans_tables(True)
        lib.ox_prans_make_lut(tables.ctypes.data, len(tables), lut.ctypes.data)
        _PRANS_NATIVE_LEGACY = True
    return _PRANS_NATIVE or None


def _prans_tables_file(td):
    """Все бакет-таблицы в бинарник: [u32 K][K x 513 x u32 cum LE]."""
    tabs = _prans_tables()
    path = os.path.join(td, 'tables.bin')
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(tabs)))
        f.write(tabs.tobytes())
    return path


def rans_enc_param_fast(symbols, bids):
    """C-версия rans_enc_param; None при неудаче."""
    native = _prans_native(bool(np.any(np.asarray(bids) < 2 * N_BUCK)))
    if native is not None:
        import ctypes
        lib, tables, _ = native
        sym = np.ascontiguousarray(symbols, dtype='<u2')
        b = np.ascontiguousarray(bids, dtype=np.uint8)
        out = np.empty(2 * len(sym) + 16, dtype=np.uint8)
        out_n = ctypes.c_size_t()
        rc = lib.ox_prans_encode(tables.ctypes.data, sym.ctypes.data,
                                 b.ctypes.data, len(sym), out.ctypes.data,
                                 len(out), ctypes.byref(out_n))
        if rc == 0:
            return out[:out_n.value].tobytes()
    import subprocess
    import tempfile
    try:
        with tempfile.TemporaryDirectory(prefix='oxpr_') as td:
            tables = _prans_tables_file(td)
            data = os.path.join(td, 'data.bin')
            out = os.path.join(td, 'out.bin')
            with open(data, 'wb') as f:
                f.write(struct.pack('<I', len(symbols)))
                f.write(symbols.astype('<u2').tobytes())
                f.write(bids.astype(np.uint8).tobytes())
            r = subprocess.run([OX_BIN, 'prans', 'c', tables, data, out],
                               capture_output=True, timeout=3600)
            if r.returncode != 0:
                return None
            return open(out, 'rb').read()
    except Exception:
        return None


def rans_dec_param_fast(blob, bids):
    """C-версия rans_dec_param; None при неудаче.
    Возвращает (symbols, consumed): consumed == len(blob), если бакеты
    совпали с энкодером; иначе поток десинхронизирован."""
    native = _prans_native(bool(np.any(np.asarray(bids) < 2 * N_BUCK)))
    if native is not None:
        import ctypes
        lib, tables, lut = native
        raw = np.frombuffer(blob, dtype=np.uint8)
        b = np.ascontiguousarray(bids, dtype=np.uint8)
        out = np.empty(len(b), dtype='<u2')
        used = ctypes.c_uint32()
        rc = lib.ox_prans_decode(tables.ctypes.data, lut.ctypes.data,
                                 raw.ctypes.data, len(raw), b.ctypes.data,
                                 len(b), out.ctypes.data, ctypes.byref(used))
        if rc == 0:
            return out.astype(np.int32), used.value
    import subprocess
    import tempfile
    try:
        with tempfile.TemporaryDirectory(prefix='oxpr_') as td:
            tables = _prans_tables_file(td)
            bl = os.path.join(td, 'blob.bin')
            bd = os.path.join(td, 'bids.bin')
            out = os.path.join(td, 'out.bin')
            open(bl, 'wb').write(blob)
            with open(bd, 'wb') as f:
                f.write(struct.pack('<I', len(bids)))
                f.write(bids.astype(np.uint8).tobytes())
            r = subprocess.run([OX_BIN, 'prans', 'd', tables, bl, bd, out],
                               capture_output=True, timeout=3600)
            if r.returncode != 0:
                return None
            raw = open(out, 'rb').read()
            if len(raw) != len(bids) * 2 + 4:
                return None
            (used,) = struct.unpack('<I', raw[-4:])
            return (np.frombuffer(raw[:-4], '<u2').astype(np.int32), used)
    except Exception:
        return None

def _bucket_center(idx):
    return 2.0 ** (LOG_B_MIN + (idx + 0.5) * LOG_B_STEP)

def _bucket_idx(logb):
    return np.clip(np.round((logb / math.log(2) - LOG_B_MIN) / LOG_B_STEP),
                   0, N_BUCK - 1).astype(np.int32)

_TABLES = {}
def _lap_table(idx):
    """cum-таблица Лапласа(0, b_бакета) над 512 символами остатка."""
    cum = _TABLES.get(idx)
    if cum is not None:
        return cum
    b = _bucket_center(idx)
    d = np.arange(-RES_RANGE, RES_RANGE)
    p = np.exp(-np.abs(d) / b)
    f = np.maximum(1, np.round(p / p.sum() * RT).astype(np.int64))
    diff = int(RT - f.sum())
    order = np.argsort(-f)
    k = 0
    while diff:
        i = order[k % len(order)]
        if diff > 0:   f[i] += 1; diff -= 1
        elif f[i] > 1: f[i] -= 1; diff += 1
        k += 1
    cum = np.zeros(513, np.uint32)
    np.cumsum(f, out=cum[1:])
    _TABLES[idx] = cum
    return cum

def _gauss_table(idx):
    """cum-таблица дискретного Гаусса N(0, sigma_бакета) над 512 символами
    остатка (контекстный поток OXN5, бакеты N_BUCK..2*N_BUCK-1)."""
    key = ('g', idx)
    cum = _TABLES.get(key)
    if cum is not None:
        return cum
    sig = _bucket_center(idx)
    d = np.arange(-RES_RANGE, RES_RANGE)
    p = np.exp(-(d * d) / (2.0 * sig * sig))
    f = np.maximum(1, np.round(p / p.sum() * RT).astype(np.int64))
    diff = int(RT - f.sum())
    order = np.argsort(-f)
    k = 0
    while diff:
        i = order[k % len(order)]
        if diff > 0:   f[i] += 1; diff -= 1
        elif f[i] > 1: f[i] -= 1; diff += 1
        k += 1
    cum = np.zeros(513, np.uint32)
    np.cumsum(f, out=cum[1:])
    _TABLES[key] = cum
    return cum

def _table_for(bid):
    if bid < N_BUCK:
        return _lap_table(bid)
    if bid < 2 * N_BUCK:
        return _gauss_table(bid - N_BUCK)
    if bid >= 5 * N_BUCK:
        return _canon15()['tables'][bid - 5 * N_BUCK]
    if bid >= 4 * N_BUCK:
        return _canon14()['tables'][bid - 4 * N_BUCK]
    return _canon8()['tables'][bid - 2 * N_BUCK]

def rans_enc_param(symbols, bids):
    fast = rans_enc_param_fast(symbols, bids)
    if fast is not None:
        return fast
    x = RL
    out = bytearray()
    for i in range(len(symbols) - 1, -1, -1):
        cum = _table_for(int(bids[i]))
        s = int(symbols[i])
        f = int(cum[s + 1] - cum[s]); c = int(cum[s])
        x_max = ((RL >> RSCALE) << 8) * f
        while x >= x_max:
            out.append(x & 0xFF); x >>= 8
        x = ((x // f) << RSCALE) + (x % f) + c
    out += struct.pack('<I', x)
    out.reverse()
    return bytes(out)

def rans_dec_param(blob, bids):
    """-> (symbols, consumed). consumed == len(blob) — поток сошёлся точно;
    иначе таблицы (бакеты) не совпали с энкодером и symbols — мусор."""
    fast = rans_dec_param_fast(blob, bids)
    if fast is not None:
        return fast
    x = struct.unpack('>I', blob[:4])[0]
    pos = 4
    res = np.empty(len(bids), np.int32)
    for i in range(len(bids)):
        cum = _table_for(int(bids[i]))
        slot = x & (RT - 1)
        s = int(np.searchsorted(cum, slot, 'right') - 1)
        res[i] = s
        f = int(cum[s + 1] - cum[s]); c = int(cum[s])
        x = f * (x >> RSCALE) + slot - c
        while x < RL:
            x = (x << 8) | (blob[pos] if pos < len(blob) else 0)
            pos += 1
    return res, pos

def _legacy_cpu_model(model):
    """Свежая копия чекпоинта, загруженная сразу на CPU: fp32-численность
    oneDNN зависит от выравнивания буферов, поэтому для переходных файлов
    воспроизводим условия «чистого» CPU-процесса максимально близко."""
    global _LEGACY_CPU_MODEL_CACHE
    path = model.__dict__.get('_ckpt_path')
    key = ('cpu32', path)
    if _LEGACY_CPU_MODEL_CACHE is None or _LEGACY_CPU_MODEL_CACHE[0] != key:
        global _FORCE_CPU
        old = _FORCE_CPU
        _FORCE_CPU = True
        try:
            m2, _ = _load(path)
        finally:
            _FORCE_CPU = old
        _LEGACY_CPU_MODEL_CACHE = (key, m2)
    return _LEGACY_CPU_MODEL_CACHE[1]


_LEGACY_CPU_MODEL_CACHE = None


def _hyper_pair(model, dev):
    """Отдельная копия hyper-ветки на нужном устройстве/точности.
    'cpu'   — fp64: канон для новых файлов (детерминировано независимо от
              числа потоков; fp32-конволюции oneDNN гуляют на уровне ulp
              от процесса к процессу и ломают энтропию);
    'cpu32'/'cpu32t1' — fp32: файлы недолгой CPU-эпохи (t1 = один поток);
    'cuda'  — файлы GPU-эпохи (TF32 + bf16-автокаст для больших картинок)."""
    import copy
    import torch
    cache = model.__dict__.setdefault('_hyper_pair_cache', {})
    if dev not in cache:
        dtype = torch.float64 if dev == 'cpu' else torch.float32
        tdev = 'cpu' if dev.startswith('cpu') else dev
        if dev.startswith('cpu32'):
            src = _legacy_cpu_model(model)
            cache[dev] = (src.hyper_enc.eval(), src.hyper_dec.eval())
        else:
            cache[dev] = (copy.deepcopy(model.hyper_enc).to(tdev, dtype).eval(),
                          copy.deepcopy(model.hyper_dec).to(tdev, dtype).eval())
    return cache[dev]


def _hyper_mu_b(model, zqd, ht, wt, dev):
    """zqd: torch (T, C) деквантизованный латент -> w ints, mu, logb (T, C).
    Всегда CPU fp64 с выключенным oneDNN: mu/buckets задают таблицы энтропии
    и обязаны совпадать бит-в-бит при любом воспроизведении — oneDNN выбирает
    ядра свёртки по состоянию процесса, и даже fp64-результат «плывёт»,
    а с ним и rANS-поток."""
    import torch
    enc, dec = _hyper_pair(model, 'cpu')
    C = zqd.shape[1]
    grid = zqd.detach().to('cpu', torch.float64).transpose(0, 1).reshape(1, C, ht, wt)
    ph, pw = (-ht) % 4, (-wt) % 4
    gpad = torch.nn.functional.pad(grid, (0, pw, 0, ph), mode='replicate')
    mkldnn_on = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        with torch.no_grad():
            w = torch.round(enc(gpad))
            mb = dec(w)
    finally:
        torch.backends.mkldnn.enabled = mkldnn_on
    mb = mb[:, :, :ht, :wt].reshape(1, 2 * C, -1)[0].transpose(0, 1)
    mu, logb = mb.chunk(2, -1)
    wq = w[0].reshape(w.shape[1], -1).transpose(0, 1)  # HC x T' -> T' x HC
    return (wq.numpy().astype(np.int32), mu.numpy(), logb.numpy())


def _hyper_from_w(model, w, ht, wt, npix, dev):
    """mu_r, bids из гипер-латента w заданной численностью (см. _hyper_pair)."""
    import torch
    HC = w.shape[1]
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    _, dec = _hyper_pair(model, dev)
    tdev = 'cpu' if dev.startswith('cpu') else dev
    dtype = torch.float64 if dev == 'cpu' else torch.float32
    grid = torch.from_numpy(w.astype(np.float32)).to(tdev, dtype)
    grid = grid.transpose(0, 1).reshape(1, HC, ht4, wt4)
    old_nt = torch.get_num_threads() if dev == 'cpu32t1' else None
    if old_nt is not None:
        torch.set_num_threads(1)
    mkldnn_on = torch.backends.mkldnn.enabled
    if dev == 'cpu':
        torch.backends.mkldnn.enabled = False   # канон: без oneDNN
    try:
        with torch.no_grad(), _maybe_autocast(tdev, npix):
            mb = dec(grid)
    finally:
        torch.backends.mkldnn.enabled = mkldnn_on
        if old_nt is not None:
            torch.set_num_threads(old_nt)
    mb = mb[:, :, :ht, :wt].reshape(1, 2 * model.latent, -1)[0].transpose(0, 1)
    mu, logb = mb.chunk(2, -1)
    # точность как у выбранной численности: fp64 для канона, fp32 для legacy
    # (bf16 вывод автокаста расширяется до fp32 без потерь; fp64 не обрезать!)
    out_dt = torch.float64 if dev == 'cpu' else torch.float32
    mu_r = np.round(mu.to(out_dt).cpu().numpy()).astype(np.int64)
    bids = _bucket_idx(logb.to(out_dt).cpu().numpy())
    return mu_r, bids


def _decode_resid(model, w, ht, wt, npix, zblob):
    """Подбирает численность гипердекодера по целостности rANS-потока:
    верные таблицы съедают ровно len(zblob) байт. Порядок: новые файлы
    (CPU fp64), переходные (CPU fp32, в т.ч. однопоточные), GPU-эпоха."""
    import torch
    cands = ['cpu', 'cpu32', 'cpu32t1']
    if torch.cuda.is_available():
        cands.append('cuda')
    used = None
    for hdev in cands:
        try:
            mu_r, bids = _hyper_from_w(model, w, ht, wt, npix, hdev)
            resid, used = rans_dec_param(zblob, bids.reshape(-1))
        except Exception:
            continue                      # кандидат недоступен (напр. VRAM=0)
        if used == len(zblob):
            return mu_r, resid
    raise RuntimeError(
        f'corrupt .oxh: entropy stream desync (read {used} of {len(zblob)})')


def _verify_stream(zblob, resid, bids):
    """Страховка энкодера: свежий поток обязан читаться своими же таблицами.
    Лучше упасть здесь, чем выдать нечитаемый файл. По умолчанию выключена
    (дорого на больших картинках) — включается флагом --verify."""
    if not _VERIFY:
        return
    back, used = rans_dec_param(zblob, bids.reshape(-1))
    if used != len(zblob) or not np.array_equal(back, resid.reshape(-1)):
        raise RuntimeError('internal: entropy stream failed self-check')


def neuro_compress_hyper(rgb, q=75, ckpt=None):
    """OXN3: hyperprior. rgb uint8 HxWx3 -> bytes"""
    import torch
    model, dev = _load(ckpt)
    p = model.patch
    s = q2s(q)
    H, W = rgb.shape[:2]
    canvas, Hp, Wp = _pad_image(rgb, p)
    ht, wt = Hp // p, Wp // p
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    if dev == 'cuda' and Hp * Wp > 16_000_000:
        x = x.to(torch.bfloat16)          # на CPU: экономия и пиковой VRAM
    x = x.to(dev)
    with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
        z = model.encode(x)[0]
    q_lat = np.round(z.float().cpu().numpy() * s).astype(np.int64)     # T x C
    T_, C = q_lat.shape
    zqd = torch.from_numpy((q_lat / s).astype(np.float32)).to(dev)
    w, mu, logb = _hyper_mu_b(model, zqd, ht, wt, dev)
    mu_r = np.round(mu).astype(np.int64)
    resid = np.clip(q_lat - mu_r, -RES_RANGE, RES_RANGE - 1) + RES_RANGE
    bids = _bucket_idx(logb)
    zblob = rans_enc_param(resid.reshape(-1).astype(np.int32),
                           bids.reshape(-1))
    _verify_stream(zblob, resid, bids)
    wblob = _encode_flat(w)          # гипер-латент — ОДНИМ потоком (1 таблица)
    HC = w.shape[1]
    out = bytearray(b'OXN3')
    out += struct.pack('<HHfBBH', H, W, s, p, C, HC)
    out += struct.pack('<I', len(wblob)) + wblob
    out += struct.pack('<I', len(zblob)) + zblob
    return bytes(out)


def _neuro_decompress_v3(blob, ckpt):
    import torch
    H, W, s, p, C, HC = struct.unpack('<HHfBBH', blob[4:16])
    model, dev = _load(ckpt)
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    ht, wt = Hp // p, Wp // p
    T_ = ht * wt
    pos = 16
    (wlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    w = _decode_flat(blob[pos:pos + wlen], ht4 * wt4 * HC).reshape(ht4 * wt4, HC)
    pos += wlen
    (zlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    mu_r, resid = _decode_resid(model, w, ht, wt, Hp * Wp,
                                blob[pos:pos + zlen])
    q_lat = resid.reshape(T_, C) - RES_RANGE + mu_r
    zq = (q_lat / s).astype(np.float32)
    z = torch.from_numpy(zq)[None].to(dev)
    lg = torch.full((1, T_), math.log(s), device=dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _encode_flat(v):
    """Плоский int32-вектор -> один rANS-поток с одной таблицей."""
    v = v.astype(np.int64).reshape(-1)
    vmin = int(v.min())
    u = v - vmin
    if u.max() < 256:
        b1 = rans_encode(u.astype(np.uint8).tobytes())
        return struct.pack('<BiI', 1, vmin, len(b1)) + b1
    b1 = rans_encode((u & 0xFF).astype(np.uint8).tobytes())
    b2 = rans_encode((u >> 8).astype(np.uint8).tobytes())
    return (struct.pack('<BiI', 2, vmin, len(b1)) + b1
            + struct.pack('<I', len(b2)) + b2)


def _decode_flat(blob, n):
    kind, vmin, l1 = struct.unpack('<BiI', blob[:9])
    pos = 9
    u = np.frombuffer(rans_decode(blob[pos:pos + l1]), np.uint8).astype(np.int64)
    pos += l1
    if kind == 2:
        (l2,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
        u2 = np.frombuffer(rans_decode(blob[pos:pos + l2]),
                           np.uint8).astype(np.int64)
        u = u + (u2 << 8)
    assert len(u) == n
    return (u + vmin).astype(np.int32)


def _pad_image(rgb, p):
    H, W = rgb.shape[:2]
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    canvas = np.zeros((Hp, Wp, 3), np.float32)
    canvas[:H, :W] = rgb.astype(np.float32) / 255.0
    if Hp > H:  # replicate-pad
        canvas[H:, :W] = canvas[H - 1:H, :W]
    if Wp > W:
        canvas[:, W:] = canvas[:, W - 1:W]
    return canvas, Hp, Wp


def _run_decode(model, z, Hp, Wp, lg, dev):
    """model.decode по горизонтальным полосам токенов с перекрытием 16 токенов
    (контекст оконного attention) — позволяет декодировать изображения
    в сотни мегапикселей с ограниченной VRAM. На обычных — прямой вызов."""
    import torch
    p = model.patch
    ht, wt = Hp // p, Wp // p
    band = 128
    if ht <= band:
        with torch.no_grad(), _decode_autocast(dev):
            xr = model.decode(z, Hp, Wp, lg)
        return xr[0].permute(1, 2, 0).float().cpu().numpy()
    ov = 16
    canvas = np.zeros((Hp, Wp, 3), np.float32)
    for y0 in range(0, ht, band):
        y1 = min(y0 + band, ht)
        a, b = max(y0 - ov, 0), min(y1 + ov, ht)
        zb = z[:, a * wt:b * wt]
        lb = lg[:, a * wt:b * wt] if lg is not None else None
        with torch.no_grad(), _decode_autocast(dev):
            xb = model.decode(zb, (b - a) * p, Wp, lb)
        canvas[y0 * p:y1 * p] = \
            xb[0].permute(1, 2, 0).float().cpu().numpy()[(y0 - a) * p:(y1 - a) * p]
    return canvas


def _run_decode_patch_mse(model, z, Hp, Wp, lg, canvas, dev):
    import torch
    p = model.patch
    ht, wt = Hp // p, Wp // p
    B = z.shape[0]
    target = torch.from_numpy(canvas).permute(2, 0, 1)[None].to(dev)
    band = 128
    if ht <= band:
        with torch.no_grad(), _decode_autocast(dev):
            xr = model.decode(z, Hp, Wp, lg)
        d = ((xr.float() - target) ** 2).mean(1)
        d = d.reshape(B, ht, p, wt, p).mean(axis=(2, 4)) * 65025.0
        return d.reshape(B, -1).cpu().numpy()
    ov = 16
    out = np.empty((B, ht * wt), np.float32)
    for y0 in range(0, ht, band):
        y1 = min(y0 + band, ht)
        a, b = max(y0 - ov, 0), min(y1 + ov, ht)
        zb = z[:, a * wt:b * wt]
        lb = lg[:, a * wt:b * wt] if lg is not None else None
        with torch.no_grad(), _decode_autocast(dev):
            xb = model.decode(zb, (b - a) * p, Wp, lb)
        tb = target[:, :, a * p:b * p]
        d = ((xb.float() - tb) ** 2).mean(1)
        d = d[:, (y0 - a) * p:(y1 - a) * p]
        d = d.reshape(B, y1 - y0, p, wt, p).mean(axis=(2, 4)) * 65025.0
        out[:, y0 * wt:y1 * wt] = d.reshape(B, -1).cpu().numpy()
    return out


def neuro_compress(rgb, q=75, ckpt=None):
    """rgb: uint8 HxWx3 -> bytes .oxn (плоский режим, одна степень s)"""
    import torch
    model, dev = _load(ckpt)
    if getattr(model, 'hyper_enc', None) is not None:
        return neuro_compress_hyper(rgb, q, ckpt)
    p = model.patch
    s = q2s(q)
    H, W = rgb.shape[:2]
    canvas, Hp, Wp = _pad_image(rgb, p)
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    if dev == 'cuda' and Hp * Wp > 16_000_000:
        x = x.to(torch.bfloat16)          # на CPU: экономия и пиковой VRAM
    x = x.to(dev)
    with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
        z = model.encode(x)[0]                    # T x C
        zq = torch.round(z * s).float().cpu().numpy().astype(np.int32)   # T x C
    T_, C = zq.shape
    body = _encode_channels(zq)
    out = bytearray(b'OXN1')
    out += struct.pack('<HHfBB', H, W, s, p, C)
    out += struct.pack('<I', len(body)) + body
    return bytes(out)


# ------------------------------------------------- адаптивный режим (OXN2) -
# Лесенка степеней сжатия («110 сетей» в одной): для каждого патча
# выбирается своя степень по критерию D + lam*R (искажение + лямбда*биты).
LADDER = (0.35, 0.6, 1.0, 1.7, 2.9, 5.0, 8.0)
TAU = 15.0       # калибровано на Kodak: lam = TAU / s0^2


# ---- boost: выбор степени маленькой conv-сетью вместо 7 декодирований -----
BOOST_FILE = os.path.join(HERE, 'ox_boost.pt')
_BOOST = None          # (net, mu, sd) или False
_BOOST_ON = False      # включается флагом --boost / чекбоксом в UI


def _boost_net(dev):
    """Свёрточная сеть над гридом латентов: z(24)+E(7)+R(7)+pix(4)+Dmid
    каналов -> log D_k на патч. Два полных декодирования вместо семи."""
    global _BOOST
    if _BOOST is None:
        if not os.path.exists(BOOST_FILE):
            _BOOST = False
        else:
            import torch
            ck = torch.load(BOOST_FILE, map_location='cpu', weights_only=True)
            h = int(ck.get('hidden', 96))
            kmax = int(ck.get('kmax', 5))
            mid = list(ck.get('mid', [1, 3]))
            state = {k[4:] if k.startswith('net.') else k: v
                     for k, v in ck['state'].items()}
            nch = int(state['0.weight'].shape[1])
            n_conv = sum(1 for k in state if k.endswith('.weight'))
            layers = [torch.nn.Conv2d(nch, h, 3, padding=1), torch.nn.GELU()]
            for _ in range(n_conv - 2):
                layers += [torch.nn.Conv2d(h, h, 3, padding=1), torch.nn.GELU()]
            layers += [torch.nn.Conv2d(h, kmax, 1)]
            net = torch.nn.Sequential(*layers)
            net.load_state_dict(state)
            _BOOST = (net.eval().to(dev), ck['mu'].to(dev), ck['sd'].to(dev),
                      mid, kmax)
    return _BOOST


def _pix_feats(canvas, ht, wt, p):
    """Как при обучении: log1p(var), log1p|gx|, log1p|gy|, комбинированная."""
    g = canvas.mean(2)
    pv = g.reshape(ht, p, wt, p).var((1, 3))
    dgx = np.pad(np.abs(np.diff(g, axis=1)), ((0, 0), (0, 1)))
    dgx = dgx.reshape(ht, p, wt, p).mean((1, 3))
    dgy = np.pad(np.abs(np.diff(g, axis=0)), ((0, 1), (0, 0)))
    dgy = dgy.reshape(ht, p, wt, p).mean((1, 3))
    return np.stack([np.log1p(pv), np.log1p(dgx), np.log1p(dgy),
                     np.log1p(pv * 0.5 + dgx * dgy)], 0).astype(np.float32)


def _boost_predict(model, canvas, zs, E, R, D_mid, ht, wt, p, dev):
    """D_k на патч: точные для mid-скейлов (2 декодирования), остальные —
    предсказание conv-сети. zs: (T, C); E, R: (7, T); D_mid: (len(mid), T)."""
    import torch
    net, mu, sd, mid, kmax = _boost_net(dev)
    px = _pix_feats(canvas, ht, wt, p)
    dm = np.log1p(D_mid).astype(np.float32)
    X = np.concatenate([zs.T, E, R, px.reshape(4, -1), dm], 0).astype(np.float32)
    nch = X.shape[0]
    with torch.no_grad():
        xb = (torch.from_numpy(X.reshape(nch, ht, wt)).to(dev)
              - mu[:, None, None]) / sd[:, None, None]
        Dh = np.exp(net(xb[None])[0].float().cpu().numpy().reshape(kmax, -1)) - 1.0
    D = np.full((len(LADDER), Dh.shape[1]), 1e9, np.float32)
    D[:kmax] = Dh
    for i, k in enumerate(mid):
        D[k] = D_mid[i]
    return D


def neuro_compress_adaptive_hyper(rgb, q=75, ckpt=None):
    """OXN4: адаптивная степень по патчам + hyperprior.
    Оценка битрейта кандидатов — по (mu, b) от hyper при базовой степени;
    финальный hyper считается от СМЕШАННОГО латента (консистентно с декодером)."""
    import torch
    model, dev = _load(ckpt)
    p = model.patch
    H, W = rgb.shape[:2]
    canvas, Hp, Wp = _pad_image(rgb, p)
    ht, wt = Hp // p, Wp // p
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    if dev == 'cuda' and Hp * Wp > 16_000_000:
        x = x.to(torch.bfloat16)
    x = x.to(dev)
    s0 = q2s(q)
    lam = TAU / (s0 * s0)
    with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
        z = model.encode(x)[0]
    del x
    if dev == 'cuda':
        torch.cuda.empty_cache()
    zf = z.float()
    T_, C = zf.shape
    zs = zf.cpu().numpy()
    # hyper при базовой степени — для оценок битрейта кандидатов
    q0 = np.round(zs * s0)
    zqd0 = torch.from_numpy((q0 / s0).astype(np.float32)).to(dev)
    _, mu0, logb0 = _hyper_mu_b(model, zqd0, ht, wt, dev)
    b0 = np.log1p(np.exp(logb0)) + 1e-2                  # softplus, как в модели
    mu0r = np.round(mu0)
    Dk, Rk, Qk, Ek = [], [], [], []
    use_boost = _BOOST_ON and _boost_net(dev) is not False
    mid = _boost_net(dev)[3] if use_boost else []
    for k, s in enumerate(LADDER):
        qk = np.round(zs * s)
        Qk.append(qk)
        Rk.append((np.abs(qk - mu0r) / b0).sum(axis=1))  # остаток к mu0
        if use_boost:
            Ek.append(((zs * s - qk) ** 2).mean(axis=1))
            if k in mid:                       # только mid-скейлы декодируем
                zt = torch.from_numpy((qk / s).astype(np.float32))[None].to(dev)
                lg = torch.full((1, T_), math.log(s), device=dev)
                xrn = _run_decode(model, zt, Hp, Wp, lg, dev)
                d = ((xrn - canvas) ** 2).mean(2)
                Dk.append(d.reshape(ht, p, wt, p).mean(axis=(1, 3)).reshape(-1) * 65025.0)
        else:
            zt = torch.from_numpy((qk / s).astype(np.float32))[None].to(dev)
            lg = torch.full((1, T_), math.log(s), device=dev)
            xrn = _run_decode(model, zt, Hp, Wp, lg, dev)
            d = ((xrn - canvas) ** 2).mean(2)
            Dk.append(d.reshape(ht, p, wt, p).mean(axis=(1, 3)).reshape(-1) * 65025.0)
    if use_boost:
        D = _boost_predict(model, canvas, zs, np.stack(Ek), np.stack(Rk),
                           np.stack(Dk), ht, wt, p, dev)
    else:
        D = np.stack(Dk)
    R = np.stack(Rk)
    choice = np.argmin(D + lam * R, axis=0).astype(np.uint8)
    zq = np.zeros_like(Qk[0], dtype=np.int64)
    for k in range(len(LADDER)):
        m = choice == k
        zq[m] = Qk[k][m]
    # финальный hyper от смешанного латента
    scales_arr = np.array(LADDER, np.float32)
    zqd = torch.from_numpy(
        (zq / scales_arr[choice][:, None]).astype(np.float32)).to(dev)
    w, mu, logb = _hyper_mu_b(model, zqd, ht, wt, dev)
    mu_r = np.round(mu).astype(np.int64)
    resid = np.clip(zq - mu_r, -RES_RANGE, RES_RANGE - 1) + RES_RANGE
    bids = _bucket_idx(logb)
    zblob = rans_enc_param(resid.reshape(-1).astype(np.int32), bids.reshape(-1))
    _verify_stream(zblob, resid, bids)
    wblob = _encode_flat(w)
    mapblob = rans_encode(choice.tobytes())
    HC = w.shape[1]
    out = bytearray(b'OXN4')
    out += struct.pack('<HHBB', H, W, p, C)
    out += struct.pack('<B', len(LADDER))
    for s_ in LADDER:
        out += struct.pack('<e', s_)
    out += struct.pack('<H', HC)
    out += struct.pack('<I', len(wblob)) + wblob
    out += struct.pack('<I', len(mapblob)) + mapblob
    out += struct.pack('<I', len(zblob)) + zblob
    return bytes(out)


def _neuro_decompress_v4(blob, ckpt):
    import torch
    H, W, p, C = struct.unpack('<HHBB', blob[4:10])
    (K,) = struct.unpack('<B', blob[10:11])
    scales = np.array(struct.unpack(f'<{K}e', blob[11:11 + 2 * K]), np.float32)
    pos = 11 + 2 * K
    (HC,) = struct.unpack('<H', blob[pos:pos + 2]); pos += 2
    model, dev = _load(ckpt)
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    ht, wt = Hp // p, Wp // p
    T_ = ht * wt
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    (wlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    w = _decode_flat(blob[pos:pos + wlen], ht4 * wt4 * HC).reshape(ht4 * wt4, HC)
    pos += wlen
    (mlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    smap = np.frombuffer(rans_decode(blob[pos:pos + mlen]), np.uint8)
    pos += mlen
    assert smap.shape[0] == T_
    (zlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    mu_r, resid = _decode_resid(model, w, ht, wt, Hp * Wp,
                                blob[pos:pos + zlen])
    q_lat = resid.reshape(T_, C) - RES_RANGE + mu_r
    zq = (q_lat / scales[smap][:, None]).astype(np.float32)
    z = torch.from_numpy(zq)[None].to(dev)
    lg = torch.log(torch.from_numpy(scales[smap].astype(np.float32)))[None].to(dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---- OXN5: контекстная энтропийная модель --------------------------------
# Идея NLAIC (arXiv 1910.06244) и последующих работ: вероятность латента
# зависит не только от hyperprior, но и от уже декодированных соседей.
# Контекст — checkerboard (He et al., 2021): токены с чётной суммой индексов
# (i+j)%2==0 («якоря») кодируются первым потоком по Лапласу от hyperprior;
# нечётные — вторым потоком по Гауссу (mu, sigma) от маленькой conv-сети,
# которая видит ТОЛЬКО якоря (маскированная свёртка 5x5 + 1x1). Декод —
# два параллельных прохода вместо попиксельной авторегрессии.
CTX_FILE = os.path.join(HERE, 'ox_ctx.pt')
CTX_HIDDEN = 64
_CTX = None


def _build_ctx(latent, hidden=CTX_HIDDEN):
    import torch
    import torch.nn as nn

    class MaskedConv(nn.Conv2d):
        """Свёртка, читающая только якорные позиции: маска зануляет веса
        на чётных (di+dj) — на checkerboard-раскраске это всегда ещё
        неизвестные соседи (включая центр)."""
        def __init__(self, cin, cout, k=5):
            super().__init__(cin, cout, k, padding=k // 2)
            m = torch.zeros(1, 1, k, k)
            for di in range(k):
                for dj in range(k):
                    if (di + dj) % 2 == 1:
                        m[0, 0, di, dj] = 1.0
            self.register_buffer('m', m)

        def forward(self, x):
            return self._conv_forward(x, self.weight * self.m, self.bias)

    class CtxNet(nn.Module):
        """якоря (C) + hyper (mu, logb = 2C) + log s (1) -> (dmu, log sig)."""
        def __init__(self):
            super().__init__()
            self.c1 = MaskedConv(latent * 3 + 1, hidden)
            self.c2 = nn.Conv2d(hidden, hidden, 1)
            self.c3 = nn.Conv2d(hidden, latent * 2, 1)

        def forward(self, x):
            h = torch.nn.functional.gelu(self.c1(x))
            h = torch.nn.functional.gelu(self.c2(h))
            return self.c3(h)

    return CtxNet()


def _ctx_net(latent, path=None):
    """Контекстная сеть на CPU fp64 — канон бит-точности, как у hyperprior
    (mu/sigma задают таблицы энтропии и обязаны совпадать у кодека)."""
    global _CTX
    import torch
    path = path or CTX_FILE
    if _CTX is None or _CTX[0] != (path, latent):
        if not os.path.exists(path):
            sys.exit(f'no context model {path} — train first: '
                     f'python3 ox_neuro.py trainctx')
        ck = torch.load(path, map_location='cpu', weights_only=True)
        ctx = _build_ctx(latent, int(ck.get('hidden', CTX_HIDDEN)))
        ctx.load_state_dict(ck['state'])
        ctx = ctx.to(torch.float64).eval()
        _CTX = ((path, latent), ctx)
    return _CTX[1]


def _anchor_mask(ht, wt):
    """(T,) bool row-major: True = якорь ((i+j) чётное)."""
    ii, jj = np.meshgrid(np.arange(ht), np.arange(wt), indexing='ij')
    return ((ii + jj) % 2 == 0).reshape(-1)


def _grid4_masks(ht, wt):
    ii, jj = np.meshgrid(np.arange(ht), np.arange(wt), indexing='ij')
    return [((ii % 2 == di) & (jj % 2 == dj)).reshape(-1)
            for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1))]


def _ctx_params(ctx, zg, mu, logb, logs_map, ht, wt):
    """zg: (T, C) латент с занулёнными не-якорями; mu, logb: (T, C) от
    hyperprior; logs_map: (T,) fp64 — log потокенной степени сжатия.
    -> dmu, log_sig (T, C) fp64 для всех токенов (используются только
    не-якорные). CPU fp64, mkldnn off — канон, как _hyper_mu_b."""
    import torch
    C = zg.shape[1]
    feats = np.concatenate([
        zg.T.reshape(1, C, ht, wt),
        mu.T.reshape(1, C, ht, wt),
        logb.T.reshape(1, C, ht, wt),
        logs_map.astype(np.float64).reshape(1, 1, ht, wt)], axis=1)
    mkldnn_on = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        with torch.no_grad():
            out = ctx(torch.from_numpy(feats))
    finally:
        torch.backends.mkldnn.enabled = mkldnn_on
    dmu, lsig = out[0].chunk(2, 0)
    return dmu.reshape(C, -1).T.numpy(), lsig.reshape(C, -1).T.numpy()


def _enc_syms(resid, bids):
    """Один параметрический rANS-поток; пустой -> b'' (тривиальный декод)."""
    r = resid.reshape(-1).astype(np.int32)
    b = np.ascontiguousarray(bids.reshape(-1))
    if len(r) == 0:
        return b''
    blob = rans_enc_param(r, b)
    if _VERIFY:
        back, used = rans_dec_param(blob, b)
        if used != len(blob) or not np.array_equal(back, r):
            raise RuntimeError('internal: entropy stream failed self-check')
    return blob


def _dec_syms(blob, bids):
    """Обратно к _enc_syms; контроль целостности по consumed == len(blob)."""
    b = np.ascontiguousarray(bids.reshape(-1))
    if len(b) == 0:
        return np.empty(0, np.int32)
    syms, used = rans_dec_param(blob, b)
    if used != len(blob):
        raise RuntimeError(
            f'corrupt .oxh: entropy stream desync (read {used} of {len(blob)})')
    return syms


def neuro_compress_v5(rgb, q=75, ckpt=None):
    """OXN5: hyperprior (якоря, Лаплас) + checkerboard-контекст (Гаусс)."""
    import torch
    model, dev = _load(ckpt)
    if getattr(model, 'hyper_enc', None) is None:
        sys.exit('OXN5 требует hyper-чекпоинт (--hyper при обучении кодека)')
    p = model.patch
    # s квантуется в float32 ДО квантизации латента: в заголовок уйдёт ровно
    # это значение, и контекстная сеть (fp64) получит на обоих концах
    # бит-идентичные zg и log s — иначе ulp-разница ломает rANS-поток
    s = float(np.float32(q2s(q)))
    H, W = rgb.shape[:2]
    canvas, Hp, Wp = _pad_image(rgb, p)
    ht, wt = Hp // p, Wp // p
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    if dev == 'cuda' and Hp * Wp > 16_000_000:
        x = x.to(torch.bfloat16)
    x = x.to(dev)
    with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
        z = model.encode(x)[0]
    q_lat = np.round(z.float().cpu().numpy() * s).astype(np.int64)   # T x C
    T_, C = q_lat.shape
    zqd = torch.from_numpy((q_lat / s).astype(np.float32)).to(dev)
    w, mu, logb = _hyper_mu_b(model, zqd, ht, wt, dev)
    anch = _anchor_mask(ht, wt)
    # якоря: Лаплас от hyperprior (бакеты 0..23)
    mu_r = np.round(mu).astype(np.int64)
    resid = np.clip(q_lat - mu_r, -RES_RANGE, RES_RANGE - 1) + RES_RANGE
    ablob = _enc_syms(resid[anch], _bucket_idx(logb)[anch])
    # не-якоря: Гаусс от контекстной сети (бакеты 24..47)
    # ВАЖНО: в контекст идёт не истинный q_lat, а его ДЕКОДИРУЕМАЯ версия
    # (resid зажат в ±256 — иначе при больших s клиппинг разводит энкодер
    # и декодер, и поток не-якорей десинхронизируется)
    zg = np.zeros((T_, C), np.float64)
    zg[anch] = (resid[anch] - RES_RANGE + mu_r[anch]) / s
    ctx = _ctx_net(C)
    dmu, lsig = _ctx_params(ctx, zg, mu, logb, np.full(T_, math.log(s)),
                            ht, wt)
    mu_nr = np.round(mu + dmu).astype(np.int64)
    resid_n = np.clip(q_lat - mu_nr, -RES_RANGE, RES_RANGE - 1) + RES_RANGE
    sig = np.log1p(np.exp(lsig)) + 1e-2
    nblob = _enc_syms(resid_n[~anch], N_BUCK + _bucket_idx(np.log(sig))[~anch])
    wblob = _encode_flat(w)
    HC = w.shape[1]
    out = bytearray(b'OXN5')
    out += struct.pack('<HHfBBH', H, W, s, p, C, HC)
    out += struct.pack('<I', len(wblob)) + wblob
    out += struct.pack('<I', len(ablob)) + ablob
    out += struct.pack('<I', len(nblob)) + nblob
    return bytes(out)


def _hyper_raw_from_w(model, w, ht, wt):
    """Сырые (mu, logb) гипердекодера из w — CPU fp64, mkldnn off (канон)."""
    import torch
    HC = w.shape[1]
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    _, dec = _hyper_pair(model, 'cpu')
    grid = torch.from_numpy(w.astype(np.float32)).to('cpu', torch.float64)
    grid = grid.transpose(0, 1).reshape(1, HC, ht4, wt4)
    mkldnn_on = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        with torch.no_grad():
            mb = dec(grid)
    finally:
        torch.backends.mkldnn.enabled = mkldnn_on
    mb = mb[:, :, :ht, :wt].reshape(1, 2 * model.latent, -1)[0].transpose(0, 1)
    mu, logb = mb.chunk(2, -1)
    return mu.numpy(), logb.numpy()


def _neuro_decompress_v5(blob, ckpt):
    import torch
    H, W, s, p, C, HC = struct.unpack('<HHfBBH', blob[4:16])
    model, dev = _load(ckpt)
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    ht, wt = Hp // p, Wp // p
    T_ = ht * wt
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    pos = 16
    (wlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    w = _decode_flat(blob[pos:pos + wlen], ht4 * wt4 * HC).reshape(ht4 * wt4, HC)
    pos += wlen
    mu, logb = _hyper_raw_from_w(model, w, ht, wt)
    anch = _anchor_mask(ht, wt)
    na, nn_ = int(anch.sum()), T_ - int(anch.sum())
    (alen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    resid_a = _dec_syms(blob[pos:pos + alen], _bucket_idx(logb)[anch])
    pos += alen
    q_lat = np.zeros((T_, C), np.int64)
    q_lat[anch] = resid_a.reshape(na, C) - RES_RANGE + np.round(mu[anch])
    zg = np.zeros((T_, C), np.float64)
    zg[anch] = q_lat[anch] / s
    ctx = _ctx_net(C)
    dmu, lsig = _ctx_params(ctx, zg, mu, logb, np.full(T_, math.log(s)),
                            ht, wt)
    sig = np.log1p(np.exp(lsig)) + 1e-2
    bids_n = (N_BUCK + _bucket_idx(np.log(sig)))[~anch]
    (nlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    resid_n = _dec_syms(blob[pos:pos + nlen], bids_n)
    q_lat[~anch] = (resid_n.reshape(nn_, C) - RES_RANGE
                    + np.round(mu + dmu)[~anch])
    zq = (q_lat / s).astype(np.float32)
    z = torch.from_numpy(zq)[None].to(dev)
    lg = torch.full((1, T_), math.log(s), device=dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def neuro_compress_v6(rgb, q=75, ckpt=None):
    """OXN6: адаптивная степень по патчам (лесенка OXN4) + контекстная
    энтропийная модель (OXN5): якоря — Лаплас от hyperprior, остальные —
    Гаусс от контекстной сети, видящей якоря и карту log s."""
    import torch
    model, dev = _load(ckpt)
    if getattr(model, 'hyper_enc', None) is None:
        sys.exit('OXN6 требует hyper-чекпоинт (--hyper при обучении кодека)')
    p = model.patch
    H, W = rgb.shape[:2]
    canvas, Hp, Wp = _pad_image(rgb, p)
    ht, wt = Hp // p, Wp // p
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    if dev == 'cuda' and Hp * Wp > 16_000_000:
        x = x.to(torch.bfloat16)
    x = x.to(dev)
    s0 = q2s(q)
    lam = TAU / (s0 * s0)
    with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
        z = model.encode(x)[0]
    del x
    if dev == 'cuda':
        torch.cuda.empty_cache()
    zf = z.float()
    T_, C = zf.shape
    zs = zf.cpu().numpy()
    # hyper при базовой степени — для оценок битрейта кандидатов
    q0 = np.round(zs * s0)
    zqd0 = torch.from_numpy((q0 / s0).astype(np.float32)).to(dev)
    _, mu0, logb0 = _hyper_mu_b(model, zqd0, ht, wt, dev)
    b0 = np.log1p(np.exp(logb0)) + 1e-2
    mu0r = np.round(mu0)
    Dk, Rk, Qk, Ek = [], [], [], []
    use_boost = _BOOST_ON and _boost_net(dev) is not False
    mid = _boost_net(dev)[3] if use_boost else []
    for k, s in enumerate(LADDER):
        qk = np.round(zs * s)
        Qk.append(qk)
        Rk.append((np.abs(qk - mu0r) / b0).sum(axis=1))
        if use_boost:
            Ek.append(((zs * s - qk) ** 2).mean(axis=1))
            if k in mid:
                zt = torch.from_numpy((qk / s).astype(np.float32))[None].to(dev)
                lg = torch.full((1, T_), math.log(s), device=dev)
                xrn = _run_decode(model, zt, Hp, Wp, lg, dev)
                d = ((xrn - canvas) ** 2).mean(2)
                Dk.append(d.reshape(ht, p, wt, p).mean(axis=(1, 3)).reshape(-1) * 65025.0)
        else:
            zt = torch.from_numpy((qk / s).astype(np.float32))[None].to(dev)
            lg = torch.full((1, T_), math.log(s), device=dev)
            xrn = _run_decode(model, zt, Hp, Wp, lg, dev)
            d = ((xrn - canvas) ** 2).mean(2)
            Dk.append(d.reshape(ht, p, wt, p).mean(axis=(1, 3)).reshape(-1) * 65025.0)
    if use_boost:
        D = _boost_predict(model, canvas, zs, np.stack(Ek), np.stack(Rk),
                           np.stack(Dk), ht, wt, p, dev)
    else:
        D = np.stack(Dk)
    R = np.stack(Rk)
    choice = np.argmin(D + lam * R, axis=0).astype(np.uint8)
    zq = np.zeros_like(Qk[0], dtype=np.int64)
    for k in range(len(LADDER)):
        m = choice == k
        zq[m] = Qk[k][m]
    # каноничные степени — float16, как в заголовке: входы контекстной сети
    # (zg и карта log s) обязаны совпасть с декодером бит-в-бит
    scales16 = np.array(LADDER, np.float16)
    stok = scales16[choice].astype(np.float32)                 # (T,)
    # финальный hyper от смешанного латента
    zqd = torch.from_numpy((zq / stok[:, None]).astype(np.float32)).to(dev)
    w, mu, logb = _hyper_mu_b(model, zqd, ht, wt, dev)
    anch = _anchor_mask(ht, wt)
    # якоря: Лаплас от hyperprior (бакеты 0..23)
    mu_r = np.round(mu).astype(np.int64)
    resid = np.clip(zq - mu_r, -RES_RANGE, RES_RANGE - 1) + RES_RANGE
    ablob = _enc_syms(resid[anch], _bucket_idx(logb)[anch])
    # не-якоря: Гаусс от контекстной сети (бакеты 24..47); в контекст идёт
    # ДЕКОДИРУЕМАЯ версия якорей (клиппинг ±256), как в OXN5
    zg = np.zeros((T_, C), np.float64)
    zg[anch] = (resid[anch] - RES_RANGE + mu_r[anch]) / stok[anch, None]
    ctx = _ctx_net(C)
    dmu, lsig = _ctx_params(ctx, zg, mu, logb,
                            np.log(stok.astype(np.float64)), ht, wt)
    mu_nr = np.round(mu + dmu).astype(np.int64)
    resid_n = np.clip(zq - mu_nr, -RES_RANGE, RES_RANGE - 1) + RES_RANGE
    sig = np.log1p(np.exp(lsig)) + 1e-2
    nblob = _enc_syms(resid_n[~anch], N_BUCK + _bucket_idx(np.log(sig))[~anch])
    wblob = _encode_flat(w)
    mapblob = rans_encode(choice.tobytes())
    HC = w.shape[1]
    out = bytearray(b'OXN6')
    out += struct.pack('<HHBB', H, W, p, C)
    out += struct.pack('<B', len(LADDER))
    for s_ in scales16:
        out += struct.pack('<e', float(s_))
    out += struct.pack('<H', HC)
    out += struct.pack('<I', len(wblob)) + wblob
    out += struct.pack('<I', len(mapblob)) + mapblob
    out += struct.pack('<I', len(ablob)) + ablob
    out += struct.pack('<I', len(nblob)) + nblob
    return bytes(out)


def _neuro_decompress_v6(blob, ckpt):
    import torch
    H, W, p, C = struct.unpack('<HHBB', blob[4:10])
    (K,) = struct.unpack('<B', blob[10:11])
    scales = np.array(struct.unpack(f'<{K}e', blob[11:11 + 2 * K]), np.float32)
    pos = 11 + 2 * K
    (HC,) = struct.unpack('<H', blob[pos:pos + 2]); pos += 2
    model, dev = _load(ckpt)
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    ht, wt = Hp // p, Wp // p
    T_ = ht * wt
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    (wlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    w = _decode_flat(blob[pos:pos + wlen], ht4 * wt4 * HC).reshape(ht4 * wt4, HC)
    pos += wlen
    (mlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    smap = np.frombuffer(rans_decode(blob[pos:pos + mlen]), np.uint8)
    pos += mlen
    assert smap.shape[0] == T_
    stok = scales[smap]                                        # (T,) float32
    mu, logb = _hyper_raw_from_w(model, w, ht, wt)
    anch = _anchor_mask(ht, wt)
    na, nn_ = int(anch.sum()), T_ - int(anch.sum())
    (alen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    resid_a = _dec_syms(blob[pos:pos + alen], _bucket_idx(logb)[anch])
    pos += alen
    q_lat = np.zeros((T_, C), np.int64)
    q_lat[anch] = resid_a.reshape(na, C) - RES_RANGE + np.round(mu[anch])
    zg = np.zeros((T_, C), np.float64)
    zg[anch] = q_lat[anch] / stok[anch, None]
    ctx = _ctx_net(C)
    dmu, lsig = _ctx_params(ctx, zg, mu, logb,
                            np.log(stok.astype(np.float64)), ht, wt)
    sig = np.log1p(np.exp(lsig)) + 1e-2
    bids_n = (N_BUCK + _bucket_idx(np.log(sig)))[~anch]
    (nlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    resid_n = _dec_syms(blob[pos:pos + nlen], bids_n)
    q_lat[~anch] = (resid_n.reshape(nn_, C) - RES_RANGE
                    + np.round(mu + dmu)[~anch])
    zq = q_lat / stok[:, None]
    z = torch.from_numpy(zq.astype(np.float32))[None].to(dev)
    lg = torch.log(torch.from_numpy(stok))[None].to(dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---- OXN7: канальный контекст (ELIC-style) --------------------------------
# Каналы латента делятся на группы; самая информативная первая группа
# (якоря) кодируется Лапласом от hyperprior, дальше каждая группа — Гауссом
# от общей ctx-сети в два прохода: сначала якоря группы (контекст —
# предыдущие группы целиком), потом не-якоря (+ якоря своей группы).
# Неизвестное зануляется: паттерн нулей детерминирован порядком декодирования
# и совпадает на обоих концах — маскированные свёртки не нужны.
CTX7_FILE = os.path.join(HERE, 'ox_ctx7.pt')
OXN8_CTX7 = CTX7_FILE
TINY_CKPT = os.path.join(HERE, 'ox_neuro_tiny.pt')
TINY_CTX7 = os.path.join(HERE, 'ox_ctx9_tiny.pt')
OXN10_CTX = os.path.join(HERE, 'ox_ctx10.pt')
OXN11_CTX = os.path.join(HERE, 'ox_ctx11.pt')
OXN12_CTX = os.path.join(HERE, 'ox_ctx12.pt')
OXN13_CTX = os.path.join(HERE, 'ox_ctx13.pt')
CTX7_GROUPS = (6, 6, 6, 6)
_CTX7 = None
_CTX10 = None


def _build_ctx7(latent, hidden=CTX_HIDDEN):
    import torch
    import torch.nn as nn

    class CtxNet7(nn.Module):
        """латент с занулённым неизвестным (C) + hyper mu,logb (2C) +
        log s (1) -> (dmu, log sig) по всем каналам (2C)."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(latent * 3 + 1, hidden, 5, padding=2), nn.GELU(),
                nn.Conv2d(hidden, hidden, 1), nn.GELU(),
                nn.Conv2d(hidden, hidden, 1), nn.GELU(),
                nn.Conv2d(hidden, latent * 2, 1))

        def forward(self, x):
            return self.net(x)

    return CtxNet7()


def _build_ctx10_scale(latent, hidden=CTX_HIDDEN):
    import torch
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(latent * 2 + 1, hidden, 5, padding=2), nn.GELU(),
        nn.Conv2d(hidden, hidden, 1), nn.GELU(),
        nn.Conv2d(hidden, latent, 1))


def _fake_quant(x, scale):
    import torch
    return x + (torch.round(x * scale) / scale - x).detach()


def _qat_scale_forward(net, x):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    for layer in net:
        if isinstance(layer, nn.Conv2d):
            w = _fake_quant(layer.weight, Q8_WSCALE)
            b = None if layer.bias is None else _fake_quant(
                layer.bias, Q8_FSCALE * Q8_WSCALE)
            x = F.conv2d(x, w, b, layer.stride, layer.padding,
                         layer.dilation, layer.groups)
            x = _fake_quant(x, Q8_FSCALE)
        elif isinstance(layer, nn.GELU):
            x = _fake_quant(F.gelu(x), Q8_FSCALE)
        else:
            raise RuntimeError(f'QAT: unsupported scale layer {layer}')
    return x


def _ctx7_net(latent, path=None):
    """-> (net fp64 CPU, groups). Канон бит-точности, как у _ctx_net."""
    global _CTX7
    import torch
    path = path or CTX7_FILE
    if _CTX7 is None or _CTX7[0] != (path, latent):
        if not os.path.exists(path):
            sys.exit(f'no context model {path} — train first: python3 '
                     f'ox_neuro.py trainctx --groups 6,6,6,6')
        ck = torch.load(path, map_location='cpu', weights_only=True)
        groups = tuple(int(v) for v in ck['groups'])
        ctx = _build_ctx7(latent, int(ck.get('hidden', CTX_HIDDEN)))
        ctx.load_state_dict(ck['state'])
        ctx = ctx.to(torch.float64).eval()
        _CTX7 = ((path, latent), ctx, groups)
    if sum(_CTX7[2]) != latent:
        sys.exit(f'{path}: groups {_CTX7[2]} не сходятся с latent={latent}')
    return _CTX7[1], _CTX7[2]


def _ctx10_net(latent, dev, path=None):
    global _CTX10
    import torch
    path = path or OXN10_CTX
    key = (path, latent, dev)
    if _CTX10 is None or _CTX10[0] != key:
        if not os.path.exists(path):
            sys.exit(f'no OXN10 context model {path} — run trainctx --one_shot')
        ck = torch.load(path, map_location='cpu', weights_only=True)
        groups = tuple(int(v) for v in ck['groups'])
        if sum(groups) != latent:
            sys.exit(f'{path}: groups {groups} не сходятся с latent={latent}')
        ctx = _build_ctx7(latent, int(ck.get('hidden', CTX_HIDDEN)))
        ctx.load_state_dict(ck['state'])
        ctx._grid4 = bool(ck.get('grid4', False))
        offset = ck.get('sigma_offset')
        if offset is None:
            ctx._sigma_offset_q = None
        else:
            offset = offset.detach().cpu().numpy().astype(np.float64)
            if offset.shape != (latent,):
                sys.exit(f'{path}: invalid sigma_offset shape {offset.shape}')
            ctx._sigma_offset_q = np.rint(
                offset * Q8_FSCALE).astype(np.int64)
        scale_state = ck.get('scale_state')
        if scale_state is None:
            ctx._scale_net = None
            ctx.__dict__['_qscale_net'] = None
        else:
            scale = _build_ctx10_scale(latent, int(ck.get('hidden', CTX_HIDDEN)))
            scale.load_state_dict(scale_state)
            ctx._scale_net = scale.to(dev, torch.float32).eval()
            ctx.__dict__['_qscale_net'] = _qnet_copy(scale)
        ctx = ctx.to(dev, torch.float32).eval()
        _CTX10 = (key, ctx, groups)
    return _CTX10[1], _CTX10[2]


def _group_cuts(groups):
    cuts = np.concatenate([[0], np.cumsum(groups)])
    return [(int(cuts[i]), int(cuts[i + 1])) for i in range(len(groups))]


def neuro_compress_v7(rgb, q=75, ckpt=None):
    """OXN7: канальные группы (ELIC) + checkerboard внутри группы."""
    import torch
    model, dev = _load(ckpt)
    if getattr(model, 'hyper_enc', None) is None:
        sys.exit('OXN7 требует hyper-чекпоинт (--hyper при обучении кодека)')
    p = model.patch
    s = float(np.float32(q2s(q)))
    H, W = rgb.shape[:2]
    canvas, Hp, Wp = _pad_image(rgb, p)
    ht, wt = Hp // p, Wp // p
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    if dev == 'cuda' and Hp * Wp > 16_000_000:
        x = x.to(torch.bfloat16)
    x = x.to(dev)
    with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
        z = model.encode(x)[0]
    q_lat = np.round(z.float().cpu().numpy() * s).astype(np.int64)   # T x C
    T_, C = q_lat.shape
    zqd = torch.from_numpy((q_lat / s).astype(np.float32)).to(dev)
    w, mu, logb = _hyper_mu_b(model, zqd, ht, wt, dev)
    ctx, groups = _ctx7_net(C)
    anch = _anchor_mask(ht, wt)
    logs_map = np.full(T_, math.log(s))
    mu_r = np.round(mu).astype(np.int64)
    bids_l = _bucket_idx(logb)
    qd = np.zeros((T_, C), np.int64)      # декодируемый латент по ходу проходов
    streams = []
    for gi, (lo, hi) in enumerate(_group_cuts(groups)):
        for phase in (True, False):       # True = якоря, False = не-якоря
            pos = anch if phase else ~anch
            if gi == 0 and phase:
                # первая группа якорей: контекста нет — Лаплас от hyper
                resid = (np.clip(q_lat - mu_r, -RES_RANGE, RES_RANGE - 1)
                         + RES_RANGE)
                streams.append(_enc_syms(resid[pos][:, lo:hi],
                                         bids_l[pos][:, lo:hi]))
                mu_ref = mu_r
            else:
                zg = (qd / s).astype(np.float64)
                dmu, lsig = _ctx_params(ctx, zg, mu, logb, logs_map, ht, wt)
                mu_nr = np.round(mu + dmu).astype(np.int64)
                resid = (np.clip(q_lat - mu_nr, -RES_RANGE, RES_RANGE - 1)
                         + RES_RANGE)
                sig = np.log1p(np.exp(lsig)) + 1e-2
                bids = N_BUCK + _bucket_idx(np.log(sig))
                streams.append(_enc_syms(resid[pos][:, lo:hi],
                                         bids[pos][:, lo:hi]))
                mu_ref = mu_nr
            # дальше в контекст идёт ДЕКОДИРУЕМАЯ версия (клиппинг ±256)
            qd[pos, lo:hi] = (resid[pos][:, lo:hi] - RES_RANGE
                              + mu_ref[pos][:, lo:hi])
    wblob = _encode_flat(w)
    HC = w.shape[1]
    out = bytearray(b'OXN7')
    out += struct.pack('<HHfBBH', H, W, s, p, C, HC)
    out += struct.pack('<B', len(groups))
    for gs in groups:
        out += struct.pack('<B', gs)
    out += struct.pack('<I', len(wblob)) + wblob
    for blob in streams:
        out += struct.pack('<I', len(blob)) + blob
    return bytes(out)


def _neuro_decompress_v7(blob, ckpt):
    import torch
    H, W, s, p, C, HC = struct.unpack('<HHfBBH', blob[4:16])
    (G,) = struct.unpack('<B', blob[16:17])
    groups = tuple(struct.unpack(f'<{G}B', blob[17:17 + G]))
    pos = 17 + G
    model, dev = _load(ckpt)
    ctx, ck_groups = _ctx7_net(C)
    if ck_groups != groups:
        sys.exit(f'ox_ctx7.pt: groups {ck_groups}, в файле {groups}')
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    ht, wt = Hp // p, Wp // p
    T_ = ht * wt
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    (wlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    w = _decode_flat(blob[pos:pos + wlen], ht4 * wt4 * HC).reshape(ht4 * wt4, HC)
    pos += wlen
    mu, logb = _hyper_raw_from_w(model, w, ht, wt)
    anch = _anchor_mask(ht, wt)
    logs_map = np.full(T_, math.log(s))
    mu_r = np.round(mu).astype(np.int64)
    bids_l = _bucket_idx(logb)
    qd = np.zeros((T_, C), np.int64)
    for gi, (lo, hi) in enumerate(_group_cuts(groups)):
        for phase in (True, False):
            posm = anch if phase else ~anch
            (slen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
            if gi == 0 and phase:
                resid = _dec_syms(blob[pos:pos + slen],
                                  bids_l[posm][:, lo:hi])
                mu_ref = mu_r
            else:
                zg = (qd / s).astype(np.float64)
                dmu, lsig = _ctx_params(ctx, zg, mu, logb, logs_map, ht, wt)
                mu_nr = np.round(mu + dmu).astype(np.int64)
                sig = np.log1p(np.exp(lsig)) + 1e-2
                bids = N_BUCK + _bucket_idx(np.log(sig))
                resid = _dec_syms(blob[pos:pos + slen],
                                  bids[posm][:, lo:hi])
                mu_ref = mu_nr
            pos += slen
            qd[posm, lo:hi] = (resid.reshape(int(posm.sum()), hi - lo)
                               - RES_RANGE + mu_ref[posm][:, lo:hi])
    zq = (qd / s).astype(np.float32)
    z = torch.from_numpy(zq)[None].to(dev)
    lg = torch.full((1, T_), math.log(s), device=dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---- OXN8: детерминированная целочисленная энтропийная модель ------------
# Слои хранят активации в Q20.12, веса — в Q20.12. Свёртка выполняется над
# целыми значениями в float64: произведения и все частичные суммы меньше 2^53,
# поэтому остаются точными целыми независимо от порядка суммирования. Между
# слоями используются только integer round/shift и замороженная GELU-таблица.
Q8_FBITS = 12
Q8_WBITS = 12
Q8_FSCALE = 1 << Q8_FBITS
Q8_WSCALE = 1 << Q8_WBITS
Q8_MAX = 256 * Q8_FSCALE
_QCTX7 = None


def _qround_shift(x, bits):
    """Round-to-nearest-even для целых NumPy/Torch, включая отрицательные."""
    floor = x >> bits
    rem = x & ((1 << bits) - 1)
    half = 1 << (bits - 1)
    return floor + ((rem > half) | ((rem == half) & ((floor & 1) != 0)))


def _qround_div(num, den):
    """Round-to-nearest-even для NumPy int64 и положительного делителя."""
    floor = np.floor_divide(num, den)
    rem = num - floor * den
    twice = rem * 2
    return floor + ((twice > den) | ((twice == den) & ((floor & 1) != 0)))


def _qdequant_latent(q_lat, scales_q, companding=0):
    if companding:
        z = companding * np.sinh(q_lat / scales_q[:, None])
        return np.clip(np.rint(z * Q8_FSCALE), -Q8_MAX, Q8_MAX).astype(np.int64)
    num = q_lat * (256 * Q8_FSCALE)
    return np.clip(_qround_div(num, scales_q[:, None]),
                   -Q8_MAX, Q8_MAX).astype(np.int64)


def _ctx10_mu(ctx, zg_q, mu_q, logb_q, logs_q, ht, wt, dev):
    import torch
    C = zg_q.shape[1]
    feats = np.concatenate([
        zg_q.T.reshape(1, C, ht, wt),
        mu_q.T.reshape(1, C, ht, wt),
        logb_q.T.reshape(1, C, ht, wt),
        logs_q.reshape(1, 1, ht, wt)], axis=1).astype(np.float32)
    feats *= 1.0 / Q8_FSCALE
    with torch.no_grad():
        out = ctx(torch.from_numpy(feats).to(dev))
    dmu = out[0].chunk(2, 0)[0]
    return dmu.float().cpu().numpy().reshape(C, -1).T


def _qgelu(x):
    import torch
    native = _prans_native()
    if native is not None:
        lib, _, _ = native
        xa = x.numpy()
        lut = np.ascontiguousarray(_canon8()['gelu'], dtype=np.int64)
        lib.oxn8_gelu(xa.ctypes.data, xa.size, lut.ctypes.data, len(lut))
        return x
    lut = torch.from_numpy(_canon8()['gelu'].astype(np.int64))
    pos = x + 8 * Q8_FSCALE
    idx = torch.clamp(pos >> 4, 0, lut.numel() - 2)
    frac = pos & 15
    y0, y1 = lut[idx], lut[idx + 1]
    y = y0 + _qround_shift((y1 - y0) * frac, 4)
    return torch.where(x <= -8 * Q8_FSCALE, torch.zeros_like(x),
                       torch.where(x >= 8 * Q8_FSCALE, x, y))


def _qnet_copy(net):
    import copy
    import torch
    import torch.nn as nn
    q = copy.deepcopy(net).to('cpu', torch.float64).eval()
    for layer in q.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            layer.weight.data.copy_(torch.round(
                layer.weight.data * Q8_WSCALE).clamp_(-32768, 32767))
            if layer.bias is not None:
                layer.bias.data.copy_(torch.round(
                    layer.bias.data * (Q8_FSCALE * Q8_WSCALE)))
    return q


def _qnet_run(layers, x):
    import torch
    import torch.nn as nn
    native = _prans_native()
    for layer in layers:
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            raw = layer(x.to('cpu', torch.float64))
            if native is not None:
                lib, _, _ = native
                raw_np = raw.detach().numpy()
                out = np.empty(raw_np.shape, dtype=np.int64)
                if lib.oxn8_requant(raw_np.ctypes.data, raw_np.size,
                                    out.ctypes.data):
                    raise RuntimeError('OXN8 integer convolution overflow')
                x = torch.from_numpy(out)
            else:
                if not torch.isfinite(raw).all():
                    raise RuntimeError('OXN8 integer convolution overflow')
                x = _qround_shift(raw.to(torch.int64), Q8_WBITS)
                x = torch.clamp(x, -Q8_MAX, Q8_MAX)
        elif isinstance(layer, nn.GELU):
            x = _qgelu(x)
        else:
            raise RuntimeError(f'OXN8: unsupported entropy layer {layer}')
    return x


def _qhyper_pair(model):
    cache = model.__dict__.get('_qhyper_pair')
    if cache is None:
        enc, dec = _hyper_pair(model, 'cpu')
        cache = (_qnet_copy(enc), _qnet_copy(dec))
        model.__dict__['_qhyper_pair'] = cache
    return cache


def _qhyper_encode(model, zqd_q, ht, wt):
    import torch
    enc, dec = _qhyper_pair(model)
    C = zqd_q.shape[1]
    grid = torch.from_numpy(zqd_q.T.reshape(1, C, ht, wt))
    ph, pw = (-ht) % 4, (-wt) % 4
    gpad = torch.nn.functional.pad(grid, (0, pw, 0, ph), mode='replicate')
    with torch.no_grad():
        w_real = _qnet_run(enc.net, gpad)
        w = _qround_shift(w_real, Q8_FBITS)
        mb = _qnet_run(dec.net, w * Q8_FSCALE)
    mb = mb[:, :, :ht, :wt].reshape(1, 2 * C, -1)[0].transpose(0, 1)
    mu_q, logb_q = mb.chunk(2, -1)
    wq = w[0].reshape(w.shape[1], -1).transpose(0, 1)
    return (wq.numpy().astype(np.int32), mu_q.numpy(), logb_q.numpy())


def _hyper_rate_estimate(model, zqd_q, ht, wt, dev):
    import torch
    C = zqd_q.shape[1]
    grid = torch.from_numpy(zqd_q.T.reshape(1, C, ht, wt).astype(np.float32)
                            / Q8_FSCALE).to(dev)
    ph, pw = (-ht) % 4, (-wt) % 4
    gpad = torch.nn.functional.pad(grid, (0, pw, 0, ph), mode='replicate')
    with torch.no_grad():
        w = model.hyper_enc(gpad)
        mb = model.hyper_dec(torch.round(w))
    mb = mb[:, :, :ht, :wt].reshape(1, 2 * C, -1)[0].transpose(0, 1)
    mu, logb = mb.chunk(2, -1)
    return (np.round(mu.float().cpu().numpy()).astype(np.int64),
            logb.float().cpu().numpy())


def _qhyper_decode(model, w, ht, wt):
    import torch
    _, dec = _qhyper_pair(model)
    HC = w.shape[1]
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    grid = torch.from_numpy(w.astype(np.int64)).transpose(0, 1).reshape(
        1, HC, ht4, wt4) * Q8_FSCALE
    with torch.no_grad():
        mb = _qnet_run(dec.net, grid)
    C = model.latent
    mb = mb[:, :, :ht, :wt].reshape(1, 2 * C, -1)[0].transpose(0, 1)
    mu_q, logb_q = mb.chunk(2, -1)
    return mu_q.numpy(), logb_q.numpy()


def _qctx7_net(latent):
    global _QCTX7
    path = CTX7_FILE
    if _QCTX7 is None or _QCTX7[0] != (path, latent):
        ctx, groups = _ctx7_net(latent)
        _QCTX7 = ((path, latent), _qnet_copy(ctx), groups)
    return _QCTX7[1], _QCTX7[2]


def _qctx7_params(ctx, zg_q, mu_q, logb_q, logs_q, ht, wt):
    import torch
    C = zg_q.shape[1]
    feats = np.concatenate([
        zg_q.T.reshape(1, C, ht, wt),
        mu_q.T.reshape(1, C, ht, wt),
        logb_q.T.reshape(1, C, ht, wt),
        logs_q.reshape(1, 1, ht, wt)], axis=1)
    with torch.no_grad():
        out = _qnet_run(ctx.net, torch.from_numpy(feats))
    dmu, lsig = out[0].chunk(2, 0)
    return dmu.reshape(C, -1).T.numpy(), lsig.reshape(C, -1).T.numpy()


def _q8_lap_buckets(logb_q):
    thresholds = _canon8()['lap_thresholds']
    return (2 * N_BUCK + np.searchsorted(thresholds, logb_q, side='right')
            ).astype(np.int32)


def _q8_gauss_buckets(lsig_q):
    thresholds = _canon8()['gauss_thresholds']
    return (3 * N_BUCK + np.searchsorted(thresholds, lsig_q, side='right')
            ).astype(np.int32)


def _q14_gauss_buckets(lsig_q):
    thresholds = _canon8()['gauss_thresholds']
    return (4 * N_BUCK + np.searchsorted(thresholds, lsig_q, side='right')
            ).astype(np.int32)


def _ctx10_gauss_buckets(mu_q, logb_q, logs_q, ht, wt, dev, ctx,
                         custom=False):
    import torch
    offset_q = ctx._sigma_offset_q
    if ctx._scale_net is not None:
        C = logb_q.shape[1]
        feats = np.concatenate([
            mu_q.T.reshape(1, C, ht, wt),
            logb_q.T.reshape(1, C, ht, wt),
            logs_q.reshape(1, 1, ht, wt)], axis=1)
        with torch.no_grad():
            lsig = _qnet_run(ctx._qscale_net, torch.from_numpy(feats))
        lsig_q = lsig[0].reshape(C, -1).T.numpy()
        if offset_q is not None:
            lsig_q += offset_q[None, :]
        return (_q14_gauss_buckets(lsig_q) if custom
                else _q8_gauss_buckets(lsig_q))
    if offset_q is None:
        return _q8_gauss_buckets(logb_q)
    logb = logb_q.astype(np.float64) / Q8_FSCALE
    sig = np.logaddexp(0.0, logb) + 1e-2
    lsig_q = np.rint(np.log(sig) * Q8_FSCALE).astype(np.int64)
    return _q8_gauss_buckets(lsig_q + offset_q[None, :])


def _collect_rans_stats(resid, bids):
    if _RANS_STATS is None:
        return
    for bucket in np.unique(bids):
        if bucket >= 5 * N_BUCK:
            row = bucket - 5 * N_BUCK
        elif bucket >= 3 * N_BUCK:
            row = bucket - 3 * N_BUCK
        else:
            continue
        values = resid[bids == bucket]
        _RANS_STATS[row] += np.bincount(values.reshape(-1), minlength=513)


def _rans_table_cost(q, mu, bids):
    resid = (np.clip(q - mu, -RES_RANGE, RES_RANGE - 1) + RES_RANGE).astype(
        np.int64)
    if np.any(bids >= 5 * N_BUCK):
        rows = bids - 5 * N_BUCK
        table = _canon15()['tables'].reshape(-1)
    elif np.any(bids >= 4 * N_BUCK):
        rows = bids - 4 * N_BUCK
        table = _canon14()['tables'].reshape(-1)
    else:
        rows = bids - 2 * N_BUCK
        table = _canon8()['tables'].reshape(-1)
    idx = rows * 513 + resid
    freq = table[idx + 1] - table[idx]
    return -np.log2(np.maximum(freq, 1) / RT)


def _gauss_table_bits(qt, mu, lsig, offset, tables, thresholds):
    import torch
    resid = (torch.round(mu) - qt).neg().clamp(-RES_RANGE, RES_RANGE - 1)
    resid = (resid + RES_RANGE).to(torch.int64)
    lsig_q = torch.round(lsig * Q8_FSCALE)
    lsig_q = lsig_q + torch.round(offset * Q8_FSCALE).reshape(1, -1, 1, 1)
    bids = torch.bucketize(lsig_q.contiguous(), thresholds, right=True)
    tab = tables[bids]
    freq = (torch.gather(tab, -1, (resid + 1).unsqueeze(-1)).squeeze(-1)
            - torch.gather(tab, -1, resid.unsqueeze(-1)).squeeze(-1))
    return -torch.log2(freq.clamp_min(1).float() / RT)


def neuro_compress_v8(rgb, q=75, ckpt=None, flat=False, _oxn10=False,
                      _oxn10_ctx=None, _rate_refine=False, _custom_tables=False,
                      _channel_tables=False, _rate_passes=2, _scale_candidates=0,
                      _companding=0):
    """OXN8/OXN10: integer hyperprior + channel-group context."""
    import torch
    if _companding < 0 or (_companding and _oxn10):
        raise ValueError('companding is only supported by the Tiny OXN8 path')
    if _companding:
        _companding = float(np.float16(_companding))
    model, dev = _load(ckpt)
    if getattr(model, 'hyper_enc', None) is None:
        sys.exit('OXN8 требует hyper-чекпоинт (--hyper при обучении кодека)')
    p = model.patch
    H, W = rgb.shape[:2]
    canvas, Hp, Wp = _pad_image(rgb, p)
    ht, wt = Hp // p, Wp // p
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    if dev == 'cuda' and Hp * Wp > 16_000_000:
        x = x.to(torch.bfloat16)
    x = x.to(dev)
    with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
        z = model.encode(x)[0]
    del x
    if dev == 'cuda':
        torch.cuda.empty_cache()
    zf = z.float()
    T_, C = zf.shape
    zs = zf.cpu().numpy()
    quantized = np.arcsinh(zs / _companding) if _companding else zs
    if flat:
        scales_q = np.array([round(float(np.float32(q2s(q))) * 256)], np.int64)
        smap = np.zeros(T_, np.uint8)
        q_lat = np.round(quantized * (scales_q[0] / 256.0)).astype(np.int64)
    else:
        s0 = q2s(q)
        lam = TAU / (s0 * s0)
        scales_q = np.rint(np.array(LADDER) * 256).astype(np.int64)
        scales = scales_q.astype(np.float32) / 256.0
        s0_q = np.array([round(float(np.float32(s0)) * 256)], np.int64)
        q0 = np.round(quantized * (s0_q[0] / 256.0)).astype(np.int64)
        zqd0_q = _qdequant_latent(q0, np.full(T_, s0_q[0], np.int64), _companding)
        mu0r, logb0 = _hyper_rate_estimate(model, zqd0_q, ht, wt, dev)
        b0 = np.log1p(np.exp(logb0)) + 1e-2
        Dk, Rk, Qk, Ek = [], [], [], []
        use_boost = _BOOST_ON and _boost_net(dev) is not False
        mid = _boost_net(dev)[3] if use_boost else []
        need = []
        for k, s in enumerate(scales):
            sf = float(s)
            qk = np.round(quantized * sf)
            Qk.append(qk)
            Rk.append((np.abs(qk - mu0r) / b0).sum(axis=1))
            if use_boost:
                Ek.append(((zs * sf - qk) ** 2).mean(axis=1))
            if not use_boost or k in mid:
                need.append(k)
        if not use_boost and _scale_candidates:
            count = min(len(need), _scale_candidates)
            center = min(range(len(scales)), key=lambda k: abs(scales[k] - s0))
            first = max(0, min(len(scales) - count, center - count // 2))
            need = list(range(first, first + count))
        per_image = max(Hp * Wp * 400, 1)
        batch = int(np.clip(_mem_budget(dev) // per_image, 1, len(need)))
        if dev != 'cuda':
            batch = 1
        for first in range(0, len(need), batch):
            ids = need[first:first + batch]
            zb = np.stack([(_companding * np.sinh(Qk[k] / float(scales[k]))
                            if _companding else Qk[k] / float(scales[k])).astype(np.float32)
                           for k in ids])
            zt = torch.from_numpy(zb).to(dev)
            lg = torch.from_numpy(np.log(scales[ids]).astype(np.float32))[:, None]
            lg = lg.expand(len(ids), T_).to(dev)
            Dk.extend(_run_decode_patch_mse(model, zt, Hp, Wp, lg, canvas, dev))
        if use_boost:
            D = _boost_predict(model, canvas, zs, np.stack(Ek), np.stack(Rk),
                               np.stack(Dk), ht, wt, p, dev)
        else:
            D = np.stack(Dk)
        R = np.stack(Rk)
        if use_boost:
            choice = np.argmin(D + lam * R, axis=0).astype(np.uint8)
        else:
            local = np.argmin(D + lam * R[need], axis=0)
            choice = np.asarray(need, dtype=np.uint8)[local]
        smap = choice
        q_lat = np.zeros_like(Qk[0], dtype=np.int64)
        for k in range(len(scales_q)):
            m = choice == k
            q_lat[m] = Qk[k][m]
    logs_q = np.rint(np.log(scales_q.astype(np.float64) / 256.0)
                     * Q8_FSCALE).astype(np.int64)
    refine = _rate_refine and _oxn10 and not flat
    for rate_pass in range(_rate_passes if refine else 1):
        refining = refine and rate_pass < _rate_passes - 1
        if not flat:
            smap = choice
        q_lat = np.zeros_like(Qk[0], np.int64) if not flat else q_lat
        if not flat:
            for k in range(len(scales_q)):
                m = choice == k
                q_lat[m] = Qk[k][m]
        stok_q = scales_q[smap]
        zqd_q = _qdequant_latent(q_lat, stok_q, _companding)
        w, mu_q, logb_q = _qhyper_encode(model, zqd_q, ht, wt)
        logs_map_q = logs_q[smap]
        if _oxn10:
            ctx, groups = _ctx10_net(C, dev, _oxn10_ctx)
            bids_g = _ctx10_gauss_buckets(mu_q, logb_q, logs_map_q,
                                          ht, wt, dev, ctx, _custom_tables)
        else:
            ctx, groups = _qctx7_net(C)
        anch = _anchor_mask(ht, wt)
        phase_masks = (_grid4_masks(ht, wt)
                       if _oxn10 and ctx._grid4 else [anch, ~anch])
        mu_r = _qround_shift(mu_q, Q8_FBITS)
        bids_l = _q8_lap_buckets(logb_q)
        qd = np.zeros((T_, C), np.int64)
        streams = []
        Rnew = np.zeros((len(scales_q), T_), np.float64) if refining else None
        for gi, (lo, hi) in enumerate(_group_cuts(groups)):
            for pi, pos in enumerate(phase_masks):
                if gi == 0 and pi == 0:
                    mu_ref = mu_r
                    bids_ref = bids_l
                else:
                    zg_q = _qdequant_latent(qd, stok_q, _companding)
                    if _oxn10:
                        dmu = _ctx10_mu(ctx, zg_q, mu_q, logb_q,
                                        logs_map_q, ht, wt, dev)
                        mu_ref = np.rint(mu_q / Q8_FSCALE + dmu).astype(np.int64)
                        bids_ref = bids_g + (gi + 1) * N_BUCK \
                            if _channel_tables else bids_g
                    else:
                        dmu_q, lsig_q = _qctx7_params(
                            ctx, zg_q, mu_q, logb_q, logs_map_q, ht, wt)
                        mu_ref = _qround_shift(mu_q + dmu_q, Q8_FBITS)
                        bids_ref = _q8_gauss_buckets(lsig_q)
                resid = (np.clip(q_lat - mu_ref, -RES_RANGE, RES_RANGE - 1)
                         + RES_RANGE)
                if refining:
                    for k in range(len(scales_q)):
                        cost = _rans_table_cost(
                            Qk[k][pos][:, lo:hi], mu_ref[pos][:, lo:hi],
                            bids_ref[pos][:, lo:hi])
                        Rnew[k][pos] += cost.sum(axis=1)
                else:
                    _collect_rans_stats(resid[pos][:, lo:hi],
                                        bids_ref[pos][:, lo:hi])
                    streams.append(_enc_syms(resid[pos][:, lo:hi],
                                             bids_ref[pos][:, lo:hi]))
                qd[pos, lo:hi] = (resid[pos][:, lo:hi] - RES_RANGE
                                  + mu_ref[pos][:, lo:hi])
        if refining:
            choice = np.argmin(D + lam * Rnew, axis=0).astype(np.uint8)
    wblob = _encode_flat(w)
    HC = w.shape[1]
    magic = (b'OXNG' if _channel_tables else b'OXNE' if _custom_tables else
             b'OXND' if ctx._grid4 and len(groups) == 2 else
             b'OXNC' if ctx._grid4 else
             b'OXNB' if ctx._scale_net is not None else b'OXNA') \
        if _oxn10 else (b'OXNU' if _companding else b'OXN8')
    out = bytearray(magic)
    out += struct.pack('<HHBBBB', H, W, p, C, int(flat), len(scales_q))
    if _companding:
        out += struct.pack('<e', _companding)
    for sq, lq in zip(scales_q, logs_q):
        out += struct.pack('<Hh', int(sq), int(lq))
    out += struct.pack('<B', len(groups))
    out += bytes(groups)
    out += struct.pack('<H', HC)
    out += struct.pack('<I', len(wblob)) + wblob
    if not flat:
        mapblob = rans_encode(smap.tobytes())
        out += struct.pack('<I', len(mapblob)) + mapblob
    for blob in streams:
        out += struct.pack('<I', len(blob)) + blob
    return bytes(out)


def neuro_compress_v9(rgb, q=75, flat=False):
    global CTX7_FILE, _BOOST_ON
    CTX7_FILE = TINY_CTX7
    old_boost = _BOOST_ON
    _BOOST_ON = False
    try:
        blob = neuro_compress_v8(rgb, q, TINY_CKPT, flat)
    finally:
        _BOOST_ON = old_boost
    return b'OXN9' + blob[4:]


def neuro_compress_tiny_v2(rgb, q=75, flat=False, ckpt=None, ctx_file=None,
                           candidate_count=0, magic=b'OXNT', use_boost=False):
    global CTX7_FILE, _BOOST_ON
    if magic not in (b'OXNT', b'OXNV'):
        raise ValueError('Tiny stream magic must be OXNT or OXNV')
    ckpt = ckpt or TINY_CKPT
    ctx_file = ctx_file or TINY_CTX7
    CTX7_FILE = ctx_file
    old_boost = _BOOST_ON
    _BOOST_ON = use_boost
    try:
        blob = neuro_compress_v8(rgb, q, ckpt, flat,
                                 _scale_candidates=candidate_count)
    finally:
        _BOOST_ON = old_boost
    return magic + blob[4:]


def neuro_compress_tiny_companding(rgb, q=75, a=0.1, ckpt=None, ctx_file=None):
    global CTX7_FILE, _BOOST_ON
    if a <= 0:
        raise ValueError('companding parameter must be positive')
    ckpt = ckpt or TINY_CKPT
    ctx_file = ctx_file or TINY_CTX7
    CTX7_FILE = ctx_file
    old_boost = _BOOST_ON
    _BOOST_ON = False
    try:
        return neuro_compress_v8(rgb, q, ckpt, False, _companding=a)
    finally:
        _BOOST_ON = old_boost


def neuro_compress_tiny_v2_target_bpp(rgb, target_bpp, ckpt=None, ctx_file=None,
                                      passes=10, magic=b'OXNT', use_boost=False):
    if target_bpp <= 0:
        raise ValueError('target bpp must be positive')
    best = None
    lo, hi = 1.0, 100.0
    for _ in range(passes):
        q = (lo + hi) / 2
        blob = neuro_compress_tiny_v2(rgb, q, False, ckpt, ctx_file, magic=magic,
                                      use_boost=use_boost)
        bpp = len(blob) * 8 / (rgb.shape[0] * rgb.shape[1])
        error = abs(bpp - target_bpp)
        if best is None or error < best[0]:
            best = error, blob
        if bpp < target_bpp:
            lo = q
        else:
            hi = q
    return best[1]


def neuro_compress_v10(rgb, q=75, flat=False, ckpt=None):
    return neuro_compress_v8(rgb, q, ckpt, flat, _oxn10=True)


def neuro_compress_v11(rgb, q=75, flat=False, ckpt=None):
    return neuro_compress_v8(rgb, q, ckpt, flat, _oxn10=True,
                             _oxn10_ctx=OXN11_CTX)


def neuro_compress_v12(rgb, q=75, flat=False, ckpt=None):
    return neuro_compress_v8(rgb, q, ckpt, flat, _oxn10=True,
                             _oxn10_ctx=OXN12_CTX)


def neuro_compress_v13(rgb, q=75, flat=False, ckpt=None):
    return neuro_compress_v8(rgb, q, ckpt, flat, _oxn10=True,
                             _oxn10_ctx=OXN13_CTX, _rate_refine=q < 100)


def neuro_compress_v14(rgb, q=75, flat=False, ckpt=None):
    return neuro_compress_v8(rgb, q, ckpt, flat, _oxn10=True,
                             _oxn10_ctx=OXN13_CTX, _rate_refine=q < 100,
                             _custom_tables=True)


def neuro_compress_v15(rgb, q=75, flat=False, ckpt=None):
    return neuro_compress_v8(rgb, q, ckpt, flat, _oxn10=True,
                             _oxn10_ctx=OXN13_CTX, _rate_refine=q < 100,
                             _custom_tables=True, _channel_tables=True)


def neuro_compress_v16(rgb, q=75, flat=False, ckpt=None):
    candidates = (neuro_compress_v14(rgb, q, flat, ckpt),
                  neuro_compress_v15(rgb, q, flat, ckpt))
    return min(candidates, key=len)


def neuro_compress_v17_fast(rgb, q=75, flat=False, ckpt=None):
    blob = neuro_compress_v8(rgb, q, ckpt or REC_CKPT, flat, _oxn10=True,
                             _oxn10_ctx=OXN13_CTX, _rate_refine=q < 100,
                             _custom_tables=True, _rate_passes=8)
    return b'OXNH' + blob[4:]


def neuro_compress_v17(rgb, q=75, flat=False, ckpt=None):
    blob = neuro_compress_v16(rgb, q, flat, ckpt or REC_CKPT)
    if blob[:4] == b'OXNE':
        return b'OXNH' + blob[4:]
    if blob[:4] == b'OXNG':
        return b'OXNI' + blob[4:]
    raise RuntimeError('unexpected OXN16 stream')


def _neuro_decompress_v8(blob, ckpt):
    import torch
    H, W, p, C, flat, K = struct.unpack('<HHBBBB', blob[4:12])
    if flat not in (0, 1):
        raise RuntimeError('corrupt OXN8: invalid rate mode')
    pos = 12
    companding = 0.0
    if blob[:4] == b'OXNU':
        (companding,) = struct.unpack('<e', blob[pos:pos + 2])
        pos += 2
        if companding <= 0:
            raise RuntimeError('corrupt OXNU: invalid companding')
    scales_q = np.empty(K, np.int64)
    logs_q = np.empty(K, np.int64)
    for k in range(K):
        scales_q[k], logs_q[k] = struct.unpack('<Hh', blob[pos:pos + 4])
        pos += 4
    if K == 0 or np.any(scales_q <= 0):
        raise RuntimeError('corrupt OXN8: invalid scales')
    (G,) = struct.unpack('<B', blob[pos:pos + 1]); pos += 1
    groups = tuple(struct.unpack(f'<{G}B', blob[pos:pos + G])); pos += G
    (HC,) = struct.unpack('<H', blob[pos:pos + 2]); pos += 2
    model, dev = _load(ckpt)
    ctx, ck_groups = _qctx7_net(C)
    if ck_groups != groups:
        sys.exit(f'ox_ctx7.pt: groups {ck_groups}, в файле {groups}')
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    ht, wt = Hp // p, Wp // p
    T_ = ht * wt
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    (wlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    w = _decode_flat(blob[pos:pos + wlen], ht4 * wt4 * HC).reshape(ht4 * wt4, HC)
    pos += wlen
    if flat:
        if K != 1:
            raise RuntimeError('corrupt OXN8: flat mode requires one scale')
        smap = np.zeros(T_, np.uint8)
    else:
        (mlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
        smap = np.frombuffer(rans_decode(blob[pos:pos + mlen]), np.uint8)
        pos += mlen
        if smap.shape[0] != T_ or np.any(smap >= K):
            raise RuntimeError('corrupt OXN8: invalid scale map')
    stok_q = scales_q[smap]
    mu_q, logb_q = _qhyper_decode(model, w, ht, wt)
    anch = _anchor_mask(ht, wt)
    logs_map_q = logs_q[smap]
    mu_r = _qround_shift(mu_q, Q8_FBITS)
    bids_l = _q8_lap_buckets(logb_q)
    qd = np.zeros((T_, C), np.int64)
    for gi, (lo, hi) in enumerate(_group_cuts(groups)):
        for phase in (True, False):
            posm = anch if phase else ~anch
            (slen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
            if gi == 0 and phase:
                resid = _dec_syms(blob[pos:pos + slen],
                                  bids_l[posm][:, lo:hi])
                mu_ref = mu_r
            else:
                zg_q = _qdequant_latent(qd, stok_q, companding)
                dmu_q, lsig_q = _qctx7_params(
                    ctx, zg_q, mu_q, logb_q, logs_map_q, ht, wt)
                mu_nr = _qround_shift(mu_q + dmu_q, Q8_FBITS)
                bids = _q8_gauss_buckets(lsig_q)
                resid = _dec_syms(blob[pos:pos + slen],
                                  bids[posm][:, lo:hi])
                mu_ref = mu_nr
            pos += slen
            qd[posm, lo:hi] = (resid.reshape(int(posm.sum()), hi - lo)
                               - RES_RANGE + mu_ref[posm][:, lo:hi])
    if pos != len(blob):
        raise RuntimeError(f'corrupt OXN8: trailing bytes ({len(blob) - pos})')
    scales = scales_q.astype(np.float32) / 256.0
    stok = scales[smap]
    normalized = qd / stok[:, None]
    zq = companding * np.sinh(normalized) if companding else normalized
    z = torch.from_numpy(zq.astype(np.float32))[None].to(dev)
    logs = logs_q[smap].astype(np.float32) / Q8_FSCALE
    lg = torch.from_numpy(logs)[None].to(dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _neuro_decompress_v10(blob, ckpt):
    import torch
    H, W, p, C, flat, K = struct.unpack('<HHBBBB', blob[4:12])
    if flat not in (0, 1):
        raise RuntimeError('corrupt OXN10: invalid rate mode')
    pos = 12
    scales_q = np.empty(K, np.int64)
    logs_q = np.empty(K, np.int64)
    for k in range(K):
        scales_q[k], logs_q[k] = struct.unpack('<Hh', blob[pos:pos + 4])
        pos += 4
    if K == 0 or np.any(scales_q <= 0):
        raise RuntimeError('corrupt OXN10: invalid scales')
    (G,) = struct.unpack('<B', blob[pos:pos + 1]); pos += 1
    groups = tuple(struct.unpack(f'<{G}B', blob[pos:pos + G])); pos += G
    (HC,) = struct.unpack('<H', blob[pos:pos + 2]); pos += 2
    model, dev = _load(ckpt)
    ctx_path = (OXN13_CTX if blob[:4] in (b'OXND', b'OXNE', b'OXNG',
                                           b'OXNH', b'OXNI') else
                OXN12_CTX if blob[:4] == b'OXNC' else
                OXN11_CTX if blob[:4] == b'OXNB' else OXN10_CTX)
    ctx, ck_groups = _ctx10_net(C, dev, ctx_path)
    if ck_groups != groups:
        sys.exit(f'{ctx_path}: groups {ck_groups}, в файле {groups}')
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    ht, wt = Hp // p, Wp // p
    T_ = ht * wt
    ht4, wt4 = (ht + 3) // 4, (wt + 3) // 4
    (wlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    w = _decode_flat(blob[pos:pos + wlen], ht4 * wt4 * HC).reshape(ht4 * wt4, HC)
    pos += wlen
    if flat:
        if K != 1:
            raise RuntimeError('corrupt OXN10: flat mode requires one scale')
        smap = np.zeros(T_, np.uint8)
    else:
        (mlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
        smap = np.frombuffer(rans_decode(blob[pos:pos + mlen]), np.uint8)
        pos += mlen
        if smap.shape[0] != T_ or np.any(smap >= K):
            raise RuntimeError('corrupt OXN10: invalid scale map')
    stok_q = scales_q[smap]
    mu_q, logb_q = _qhyper_decode(model, w, ht, wt)
    anch = _anchor_mask(ht, wt)
    phase_masks = _grid4_masks(ht, wt) if ctx._grid4 else [anch, ~anch]
    logs_map_q = logs_q[smap]
    mu_r = _qround_shift(mu_q, Q8_FBITS)
    bids_l = _q8_lap_buckets(logb_q)
    bids_g = _ctx10_gauss_buckets(mu_q, logb_q, logs_map_q,
                                  ht, wt, dev, ctx,
                                  blob[:4] in (b'OXNE', b'OXNG', b'OXNH', b'OXNI'))
    residuals = []
    for gi, (lo, hi) in enumerate(_group_cuts(groups)):
        for pi, posm in enumerate(phase_masks):
            (slen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
            if gi == 0 and pi == 0:
                bids = bids_l
            elif blob[:4] in (b'OXNG', b'OXNI'):
                bids = bids_g + (gi + 1) * N_BUCK
            else:
                bids = bids_g
            residuals.append(_dec_syms(blob[pos:pos + slen],
                                       bids[posm][:, lo:hi]))
            pos += slen
    if pos != len(blob):
        raise RuntimeError(f'corrupt OXN10: trailing bytes ({len(blob) - pos})')
    qd = np.zeros((T_, C), np.int64)
    ri = 0
    for gi, (lo, hi) in enumerate(_group_cuts(groups)):
        for pi, posm in enumerate(phase_masks):
            resid = residuals[ri]
            ri += 1
            if gi == 0 and pi == 0:
                mu_ref = mu_r
            else:
                zg_q = _qdequant_latent(qd, stok_q)
                dmu = _ctx10_mu(ctx, zg_q, mu_q, logb_q,
                                logs_map_q, ht, wt, dev)
                mu_ref = np.rint(mu_q / Q8_FSCALE + dmu).astype(np.int64)
            qd[posm, lo:hi] = (resid.reshape(int(posm.sum()), hi - lo)
                               - RES_RANGE + mu_ref[posm][:, lo:hi])
    scales = scales_q.astype(np.float32) / 256.0
    stok = scales[smap]
    zq = qd / stok[:, None]
    z = torch.from_numpy(zq.astype(np.float32))[None].to(dev)
    logs = logs_q[smap].astype(np.float32) / Q8_FSCALE
    lg = torch.from_numpy(logs)[None].to(dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def cmd_trainctx(args):
    """Обучение контекстной ветки OXN5/OXN6: энкодер и hyperprior
    заморожены, минимизируется энтропия не-якорных токенов под дискретным
    Гауссом."""
    import torch
    model, dev = _load(args.ckpt)
    if getattr(model, 'hyper_enc', None) is None:
        sys.exit('trainctx требует hyper-чекпоинт (--hyper при обучении)')
    for prm in model.parameters():
        prm.requires_grad_(False)
    C = model.latent
    p = model.patch
    crop = args.crop
    g = crop // p                                # токенов по стороне кропа
    anch_np = _anchor_mask(g, g)
    anch = torch.from_numpy(anch_np).to(dev)
    nanch = torch.from_numpy(~anch_np).to(dev)
    if args.grid4:
        train_masks = [torch.from_numpy(m).to(dev) for m in _grid4_masks(g, g)]
    else:
        train_masks = [anch, nanch]
    groups = None
    if args.groups:
        groups = tuple(int(v) for v in args.groups.split(','))
        if sum(groups) != C:
            sys.exit(f'--groups {groups} не сходятся с latent={C}')
        ctx = _build_ctx7(C, args.ctx_hidden).to(dev)
        out_file = args.ctx_out or (OXN10_CTX if args.one_shot else args.ctx_file)
    else:
        ctx = _build_ctx(C, args.ctx_hidden).to(dev)
        out_file = args.ctx_out or CTX_FILE
    sigma_offset = None
    scale_net = None
    if args.one_shot:
        if not args.groups:
            sys.exit('--one_shot requires --groups')
        sigma_offset = torch.nn.Parameter(torch.zeros(C, device=dev))
    if args.scale_net:
        if not args.one_shot:
            sys.exit('--scale_net requires --one_shot')
        scale_net = _build_ctx10_scale(C, args.ctx_hidden).to(dev)
    if args.table_loss and scale_net is None:
        sys.exit('--table_loss requires --scale_net')
    if args.qat and scale_net is None:
        sys.exit('--qat requires --scale_net')
    if args.sigma_only:
        if sigma_offset is None:
            sys.exit('--sigma_only requires --one_shot')
        ctx.requires_grad_(False)
    if args.ctx_only:
        if scale_net is None or args.sigma_only:
            sys.exit('--ctx_only requires --scale_net and excludes --sigma_only')
        scale_net.requires_grad_(False)
        sigma_offset.requires_grad_(False)
    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location='cpu', weights_only=True)
        ctx.load_state_dict(ck['state'])
        if sigma_offset is not None and ck.get('sigma_offset') is not None:
            with torch.no_grad():
                sigma_offset.copy_(ck['sigma_offset'].to(dev))
        if scale_net is not None and ck.get('scale_state') is not None:
            scale_net.load_state_dict(ck['scale_state'])
        print(f'init ctx from {args.init}')
    n_params = sum(prm.numel() for prm in ctx.parameters())
    print(f'ctx parameters: {n_params / 1e3:.0f}K, device: {dev}')
    exts = ('.png', '.jpg', '.jpeg')
    files = [os.path.join(args.data, f) for f in sorted(os.listdir(args.data))
             if f.lower().endswith(exts)]
    # в памяти держим только кропы uint8: полные кадры DIV2K не влезают в RAM
    crops = []
    for f in files:
        a = load_rgb(f)
        H, W = a.shape[:2]
        if H < crop or W < crop:
            continue
        for _ in range(2):
            y = np.random.randint(0, H - crop + 1)
            x = np.random.randint(0, W - crop + 1)
            crops.append(torch.from_numpy(a[y:y + crop, x:x + crop].copy()))
    print(f'trainctx: {len(crops)} crops ({crop}px) from '
          f'{len(files)} images in {args.data}')
    if not 0 <= args.resume_step < args.steps:
        sys.exit('--resume_step must be >= 0 and less than --steps')
    params = [] if args.sigma_only else list(ctx.parameters())
    if scale_net is not None and not args.ctx_only:
        params += list(scale_net.parameters())
    if sigma_offset is not None and not args.ctx_only:
        params.append(sigma_offset)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    if args.resume_step:
        for group in opt.param_groups:
            group['initial_lr'] = args.lr
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr / 20,
        last_epoch=args.resume_step or -1)
    if args.table_loss:
        canon = _canon8()
        gauss_tables = torch.from_numpy(
            canon['tables'][N_BUCK:].astype(np.int64)).to(dev)
        gauss_thresholds = torch.from_numpy(
            canon['gauss_thresholds'].astype(np.int64)).to(dev)
    else:
        gauss_tables = gauss_thresholds = None
    sq2 = math.sqrt(2.0)
    msel = nanch.reshape(1, 1, g, g)
    t0 = time.time()
    for step in range(args.resume_step + 1, args.steps + 1):
        idx = np.random.randint(0, len(crops), size=args.batch)
        xs = []
        for i in idx:
            c = crops[i]
            if np.random.rand() < 0.5:
                c = torch.flip(c, dims=[1])
            xs.append(c)
        xb = (torch.stack(xs).to(dev, non_blocking=True)
              .permute(0, 3, 1, 2).float() / 255.0)
        # поле степеней: 40% равномерное, 60% кусочно-постоянное — как при
        # обучении кодека; так одна ctx-сеть обслуживает и OXN5, и OXN6
        T_ = g * g
        if np.random.rand() < 0.4:
            s0 = math.exp(np.random.uniform(math.log(S_MIN), math.log(S_MAX)))
            smap = torch.full((args.batch, T_), s0, device=dev)
        else:
            bs = int(np.random.choice([v for v in (2, 4, 8, 16) if v <= g]))
            nb = g // bs
            fields = np.exp(np.random.uniform(math.log(S_MIN), math.log(S_MAX),
                                              size=(args.batch, nb, nb)))
            field = np.repeat(np.repeat(fields, bs, 1), bs, 2)
            smap = torch.from_numpy(
                field.reshape(args.batch, T_).astype(np.float32)).to(dev)
        with torch.no_grad():
            z = model.encode(xb)                              # B x T x C
            q_lat = torch.round(z * smap.unsqueeze(-1))
            grid = (q_lat / smap.unsqueeze(-1)).transpose(1, 2).reshape(
                args.batch, C, g, g)
            w = torch.round(model.hyper_enc(grid))
            mu_h, logb_h = model.hyper_dec(w).chunk(2, 1)     # B x C x g x g
        zg = grid * anch.reshape(1, 1, g, g)                  # не-якоря = 0
        if groups is None:
            feats = torch.cat([zg, mu_h, logb_h,
                               torch.log(smap).reshape(args.batch, 1, g, g)], 1)
            dmu, lsig = ctx(feats).chunk(2, 1)
            mu = mu_h + dmu
            sig = torch.nn.functional.softplus(lsig) + 1e-2
            qg = q_lat.transpose(1, 2).reshape(args.batch, C, g, g)
            cdf = 0.5 * (1 + torch.erf(
                (torch.stack([qg + 0.5, qg - 0.5]) - mu) / (sig * sq2)))
            prob = (cdf[0] - cdf[1]).clamp_min(1e-9)
            bits = -torch.log2(prob)                          # B x C x g x g
            loss = (bits * msel).sum() / (msel.sum() * args.batch)
        else:
            # OXN7: проходы по группам и фазам, как при декодировании;
            # известное подставляем учительски (истинные q/s), неизвестное — 0
            qg = q_lat.transpose(1, 2).reshape(args.batch, C, g, g)
            logs_ch = torch.log(smap).reshape(args.batch, 1, g, g)
            lsig_one = None
            if scale_net is not None:
                if args.qat:
                    mu_s = _fake_quant(mu_h, Q8_FSCALE)
                    logb_s = _fake_quant(logb_h, Q8_FSCALE)
                    logs_s = _fake_quant(logs_ch, Q8_FSCALE)
                    lsig_one = _qat_scale_forward(
                        scale_net, torch.cat([mu_s, logb_s, logs_s], 1))
                else:
                    lsig_one = scale_net(torch.cat([mu_h, logb_h, logs_ch], 1))
            qd = torch.zeros_like(grid)
            loss = 0.0
            npass = 0
            for gi, (lo, hi) in enumerate(_group_cuts(groups)):
                for pi, mask in enumerate(train_masks):
                    pm = mask.reshape(1, 1, g, g)
                    if gi == 0 and pi == 0:
                        qd[:, lo:hi] = grid[:, lo:hi] * pm
                        continue
                    feats = torch.cat([qd, mu_h, logb_h, logs_ch], 1)
                    dmu, lsig = ctx(feats).chunk(2, 1)
                    mu = mu_h[:, lo:hi] + dmu[:, lo:hi]
                    if scale_net is not None:
                        sig0 = (torch.nn.functional.softplus(
                            lsig_one[:, lo:hi]) + 1e-2)
                        sig = sig0 * torch.exp(
                            sigma_offset[lo:hi].reshape(1, -1, 1, 1))
                    elif args.one_shot:
                        sig0 = (torch.nn.functional.softplus(logb_h[:, lo:hi])
                                + 1e-2)
                        sig = sig0 * torch.exp(
                            sigma_offset[lo:hi].reshape(1, -1, 1, 1))
                    else:
                        sig = torch.nn.functional.softplus(lsig[:, lo:hi]) + 1e-2
                    qt = qg[:, lo:hi]
                    cdf = 0.5 * (1 + torch.erf(
                        (torch.stack([qt + 0.5, qt - 0.5]) - mu) / (sig * sq2)))
                    prob = (cdf[0] - cdf[1]).clamp_min(1e-9)
                    bits = -torch.log2(prob)                  # B x gc x g x g
                    if args.table_loss:
                        table_bits = _gauss_table_bits(
                            qt, mu, lsig_one[:, lo:hi], sigma_offset[lo:hi],
                            gauss_tables, gauss_thresholds)
                        bits = bits + (table_bits - bits).detach()
                    pmf = pm.float()
                    loss = loss + (bits * pmf).sum() / (pmf.sum() * args.batch)
                    npass += 1
                    qd[:, lo:hi] = torch.where(pm, grid[:, lo:hi],
                                               qd[:, lo:hi])
            loss = loss / npass
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                b_h = torch.nn.functional.softplus(logb_h) + 1e-2
                bits_h = ((qg - mu_h).abs() / (b_h * math.log(2))
                          + torch.log2(2 * b_h))
                bh = (bits_h * msel).sum() / (msel.sum() * args.batch)
            print(f'  step {step:5d}/{args.steps}  '
                  f'bits/ctx-tok {loss.item():.3f}  hyper-only {bh.item():.3f} '
                  f'(s~{smap.mean().item():.2f})  ({time.time() - t0:.0f} с)')
        if step % 2000 == 0:
            torch.save({'state': ctx.state_dict(), 'latent': C,
                        'hidden': args.ctx_hidden, 'groups': groups,
                        'step': step, 'grid4': args.grid4,
                        'sigma_offset': None if sigma_offset is None else
                        sigma_offset.detach().cpu(),
                        'scale_state': None if scale_net is None else
                        scale_net.state_dict()}, out_file + '.part')
    torch.save({'state': ctx.state_dict(), 'latent': C,
                'hidden': args.ctx_hidden, 'groups': groups, 'step': args.steps,
                'grid4': args.grid4,
                'sigma_offset': None if sigma_offset is None else
                sigma_offset.detach().cpu(),
                'scale_state': None if scale_net is None else
                scale_net.state_dict()}, out_file)
    print(f'saved: {out_file}  ({time.time() - t0:.0f} s)')


def neuro_compress_adaptive(rgb, q=75, ckpt=None):

    """Адаптивная степень сжатия по патчам. rgb: uint8 HxWx3 -> bytes .oxn"""
    import torch
    model, dev = _load(ckpt)
    if getattr(model, 'hyper_enc', None) is not None:
        return neuro_compress_adaptive_hyper(rgb, q, ckpt)
    p = model.patch
    H, W = rgb.shape[:2]
    canvas, Hp, Wp = _pad_image(rgb, p)
    ht, wt = Hp // p, Wp // p
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    if dev == 'cuda' and Hp * Wp > 16_000_000:
        x = x.to(torch.bfloat16)          # на CPU: экономия и пиковой VRAM
    x = x.to(dev)
    s0 = q2s(q)
    lam = TAU / (s0 * s0)
    with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
        z = model.encode(x)[0]                              # T x C
    del x                                                   # вход больше не нужен
    if dev == 'cuda':
        torch.cuda.empty_cache()
    with torch.no_grad():
        b = torch.nn.functional.softplus(model.log_b) + 1e-3
        b = b.float().cpu().numpy()                         # C
        zs = z.float().cpu().numpy()                        # T x C
        Dk, Rk, Qk = [], [], []
        for s in LADDER:
            qk = np.round(zs * s)
            Qk.append(qk)
            Rk.append((np.abs(qk) / b).sum(axis=1))         # биты ~ сумма |q|/b (T)
            zt = torch.from_numpy((qk / s).astype(np.float32))[None].to(dev)
            lg = torch.full((1, zt.shape[1]), math.log(s), device=dev)
            xrn = _run_decode(model, zt, Hp, Wp, lg, dev)   # Hp x Wp x 3 numpy [0..1]
            d = ((xrn - canvas) ** 2).mean(2)               # numpy, без GPU-пиков
            dp = d.reshape(ht, p, wt, p).mean(axis=(1, 3)).reshape(-1) * 65025.0
            Dk.append(dp)                                   # искажение патча (T)
        D = np.stack(Dk)                                    # K x T
        R = np.stack(Rk)                                    # K x T
        choice = np.argmin(D + lam * R, axis=0).astype(np.uint8)   # T
        zq = np.zeros_like(Qk[0], dtype=np.int32)
        for k in range(len(LADDER)):
            m = choice == k
            zq[m] = Qk[k][m].astype(np.int32)
    T_, C = zq.shape
    mapblob = rans_encode(choice.tobytes())
    body = _encode_channels(zq)
    out = bytearray(b'OXN2')
    out += struct.pack('<HHBB', H, W, p, C)
    out += struct.pack('<B', len(LADDER))
    for s in LADDER:
        out += struct.pack('<e', s)
    out += struct.pack('<I', len(mapblob)) + mapblob
    out += struct.pack('<I', len(body)) + body
    return bytes(out)


def _neuro_decompress_v2(blob, ckpt):
    import torch
    H, W, p, C = struct.unpack('<HHBB', blob[4:10])
    (K,) = struct.unpack('<B', blob[10:11])
    scales = np.array(struct.unpack(f'<{K}e', blob[11:11 + 2 * K]), np.float32)
    pos = 11 + 2 * K
    model, dev = _load(ckpt)
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    T_ = (Hp // p) * (Wp // p)
    (mlen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    smap = np.frombuffer(rans_decode(blob[pos:pos + mlen]), np.uint8)
    pos += mlen
    assert smap.shape[0] == T_
    (blen,) = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
    zq = _decode_channels(blob[pos:pos + blen], C, T_)
    zq = zq / scales[smap][:, None]
    z = torch.from_numpy(zq.astype(np.float32))[None].to(dev)
    lg = torch.log(torch.from_numpy(scales[smap].astype(np.float32)))[None].to(dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def neuro_decompress(blob, ckpt=None, ctx_file=None):
    import torch
    global CTX7_FILE
    magic = blob[:4]
    if magic == b'OXN9':
        ckpt = ckpt or TINY_CKPT
        CTX7_FILE = TINY_CTX7
    elif magic in (b'OXNT', b'OXNU', b'OXNV'):
        ckpt = ckpt or TINY_CKPT
        CTX7_FILE = ctx_file or TINY_CTX7
    elif magic in (b'OXNH', b'OXNI'):
        ckpt = ckpt or REC_CKPT
    elif magic in (b'OXN7', b'OXN8'):
        CTX7_FILE = ctx_file or OXN8_CTX7
    # hyper-форматы требуют hyper-модель; если выбранная без неё — ищем подходящую
    if magic in (b'OXN3', b'OXN4', b'OXN5', b'OXN6', b'OXN7', b'OXN8',
                 b'OXN9', b'OXNT', b'OXNU', b'OXNV', b'OXNA', b'OXNB', b'OXNC', b'OXND', b'OXNE', b'OXNG',
                 b'OXNH', b'OXNI'):
        model, _ = _load(ckpt)
        if getattr(model, 'hyper_enc', None) is None:
            for alt in ('ox_neuro_v10.pt', 'ox_neuro_v9.pt'):
                p = os.path.join(HERE, alt)
                if os.path.exists(p):
                    ckpt = p
                    break
    if magic in (b'OXNA', b'OXNB', b'OXNC', b'OXND', b'OXNE', b'OXNG',
                 b'OXNH', b'OXNI'):
        return _neuro_decompress_v10(blob, ckpt)
    if magic in (b'OXN8', b'OXN9', b'OXNT', b'OXNU', b'OXNV'):
        return _neuro_decompress_v8(blob, ckpt)
    if magic == b'OXN7':
        return _neuro_decompress_v7(blob, ckpt)
    if magic == b'OXN6':
        return _neuro_decompress_v6(blob, ckpt)
    if magic == b'OXN5':
        return _neuro_decompress_v5(blob, ckpt)
    if magic == b'OXN4':
        return _neuro_decompress_v4(blob, ckpt)
    if magic == b'OXN3':
        return _neuro_decompress_v3(blob, ckpt)
    if blob[:4] == b'OXN2':
        return _neuro_decompress_v2(blob, ckpt)
    if blob[:4] != b'OXN1':
        sys.exit('not an OXN file')
    H, W, s, p, C = struct.unpack('<HHfBB', blob[4:14])
    model, dev = _load(ckpt)
    Hp = (H + p - 1) // p * p
    Wp = (W + p - 1) // p * p
    T_ = (Hp // p) * (Wp // p)
    (blen,) = struct.unpack('<I', blob[14:18])
    zq = _decode_channels(blob[18:18 + blen], C, T_)
    z = torch.from_numpy(zq / s)[None].to(dev)
    lg = torch.full((1, T_), math.log(s), device=dev)
    xr = _run_decode(model, z, Hp, Wp, lg, dev)
    return np.clip(xr[:H, :W] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else -10 * math.log10(mse / (255.0 * 255.0))


def cmd_bench(args):
    from PIL import Image
    rgb = load_rgb(args.args[0])
    n = rgb.shape[0] * rgb.shape[1]
    print(f'{args.args[0]}: {rgb.shape[1]}x{rgb.shape[0]}')
    print(f'{"codec":>16} {"bpp":>7} {"PSNR":>7}')
    for q in (5, 15, 30, 50, 75, 92):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, 'JPEG', quality=q)
        rec = np.asarray(Image.open(buf).convert('RGB'))
        print(f'{"JPEG q=" + str(q):>16} {len(buf.getvalue()) * 8 / n:7.3f} '
              f'{psnr(rgb, rec):7.2f}')
    for q in (5, 15, 30, 50, 75, 92):
        blob = neuro_compress(rgb, q, args.ckpt)
        rec = neuro_decompress(blob, args.ckpt, args.ctx_file)
        print(f'{"OXN q=" + str(q):>16} {len(blob) * 8 / n:7.3f} '
              f'{psnr(rgb, rec):7.2f}')
    for q in (5, 15, 30, 50, 75, 92):
        blob = neuro_compress_adaptive(rgb, q, args.ckpt)
        rec = neuro_decompress(blob, args.ckpt, args.ctx_file)
        print(f'{"OXN-адапт q=" + str(q):>16} {len(blob) * 8 / n:7.3f} '
              f'{psnr(rgb, rec):7.2f}')
    if os.path.exists(CTX_FILE):
        for q in (5, 15, 30, 50, 75, 92):
            blob = neuro_compress_v5(rgb, q, args.ckpt)
            rec = neuro_decompress(blob, args.ckpt, args.ctx_file)
            print(f'{"OXN5-ctx q=" + str(q):>16} {len(blob) * 8 / n:7.3f} '
                  f'{psnr(rgb, rec):7.2f}')
        for q in (5, 15, 30, 50, 75, 92):
            blob = neuro_compress_v6(rgb, q, args.ckpt)
            rec = neuro_decompress(blob, args.ckpt, args.ctx_file)
            print(f'{"OXN6-адапт q=" + str(q):>16} {len(blob) * 8 / n:7.3f} '
                  f'{psnr(rgb, rec):7.2f}')
    if os.path.exists(CTX7_FILE):
        for q in (5, 15, 30, 50, 75, 92):
            blob = neuro_compress_v7(rgb, q, args.ckpt)
            rec = neuro_decompress(blob, args.ckpt, args.ctx_file)
            print(f'{"OXN7-кан q=" + str(q):>16} {len(blob) * 8 / n:7.3f} '
                  f'{psnr(rgb, rec):7.2f}')
        for flat in (False, True):
            for q in (5, 15, 30, 50, 75, 92):
                blob = neuro_compress_v8(rgb, q, args.ckpt, flat)
                rec = neuro_decompress(blob, args.ckpt, args.ctx_file)
                mode = 'flat' if flat else 'адапт'
                print(f'{"OXN8-" + mode + " q=" + str(q):>16} '
                      f'{len(blob) * 8 / n:7.3f} {psnr(rgb, rec):7.2f}')


def main():
    global CTX7_FILE
    ap = argparse.ArgumentParser(description='OX-Neuro — learned image codec')
    ap.add_argument('cmd', choices=['train', 'trainctx', 'c', 'd', 'bench'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--resume_step', type=int, default=0,
                    help='trainctx: resume LR schedule and numbering at this step')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'kodak'))
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='filename substrings excluded from the training corpus')
    ap.add_argument('--q', type=int, default=75)
    ap.add_argument('--dim', type=int, default=DIM)
    ap.add_argument('--depth', type=int, default=DEPTH)
    ap.add_argument('--latent', type=int, default=LATENT)
    ap.add_argument('--heads', type=int, default=HEADS)
    ap.add_argument('--hyper_width', type=int, default=64)
    ap.add_argument('--hyper_c', type=int, default=32)
    ap.add_argument('--pos_tokens', type=int, default=4096)
    ap.add_argument('--shared', action='store_true',
                    help='single block looped depth times (weight tying)')
    ap.add_argument('--moe', type=int, default=0,
                    help='FFN experts per decoder block (0 = off)')
    ap.add_argument('--hyper', action='store_true',
                    help='hyperprior: network predicts the latent distribution')
    ap.add_argument('--window', type=int, default=0,
                    help='windowed attention, N x N tokens (0 = global)')
    ap.add_argument('--recurrent', type=int, default=0,
                    help='shared zero-init reconstruction iterations')
    ap.add_argument('--flow', type=int, default=0,
                    help='conditional rectified-flow refinement steps')
    ap.add_argument('--w_flow', type=float, default=0.0,
                    help='flow-matching loss weight')
    ap.add_argument('--refine', action='store_true',
                    help='conv refinement head against patch seams')
    ap.add_argument('--se', action='store_true',
                    help='SE-гейты на латенте (attention-маски NLAM)')
    ap.add_argument('--ctx7', action='store_true',
                    help='train: совместный ретрен с OXN7-энтропией в лоссе '
                         '(нужен обученный ox_ctx7.pt)')
    ap.add_argument('--ctx_file', default=CTX7_FILE,
                    help='файл OXN7-контекста (train/trainctx/c/d/bench)')
    ap.add_argument('--w_hp', type=float, default=0.0,
                    help='high-pass loss weight (Sobel, sharpness)')
    ap.add_argument('--w_tal', type=float, default=0.0,
                    help='multi-resolution tiling artifact L1 weight')
    ap.add_argument('--w_color', type=float, default=0.0,
                    help='per-patch low-frequency color balance weight')
    ap.add_argument('--w_seam', type=float, default=0.0,
                    help='patch-boundary continuity loss weight')
    ap.add_argument('--w_text', type=float, default=0.0,
                    help='gradient-saliency text/detail fidelity weight')
    ap.add_argument('--init', default=None, help='checkpoint to fine-tune from')
    ap.add_argument('--w_mse', type=float, default=1000.0)
    ap.add_argument('--w_msssim', type=float, default=0.0,
                    help='MS-SSIM perceptual loss weight (0 = off)')
    ap.add_argument('--w_lpips', type=float, default=0.0,
                    help='LPIPS perceptual loss weight (0 = off)')
    ap.add_argument('--freeze_enc', action='store_true',
                    help='fine-tune decoder only (encoder/quantizer frozen)')
    ap.add_argument('--out', default=CKPT, help='where to save the checkpoint (train)')
    ap.add_argument('--ckpt', default=None, help='checkpoint for c/d/bench')
    ap.add_argument('--flat', action='store_true',
                    help='плоский OXN8 вместо адаптивного OXN8 (по умолчанию)')
    ap.add_argument('--boost', action='store_true',
                    help='OXN8: степень по патчам выбирает маленькая '
                         'conv-сеть (ox_boost.pt) вместо 7 прогонов декодера')
    ap.add_argument('--tiny', action='store_true',
                    help='legacy OXN9: ox_neuro_tiny.pt + ox_ctx9_tiny.pt')
    ap.add_argument('--oxn10', action='store_true',
                    help='experimental OXN10: one-shot scale + GPU context')
    ap.add_argument('--oxn11', action='store_true',
                    help='experimental OXN11: spatial one-shot sigma + GPU context')
    ap.add_argument('--oxn12', action='store_true',
                    help='experimental OXN12: 2x2 grid context + spatial sigma')
    ap.add_argument('--oxn13', action='store_true',
                    help='experimental OXN13: 2x2 grid with two channel groups')
    ap.add_argument('--oxn14', action='store_true',
                    help='experimental OXN14: OXN13 with empirical rANS tables')
    ap.add_argument('--oxn15', action='store_true',
                    help='experimental OXN15: channel-group empirical rANS tables')
    ap.add_argument('--oxn16', action='store_true',
                    help='experimental OXN16: smallest of OXN14/OXN15 per image')
    ap.add_argument('--oxn17', action='store_true',
                    help='experimental OXN17: OXN16 with recurrent decoder')
    ap.add_argument('--oxn17_fast', action='store_true',
                    help='OXN17 with a single OXN14 encode instead of best-of')
    ap.add_argument('--one_shot', action='store_true',
                    help='trainctx: train OXN10 mu with fixed hyperprior sigma')
    ap.add_argument('--sigma_only', action='store_true',
                    help='trainctx: optimize only OXN10 one-shot sigma')
    ap.add_argument('--scale_net', action='store_true',
                    help='trainctx: add spatial one-shot log-sigma network')
    ap.add_argument('--table_loss', action='store_true',
                    help='trainctx: straight-through loss with exact rANS table cost')
    ap.add_argument('--qat', action='store_true',
                    help='trainctx: Q20.12 quantization-aware scale training')
    ap.add_argument('--grid4', action='store_true',
                    help='trainctx: use four 2x2 spatial phases')
    ap.add_argument('--ctx_only', action='store_true',
                    help='trainctx: freeze one-shot sigma and train only context')
    ap.add_argument('--ctx_out', default=None,
                    help='trainctx: output checkpoint path')
    ap.add_argument('--verify', action='store_true',
                    help='после сжатия декодировать энтропийный поток обратно '
                         'и сверить (медленно на больших изображениях)')
    ap.add_argument('--groups', default='',
                    help='trainctx: канальные группы для OXN7, напр. 6,6,6,6')
    ap.add_argument('--ctx_hidden', type=int, default=CTX_HIDDEN,
                    help='ширина контекстной сети (trainctx)')
    a = ap.parse_args()
    CTX7_FILE = a.ctx_file
    if a.verify:
        global _VERIFY
        _VERIFY = True
    if a.boost:
        global _BOOST_ON
        _BOOST_ON = True

    def with_cpu_fallback(fn, *fargs):
        """При нехватке VRAM — повтор на CPU (в RAM)."""
        global _FORCE_CPU
        try:
            return fn(*fargs)
        except Exception as e:
            if 'CUDA' in str(e) and 'memory' in str(e) and not _FORCE_CPU:
                print('CUDA out of memory — retrying on CPU '
                      '(slower, but will fit)', file=sys.stderr)
                _FORCE_CPU = True
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return fn(*fargs)
            raise

    if a.cmd == 'train':
        cmd_train(a)
    elif a.cmd == 'trainctx':
        cmd_trainctx(a)
    elif a.cmd == 'c':
        from PIL import Image
        rgb = load_rgb(a.args[0])
        t0 = time.time()
        if sum((a.tiny, a.oxn10, a.oxn11, a.oxn12, a.oxn13, a.oxn14,
                a.oxn15, a.oxn16, a.oxn17, a.oxn17_fast)) > 1:
            ap.error('experimental model flags are mutually exclusive')
        if a.tiny:
            blob = with_cpu_fallback(neuro_compress_tiny_v2, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn17_fast:
            blob = with_cpu_fallback(neuro_compress_v17_fast, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn17:
            blob = with_cpu_fallback(neuro_compress_v17, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn16:
            blob = with_cpu_fallback(neuro_compress_v16, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn15:
            blob = with_cpu_fallback(neuro_compress_v15, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn14:
            blob = with_cpu_fallback(neuro_compress_v14, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn13:
            blob = with_cpu_fallback(neuro_compress_v13, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn12:
            blob = with_cpu_fallback(neuro_compress_v12, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn11:
            blob = with_cpu_fallback(neuro_compress_v11, rgb, a.q, a.flat, a.ckpt)
        elif a.oxn10:
            blob = with_cpu_fallback(neuro_compress_v10, rgb, a.q, a.flat, a.ckpt)
        else:
            blob = with_cpu_fallback(neuro_compress_v8, rgb, a.q, a.ckpt, a.flat)
        open(a.args[1], 'wb').write(blob)
        n = rgb.shape[0] * rgb.shape[1]
        print(f'{a.args[0]} -> {a.args[1]}: {len(blob)} байт '
              f'({len(blob) * 8 / n:.3f} bpp, s={q2s(a.q):.2f}) in {time.time() - t0:.2f} s')
    elif a.cmd == 'd':
        blob_data = open(a.args[0], 'rb').read()
        rec = with_cpu_fallback(neuro_decompress, blob_data, a.ckpt, a.ctx_file)
        from PIL import Image
        if a.args[1].lower().endswith(('.jpg', '.jpeg')):
            Image.fromarray(rec).save(a.args[1], quality=95, subsampling=0)
        else:
            Image.fromarray(rec).save(a.args[1])
        print(f'{a.args[0]} -> {a.args[1]}')
    elif a.cmd == 'bench':
        cmd_bench(a)


if __name__ == '__main__':
    main()
