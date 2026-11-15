#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OX-Neuro — learned image codec (lossy): a transformer autoencoder trained
to reconstruct itself; the compression ratio is set by a latent SCALER s:
z -> round(z*s)/s. One model covers the whole quality range.

Entropy coding: static rANS from ox.py per latent channel; with --hyper a
hypernetwork predicts a per-token Laplace (mu, b) and rANS codes only the
residual (parametric rANS, optionally via the C accelerator `ox prans`).

Usage:
  python3 ox_neuro.py train [--steps 40000] [--data data/train]
  python3 ox_neuro.py c in.png out.oxh [--q 100] [--adaptive]
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
PATCH = 16
DIM = 256
DEPTH = 4
LATENT = 16
HEADS = 8
S_MIN, S_MAX = 0.35, 8.0          # диапазон скейлера (степени сжатия)

sys.path.insert(0, HERE)
from ox import rans_encode, rans_decode      # энтропийный движок OX


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
                shared=False, moe=0, refine=False, hyper=False, window=0):
    import torch
    import torch.nn as nn

    HYPER_C = 32

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
                nn.Conv2d(latent, 64, 3, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, HYPER_C, 3, stride=2, padding=1))

        def forward(self, g):
            return self.net(g)

    class HyperDec(nn.Module):
        """Гипер-латент -> (mu, log b) распределения Лапласа на токен×канал."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(HYPER_C, 64, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, latent * 2, 3, padding=1))

        def forward(self, w):
            return self.net(w)

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
            self.pos = nn.Parameter(torch.zeros(1, 4096, dim))
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
            self.refine = RefineHead() if refine else None
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
            return _linc(self.to_lat, h)

        def decode(self, z, H, W, logs=None):
            p = self.patch
            ht, wt = H // p, W // p
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
            return xr, bpp

    return OxFormer()


# ------------------------------------------------------------ обучение -----
def cmd_train(args):
    import torch
    from PIL import Image

    crop_min = args.crop
    exts = ('.png', '.jpg', '.jpeg')
    files = sorted(f for f in os.listdir(args.data)
                   if f.lower().endswith(exts))
    files = [os.path.join(args.data, f) for f in files]
    imgs = []
    for f in files:
        a = load_rgb(f).astype(np.float32) / 255.0
        if a.shape[0] >= crop_min and a.shape[1] >= crop_min:
            imgs.append(torch.from_numpy(a).permute(2, 0, 1))
    print(f'train: {len(imgs)} images from {args.data}')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(dim=args.dim, depth=args.depth, latent=args.latent,
                        shared=args.shared, moe=args.moe,
                        refine=args.refine, hyper=args.hyper,
                        window=args.window).to(dev)
    if args.init:
        ck = torch.load(args.init, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'], strict=False)
        print(f'init from {args.init}')
    if args.freeze_enc:
        n_frozen = 0
        for name, prm in model.named_parameters():
            if not (name.startswith(('dec', 'to_pix', 'from_lat', 'scale_emb'))):
                prm.requires_grad_(False)
                n_frozen += prm.numel()
        print(f'frozen encoder parameters: {n_frozen / 1e6:.2f}M')
    gk = _gauss_kernel().to(dev)
    sk = _sobel_kernel().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'parameters: {n_params / 1e6:.2f}M, device: {dev}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr / 20)
    crop = args.crop
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # случайный батч кропов
        xs = []
        for _ in range(args.batch):
            im = imgs[np.random.randint(len(imgs))]
            _, H, W = im.shape
            y = np.random.randint(0, H - crop + 1)
            x = np.random.randint(0, W - crop + 1)
            c = im[:, y:y + crop, x:x + crop]
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
            bs = int(np.random.choice([2, 4, 8, 16]))
            nb = g // bs
            fields = np.exp(np.random.uniform(math.log(S_MIN), math.log(S_MAX),
                                              size=(args.batch, nb, nb)))
            field = np.repeat(np.repeat(fields, bs, 1), bs, 2)
            smap = torch.from_numpy(
                field.reshape(args.batch, T_).astype(np.float32)).to(dev)
        lam = 25.0 / (smap * smap)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            xr, bpp = model(xb, smap)
            mse = torch.mean((xr - xb) ** 2)
            loss = args.w_mse * mse + (lam * bpp).mean()
            if args.w_msssim > 0:
                # MS-SSIM считаем строго в fp32: в bf16 дисперсии/ковариации
                # теряют точность и градиент лосса становится разрушительным
                with torch.autocast('cuda', enabled=False):
                    loss = loss + args.w_msssim * (
                        1 - ms_ssim(xr.float().clamp(0, 1), xb.float(), gk))
            if args.w_hp > 0:
                loss = loss + args.w_hp * highpass_mse(
                    xr.float().clamp(0, 1), xb.float(), sk)
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
            torch.save({'model': model.state_dict(),
                        'arch': {'patch': PATCH, 'dim': args.dim,
                                 'depth': args.depth, 'latent': args.latent,
                                 'heads': HEADS, 'shared': args.shared,
                                 'moe': args.moe, 'refine': args.refine,
                                 'hyper': args.hyper, 'window': args.window}},
                        args.out + '.part')
    torch.save({'model': model.state_dict(),
                'arch': {'patch': PATCH, 'dim': args.dim, 'depth': args.depth,
                         'latent': args.latent, 'heads': HEADS,
                         'shared': args.shared, 'moe': args.moe,
                         'refine': args.refine, 'hyper': args.hyper,
                         'window': args.window}}, args.out)
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


_FORCE_CPU = False     # после CUDA OOM переключаемся на CPU насовсем
_VERIFY = False        # самопроверка энтропийного потока после сжатия (--verify)

def _load(path=None):
    import torch
    path = path or CKPT
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
    model = build_model(**arch)
    model.load_state_dict(ck['model'], strict=False)  # старые чекпоинты без scale_emb
    model.__dict__['_ckpt_path'] = path
    dev = 'cpu' if _FORCE_CPU or not torch.cuda.is_available() else 'cuda'
    return model.eval().to(dev), dev


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
RT, RSCALE, RL = 4096, 12, 1 << 23

# Ускорение: параметрический rANS через C-бинарник ox (с fallback на Python)
OX_BIN = os.path.join(HERE, 'ox')


def _prans_tables_file(td):
    """Все бакет-таблицы в бинарник: [u32 K][K x 513 x u32 cum LE]."""
    tabs = np.stack([_lap_table(i) for i in range(N_BUCK)])
    path = os.path.join(td, 'tables.bin')
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', N_BUCK))
        f.write(tabs.astype('<u4').tobytes())
    return path


def rans_enc_param_fast(symbols, bids):
    """C-версия rans_enc_param через subprocess; None при неудаче."""
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
    """C-версия rans_dec_param через subprocess; None при неудаче.
    Возвращает (symbols, consumed): consumed == len(blob), если бакеты
    совпали с энкодером; иначе поток десинхронизирован."""
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

def rans_enc_param(symbols, bids):
    fast = rans_enc_param_fast(symbols, bids)
    if fast is not None:
        return fast
    x = RL
    out = bytearray()
    for i in range(len(symbols) - 1, -1, -1):
        cum = _lap_table(int(bids[i]))
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
        cum = _lap_table(int(bids[i]))
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
        with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
            xr = model.decode(z, Hp, Wp, lg)
        return xr[0].permute(1, 2, 0).float().cpu().numpy()
    ov = 16
    canvas = np.zeros((Hp, Wp, 3), np.float32)
    for y0 in range(0, ht, band):
        y1 = min(y0 + band, ht)
        a, b = max(y0 - ov, 0), min(y1 + ov, ht)
        zb = z[:, a * wt:b * wt]
        lb = lg[:, a * wt:b * wt] if lg is not None else None
        with torch.no_grad(), _maybe_autocast(dev, Hp * Wp):
            xb = model.decode(zb, (b - a) * p, Wp, lb)
        canvas[y0 * p:y1 * p] = \
            xb[0].permute(1, 2, 0).float().cpu().numpy()[(y0 - a) * p:(y1 - a) * p]
    return canvas


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


def neuro_decompress(blob, ckpt=None):
    import torch
    magic = blob[:4]
    # hyper-форматы требуют hyper-модель; если выбранная без неё — ищем подходящую
    if magic in (b'OXN3', b'OXN4'):
        model, _ = _load(ckpt)
        if getattr(model, 'hyper_enc', None) is None:
            for alt in ('ox_neuro_v10.pt', 'ox_neuro_v9.pt'):
                p = os.path.join(HERE, alt)
                if os.path.exists(p):
                    ckpt = p
                    break
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
        rec = neuro_decompress(blob, args.ckpt)
        print(f'{"OXN q=" + str(q):>16} {len(blob) * 8 / n:7.3f} '
              f'{psnr(rgb, rec):7.2f}')
    for q in (5, 15, 30, 50, 75, 92):
        blob = neuro_compress_adaptive(rgb, q, args.ckpt)
        rec = neuro_decompress(blob, args.ckpt)
        print(f'{"OXN-адапт q=" + str(q):>16} {len(blob) * 8 / n:7.3f} '
              f'{psnr(rgb, rec):7.2f}')


def main():
    ap = argparse.ArgumentParser(description='OX-Neuro — learned image codec')
    ap.add_argument('cmd', choices=['train', 'c', 'd', 'bench'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'kodak'))
    ap.add_argument('--q', type=int, default=75)
    ap.add_argument('--dim', type=int, default=DIM)
    ap.add_argument('--depth', type=int, default=DEPTH)
    ap.add_argument('--latent', type=int, default=LATENT)
    ap.add_argument('--shared', action='store_true',
                    help='single block looped depth times (weight tying)')
    ap.add_argument('--moe', type=int, default=0,
                    help='FFN experts per decoder block (0 = off)')
    ap.add_argument('--hyper', action='store_true',
                    help='hyperprior: network predicts the latent distribution')
    ap.add_argument('--window', type=int, default=0,
                    help='windowed attention, N x N tokens (0 = global)')
    ap.add_argument('--refine', action='store_true',
                    help='conv refinement head against patch seams')
    ap.add_argument('--w_hp', type=float, default=0.0,
                    help='high-pass loss weight (Sobel, sharpness)')
    ap.add_argument('--init', default=None, help='checkpoint to fine-tune from')
    ap.add_argument('--w_mse', type=float, default=1000.0)
    ap.add_argument('--w_msssim', type=float, default=0.0,
                    help='MS-SSIM perceptual loss weight (0 = off)')
    ap.add_argument('--freeze_enc', action='store_true',
                    help='fine-tune decoder only (encoder/quantizer frozen)')
    ap.add_argument('--out', default=CKPT, help='where to save the checkpoint (train)')
    ap.add_argument('--ckpt', default=None, help='checkpoint for c/d/bench')
    ap.add_argument('--adaptive', action='store_true',
                    help='per-patch adaptive rate (OXN2/OXN4)')
    ap.add_argument('--verify', action='store_true',
                    help='после сжатия декодировать энтропийный поток обратно '
                         'и сверить (медленно на больших изображениях)')
    ap.add_argument('--boost', action='store_true',
                    help='adaptive: степень по патчам выбирает маленькая '
                         'conv-сеть (ox_boost.pt) вместо 7 прогонов декодера')
    a = ap.parse_args()
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
    elif a.cmd == 'c':
        from PIL import Image
        rgb = load_rgb(a.args[0])
        t0 = time.time()
        if a.adaptive:
            blob = with_cpu_fallback(neuro_compress_adaptive, rgb, a.q, a.ckpt)
        else:
            blob = with_cpu_fallback(neuro_compress, rgb, a.q, a.ckpt)
        open(a.args[1], 'wb').write(blob)
        n = rgb.shape[0] * rgb.shape[1]
        print(f'{a.args[0]} -> {a.args[1]}: {len(blob)} байт '
              f'({len(blob) * 8 / n:.3f} bpp, s={q2s(a.q):.2f}) in {time.time() - t0:.2f} s')
    elif a.cmd == 'd':
        blob_data = open(a.args[0], 'rb').read()
        rec = with_cpu_fallback(neuro_decompress, blob_data, a.ckpt)
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
