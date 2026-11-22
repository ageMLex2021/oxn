#!/usr/bin/env python3
"""PICO architecture reimplementation from the CVPR 2026 paper."""

import argparse
import concurrent.futures
import copy
import math
import os
import platform
import struct
import sys
import threading
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ox_data
import ox_neuro as on
from ox_neuro import load_rgb, q2s

CKPT = os.path.join(HERE, 'oxn_pico_v8_moe6_final.pt')


def _stream_bids(sig, empirical):
    if empirical:
        thresholds = on._canon8()['gauss_thresholds']
        lq = np.rint(np.log(sig) * on.Q8_FSCALE).astype(np.int64)
        return 4 * on.N_BUCK + np.searchsorted(thresholds, lq, side='right')
    return on.N_BUCK + on._bucket_idx(np.log(sig))


def _stream_encode(q, sig, empirical=True):
    residual = np.clip(q, -256, 255).astype(np.int64) + 256
    bids = np.broadcast_to(_stream_bids(sig, empirical), q.shape)
    return on.rans_enc_param(residual.reshape(-1).astype(np.int32),
                             bids.reshape(-1).astype(np.int32))


def _stream_decode(blob, sig, shape, empirical=True):
    bids = np.broadcast_to(_stream_bids(sig, empirical), shape)
    residual, used = on.rans_dec_param(blob, bids.reshape(-1).astype(np.int32))
    if used != len(blob):
        raise RuntimeError(f'PICO entropy stream desync: consumed {used} of {len(blob)} bytes')
    return residual.reshape(shape).astype(np.int64) - 256


def _sparse_stream_encode(q, sig):
    mask = q != 0
    packed = np.packbits(mask.reshape(-1)).tobytes()
    compressed = zlib.compress(packed, 9)
    nonzero = q[mask]
    if nonzero.size:
        residual = _stream_encode(nonzero, np.broadcast_to(sig, q.shape)[mask])
    else:
        residual = b''
    sparse = struct.pack('<II', len(compressed), len(residual)) + compressed + residual
    if not nonzero.size and q.size >= 1024:
        return b'\x01' + sparse
    plain = _stream_encode(q, sig)
    return (b'\x01' + sparse) if len(sparse) < len(plain) else (b'\x00' + plain)


def _encode_verified(q, sig, sparse, verify):
    blob = _sparse_stream_encode(q, sig) if sparse else _stream_encode(q, sig)
    if verify:
        back = _sparse_stream_decode(blob, sig, q.shape) if sparse else \
            _stream_decode(blob, sig, q.shape)
        if not np.array_equal(back, q):
            raise RuntimeError('PICO entropy verification failed')
    return blob


def _sparse_stream_decode(blob, sig, shape):
    method = blob[0]
    if method == 0:
        return _stream_decode(blob[1:], sig, shape)
    if method != 1:
        raise RuntimeError('invalid PICO sparse stream')
    mask_n, residual_n = struct.unpack('<II', blob[1:9])
    mask_end = 9 + mask_n
    compressed = blob[9:mask_end]
    residual_blob = blob[mask_end:mask_end + residual_n]
    if mask_end + residual_n != len(blob):
        raise RuntimeError('trailing PICO sparse bytes')
    count = int(np.prod(shape))
    mask = np.unpackbits(np.frombuffer(zlib.decompress(compressed), np.uint8),
                         count=count).reshape(shape).astype(bool)
    nonzero = mask.sum()
    if nonzero:
        residual = _stream_decode(residual_blob, np.broadcast_to(sig, shape)[mask],
                                  (int(nonzero),))
    else:
        residual = np.empty(0, np.int64)
        if residual_blob:
            raise RuntimeError('invalid empty PICO sparse stream')
    out = np.zeros(shape, np.int64)
    out[mask] = residual
    return out


def _empirical_bits(q, sigma, tables, thresholds):
    sigma = sigma.expand_as(q)
    residual = (q.clamp(-256, 255) + 256).to(torch.int64)
    lq = torch.round(torch.log(sigma) * on.Q8_FSCALE)
    bids = torch.bucketize(lq.contiguous(), thresholds, right=True)
    table = tables[bids]
    freq = (torch.gather(table, -1, (residual + 1).unsqueeze(-1)).squeeze(-1) -
            torch.gather(table, -1, residual.unsqueeze(-1)).squeeze(-1))
    return -torch.log2(freq.clamp_min(1).float() / on.RT)


def _rate_bits(q, sigma, tables, thresholds):
    from oxn_mr import _gauss_bits, _table_sigma
    sigma = sigma.expand_as(q)
    smooth = _gauss_bits(q, torch.zeros_like(q), _table_sigma(sigma))
    exact = _empirical_bits(q, sigma, tables, thresholds)
    return smooth + (exact - smooth).detach()


def _sparse_rate_bits(q, sigma, tables, thresholds, spatial_mask):
    nonzero = (q != 0).to(q.dtype) * spatial_mask
    residual = (_rate_bits(q, sigma, tables, thresholds) * nonzero).sum()
    count = spatial_mask.sum()
    density = (nonzero.sum() / count).clamp(1e-6, 1 - 1e-6)
    mask = -count * (density * torch.log2(density) +
                     (1 - density) * torch.log2(1 - density))
    return residual + mask + 72


def level_vector(level, device=None, dtype=None):
    pos = float(level) * 7 / 100
    lo = int(math.floor(pos))
    hi = min(7, lo + 1)
    frac = pos - lo
    v = np.zeros(8, np.float32)
    v[lo] = 1 - frac
    v[hi] += frac
    return torch.from_numpy(v).to(device=device, dtype=dtype or torch.float32)


def _rate_expert(bpp, moe):
    if moe != 6:
        return -1
    return int(np.searchsorted((0.075, 0.15, 0.3, 0.75, 1.5), bpp, side='right'))


def _groups(cin, cout):
    return max(1, min(cin, cout) // 32)


class ConvScale(nn.Conv2d):
    def __init__(self, cin, cout, kernel, stride=1, padding=None, groups=None):
        if padding is None:
            padding = kernel // 2
        groups = groups or (_groups(cin, cout) if kernel == 3 else 1)
        super().__init__(cin, cout, kernel, stride, padding, groups=groups)
        self.sin = nn.Parameter(torch.ones(1, cin // groups, 1, 1))
        self.sout = nn.Parameter(torch.ones(cout, 1, 1, 1))
        self._fused = False

    @torch.no_grad()
    def reparameterize(self):
        weight = self.weight * self.sin * self.sout
        self.weight = nn.Parameter(weight, requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias * self.sout[:, 0, 0, 0],
                                     requires_grad=False)
        self._fused = True

    def forward(self, x):
        if self._fused:
            return self._conv_forward(x, self.weight, self.bias)
        w = self.weight * self.sin * self.sout
        b = self.bias * self.sout[:, 0, 0, 0] if self.bias is not None else None
        return self._conv_forward(x, w, b)


class CS311(nn.Module):
    def __init__(self, channels, expansion, expansion2):
        super().__init__()
        self.net = nn.Sequential(
            ConvScale(channels, channels * expansion, 3), nn.GELU(),
            ConvScale(channels * expansion, channels * expansion2, 1), nn.GELU(),
            ConvScale(channels * expansion2, channels, 1))

    def forward(self, x):
        return x + self.net(x)


def chain(channels, repeats, expansion, expansion2):
    return nn.Sequential(*(CS311(channels, expansion, expansion2)
                           for _ in range(repeats)))


class HaarDown(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin * 4, cout, 1)

    def forward(self, x):
        a, b = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
        c, d = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
        h = torch.cat((a + b + c + d, -a - b + c + d,
                       -a + b - c + d, a - b - c + d), 1) * 0.5
        return self.conv(h)


class HaarUp(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cout = cout
        self.conv = nn.Conv2d(cin, cout * 4, 1)

    def forward(self, x):
        ll, lh, hl, hh = self.conv(x).chunk(4, 1)
        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5
        out = torch.empty(x.shape[0], self.cout, x.shape[2] * 2, x.shape[3] * 2,
                          device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = a
        out[:, :, 0::2, 1::2] = b
        out[:, :, 1::2, 0::2] = c
        out[:, :, 1::2, 1::2] = d
        return out


class LearnedScale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x):
        return x * self.scale


class ScaleDecoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.start = ConvScale(192, 128, 1)
        self.c1 = chain(128, 2, 4, 1)
        self.u1 = HaarUp(128, 128)
        self.c2 = chain(128, 2, 4, 1)
        self.u2 = HaarUp(128, 128)
        self.c3 = chain(128, 2, 4, 1)
        self.end = ConvScale(128, latent, 1)

    def forward(self, x):
        return self.end(self.c3(self.u2(self.c2(self.u1(self.c1(self.start(x)))))))


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1))

    def forward(self, x):
        return self.net(x)


class PicoSynthesis(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.dec_level = HaarUp(latent + 8, 160)
        self.d11 = chain(160, 3, 1, 1)
        self.d12 = chain(160, 2, 1, 2)
        self.ds1 = LearnedScale(160)
        self.up1 = HaarUp(160, 64)
        self.d21 = chain(64, 2, 1, 1)
        self.d22 = chain(64, 1, 4, 2)
        self.ds2 = LearnedScale(64)
        self.up2 = HaarUp(64, 32)
        self.d31 = chain(32, 2, 2, 2)
        self.d32 = chain(32, 1, 3, 2)
        self.ds3 = LearnedScale(32)
        self.to_rgb = HaarUp(32, 3)

    def forward(self, y, level):
        lv = level[None, :, None, None].expand(y.shape[0], -1, y.shape[2], y.shape[3])
        h = self.dec_level(torch.cat((y, lv), 1))
        h = self.ds1(self.d12(self.d11(h)))
        h = self.up1(h)
        h = self.ds2(self.d22(self.d21(h)))
        h = self.up2(h)
        h = self.ds3(self.d32(self.d31(h)))
        return self.to_rgb(h)


_SYNTHESIS_NAMES = ('dec_level', 'd11', 'd12', 'ds1', 'up1', 'd21', 'd22', 'ds2',
                    'up2', 'd31', 'd32', 'ds3', 'to_rgb')


class PicoModel(nn.Module):
    def __init__(self, latent=192, channel_mask=False):
        super().__init__()
        self.latent = latent
        self.channel_mask = channel_mask
        self.down0 = HaarDown(3, 64)
        self.enc_level = ConvScale(72, 64, 1)
        self.e11 = chain(64, 1, 1, 2)
        self.e12 = chain(64, 1, 4, 2)
        self.es1 = LearnedScale(64)
        self.down1 = HaarDown(64, 96)
        self.e21 = chain(96, 2, 1, 1)
        self.e22 = chain(96, 3, 4, 1)
        self.es2 = LearnedScale(96)
        self.down2 = HaarDown(96, 96)
        self.e31 = chain(96, 2, 1, 1)
        self.e32 = chain(96, 4, 1, 1)
        self.es3 = LearnedScale(96)
        self.to_latent = HaarDown(96, latent)
        self.h1 = HaarDown(96, 128)
        self.hc = chain(128, 1, 2, 1)
        self.h2 = HaarDown(128, 192)
        self.h3 = HaarDown(192, 192)
        self.scale_decoder = ScaleDecoder(latent)
        self.context_decoder = ScaleDecoder(latent)
        self.predictor = nn.Sequential(
            nn.Conv2d(latent * 2 + 8, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 256, 1), nn.GELU(),
            nn.Conv2d(256, latent * 2, 1))
        self.dec_level = HaarUp(latent + 8, 160)
        self.d11 = chain(160, 3, 1, 1)
        self.d12 = chain(160, 2, 1, 2)
        self.ds1 = LearnedScale(160)
        self.up1 = HaarUp(160, 64)
        self.d21 = chain(64, 2, 1, 1)
        self.d22 = chain(64, 1, 4, 2)
        self.ds2 = LearnedScale(64)
        self.up2 = HaarUp(64, 32)
        self.d31 = chain(32, 2, 2, 2)
        self.d32 = chain(32, 1, 3, 2)
        self.ds3 = LearnedScale(32)
        self.to_rgb = HaarUp(32, 3)
        self.level_gain = nn.Parameter(torch.ones(8, latent, 1, 1))
        if channel_mask:
            self.level_mask = nn.Parameter(torch.full((8, latent, 1, 1), 4.0))
        self.hyper_logsig = nn.Parameter(torch.full((192,), -0.5))

    def reparameterize(self):
        for module in self.modules():
            if isinstance(module, ConvScale):
                module.reparameterize()

    def analysis(self, x, level):
        h = self.down0(x)
        lv = level[None, :, None, None].expand(x.shape[0], -1, h.shape[2], h.shape[3])
        h = self.enc_level(torch.cat((h, lv), 1))
        h = self.es1(self.e12(self.e11(h)))
        h = self.down1(h)
        h = self.es2(self.e22(self.e21(h)))
        h = self.down2(h)
        h = self.es3(self.e32(self.e31(h)))
        return self.to_latent(h), self.h3(self.h2(self.hc(self.h1(h))))

    def synthesis(self, y, level):
        lv = level[None, :, None, None].expand(y.shape[0], -1, y.shape[2], y.shape[3])
        h = self.dec_level(torch.cat((y, lv), 1))
        h = self.ds1(self.d12(self.d11(h)))
        h = self.up1(h)
        h = self.ds2(self.d22(self.d21(h)))
        h = self.up2(h)
        h = self.ds3(self.d32(self.d31(h)))
        return self.to_rgb(h)

    @staticmethod
    def masks(h, w, device):
        yy, xx = torch.meshgrid(torch.arange(h, device=device),
                                torch.arange(w, device=device), indexing='ij')
        return [((yy % 2 == a) & (xx % 2 == b))[None, None]
                for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]

    def quantization_gain(self, y, level):
        gain = torch.einsum('l,lcxy->cxy', level, self.level_gain)[None]
        if self.channel_mask:
            soft = torch.einsum('l,lcxy->cxy', level, self.level_mask)[None].sigmoid()
            hard = (soft > 0.5).to(soft.dtype)
            mask = hard + (soft - soft.detach())
            return y * gain, gain, mask
        return y * gain, gain, 1.0

    def forward(self, x, level):
        y, hyper = self.analysis(x, level)
        hs = hyper * 2
        hq = hs + (torch.round(hs) - hs).detach()
        hqd = hq / 2
        sigma = F.softplus(self.scale_decoder(hqd)) + 0.03
        prior = self.context_decoder(hqd)
        yg, gain, quantization_mask = self.quantization_gain(y, level)
        decoded = torch.zeros_like(yg)
        yq = torch.zeros_like(yg)
        bpp = torch.zeros(1, device=x.device)
        sigh = F.softplus(self.hyper_logsig)[None, :, None, None] + 0.03
        tables = torch.from_numpy(on._canon14()['tables']).to(x.device, torch.float32)
        thresholds = torch.from_numpy(on._canon8()['gauss_thresholds']).to(x.device)
        bpp = bpp + _rate_bits(hq, sigh, tables, thresholds).sum() / (
            x.shape[0] * x.shape[2] * x.shape[3])
        lv = level[None, :, None, None].expand(y.shape[0], -1, y.shape[2], y.shape[3])
        for mask in self.masks(y.shape[2], y.shape[3], x.device):
            mu, lq = self.predictor(torch.cat((prior, decoded, lv), 1)).chunk(2, 1)
            qw = F.softplus(lq) + 0.02
            zn = (yg - mu) / qw
            zi = (zn + (torch.round(zn) - zn).detach()) * quantization_mask
            deq = zi * qw + mu
            yq = torch.where(mask, deq, yq)
            decoded = torch.where(mask, deq, decoded)
            if self.channel_mask:
                bits = _sparse_rate_bits(zi, sigma, tables, thresholds, mask)
                bpp = bpp + bits / (x.shape[0] * x.shape[2] * x.shape[3])
            else:
                bits = _rate_bits(zi, sigma, tables, thresholds)
                bpp = bpp + (bits * mask).sum() / (x.shape[0] * x.shape[2] * x.shape[3])
        xr = self.synthesis(yq / gain, level)
        return xr, bpp


class PicoMoeModel(PicoModel):
    def __init__(self, latent=192, channel_mask=False, moe=3):
        super().__init__(latent, channel_mask)
        if moe not in (2, 3, 6):
            raise ValueError('PICO MoE supports two, three or six experts')
        self.moe = moe
        self.synthesis_experts = nn.ModuleList(
            PicoSynthesis(latent) for _ in range(moe))
        for name in _SYNTHESIS_NAMES:
            delattr(self, name)

    def expert_for_level(self, level):
        pos = int(level.argmax().item())
        if self.moe == 2:
            return int(pos >= 2)
        if self.moe == 3:
            return 0 if pos < 2 else 1 if pos < 5 else 2
        return (0, 0, 1, 2, 3, 4, 4, 5)[pos]

    def quantization_gain(self, y, level, expert=-1):
        if expert < 0:
            expert = self.expert_for_level(level)
        yg, gain, channel_mask = super().quantization_gain(y, level)
        if self.moe == 6 and expert == 0:
            yy, xx = torch.meshgrid(torch.arange(y.shape[2], device=y.device),
                                    torch.arange(y.shape[3], device=y.device), indexing='ij')
            spatial = ((yy % 2 == 0) & (xx % 2 == 0))[None, None].to(y.dtype)
            channel_mask = channel_mask * spatial
        return yg, gain, channel_mask

    def synthesis(self, y, level, expert=-1):
        if expert < 0:
            expert = self.expert_for_level(level)
        if expert >= self.moe:
            raise ValueError('invalid PICO MoE expert')
        return self.synthesis_experts[expert](y, level)


def _pad64(rgb):
    h, w = rgb.shape[:2]
    hp, wp = (h + 63) // 64 * 64, (w + 63) // 64 * 64
    if hp != h or wp != w:
        rgb = np.pad(rgb, ((0, hp - h), (0, wp - w), (0, 0)), mode='reflect')
    return rgb.astype(np.float32) / 255, hp, wp


def _load(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    arch = ck.get('arch', {})
    model = PicoMoeModel(**arch) if arch.get('moe') else PicoModel(**arch)
    model.load_state_dict(ck['model'])
    return model.eval(), ck


_RUNTIME_CACHE = {}


def _runtime(ckpt, fast=False):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_triton = (fast and dev == 'cuda' and
                  platform.machine().lower() in ('aarch64', 'arm64') and
                  os.environ.get('OXP9_TRITON', '1') != '0')
    key = (os.path.abspath(ckpt), fast, use_triton)
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    model, _ = _load(ckpt)
    if fast and dev != 'cuda':
        raise RuntimeError('OXP5 requires CUDA')
    model.to(dev)
    model.reparameterize()
    if fast:
        torch.backends.cudnn.allow_tf32 = True
        model.to(memory_format=torch.channels_last)
        if hasattr(model, 'synthesis_experts'):
            model.synthesis_experts.to(memory_format=torch.contiguous_format)
        else:
            for name in _SYNTHESIS_NAMES:
                getattr(model, name).to(memory_format=torch.contiguous_format)
        if use_triton:
            try:
                from oxn_pico_triton import patch_analysis
                patch_analysis(model)
            except ImportError:
                pass
        canon = model.scale_decoder
        hyper_logsig = model.hyper_logsig
    else:
        canon = copy.deepcopy(model.scale_decoder).to('cpu', torch.float64)
        hyper_logsig = model.hyper_logsig.detach().to('cpu', torch.float64).clone()
    runtime = model, canon, hyper_logsig, dev
    _RUNTIME_CACHE[key] = runtime
    return runtime


def _default_fast(dev, pixels=0):
    return dev == 'cuda' and platform.machine().lower() in ('aarch64', 'arm64')


def _available_memory(dev):
    try:
        free = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, OSError, ValueError):
        free = 4 * 1024 ** 3
    if dev == 'cuda':
        free = min(free, torch.cuda.mem_get_info()[0])
    return free


def _auto_tile_size(h, w, dev, workers=1, overlap=0):
    working_pixels = (_available_memory(dev) // 2) // (2048 * workers)
    if h * w <= working_pixels:
        return max(h, w)
    side = int(math.sqrt(working_pixels)) - 2 * overlap
    return max(64, side // 64 * 64)


def _auto_workers(tile_pixels, dev):
    if dev != 'cuda':
        return 1
    budget_per_tile = tile_pixels * 2048
    return min(2, max(1, round((_available_memory(dev) // 2) / budget_per_tile)))


def _performance_tile_cap(dev):
    if dev != 'cuda':
        return 2048
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    throughput = props.multi_processor_count * props.clock_rate
    baseline = 48 * 2418000
    if throughput <= 0:
        return 2048
    cap = 2048 * math.sqrt(throughput / baseline)
    return int(min(4096, max(1024, cap))) // 64 * 64


def _run_tiles(fn, tasks, workers, dev):
    if workers <= 1:
        for task in tasks:
            yield fn(*task)
        return
    on._canon8()
    on._canon14()
    on._prans_native(False)
    local = threading.local()

    def run(task):
        stream = getattr(local, 'stream', None)
        if stream is None:
            stream = local.stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            return fn(*task)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(run, tasks)


def _tile_grid(h, w, tile_size):
    return [(y, x, min(tile_size, h - y), min(tile_size, w - x))
            for y in range(0, h, tile_size) for x in range(0, w, tile_size)]


def _compress_tile_impl(rgb, q, verify, model, canon, hyper_logsig, dev, fast, sparse=False, expert=-1):
    h, w = rgb.shape[:2]
    canvas, hp, wp = _pad64(rgb)
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None].to(dev)
    if fast:
        x = x.contiguous(memory_format=torch.channels_last)
    level = level_vector(q, dev)
    with torch.no_grad():
        y, hyper = model.analysis(x, level)
        if fast:
            hq_dev = (hyper * 2).round().to(torch.int64)
            hqd = hq_dev / 2
            hqd = hqd.contiguous(memory_format=torch.channels_last)
            hq = hq_dev[0].cpu().numpy()
            hqd_cpu = None
        else:
            ha = hyper[0].float().cpu().numpy().astype(np.float64)
            hq = np.rint(ha * 2).astype(np.int64)
            hqd_cpu = torch.from_numpy(hq)[None].to(torch.float64) / 2
            hqd = torch.from_numpy(hq)[None].to(dev) / 2
    sigh = (F.softplus(hyper_logsig)[:, None, None] + 0.03).detach()
    hyper_sig = sigh.float().cpu().numpy() if fast else sigh.numpy()
    pool = None
    hyper_future = None
    if fast:
        on._prans_native(False)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        hyper_future = pool.submit(_encode_verified, hq, hyper_sig, sparse, verify)
    with torch.no_grad():
        sigma = F.softplus(canon(hqd if fast else hqd_cpu)) + 0.03
        prior = model.context_decoder(hqd)
        yg, gain, quantization_mask = model.quantization_gain(y, level, expert) \
            if hasattr(model, 'moe') else model.quantization_gain(y, level)
        decoded = torch.zeros_like(yg)
        lv = level[None, :, None, None].expand(-1, -1, yg.shape[2], yg.shape[3])
    streams = []
    if not fast:
        streams.append(_encode_verified(hq, hyper_sig, sparse, verify))
    latent_data = []
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        with torch.no_grad():
            mu, lq = model.predictor(torch.cat((prior, decoded, lv), 1)).chunk(2, 1)
            mu = mu[0, :, a::2, b::2]
            qw = F.softplus(lq[0, :, a::2, b::2]) + 0.02
            zi = ((yg[0, :, a::2, b::2] - mu) / qw).round().clamp(-256, 255)
            qm = quantization_mask[0, :, a::2, b::2] \
                if quantization_mask.shape[-2:] == yg.shape[-2:] else quantization_mask[0]
            zi = zi * qm
            decoded[0, :, a::2, b::2] = zi * qw + mu
        if fast:
            latent_data.append((zi.to(torch.int16), sigma[0, :, a::2, b::2]))
            continue
        zim = zi.cpu().numpy().astype(np.int64)
        sm = sigma[0, :, a::2, b::2].numpy()
        streams.append(_encode_verified(zim, sm, sparse, verify))
    if fast:
        host_data = []
        for zim_dev, sm_dev in latent_data:
            zim_host = torch.empty(zim_dev.shape, dtype=torch.int16, pin_memory=True)
            sm_host = torch.empty(sm_dev.shape, dtype=torch.float32, pin_memory=True)
            zim_host.copy_(zim_dev, non_blocking=True)
            sm_host.copy_(sm_dev.float(), non_blocking=True)
            host_data.append((zim_host, sm_host))
        torch.cuda.synchronize()
        futures = []
        for zim_host, sm_host in host_data:
            zim = zim_host.numpy().astype(np.int64)
            sm = sm_host.numpy()
            futures.append(pool.submit(_encode_verified, zim, sm, sparse, verify))
        streams.append(hyper_future.result())
        streams.extend(future.result() for future in futures)
        pool.shutdown()
    out = bytearray(b'OXP7' if sparse else b'OXP2')
    out += struct.pack('<IIff', h, w, float(q), float(model.latent))
    for blob in streams:
        out += struct.pack('<I', len(blob)) + blob
    return bytes(out)


def _compress_tile(rgb, q, verify, model, canon, hyper_logsig, dev, fast=False, sparse=False, expert=-1):
    with torch.no_grad():
        return _compress_tile_impl(rgb, q, verify, model, canon, hyper_logsig, dev, fast, sparse, expert)


def compress(rgb, q=75, ckpt=CKPT, verify=False, tile_size=0, workers=0, overlap=-1,
             fast=None, stream_version=5, expert=-1):
    if stream_version not in (5, 6, 7, 8, 9):
        raise ValueError('PICO stream version must be 5, 6, 7, 8 or 9')
    if tile_size < 0 or tile_size % 64:
        raise ValueError('PICO tile size must be zero (auto) or a positive multiple of 64')
    if workers not in (0, 1, 2):
        raise ValueError('PICO workers must be 0 (auto), 1 or 2')
    if overlap < -1:
        raise ValueError('PICO overlap must be -1 (auto) or nonnegative')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = rgb.shape[:2]
    fast = _default_fast(dev, h * w) if fast is None else fast
    model, canon, hyper_logsig, dev = _runtime(ckpt, fast)
    if expert >= 0 and (not hasattr(model, 'moe') or expert >= model.moe):
        raise ValueError('invalid PICO MoE expert')
    if tile_size:
        overlap = min(64, tile_size // 2) if overlap < 0 else overlap
        workers = workers or _auto_workers((tile_size + 2 * overlap) ** 2, dev)
    else:
        whole_tile = _auto_tile_size(h, w, dev)
        if h <= whole_tile and w <= whole_tile:
            workers, tile_size = 1, whole_tile
        else:
            workers = workers or (2 if dev == 'cuda' else 1)
            reserved_overlap = 64 if overlap < 0 else overlap
            tile_size = _auto_tile_size(h, w, dev, workers, reserved_overlap)
            tile_size = min(tile_size, _performance_tile_cap(dev))
        overlap = min(64, tile_size // 2) if overlap < 0 else overlap
    whole = h <= tile_size and w <= tile_size
    if whole and not fast and stream_version == 5:
        return _compress_tile(rgb, q, verify, model, canon, hyper_logsig, dev)
    if whole:
        overlap = 0
        tile_size = (max(h, w) + 63) // 64 * 64
    grid = _tile_grid(h, w, tile_size)
    tasks = []
    for y, x, th, tw in grid:
        ey, ex = max(0, y - overlap), max(0, x - overlap)
        eh, ew = min(h, y + th + overlap), min(w, x + tw + overlap)
        tasks.append((rgb[ey:eh, ex:ew], q, verify, model, canon, hyper_logsig, dev, fast,
                      stream_version >= 7, expert))
    magic = b'OXP9' if stream_version == 9 else b'OXP8' if stream_version == 8 else \
        b'OXP7' if stream_version == 7 else b'OXP6' if stream_version == 6 else \
        b'OXP5' if fast else (b'OXP4' if overlap else b'OXP3')
    out = bytearray(magic)
    if stream_version == 9:
        out += struct.pack('<IIffIIBB', h, w, float(q), float(model.latent),
                           tile_size, overlap, int(fast), expert if expert >= 0 else 255)
    elif stream_version in (6, 7, 8):
        out += struct.pack('<IIffIIB', h, w, float(q), float(model.latent),
                           tile_size, overlap, int(fast))
    elif fast or overlap:
        out += struct.pack('<IIffII', h, w, float(q), float(model.latent), tile_size, overlap)
    else:
        out += struct.pack('<IIffI', h, w, float(q), float(model.latent), tile_size)
    for (y, x, th, tw), blob in zip(grid, _run_tiles(
            _compress_tile, tasks, workers, dev)):
        out += struct.pack('<IIIII', y, x, th, tw, len(blob)) + blob
    return bytes(out)


def compress_target_bpp(rgb, target_bpp, ckpt=CKPT, verify=False, passes=10,
                        stream_version=5, expert=-1):
    if target_bpp <= 0:
        raise ValueError('target bpp must be positive')
    if stream_version == 9 and expert < 0:
        model, _ = _load(ckpt)
        expert = _rate_expert(target_bpp, model.moe)
    best = None
    lo, hi = 1.0, 100.0
    for _ in range(passes):
        q = (lo + hi) / 2
        blob = compress(rgb, q, ckpt, verify, stream_version=stream_version, expert=expert)
        bpp = len(blob) * 8 / (rgb.shape[0] * rgb.shape[1])
        error = abs(bpp - target_bpp)
        if best is None or error < best[0]:
            best = error, blob
        if bpp < target_bpp:
            lo = q
        else:
            hi = q
    return best[1]


def _decompress_tile_impl(blob, model, canon, hyper_logsig, dev, fast, sparse=False, expert=-1):
    if blob[:4] not in (b'OXP1', b'OXP2', b'OXP7'):
        raise ValueError('not a PICO prototype stream')
    empirical = blob[:4] in (b'OXP2', b'OXP7')
    h, w, q, latent = struct.unpack('<IIff', blob[4:20])
    if int(latent) != model.latent:
        raise ValueError('latent mismatch')
    level = level_vector(q, dev)
    hp, wp = (h + 63) // 64 * 64, (w + 63) // 64 * 64
    streams, pos = [], 20
    for _ in range(5):
        n, = struct.unpack('<I', blob[pos:pos + 4]); pos += 4
        streams.append(blob[pos:pos + n]); pos += n
    if pos != len(blob):
        raise ValueError('trailing OXP1 bytes')
    sigh = (F.softplus(hyper_logsig)[:, None, None] + 0.03).detach()
    hyper_sig = sigh.float().cpu().numpy() if fast else sigh.numpy()
    hq = _sparse_stream_decode(streams[0], hyper_sig, (192, hp // 64, wp // 64)) \
        if sparse else _stream_decode(streams[0], hyper_sig, (192, hp // 64, wp // 64), empirical)
    hqd_cpu = None if fast else torch.from_numpy(hq)[None].to(torch.float64) / 2
    hqd = torch.from_numpy(hq)[None].to(dev) / 2
    if fast:
        hqd = hqd.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        sigma = F.softplus(canon(hqd if fast else hqd_cpu)) + 0.03
        prior = model.context_decoder(hqd)
        shape = (1, model.latent, hp // 16, wp // 16)
        decoded = torch.zeros(shape, device=dev)
        yq = torch.zeros(shape, device=dev)
        _, gain, _ = model.quantization_gain(yq, level, expert) \
            if hasattr(model, 'moe') else model.quantization_gain(yq, level)
        lv = level[None, :, None, None].expand(-1, -1, shape[2], shape[3])
    sigma = sigma[0].float().cpu().numpy() if fast else sigma[0].numpy()
    for i, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        with torch.no_grad():
            mu, lq = model.predictor(torch.cat((prior, decoded, lv), 1)).chunk(2, 1)
            mu = mu[0, :, a::2, b::2]
            qw = F.softplus(lq[0, :, a::2, b::2]) + 0.02
        sm = sigma[:, a::2, b::2].reshape(model.latent, -1)
        zi_np = _sparse_stream_decode(streams[i + 1], sm, sm.shape) if sparse else \
            _stream_decode(streams[i + 1], sm, sm.shape, empirical)
        zi = torch.from_numpy(zi_np).to(dev, mu.dtype).reshape(mu.shape)
        deq = zi * qw + mu
        yq[0, :, a::2, b::2] = deq
        decoded[0, :, a::2, b::2] = deq
    z = (yq / gain).float()
    if fast:
        z = z.contiguous()
    with torch.no_grad():
        xr = model.synthesis(z, level, expert) if hasattr(model, 'moe') else \
            model.synthesis(z, level)
    if fast:
        rec = xr[0].permute(1, 2, 0).float().mul(255).add(0.5).clamp(0, 255)
        return rec.to(torch.uint8).contiguous().cpu().numpy()[:h, :w]
    rec = xr[0].permute(1, 2, 0).float().contiguous().cpu().numpy()
    return np.clip(rec[:h, :w] * 255 + 0.5, 0, 255).astype(np.uint8)


def _decompress_tile(blob, model, canon, hyper_logsig, dev, fast=False, sparse=False, expert=-1):
    with torch.no_grad():
        return _decompress_tile_impl(blob, model, canon, hyper_logsig, dev, fast, sparse, expert)


def decompress(blob, ckpt=CKPT, workers=0):
    if blob[:4] not in (b'OXP1', b'OXP2', b'OXP3', b'OXP4', b'OXP5', b'OXP6', b'OXP7', b'OXP8', b'OXP9'):
        raise ValueError('not a PICO prototype stream')
    if workers not in (0, 1, 2):
        raise ValueError('PICO workers must be 0 (auto), 1 or 2')
    if blob[:4] in (b'OXP6', b'OXP7', b'OXP8') and (len(blob) < 29 or blob[28] not in (0, 1)):
        raise ValueError('invalid PICO versioned header')
    if blob[:4] == b'OXP9' and (len(blob) < 30 or blob[28] not in (0, 1)):
        raise ValueError('invalid OXP9 header')
    fast = blob[:4] == b'OXP5' or (blob[:4] in (b'OXP6', b'OXP7', b'OXP8', b'OXP9') and bool(blob[28]))
    model, canon, hyper_logsig, dev = _runtime(ckpt, fast)
    if blob[:4] in (b'OXP1', b'OXP2'):
        return _decompress_tile(blob, model, canon, hyper_logsig, dev)
    expert = -1
    if blob[:4] == b'OXP9':
        h, w, q, latent, tile_size, overlap, fast, expert = struct.unpack('<IIffIIBB', blob[4:30])
        pos = 30
        if expert == 255:
            expert = -1
        elif expert >= model.moe:
            raise ValueError('invalid OXP9 expert')
    elif blob[:4] in (b'OXP6', b'OXP7', b'OXP8'):
        h, w, q, latent, tile_size, overlap, fast = struct.unpack('<IIffIIB', blob[4:29])
        pos = 29
    elif blob[:4] in (b'OXP4', b'OXP5'):
        h, w, q, latent, tile_size, overlap = struct.unpack('<IIffII', blob[4:28])
        pos = 28
    else:
        h, w, q, latent, tile_size = struct.unpack('<IIffI', blob[4:24])
        overlap, pos = 0, 24
    if int(latent) != model.latent:
        raise ValueError('latent mismatch')
    if tile_size < 64 or tile_size % 64:
        raise ValueError('invalid PICO tile size')
    grid = _tile_grid(h, w, tile_size)
    tiles, tile_pos = [], []
    for y, x, th, tw in grid:
        if pos + 20 > len(blob):
            raise ValueError('truncated PICO tile header')
        ty, tx, nth, ntw, n = struct.unpack('<IIIII', blob[pos:pos + 20])
        pos += 20
        if (ty, tx, nth, ntw) != (y, x, th, tw) or pos + n > len(blob):
            raise ValueError('invalid PICO tile')
        tile = blob[pos:pos + n]
        pos += n
        ey, ex = max(0, y - overlap), max(0, x - overlap)
        eh, ew = min(h, y + th + overlap), min(w, x + tw + overlap)
        if len(tile) < 20 or struct.unpack('<II', tile[4:12]) != (eh - ey, ew - ex):
            raise ValueError('PICO tile dimensions mismatch')
        tiles.append(tile)
        tile_pos.append((y - ey, x - ex))
    if pos != len(blob):
        raise ValueError('trailing PICO bytes')
    workers = workers or (1 if len(tiles) == 1 else
                          _auto_workers((tile_size + 2 * overlap) ** 2, dev))
    tasks = [(tile, model, canon, hyper_logsig, dev, fast,
              blob[:4] in (b'OXP7', b'OXP8', b'OXP9'), expert) for tile in tiles]
    rec = np.empty((h, w, 3), dtype=np.uint8)
    results = _run_tiles(_decompress_tile, tasks, workers, dev)
    for (y, x, th, tw), (top, left), tile_rec in zip(grid, tile_pos, results):
        rec[y:y + th, x:x + tw] = tile_rec[top:top + th, left:left + tw]
    return rec


def _train_files(path, crop, exclude=()):
    return ox_data.train_files(path, crop, exclude)


def _random_crop(path, crop):
    return ox_data.random_crop(path, crop)


def train(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PicoMoeModel(args.latent, args.channel_mask, args.moe).to(dev) \
        if args.moe else PicoModel(args.latent, args.channel_mask).to(dev)
    start = 0
    disc_state = None
    if args.init:
        ck = torch.load(args.init, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'], strict=False)
        start = int(ck.get('step', 0))
        disc_state = ck.get('disc')
    anchors = {}
    for key, path in (('low', args.low_anchor), ('high', args.high_anchor)):
        if path:
            ck_anchor = torch.load(path, map_location=dev, weights_only=True)
            anchors[key] = {k: v.detach() for k, v in ck_anchor['model'].items()}
    teacher = None
    if args.high_teacher:
        ck_teacher = torch.load(args.high_teacher, map_location='cpu', weights_only=True)
        teacher_arch = ck_teacher.get('arch', {})
        teacher = PicoMoeModel(**teacher_arch).to(dev) if teacher_arch.get('moe') else \
            PicoModel(**teacher_arch).to(dev)
        teacher.load_state_dict(ck_teacher['model'])
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
    if not 0 < args.q_min <= args.q_max <= 100:
        raise ValueError('training q range must satisfy 0 < q_min <= q_max <= 100')
    imgs = _train_files(args.data, args.crop, args.exclude)
    print(f'train {len(imgs)} images, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {dev}', flush=True)
    analysis_prefixes = ('down0', 'enc_level', 'e11', 'e12', 'es1', 'down1', 'e21',
                         'e22', 'es2', 'down2', 'e31', 'e32', 'es3', 'to_latent', 'h1',
                         'hc', 'h2', 'h3', 'predictor', 'level_gain')
    frozen = ('scale_decoder', 'context_decoder', 'hyper_logsig') if args.freeze_entropy else ()
    if args.freeze_analysis:
        frozen += analysis_prefixes
    if args.train_expert >= 0:
        if not args.moe or args.train_expert >= args.moe:
            raise ValueError('train_expert must address an existing MoE expert')
        for name, param in model.named_parameters():
            if not name.startswith(f'synthesis_experts.{args.train_expert}.') and \
                    (args.train_expert != 0 or name != 'level_mask'):
                param.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith(frozen):
            param.requires_grad_(False)
    mask_param = model.level_mask if model.channel_mask else None
    analysis_params = [p for name, p in model.named_parameters()
                       if args.analysis_lr > 0 and name.startswith(analysis_prefixes)
                       and p.requires_grad and p is not mask_param]
    analysis_ids = {id(p) for p in analysis_params}
    params = [p for p in model.parameters()
              if p.requires_grad and p is not mask_param and id(p) not in analysis_ids]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    if analysis_params:
        opt.add_param_group({'params': analysis_params, 'lr': args.analysis_lr})
    if mask_param is not None and args.mask_lr > 0:
        opt.add_param_group({'params': [mask_param], 'lr': args.mask_lr})
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 20)
    lp = None
    if args.w_lpips:
        import lpips
        lp = lpips.LPIPS(net='alex').to(dev).eval()
    gk = on._gauss_kernel().to(dev) if args.w_msssim else None
    disc = PatchDiscriminator().to(dev) if args.w_gan else None
    if disc is not None and disc_state is not None:
        disc.load_state_dict(disc_state)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr) if disc is not None else None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.multi_resolution:
            crop = int(np.random.choice([args.crop // 2, args.crop]))
        crops = []
        for _ in range(args.batch):
            c = _random_crop(imgs[np.random.randint(len(imgs))], args.crop)
            if np.random.rand() < 0.5:
                c = torch.flip(c, (2,))
            crops.append(c)
        xb = torch.stack(crops).to(dev)
        if args.rate_balance:
            bucket = np.random.rand()
            if bucket < 0.5:
                q = float(np.exp(np.random.uniform(np.log(1), np.log(14))))
            elif bucket < 0.75:
                q = float(np.random.uniform(14, 50))
            else:
                q = float(np.random.uniform(50, 100))
        else:
            q = float(np.random.uniform(np.log(args.q_min), np.log(args.q_max))) \
                if args.q_log else float(np.random.uniform(args.q_min, args.q_max))
            if args.q_log:
                q = float(np.exp(q))
        level = level_vector(q, dev)
        teacher_xr = None
        if teacher is not None and q >= 14 and (args.teacher_augment or args.w_distill):
            with torch.no_grad():
                teacher_xr, _ = teacher(xb, level)
        model_input = xb
        if teacher_xr is not None and np.random.rand() < args.teacher_augment:
            model_input = teacher_xr.float().clamp(0, 1)
        with torch.autocast('cuda', enabled=False):
            xr, bpp = model(model_input, level)
            mse = ((xr - xb) ** 2).mean()
            loss = args.w_mse * mse + (args.rate_weight / q2s(q) ** 2) * bpp
            if args.w_mask and model.channel_mask:
                loss = loss + args.w_mask * model.level_mask.sigmoid().mean()
            ref = anchors.get('low' if q < 14 else 'high')
            if args.w_anchor and ref is not None:
                anchor_loss = []
                for name, param in model.named_parameters():
                    if param.requires_grad and name in ref:
                        anchor_loss.append(((param - ref[name]) ** 2).mean())
                if anchor_loss:
                    loss = loss + args.w_anchor * torch.stack(anchor_loss).mean()
            if teacher_xr is not None and args.w_distill:
                loss = loss + args.w_distill * ((xr - teacher_xr) ** 2).mean()
            if lp is not None:
                with torch.autocast('cuda', enabled=False):
                    loss = loss + args.w_lpips * lp(xr.float().clamp(0, 1) * 2 - 1,
                                                    xb * 2 - 1).mean()
            if gk is not None:
                loss = loss + args.w_msssim * (1 - on.ms_ssim(
                    xr.float().clamp(0, 1), xb, gk))
            if args.w_tal:
                loss = loss + args.w_tal * on.tiling_artifact_loss(xr, xb)
            if args.w_color:
                loss = loss + args.w_color * on.color_balance_loss(xr, xb, 16)
            if args.w_seam:
                loss = loss + args.w_seam * on.patch_seam_loss(xr, xb, 16)
            if args.w_text:
                loss = loss + args.w_text * on.text_fidelity_loss(xr, xb)
        d_loss = None
        if disc is not None:
            with torch.autocast('cuda', enabled=False):
                real = disc(xb.float())
                fake = disc(xr.float().clamp(0, 1).detach())
                d_loss = (F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) +
                          F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                warm = min(1.0, step / max(1, args.steps * 0.3))
                gen = disc(xr.float().clamp(0, 1))
                gan_loss = F.binary_cross_entropy_with_logits(
                    gen, torch.ones_like(gen), reduction='none')
                if args.w_text:
                    mask = F.adaptive_avg_pool2d(on.text_saliency(xb), gen.shape[-2:])
                    gan_loss = gan_loss * (1 - 0.75 * mask)
                loss = loss + args.w_gan * warm * gan_loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step(); sched.step()
        if step == 1 or step % 100 == 0:
            print(f'step {start + step} loss {loss.item():.4f} mse {mse.item():.6f} '
                  f'psnr~{-10 * math.log10(max(mse.item(), 1e-9)):.2f} '
                  f'bpp~{bpp.item():.4f} q={q:.1f} ({time.time() - t0:.0f}s)', flush=True)
        if step % 1000 == 0:
            torch.save({'model': model.state_dict(), 'arch': {'latent': args.latent, 'channel_mask': args.channel_mask, 'moe': args.moe},
                        'step': start + step,
                        'disc': disc.state_dict() if disc is not None else None,
                        'training': {'args': dict(vars(args)), 'image_count': len(imgs)}},
                       args.out + '.part')
    torch.save({'model': model.state_dict(), 'arch': {'latent': args.latent, 'channel_mask': args.channel_mask, 'moe': args.moe},
                'step': start + args.steps,
                'disc': disc.state_dict() if disc is not None else None,
                'training': {'args': dict(vars(args)), 'image_count': len(imgs)}}, args.out)
    print('saved', args.out, flush=True)


def evaluate(args):
    import lpips
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lp = lpips.LPIPS(net='alex').to(dev).eval()
    gk = on._gauss_kernel().to(dev)
    files = sorted(f for f in os.listdir(args.data)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if args.limit:
        files = files[-args.limit:]
    rows = []
    for name in files:
        x = load_rgb(os.path.join(args.data, name))
        t0 = time.time()
        blob = compress(x, args.q, args.ckpt, args.verify, args.tile, args.workers,
                        args.overlap, args.fast, args.stream_version, args.expert)
        enc = time.time() - t0
        t0 = time.time()
        rec = decompress(blob, args.ckpt, args.workers)
        dec = time.time() - t0
        a = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].to(dev).float() / 255
        b = torch.from_numpy(rec.copy()).permute(2, 0, 1)[None].to(dev).float() / 255
        with torch.no_grad():
            ms = on.ms_ssim(a, b, gk).item()
            lv = lp(a * 2 - 1, b * 2 - 1).item()
        row = {'name': name, 'bytes': len(blob),
               'bpp': len(blob) * 8 / (x.shape[0] * x.shape[1]),
               'psnr': on.psnr(x, rec), 'ms_ssim': ms, 'lpips': lv,
               'encode_s': enc, 'decode_s': dec}
        rows.append(row)
        print(row, flush=True)
    import json
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    print('mean', {k: float(np.mean([r[k] for r in rows]))
                   for k in ('bpp', 'psnr', 'ms_ssim', 'lpips', 'encode_s', 'decode_s')})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=('train', 'c', 'd', 'eval'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--analysis_lr', type=float, default=0)
    ap.add_argument('--low_anchor')
    ap.add_argument('--high_anchor')
    ap.add_argument('--high_teacher')
    ap.add_argument('--teacher_augment', type=float, default=0,
                    help='probability of high-rate teacher reconstruction as model input')
    ap.add_argument('--w_anchor', type=float, default=0)
    ap.add_argument('--w_distill', type=float, default=0)
    ap.add_argument('--latent', type=int, default=192)
    ap.add_argument('--moe', type=int, choices=(0, 2, 3, 6), default=0,
                    help='shared analysis with 2, 3 or 6 rate-gated synthesis experts')
    ap.add_argument('--train_expert', type=int, default=-1,
                    help='freeze everything except this synthesis expert')
    ap.add_argument('--multi_resolution', action='store_true',
                    help='alternate full and half-size random crops per step')
    ap.add_argument('--channel_mask', action='store_true',
                    help='learn level/channel latent allocation instead of inverse-only gains')
    ap.add_argument('--freeze_entropy', action='store_true',
                    help='freeze scale/context/hyper weights during selective fine-tuning')
    ap.add_argument('--freeze_analysis', action='store_true',
                    help='freeze analysis/predictor and train only synthesis/allocation')
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'train'))
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--out', default=CKPT)
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--init')
    ap.add_argument('--q', type=float, default=75)
    ap.add_argument('--q_min', type=float, default=1)
    ap.add_argument('--q_max', type=float, default=100)
    ap.add_argument('--q_log', action='store_true',
                    help='sample training quality log-uniformly to emphasize low rates')
    ap.add_argument('--rate_balance', action='store_true',
                    help='sample 50% low, 25% middle and 25% high rates')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--tile', type=int, default=0,
                    help='tile size: 0 = auto from free memory, or a multiple of 64')
    ap.add_argument('--workers', type=int, default=0, choices=(0, 1, 2),
                    help='parallel CUDA tile workers: 0 = auto from the memory budget')
    ap.add_argument('--overlap', type=int, default=-1,
                    help='tile halo pixels: -1 = auto, 0 = no overlap')
    ap.add_argument('--fast', action=argparse.BooleanOptionalAction, default=None,
                    help='OXP5 TF32/channels-last CUDA inference (default on ARM)')
    ap.add_argument('--stream-version', type=int, choices=(5, 6, 7, 8, 9), default=9)
    ap.add_argument('--expert', type=int, default=-1,
                    help='explicit MoE expert for OXP9; -1 = target-bpp/q gate')
    ap.add_argument('--w_mse', type=float, default=1000)
    ap.add_argument('--w_mask', type=float, default=0)
    ap.add_argument('--mask_lr', type=float, default=0)
    ap.add_argument('--rate_weight', type=float, default=25)
    ap.add_argument('--w_lpips', type=float, default=0)
    ap.add_argument('--w_msssim', type=float, default=0)
    ap.add_argument('--w_gan', type=float, default=0)
    ap.add_argument('--w_tal', type=float, default=0)
    ap.add_argument('--w_color', type=float, default=0)
    ap.add_argument('--w_seam', type=float, default=0)
    ap.add_argument('--w_text', type=float, default=0)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a)
    elif a.cmd == 'c':
        rgb = load_rgb(a.args[0]); t0 = time.time()
        blob = compress(rgb, a.q, a.ckpt, a.verify, a.tile, a.workers, a.overlap, a.fast,
                        a.stream_version, a.expert)
        with open(a.args[1], 'wb') as f:
            f.write(blob)
        print(f'{len(blob)} bytes, {len(blob) * 8 / (rgb.shape[0] * rgb.shape[1]):.4f} bpp, '
              f'{time.time() - t0:.2f}s')
    elif a.cmd == 'd':
        blob = open(a.args[0], 'rb').read(); t0 = time.time()
        rec = decompress(blob, a.ckpt, a.workers)
        Image.fromarray(rec).save(a.args[1])
        print(f'decoded in {time.time() - t0:.2f}s')
    else:
        if a.out == CKPT:
            ap.error('eval requires an explicit --out JSON path')
        evaluate(a)


if __name__ == '__main__':
    main()
