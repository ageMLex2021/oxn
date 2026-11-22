import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _direct_kernel(xp, wp, bp, op, h, w, cin, cout, groups,
                   pixels, M: tl.constexpr):
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
        iy = yy + r // 3 - 1
        ix = xx + r % 3 - 1
        valid = (pos < pixels) & (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
        xv = tl.load(xp + iy[:, None] * (w * cin) + ix[:, None] * cin +
                     group * 32 + k[None, :], mask=valid[:, None], other=0.0)
        wv = tl.load(wbase + r * (32 * cout_group) + k[:, None] * cout_group)
        acc = tl.dot(xv.to(tl.float16), wv, acc)
    acc += tl.load(bp + out)[None, :]
    acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476))
    tl.store(op + yy[:, None] * (w * cout) + xx[:, None] * cout + out[None, :],
             acc, mask=(pos < pixels)[:, None])


def _transform_weight(weight, groups):
    cout, cin_group = weight.shape[:2]
    cout_group = cout // groups
    return weight.reshape(groups, cout_group, cin_group, 9).permute(0, 3, 2, 1).contiguous()


class DirectGrouped3x3:
    def __init__(self, conv, block=64):
        self.groups = conv.groups
        self.cout = conv.out_channels
        self.weight = _transform_weight(conv.weight.detach(), self.groups).to(torch.float16)
        self.bias = conv.bias.detach()
        self.block = block

    def __call__(self, x):
        h, w, cin = x.shape
        out = torch.empty((h, w, self.cout), device=x.device, dtype=x.dtype)
        pixels = h * w
        grid = (triton.cdiv(pixels, self.block), triton.cdiv(self.cout, 32))
        _direct_kernel[grid](x, self.weight, self.bias, out, h, w, cin, self.cout,
                             self.groups, pixels, M=self.block, num_warps=4)
        return out


class FusedConvGelu(nn.Module):
    def __init__(self, conv, block):
        super().__init__()
        self.conv = conv
        self.kernel = DirectGrouped3x3(conv, block)
        self.failed = False

    def forward(self, x):
        if not self.failed and x.shape[0] == 1:
            try:
                out = self.kernel(x[0].permute(1, 2, 0).contiguous())
                return out.permute(2, 0, 1)[None]
            except Exception as exc:
                self.failed = True
                warnings.warn(f'PICO Triton analysis fallback: {exc}')
        return F.gelu(self.conv(x))


def patch_analysis(model, block=64):
    from oxn_pico import CS311

    selected = []
    for name, module in model.named_modules():
        if not name.startswith(('e11.', 'e12.', 'e21.', 'e22.', 'e31.', 'e32.')):
            continue
        if not isinstance(module, CS311):
            continue
        conv = module.net[0]
        if conv.groups == 1 or conv.in_channels // conv.groups != 32:
            continue
        module.net[0] = FusedConvGelu(conv, block)
        module.net[1] = nn.Identity()
        selected.append(name)
    if not selected:
        raise RuntimeError('no compatible PICO analysis convolutions')
    model._triton_analysis = True
    return selected
