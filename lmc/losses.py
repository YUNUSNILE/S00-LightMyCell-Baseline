"""Masked loss = L1 + (1 - SSIM) + (1 - Pearson), per organelle channel.

Channels without ground truth (mask 0) contribute nothing, so one network
can learn all four organelles from studies that only label some of them.
"""
import torch
import torch.nn.functional as F


def _gaussian_kernel(channels, size=11, sigma=1.5, device=None):
    ax = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim_per_channel(x, y, data_range=1.0):
    """Mean SSIM (Wang et al., 2004) for every (sample, channel): shape (B, C)."""
    c = x.shape[1]
    w = _gaussian_kernel(c, device=x.device)
    mu_x, mu_y = F.conv2d(x, w, groups=c), F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x**2
    syy = F.conv2d(y * y, w, groups=c) - mu_y**2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(2, 3))


def pearson_per_channel(x, y, eps=1e-6):
    """Pearson correlation for every (sample, channel): shape (B, C)."""
    x = x.flatten(2)
    y = y.flatten(2)
    x = x - x.mean(-1, keepdim=True)
    y = y - y.mean(-1, keepdim=True)
    return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1) + eps)


def masked_loss(pred, target, mask, w_l1=1.0, w_ssim=0.5, w_pcc=0.5):
    pred, target = pred.float(), target.float()
    l1 = (pred - target).abs().mean(dim=(2, 3))
    ssim = ssim_per_channel(pred, target)
    pcc = pearson_per_channel(pred, target)
    per = w_l1 * l1 + w_ssim * (1 - ssim) + w_pcc * (1 - pcc)
    n = mask.sum().clamp_min(1.0)
    loss = (per * mask).sum() / n
    parts = {k: float((v.detach() * mask).sum() / n) for k, v in (("l1", l1), ("ssim", ssim), ("pcc", pcc))}
    return loss, parts, ssim.detach(), pcc.detach()
