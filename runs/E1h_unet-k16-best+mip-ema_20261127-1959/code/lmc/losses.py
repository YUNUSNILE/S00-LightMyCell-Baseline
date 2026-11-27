"""Masked loss = w_l1 * L1 + w_ssim * (1 - SSIM) + w_pcc * (1 - PCC), per organelle channel.

Copied from the teammate's baseline: baseline_teammate/lmc/losses.py
(github.com/YUNUSNILE/S00-LightMyCell-Baseline, commit fc2ab9a), unchanged except for comments.
Channels without ground truth (mask 0) contribute nothing, so one network learns all four
organelles from studies that label only some of them.
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
    """Returns (loss, parts dict, per-(sample, channel) SSIM, per-(sample, channel) PCC)."""
    pred, target = pred.float(), target.float()
    l1 = (pred - target).abs().mean(dim=(2, 3))
    ssim = ssim_per_channel(pred, target)
    pcc = pearson_per_channel(pred, target)
    per = w_l1 * l1 + w_ssim * (1 - ssim) + w_pcc * (1 - pcc)
    n = mask.sum().clamp_min(1.0)
    loss = (per * mask).sum() / n
    parts = {k: float((v.detach() * mask).sum() / n) for k, v in (("l1", l1), ("ssim", ssim), ("pcc", pcc))}
    return loss, parts, ssim.detach(), pcc.detach()


def focus_target(bz, k, sigma=1.0):
    """Gaussian over the K resampled plane indices centred on bz (B, 4), normalised over z: (B, 4, K).
    NaN bz (missing organelle) gives a NaN row; it is masked out by focus_kl."""
    z = torch.arange(k, dtype=torch.float32, device=bz.device)
    q = torch.exp(-0.5 * ((z[None, None, :] - bz[..., None]) / sigma) ** 2)
    return q / q.sum(-1, keepdim=True)


def focus_kl(attn_mean, bz, mask, sigma=1.0, eps=1e-8):
    """E4 auxiliary loss: KL(q || p) between the Gaussian target q at each organelle's resampled best_z and
    p = the organelle's attention over z averaged over the patch pixels (B, 4, K).
    Masked for organelles without ground truth or without best_z; mean over the valid (sample, organelle)."""
    valid = (mask > 0) & torch.isfinite(bz)
    if not valid.any():
        return attn_mean.sum() * 0.0
    q = focus_target(torch.where(valid, bz, torch.zeros_like(bz)), attn_mean.shape[-1], sigma)
    p = attn_mean.float().clamp_min(eps)
    kl = (q * (torch.log(q.clamp_min(eps)) - torch.log(p))).sum(-1)  # (B, 4)
    return (kl * valid).sum() / valid.sum()
