"""Backbones and the E3-E5 modules. Input: K TL planes (or 1 centre plane); output: 4 organelle channels.

Every model has forward(x, zpos=None, meta=None):
  x     (B, C, H, W)  normalised TL input (C = K planes, or 1)
  zpos  (B, C)        position of each plane in um relative to the stack centre (from dz; test-time info)
  meta  (B, 5)        one-hot modality (BF, PC, DIC), log(px / 0.1 um), log(dz / 0.5 um) (OME; test-time info)
No model ever receives a best_z (oracle); the focus loss (E4) reads `model.aux["attn_mean"]` in training only.

- `unet`         plain 2D U-Net from scratch. Copied from baseline_teammate/lmc/model.py (class UNet,
                 github.com/YUNUSNILE/S00-LightMyCell-Baseline, commit fc2ab9a); only in_ch changes.
- `smp_resnet34` U-Net with an ImageNet-pretrained ResNet-34 encoder (segmentation_models_pytorch).
                 smp adapts the RGB first conv to the input channel count by cycling the pretrained RGB
                 filters and rescaling them. Input standardised with grayscale ImageNet statistics, as in
                 the teammate's PretrainedUNet.
- `smp_convnext_tiny`  (S01 v2, E1d) U-Net with a pretrained ConvNeXt-Tiny encoder (timm via smp `tu-convnext_tiny`;
                 timm's default pretrained tag, convnext_tiny.in12k_ft_in1k = ImageNet-12k then ImageNet-1k).
                 timm adapts the RGB patch-embedding stem (4x4, stride 4) to K planes with `adapt_input_conv`:
                 the pretrained filters are repeated over the planes and scaled by 3/K. Same input
                 standardisation as smp_resnet34. Depthwise convs get NCHW inputs (`nchw_depthwise_convs`, a bf16
                 GPU bug workaround from S01 v1). `smp_convnextv2_tiny` is the same with ConvNeXt V2-Tiny.
                 `model.encoder` (optional) overrides the smp encoder name, e.g. tu-convnext_tiny.fb_in1k.
- `runet`        residual encoder-decoder of Wodzinski & Mueller (LightMyCells 2024, 3rd place).
                 Ported from github.com/MWod/LightMyCells, src/networks/runet.py, building_blocks.py,
                 configs.py (default_encoder_config / default_decoder_config), Apache-2.0.
                 Changes: K input channels; ONE shared encoder and ONE decoder with 4 output channels
                 (they trained one network per organelle); the unused image_size resampling branch removed.

Config flags (defaults in code, so older configs keep their hash):
  model.zattn.mode        none | softmax (E3) | mean (E3m control)
  model.zattn.features    per-plane stem features F (default 16)
  model.zattn.pe          add the z positional encoding to the per-plane features (default true)
  model.zattn.checkpoint  gradient checkpointing of the attention block (default false: at scale 2 it is not
                          needed for memory and made steps 6-30x slower on this GPU)
  model.zattn.scale       resolution divisor of the attention block (default 2): the stack is average-pooled
                          by `scale`, attention/aggregation run there, the 4 organelle maps are upsampled
                          (bilinear) back to full size. At 512^2 x batch 2 on the GTX 1650, full resolution
                          (scale 1) runs out of memory at F=16 and takes 4.5 s/step at F=8; scale 2 matches the
                          resolution of the ResNet-34 encoder's first (stride-2) level.
  model.film.enabled      FiLM on the decoder stages from `meta` (E5)
  model.film.hidden       MLP width (default 32)
  data.tasks              [best] (default) | [best, mip]: output channels = 4 x number of tasks; channels 0-3 are
                          the best-focus organelles (Task 1), 4-7 their maximum-intensity projections (Task 2)

All models need H, W divisible by `model.divisor` (inference pads to it).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from lmc import ORGANELLES

_IMNET_MEAN, _IMNET_STD = 0.449, 0.226
N_ORG = len(ORGANELLES)
META_DIM = 5


def zcfg(cfg):
    z = dict(mode="none", features=16, pe=True, checkpoint=False, scale=2)
    z.update((cfg.get("model", {}).get("zattn") or {}))
    return z


def filmcfg(cfg):
    f = dict(enabled=False, hidden=32)
    f.update((cfg.get("model", {}).get("film") or {}))
    return f


# ---------------------------------------------------------------- plain U-Net (teammate)
class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, base=32, depth=4):
        super().__init__()
        ch = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([ConvBlock(in_ch, ch[0])] + [ConvBlock(ch[i], ch[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([ConvBlock(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], out_ch, 1)
        self.divisor = 2**depth

    def forward(self, x, zpos=None, meta=None):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


# ---------------------------------------------------------------- FiLM (E5)
class DecoderFiLM(nn.Module):
    """meta -> (gamma, beta) per smp decoder stage; applied by forward hooks as y * (1 + gamma) + beta.
    The last layer is zero-initialised, so the network starts exactly as without FiLM."""

    def __init__(self, decoder_blocks, hidden=32):
        super().__init__()
        self.channels = [b.conv2[0].out_channels for b in decoder_blocks]
        self.mlp = nn.Sequential(nn.Linear(META_DIM, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 2 * sum(self.channels)))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self._params = None
        for i, b in enumerate(decoder_blocks):
            b.register_forward_hook(self._hook(i))

    def set_meta(self, meta):
        self._params = None if meta is None else torch.split(self.mlp(meta), [c for c in self.channels for _ in (0, 1)], 1)

    def _hook(self, i):
        def hook(module, inp, out):
            if self._params is None:
                return out
            g, b = self._params[2 * i], self._params[2 * i + 1]
            return out * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return hook


# ---------------------------------------------------------------- bf16 depthwise-conv workaround (S01 v1)
def _contiguous_input(module, args):
    return (args[0].contiguous(),) + tuple(args[1:])


def nchw_depthwise_convs(model):
    """Give every depthwise conv a plain NCHW (contiguous) input. Returns the number of convs hooked.

    Ported from S01 v1 (E:/DeepLearningCompetition/lmc/model.py, nchw_depthwise_convs). On the RTX 3050 laptop
    (torch 2.8 + cuDNN 9.10) a bf16 depthwise conv with a channels-last strided input crashed with "illegal memory
    access" or returned NaN; timm's ConvNeXt produces such inputs (it permutes to NHWC for LayerNorm and back).
    A contiguous input passed the same stress test (docs/02_pipeline.md section 6 of the course project).
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) and mod.groups > 1 and mod.groups == mod.in_channels:
            mod.register_forward_pre_hook(_contiguous_input)
            n += 1
    return n


# ---------------------------------------------------------------- smp U-Nets (ResNet-34, ConvNeXt)
class SmpUNet(nn.Module):
    def __init__(self, in_ch, out_ch=4, encoder="resnet34", weights="imagenet", film=None):
        super().__init__()
        import segmentation_models_pytorch as smp

        if encoder.startswith("tu-"):  # smp wants True/None for timm encoders (the tag is part of the name)
            weights = None if weights is None else True
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=in_ch, classes=out_ch)
        self.n_depthwise_hooked = nchw_depthwise_convs(self.net) if encoder.startswith("tu-") else 0
        self.register_buffer("mean", torch.tensor(_IMNET_MEAN))
        self.register_buffer("std", torch.tensor(_IMNET_STD))
        self.film = DecoderFiLM(self.net.decoder.blocks, film["hidden"]) if film and film["enabled"] else None
        self.divisor = 32

    @property
    def encoder(self):
        return self.net.encoder

    def forward(self, x, zpos=None, meta=None):
        if self.film is not None:
            self.film.set_meta(meta)
        return self.net((x - self.mean) / self.std)


# ---------------------------------------------------------------- organelle-specific z-attention (E3 / E3m / E4)
def fourier_z(zpos, lengths=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0)):
    """zpos (B, K) in um -> (B, K, 1 + 2*len(lengths)) features: z/10, sin/cos(2 pi z / L)."""
    f = [zpos[..., None] / 10.0]
    for L in lengths:
        f += [torch.sin(2 * math.pi * zpos[..., None] / L), torch.cos(2 * math.pi * zpos[..., None] / L)]
    return torch.cat(f, -1)


class ZAttention(nn.Module):
    """Shared per-plane stem (2 x conv3x3-GN-ReLU, F features) + z positional encoding, then for each
    organelle a 1x1-conv scorer and a softmax over z per pixel (mode 'softmax'), or a plain mean over z
    (mode 'mean', control). Output: concatenation of the 4 organelle maps, (B, 4F, H, W).
    Also returns the pixel-averaged attention (B, 4, K), used by the E4 focus loss and for logging."""

    def __init__(self, features=16, mode="softmax", pe=True):
        super().__init__()
        self.F, self.mode, self.use_pe = features, mode, pe
        self.stem = nn.Sequential(nn.Conv2d(1, features, 3, padding=1), nn.GroupNorm(4, features), nn.ReLU(inplace=True),
                                  nn.Conv2d(features, features, 3, padding=1), nn.GroupNorm(4, features),
                                  nn.ReLU(inplace=True))
        self.pe = nn.Linear(13, features) if pe else None
        if mode == "softmax":
            self.scorer = nn.Conv2d(features, N_ORG, 1)
            nn.init.zeros_(self.scorer.weight)  # start from uniform attention
            nn.init.zeros_(self.scorer.bias)

    def _plane(self, xz, pez):
        f = self.stem(xz)
        return f + pez[..., None, None] if pez is not None else f

    def _vectorised(self, x, pe):
        """All planes at once: (B, K, F, H, W) features; organelle maps (B, 4F, H, W); attention (B, K, 4, H, W)."""
        b, k, h, w = x.shape
        f = self.stem(x.reshape(b * k, 1, h, w)).reshape(b, k, self.F, h, w)
        if pe is not None:
            f = f + pe[..., None, None]
        if self.mode != "softmax":
            return f.mean(1).repeat(1, N_ORG, 1, 1), None
        a = torch.softmax(self.scorer(f.reshape(b * k, self.F, h, w)).reshape(b, k, N_ORG, h, w), dim=1)
        return torch.cat([(a[:, :, o:o + 1] * f).sum(1) for o in range(N_ORG)], 1), a

    def forward(self, x, zpos, keep_maps=False, ckpt=False, per_plane=False):
        """per_plane=False: vectorised over planes (fast; used at reduced resolution), the whole block
        checkpointed if ckpt. per_plane=True: plane-by-plane (memory: one (B, F, H, W) map per plane kept,
        no broadcast product), each plane's stem checkpointed if ckpt (for full resolution)."""
        b, k, h, w = x.shape
        pe = self.pe(fourier_z(zpos)) if self.use_pe else None  # (B, K, F)
        if not per_plane:
            if ckpt and torch.is_grad_enabled():
                out, a = checkpoint(self._vectorised, x, pe, use_reentrant=False)
            else:
                out, a = self._vectorised(x, pe)
            if a is None:
                a_mean = torch.full((b, N_ORG, k), 1.0 / k, device=x.device, dtype=x.dtype)
            else:
                a_mean = a.mean((3, 4)).transpose(1, 2)
            return out, a_mean, (a.detach() if keep_maps and a is not None else None)
        feats = []
        for z in range(k):
            pz = pe[:, z] if pe is not None else None
            if ckpt and torch.is_grad_enabled():
                feats.append(checkpoint(self._plane, x[:, z:z + 1], pz, use_reentrant=False))
            else:
                feats.append(self._plane(x[:, z:z + 1], pz))
        if self.mode == "softmax":
            a = torch.softmax(torch.stack([self.scorer(f) for f in feats], 1), dim=1)  # (B, K, 4, H, W)
            outs = []
            for o in range(N_ORG):
                acc = None
                for z in range(k):
                    t = a[:, z, o:o + 1] * feats[z]
                    acc = t if acc is None else acc + t
                outs.append(acc)
            out = torch.cat(outs, 1)
            a_mean = a.mean((3, 4)).transpose(1, 2)  # (B, 4, K)
        else:  # mean over z: identical map for every organelle
            acc = feats[0]
            for f in feats[1:]:
                acc = acc + f
            out = (acc / k).repeat(1, N_ORG, 1, 1)
            a_mean = torch.full((b, N_ORG, k), 1.0 / k, device=x.device, dtype=x.dtype)
            a = None
        return out, a_mean, (a.detach() if keep_maps and a is not None else None)


class ZAttnSmpUNet(nn.Module):
    def __init__(self, k, out_ch=4, z=None, film=None, encoder="resnet34", weights="imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.k, self.ckpt, self.scale = k, z["checkpoint"], int(z["scale"])
        self.has_attention = z["mode"] == "softmax"  # learned weights over z (logged / supervised)
        self.zattn = ZAttention(z["features"], z["mode"], z["pe"])
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=N_ORG * z["features"],
                            classes=out_ch)
        self.film = DecoderFiLM(self.net.decoder.blocks, film["hidden"]) if film and film["enabled"] else None
        self.divisor = 32
        self.aux = {}
        self.keep_maps = False  # set True to keep full attention maps (figures)
        self.last_maps = None

    @property
    def encoder(self):
        return self.net.encoder

    def forward(self, x, zpos=None, meta=None):
        if zpos is None:
            raise ValueError("z-attention needs zpos (plane positions in um, from dz)")
        xs = F.avg_pool2d(x, self.scale) if self.scale > 1 else x
        feats, a_mean, maps = self.zattn(xs, zpos, self.keep_maps, ckpt=self.ckpt and self.training,
                                         per_plane=self.scale == 1)
        if self.scale > 1:
            feats = F.interpolate(feats, size=x.shape[-2:], mode="bilinear", align_corners=False)
        self.aux = {"attn_mean": a_mean}
        self.last_maps = maps
        if self.film is not None:
            self.film.set_meta(meta)
        return self.net(feats)


# ---------------------------------------------------------------- RUNet (Wodzinski & Mueller)
class ResidualBlock(nn.Module):
    """building_blocks.ResidualBlock: two conv3x3-GN-LeakyReLU plus a 1x1 projection shortcut."""

    def __init__(self, cin, cout, leaky_alpha=0.01):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(cout, cout), nn.LeakyReLU(leaky_alpha, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(cout, cout), nn.LeakyReLU(leaky_alpha, inplace=True),
        )
        self.conv = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.module(x) + self.conv(x)


def _pad_to(x, template):
    """building_blocks.pad: zero-pad x to template's spatial size."""
    py, px = abs(x.size(2) - template.size(2)), abs(x.size(3) - template.size(3))
    return F.pad(x, (math.floor(px / 2), math.ceil(px / 2), math.floor(py / 2), math.ceil(py / 2)))


class RUNet(nn.Module):
    def __init__(self, in_ch, out_ch=4, channels=(16, 64, 128, 256, 512), enc_blocks=(2, 2, 2, 2, 2),
                 dec_blocks=(1, 1, 1, 1, 1)):
        super().__init__()
        ins = (in_ch,) + tuple(channels[:-1])
        self.encoders = nn.ModuleList()
        for ic, oc, nb in zip(ins, channels, enc_blocks):  # RUNetEncoder
            mods = [ResidualBlock(ic if j == 0 else oc, oc) for j in range(nb)]
            mods += [nn.Conv2d(oc if nb else ic, oc, 4, stride=2, padding=1), nn.GroupNorm(oc, oc),
                     nn.LeakyReLU(0.01, inplace=True)]
            self.encoders.append(nn.Sequential(*mods))
        n = len(channels)
        self.decoders = nn.ModuleList()
        for i, (oc, nb) in enumerate(zip(channels, dec_blocks)):  # RUNetDecoder
            coc = oc if i == n - 1 else oc + channels[i + 1]
            mods = [ResidualBlock(coc if j == 0 else oc, oc) for j in range(nb)]
            mods += [nn.ConvTranspose2d(oc if nb else coc, oc, 4, stride=2, padding=1), nn.GroupNorm(oc, oc),
                     nn.LeakyReLU(0.01, inplace=True)]
            self.decoders.append(nn.Sequential(*mods))
        self.last_layer = nn.Conv2d(channels[0], out_ch, 1)
        self.divisor = 2**n

    def forward(self, x, zpos=None, meta=None):
        emb = []
        for enc in self.encoders:
            x = enc(x)
            emb.append(x)
        cx = None
        for i in range(len(self.decoders) - 1, -1, -1):
            if cx is None:
                cx = self.decoders[i](emb[i])
            else:
                cx = self.decoders[i](torch.cat((_pad_to(cx, emb[i]), emb[i]), dim=1))
        return self.last_layer(cx)


# ---------------------------------------------------------------- factory
def build_model(cfg, pretrained=True):
    """cfg: the resolved run config dict. pretrained=False skips weight download (checkpoint loaded after)."""
    in_ch = int(cfg["data"]["in_channels"])
    n_out = N_ORG * n_tasks(cfg)
    m = cfg["model"]
    name = m["backbone"]
    z, fm = zcfg(cfg), filmcfg(cfg)
    weights = m.get("encoder_weights", "imagenet") if pretrained else None
    if z["mode"] != "none":
        if name != "smp_resnet34" or cfg["data"]["input"] != "stack":
            raise ValueError("z-attention is implemented for backbone smp_resnet34 with input 'stack'")
        return ZAttnSmpUNet(in_ch, n_out, z, fm, "resnet34", weights)
    if fm["enabled"] and name != "smp_resnet34":
        raise ValueError("FiLM is implemented for backbone smp_resnet34")
    if name == "unet":
        return UNet(in_ch, n_out, base=int(m.get("base", 32)), depth=int(m.get("depth", 4)))
    if name in SMP_ENCODERS:
        return SmpUNet(in_ch, n_out, m.get("encoder") or SMP_ENCODERS[name], weights, fm)
    if name == "runet":
        return RUNet(in_ch, n_out)
    raise ValueError(f"unknown backbone {name!r}")


# backbone name -> smp encoder name (S01 v2 adds the ConvNeXt encoders)
SMP_ENCODERS = {"smp_resnet34": "resnet34", "smp_convnext_tiny": "tu-convnext_tiny",
                "smp_convnextv2_tiny": "tu-convnextv2_tiny"}


def task_list(cfg):
    """Prediction tasks: ['best'] (Task 1, default), ['best', 'mip'] (Task 1 + Task 2 heads) or ['mip'] (Task 2 only)."""
    tasks = list(cfg.get("data", {}).get("tasks") or ["best"])
    if tasks not in (["best"], ["best", "mip"], ["mip"]):
        raise ValueError(f"data.tasks must be [best], [best, mip] or [mip], got {tasks}")
    return tasks


def n_tasks(cfg):
    return len(task_list(cfg))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_groups(model, lr, enc_lr_mult=1.0):
    """AdamW parameter groups; the pretrained encoder can get lr x enc_lr_mult (discriminative fine-tuning,
    Howard & Ruder 2018). Ported from S01 v1 (E:/DeepLearningCompetition/lmc/model.py, param_groups).
    enc_lr_mult = 1.0 (or a model without `.encoder`) gives the single group used by every run before S01 v2."""
    enc = getattr(model, "encoder", None)
    if enc is None or enc_lr_mult == 1.0:
        return [{"params": list(model.parameters()), "lr": lr}]
    enc_ids = {id(p) for p in enc.parameters()}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    return [{"params": list(enc.parameters()), "lr": lr * enc_lr_mult}, {"params": rest, "lr": lr}]
