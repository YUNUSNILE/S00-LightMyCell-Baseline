"""Models: 1 TL channel in, 4 organelle channels out.

- `UNet`: plain 2D U-Net (Ronneberger et al., 2015), trained from scratch, used by S00.
- `PretrainedUNet`: U-Net / U-Net++ with a pretrained encoder from
  segmentation_models_pytorch / timm, used by S01 (docs/03_strategies.md).
- `build_model(cfg)`: builds either one from a training config (the `args` dict
  stored in every checkpoint), so train.py and evaluate.py stay in sync.
"""
import torch
from torch import nn

from lmc import ORGANELLES

# Grayscale ImageNet statistics (mean of the RGB values). smp collapses the pretrained
# RGB first conv to 1 channel by summing its weights, so a grey image standardised like
# this looks to the encoder like an ImageNet image.
_IMNET_MEAN, _IMNET_STD = 0.449, 0.226


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Input height/width must be divisible by 2**depth."""

    def __init__(self, in_ch=1, out_ch=4, base=32, depth=4):
        super().__init__()
        ch = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([ConvBlock(in_ch, ch[0])] + [ConvBlock(ch[i], ch[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([ConvBlock(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def _contiguous_input(module, args):
    return (args[0].contiguous(),) + tuple(args[1:])


def nchw_depthwise_convs(model):
    """Give every depthwise conv a plain NCHW (contiguous) input.

    Works around a GPU bug in our environment (torch 2.8 + cuDNN 9.10, RTX 3050):
    a bf16 depthwise conv whose input is channels-last strided crashes with
    "illegal memory access" or yields NaN, intermittently. timm's ConvNeXt makes
    such inputs (it permutes to NHWC for LayerNorm and back). Reproduced and checked
    with a stress test: channels-last input fails, the same input made contiguous
    passes (docs/02_pipeline.md, section 6). Returns the number of convs hooked.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels:
            m.register_forward_pre_hook(_contiguous_input)
            n += 1
    return n


class PretrainedUNet(nn.Module):
    """U-Net or U-Net++ decoder on a pretrained smp/timm encoder (strategy S01).

    encoder: smp name, e.g. "resnet34", "efficientnet-b3", or a timm one as
             "tu-convnext_tiny".
    weights: "imagenet" or None (same network, random init: the pretraining ablation).
    freeze_bn: keep the encoder's BatchNorm running statistics fixed (eval mode).
    Input height/width must be divisible by 32.
    """

    def __init__(self, encoder="resnet34", weights="imagenet", decoder="unet", out_ch=4, freeze_bn=False):
        super().__init__()
        import segmentation_models_pytorch as smp  # optional dependency, only needed for S01

        cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus}[decoder]
        self.net = cls(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=out_ch)
        nchw_depthwise_convs(self.net)  # bf16 channels-last depthwise conv bug, see the function
        self.freeze_bn = freeze_bn
        self.register_buffer("mean", torch.tensor(_IMNET_MEAN))
        self.register_buffer("std", torch.tensor(_IMNET_STD))

    @property
    def encoder(self):
        return self.net.encoder

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.net.encoder.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def _weights(cfg):
    w = cfg.get("encoder_weights", "imagenet")
    return None if w in (None, "", "none", "None") else w


def build_model(cfg, pretrained=True):
    """Model from a config dict (train.py args). pretrained=False skips downloading
    encoder weights, for when a checkpoint is loaded right after (evaluate.py)."""
    arch = cfg.get("arch", "unet")  # checkpoints from before S01 have no "arch": plain U-Net
    if arch == "unet":
        return UNet(1, len(ORGANELLES), base=cfg.get("base", 32))
    if arch in ("smp-unet", "smp-unetpp"):
        return PretrainedUNet(encoder=cfg.get("encoder", "resnet34"),
                              weights=_weights(cfg) if pretrained else None,
                              decoder="unet" if arch == "smp-unet" else "unetpp",
                              out_ch=len(ORGANELLES), freeze_bn=cfg.get("freeze_bn", False))
    raise ValueError(f"unknown arch {arch!r}")


def model_name(cfg):
    if cfg.get("arch", "unet") == "unet":
        return f"U-Net (base {cfg.get('base', 32)}, scratch)"
    dec = "U-Net" if cfg["arch"] == "smp-unet" else "U-Net++"
    return f"{dec} + {cfg.get('encoder')} encoder ({_weights(cfg) or 'random init'})"


def param_groups(model, lr, enc_lr_mult=1.0):
    """AdamW groups: the encoder can get a smaller lr (discriminative fine-tuning)."""
    enc = getattr(model, "encoder", None)
    if enc is None or enc_lr_mult == 1.0:
        return [{"params": list(model.parameters()), "lr": lr}]
    enc_ids = {id(p) for p in enc.parameters()}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    return [{"params": list(enc.parameters()), "lr": lr * enc_lr_mult}, {"params": rest, "lr": lr}]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
