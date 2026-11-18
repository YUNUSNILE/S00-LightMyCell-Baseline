"""Train the baseline U-Net (TL image -> 4 organelles) with a masked loss.

Writes to --out: log.csv (one row per epoch), curves.png, best.pt, last.pt,
config.json and split.json.

Examples:
  # smoke test on one study (image-level split, numbers are not meaningful)
  python train.py --val-frac 0.2 --epochs 2 --samples-per-epoch 64 --out runs/smoke
  # baseline with held-out studies
  python train.py --val-studies 3,6,21 --epochs 40 --out runs/baseline
  # S01: U-Net with an ImageNet-pretrained ResNet-34 encoder, encoder lr = 0.1 x lr
  python train.py --arch smp-unet --encoder resnet34 --encoder-weights imagenet \\
      --enc-lr-mult 0.1 --val-studies 3,6,21 --out runs/S01/r34-imagenet
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from lmc import ORGANELLES
from lmc.data import LMCPatches, load_index, split_index, study_balanced_weights
from lmc.losses import masked_loss
from lmc.model import build_model, count_params, model_name, param_groups


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed")
    p.add_argument("--train-studies", default="", help="default: every study not used for validation")
    p.add_argument("--val-studies", default="")
    p.add_argument("--val-frac", type=float, default=0.0, help="image-level split, smoke tests only")
    p.add_argument("--target", choices=["best", "mip"], default="best")
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--batch", type=int, default=4, help="micro-batch size (what goes through the GPU at once)")
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation: one optimizer step per N micro-batches (effective batch = batch x N)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--samples-per-epoch", type=int, default=800)
    p.add_argument("--val-crops", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--arch", choices=["unet", "smp-unet", "smp-unetpp"], default="unet",
                   help="unet = plain U-Net from scratch (S00); smp-unet / smp-unetpp = pretrained encoder (S01)")
    p.add_argument("--base", type=int, default=32, help="plain U-Net width (--arch unet)")
    p.add_argument("--encoder", default="resnet34",
                   help="S01 encoder: smp name (resnet34, efficientnet-b3) or timm via tu- (tu-convnext_tiny)")
    p.add_argument("--encoder-weights", default="imagenet", help="S01: 'imagenet' or 'none' (random init)")
    p.add_argument("--enc-lr-mult", type=float, default=0.1, help="S01: encoder lr = lr x this")
    p.add_argument("--freeze-epochs", type=int, default=0, help="S01: keep the encoder frozen for N epochs")
    p.add_argument("--freeze-bn", action="store_true", help="S01: keep encoder BatchNorm statistics fixed")
    p.add_argument("--w-l1", type=float, default=1.0)
    p.add_argument("--w-ssim", type=float, default=0.5)
    p.add_argument("--w-pcc", type=float, default=0.5)
    p.add_argument("--balance", choices=["study", "none"], default="study")
    p.add_argument("--focus-aug", action="store_true", help="random TL plane near best focus")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="bf16 has the fp32 range; fp16 overflowed (inf BatchNorm stats) in our first run")
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping, 0 = off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/baseline")
    return p.parse_args()


def ids(s):
    return [int(v) for v in s.split(",") if v.strip()]


def autocast(device, amp_dtype):
    """Mixed-precision context; amp_dtype None means plain fp32."""
    return torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None)


def bn_stats_finite(model):
    return all(torch.isfinite(b).all() for n, b in model.named_buffers() if "running" in n)


@torch.no_grad()
def validate(model, loader, device, amp_dtype, w):
    model.eval()
    n_org = len(ORGANELLES)
    s_sum, p_sum, cnt = np.zeros(n_org), np.zeros(n_org), np.zeros(n_org)
    loss_sum, n_batches = 0.0, 0
    for b in loader:
        x, y, m = b["x"].to(device), b["y"].to(device), b["m"].to(device)
        with autocast(device, amp_dtype):
            pred = model(x)
        loss, _, ssim, pcc = masked_loss(pred, y, m, *w)
        loss_sum += float(loss)
        n_batches += 1
        m_np = m.cpu().numpy()
        s_sum += (ssim.cpu().numpy() * m_np).sum(0)
        p_sum += (pcc.cpu().numpy() * m_np).sum(0)
        cnt += m_np.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ssim_org = np.where(cnt > 0, s_sum / cnt, np.nan)
        pcc_org = np.where(cnt > 0, p_sum / cnt, np.nan)
    return loss_sum / max(n_batches, 1), ssim_org, pcc_org


def plot_curves(log_path, out_png):
    with open(log_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    ax[0].plot(ep, [float(r["val_loss"]) for r in rows], label="val")
    ax[0].set_title("Masked loss")
    for k, metric in enumerate(("ssim", "pcc"), start=1):
        for org in ORGANELLES:
            v = [float(r[f"val_{metric}_{org}"]) for r in rows]
            if not all(math.isnan(t) for t in v):
                ax[k].plot(ep, v, label=org)
        ax[k].set_title(f"Validation {metric.upper()} per organelle")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda" and args.precision != "fp32":
        amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("bf16 not supported on this GPU, falling back to fp32")
            amp_dtype = None

    df = load_index(args.data)
    train_df, val_df = split_index(df, ids(args.val_studies), args.val_frac, args.seed)
    if args.train_studies:
        train_df = train_df[train_df.study.isin(ids(args.train_studies))].reset_index(drop=True)
    if args.target == "mip":
        has = [f"has_mip_{o}" for o in ORGANELLES]
        train_df = train_df[train_df[has].sum(1) > 0].reset_index(drop=True)
        val_df = val_df[val_df[has].sum(1) > 0].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0:
        raise SystemExit("empty train or val split, check --val-studies / --train-studies")

    split = {"train_studies": sorted(train_df.study.unique().tolist()),
             "val_studies": sorted(val_df.study.unique().tolist()),
             "val_images": val_df[["study", "image_id"]].drop_duplicates().values.tolist(),
             "image_level_split": not args.val_studies}
    (out / "split.json").write_text(json.dumps(split, indent=1))
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    print(f"train: {len(train_df)} samples from studies {split['train_studies']}")
    print(f"val:   {len(val_df)} samples from studies {split['val_studies']}"
          + ("  [image-level split: smoke test only]" if split["image_level_split"] else ""))

    train_ds = LMCPatches(train_df, args.data, args.patch, train=True, target=args.target, focus_aug=args.focus_aug)
    val_ds = LMCPatches(val_df, args.data, args.patch, train=False, target=args.target, val_crops=args.val_crops)
    weights = study_balanced_weights(train_df) if args.balance == "study" else np.ones(len(train_df))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), args.samples_per_epoch, replacement=True)
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.batch, sampler=sampler, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, **kw)

    # free VRAM before the model: above this, Windows (WDDM) spills into shared system memory,
    # which is slow and was associated with GPU faults in the S01 screen (docs/05_experiment_log.md)
    vram_free = torch.cuda.mem_get_info()[0] if device.type == "cuda" else None
    model = build_model(vars(args)).to(device)
    print(f"{model_name(vars(args))}: {count_params(model) / 1e6:.1f}M parameters, device {device}, "
          f"precision {str(amp_dtype).replace('torch.', '') if amp_dtype else 'fp32'}")
    groups = param_groups(model, args.lr, args.enc_lr_mult)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_dl) // args.accum
    if steps_per_epoch == 0:
        raise SystemExit("--accum is larger than the number of batches per epoch")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=args.epochs * steps_per_epoch, pct_start=0.05)
    encoder = getattr(model, "encoder", None)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    w = (args.w_l1, args.w_ssim, args.w_pcc)
    bn_bufs = [b for n, b in model.named_buffers() if "running" in n]
    spill_warned = False

    log_path = out / "log.csv"
    fields = ["epoch", "lr", "train_loss", "train_l1", "train_ssim", "train_pcc", "val_loss"]
    fields += [f"val_{m}_{o}" for m in ("ssim", "pcc") for o in ORGANELLES] + ["val_score", "skipped_batches", "seconds"]
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    best = -math.inf
    for epoch in range(1, args.epochs + 1):
        if encoder is not None and args.freeze_epochs:
            for prm in encoder.parameters():
                prm.requires_grad_(epoch > args.freeze_epochs)
        model.train()
        t0 = time.time()
        tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "pcc": 0.0}
        skipped, n_ok, bad = 0, 0, False  # skipped = optimizer steps not taken (non-finite loss or gradient)
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(train_dl):
            if i >= steps_per_epoch * args.accum:  # leftover micro-batches that cannot fill a step
                break
            x, y, m = b["x"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True), b["m"].to(device)
            bn_backup = [t.clone() for t in bn_bufs]
            with autocast(device, amp_dtype):
                pred = model(x)
            loss, parts, _, _ = masked_loss(pred, y, m, *w)
            if torch.isfinite(loss):
                scaler.scale(loss / args.accum).backward()
                n_ok += 1
                tot["loss"] += loss.item()
                for k, v in parts.items():
                    tot[k] += v
            else:  # a non-finite forward also polluted the BatchNorm running statistics: undo that
                bad = True
                for t, s in zip(bn_bufs, bn_backup):
                    t.copy_(s)
            if (i + 1) % args.accum:
                continue
            if not bad:
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else math.inf)
                # never apply a non-finite gradient: clipping turns it into NaN for every weight
                # (fp16: the GradScaler skips such steps itself and lowers its scale)
                bad = not scaler.is_enabled() and not torch.isfinite(gnorm)
            if bad:
                skipped += 1
            else:
                scaler.step(opt)
                scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            bad = False
        nb = max(n_ok, 1)
        if skipped:
            print(f"  warning: skipped {skipped}/{steps_per_epoch} optimizer steps (non-finite loss or gradient)",
                  flush=True)
        if vram_free is not None and not spill_warned and torch.cuda.max_memory_reserved() > vram_free:
            spill_warned = True
            print(f"  warning: reserved GPU memory {torch.cuda.max_memory_reserved() / 2**30:.2f} GB exceeds the "
                  f"{vram_free / 2**30:.2f} GB that were free; Windows spills the rest into shared memory "
                  "(slow, possibly unstable). Use a smaller --batch with --accum.", flush=True)
        if not bn_stats_finite(model):
            raise SystemExit(f"BatchNorm statistics became non-finite in epoch {epoch}; stopping")
        val_loss, ssim_org, pcc_org = validate(model, val_dl, device, amp_dtype, w)
        score = float(np.nanmean(np.concatenate([ssim_org, pcc_org])))
        row = [epoch, sched.get_last_lr()[-1], tot["loss"] / nb, tot["l1"] / nb, tot["ssim"] / nb, tot["pcc"] / nb, val_loss]
        row += list(ssim_org) + list(pcc_org) + [score, skipped, time.time() - t0]
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        plot_curves(log_path, out / "curves.png")

        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_score": score}
        torch.save(ckpt, out / "last.pt")
        if score > best:
            best = score
            torch.save(ckpt, out / "best.pt")
        per_org = "  ".join(f"{o[:4]} {s:.3f}/{p:.3f}" for o, s, p in zip(ORGANELLES, ssim_org, pcc_org))
        mem = f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else ""
        print(f"epoch {epoch:3d}  train {tot['loss'] / nb:.4f}  val {val_loss:.4f}  score {score:.3f}  "
              f"[SSIM/PCC {per_org}]  {time.time() - t0:.0f}s{mem}", flush=True)
    print(f"best val score {best:.3f}; outputs in {out}")


if __name__ == "__main__":
    main()
