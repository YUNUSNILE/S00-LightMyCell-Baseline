"""Evaluate a checkpoint on full validation images (tiled inference).

Works for every architecture in lmc.model (rebuilt from the checkpoint's config).
Writes to <checkpoint folder>/eval (or --out): metrics_per_image.csv,
summary_by_organelle.csv, summary_by_organelle_modality.csv, scores.json
(LMC-score) and example figures (TL | ground truth | prediction).

Metric columns: off_* = official LMC metric (what the leaderboard scores; used for
strategy selection), ssim / pcc = our original metric (kept for comparison with
older runs).

Example:
  python evaluate.py --ckpt runs/baseline/best.pt
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch

from lmc import ORGANELLES
from lmc.data import _norm, load_index
from lmc.metrics import lmc_scores, official_metrics, pcc, ssim
from lmc.model import build_model, model_name


@torch.no_grad()
def predict_full(model, img, device, tile=512, overlap=128, amp_dtype=torch.bfloat16, batch=4):
    """Sliding-window prediction with Hann-weighted blending; returns (4, H, W)."""
    h, w = img.shape
    ph, pw = max(tile, -(-h // 16) * 16), max(tile, -(-w // 16) * 16)
    x = np.pad(img, ((0, ph - h), (0, pw - w)), mode="reflect")
    stride = tile - overlap

    def starts(n):
        s = list(range(0, n - tile + 1, stride))
        return s if s[-1] == n - tile else s + [n - tile]

    pos = [(y, z) for y in starts(ph) for z in starts(pw)]
    win = torch.from_numpy(np.clip(np.outer(np.hanning(tile), np.hanning(tile)), 1e-3, None)).float().to(device)
    out = torch.zeros(len(ORGANELLES), ph, pw, device=device)
    norm = torch.zeros(1, ph, pw, device=device)
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        xb = torch.from_numpy(np.stack([x[y:y + tile, z:z + tile] for y, z in chunk])[:, None]).float().to(device)
        with torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            pb = model(xb).float()
        for (y, z), p in zip(chunk, pb):
            out[:, y:y + tile, z:z + tile] += p * win
            norm[:, y:y + tile, z:z + tile] += win
    return (out / norm)[:, :h, :w].cpu().numpy()


def save_figure(path, tl, gts, preds, scores, title, crop=1024):
    h, w = tl.shape
    cy, cx = max((h - crop) // 2, 0), max((w - crop) // 2, 0)
    sl = (slice(cy, cy + crop), slice(cx, cx + crop))
    orgs = list(gts)
    fig, ax = plt.subplots(2, len(orgs) + 1, figsize=(3.2 * (len(orgs) + 1), 6.6), squeeze=False)
    lo, hi = np.percentile(tl[sl], [1, 99.8])
    ax[0, 0].imshow(tl[sl], cmap="gray", vmin=lo, vmax=hi)
    ax[0, 0].set_title("transmitted light")
    ax[1, 0].axis("off")
    for j, org in enumerate(orgs, start=1):
        lo, hi = np.percentile(gts[org][sl], [1, 99.8])
        ax[0, j].imshow(gts[org][sl], cmap="magma", vmin=lo, vmax=hi)
        ax[0, j].set_title(f"{org} (ground truth)")
        ax[1, j].imshow(preds[org][sl], cmap="magma", vmin=lo, vmax=hi)
        s, p = scores[org]
        ax[1, j].set_title(f"predicted  SSIM {s:.2f}  PCC {p:.2f}")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=None, help="default: the --data used for training")
    p.add_argument("--studies", default="", help="default: the validation images of the run")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--n-figures", type=int, default=6)
    p.add_argument("--save-tiff", action="store_true", help="also write predictions as OME-TIFF")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    data_root = Path(args.data or targs["data"])
    target = targs.get("target", "best")
    suffix, pre = ("", "") if target == "best" else ("_mip", "mip_")
    out = Path(args.out) if args.out else ckpt_path.parent / "eval"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_index(data_root)
    if args.studies:
        df = df[df.study.isin([int(s) for s in args.studies.split(",")])]
    else:
        split = json.loads((ckpt_path.parent / "split.json").read_text())
        keep = {tuple(k) for k in split["val_images"]}
        df = df[[(s, i) in keep for s, i in zip(df.study, df.image_id)]]
    df = df.reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(targs, pretrained=False).to(device)  # weights come from the checkpoint
    model.load_state_dict(ckpt["model"])
    model.eval()
    # bf16 keeps the fp32 range (fp16 can overflow); plain fp32 on CPU or older GPUs
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    print(f"{model_name(targs)}, checkpoint epoch {ckpt['epoch']}, evaluating {len(df)} samples on {device}")

    # figures for the first image of every study, then the second, ... up to n_figures
    order = df.assign(k=df.groupby("study").cumcount()).sort_values(["k", "study"]).index
    fig_rows = set(order[: args.n_figures])
    records = []
    for i, r in df.iterrows():
        d = data_root / r.dir
        x = _norm(np.load(d / f"{r.tl}.npy"), r.tl_lo, r.tl_hi)
        pred = predict_full(model, x, device, args.tile, args.overlap, amp_dtype)
        gts, preds, scores = {}, {}, {}
        for c, org in enumerate(ORGANELLES):
            if args.save_tiff:
                tifffile.imwrite(out / f"S{r.study:02d}_image_{r.image_id}_{r.tl}_{org}{suffix}.ome.tiff",
                                 pred[c].astype(np.float32), ome=True, metadata={"axes": "YX"})
            if r[f"has_{pre}{org}"] != 1:
                continue
            gt = _norm(np.load(d / f"{org}{suffix}.npy"), r[f"lo_{pre}{org}"], r[f"hi_{pre}{org}"])
            s, c_ = ssim(gt, pred[c]), pcc(gt, pred[c])
            off = official_metrics(org, gt, pred[c])
            records.append(dict(study=r.study, image_id=r.image_id, tl=r.tl, organelle=org, ssim=s, pcc=c_, **off))
            gts[org], preds[org], scores[org] = gt, pred[c], (off["off_ssim"], off["off_pcc"])
        if i in fig_rows and gts:
            save_figure(out / "figures" / f"S{r.study:02d}_image_{r.image_id}_{r.tl}.png", x, gts, preds, scores,
                        f"Study {r.study}, image {r.image_id}, {r.tl}")
        print(f"  [{i + 1}/{len(df)}] S{r.study:02d} image {r.image_id} {r.tl} (official SSIM/PCC): "
              + "  ".join(f"{o} {s:.3f}/{c_:.3f}" for o, (s, c_) in scores.items()), flush=True)

    res = pd.DataFrame(records)
    res.to_csv(out / "metrics_per_image.csv", index=False)
    cols = [c for c in ("off_ssim", "off_pcc", "off_mae", "off_edist", "off_cdist", "ssim", "pcc") if c in res]
    by_org = res.groupby("organelle")[cols].agg(["mean", "std", "count"]).reindex(ORGANELLES).dropna(how="all")
    by_org_mod = res.groupby(["organelle", "tl"])[["off_ssim", "off_pcc", "ssim", "pcc"]].mean().round(3)
    by_org.to_csv(out / "summary_by_organelle.csv")
    by_org_mod.to_csv(out / "summary_by_organelle_modality.csv")
    summary = lmc_scores(res)
    summary.update(checkpoint=str(ckpt_path), epoch=ckpt["epoch"], model=model_name(targs))
    (out / "scores.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 200)
    print("\nMean over validation images (off_* = official metric):")
    print(by_org.xs("mean", axis=1, level=1).round(3).to_string())
    print("\nBy organelle and modality:")
    print(by_org_mod.to_string())
    print(f"\nLMC-score (official, study-macro) {summary['lmc_score']:.4f}   "
          f"image-mean {summary['lmc_score_image']:.4f}")
    print("per organelle:", summary["per_organelle_study_macro"])
    print(f"\nresults in {out}")


if __name__ == "__main__":
    main()
