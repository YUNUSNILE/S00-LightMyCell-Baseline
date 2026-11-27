"""Full-image tiled inference and official-metric evaluation of a model on a set of cache rows.

`predict_full` is adapted from baseline_teammate/evaluate.py (predict_full: sliding window with Hann
blending; github.com/YUNUSNILE/S00-LightMyCell-Baseline, commit fc2ab9a). Changes: C input channels,
plane positions (zpos) and metadata passed to the model, fp32 (fp16/bf16 are slower on the GTX 1650),
padding to the model's divisor, optional attention read-out for z-attention models.
Metrics run in worker processes while the GPU predicts the next image.

Attention logging (models with `has_attention`): the per-organelle attention over z, averaged over all
pixels of all tiles, is compared with the organelle's resampled best_z AFTER prediction (best_z never
reaches the model): peak error |argmax - best_zr| in planes and in um (x resampled plane spacing).
"""
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import tifffile
import torch

from lmc import ORGANELLES
from lmc.data import load_input, load_targets, meta_vector, plane_positions
from lmc.metrics import breakdown, lmc_scores, official_metrics

SECONDARY_EXCLUDE = (20,)  # secondary LMC-score without Study 20 (reported only; selection uses all of Val)


@torch.no_grad()
def predict_full(model, img, device, tile=512, overlap=128, batch=2, divisor=32, zpos=None, meta=None,
                 return_attention=False, tta=False):
    """img (C, H, W) float32 -> prediction (C_out, H, W) float32 (C_out = 4 per predicted task; 4 before S01 v2).
    tta: average over the 8 dihedral transforms of each tile (no attention read-out then).
    return_attention: also return the pixel-averaged attention profile (4, K) and, if model.keep_maps,
    the per-pixel expected plane index map (4, H, W)."""
    c, h, w = img.shape
    ph = max(tile, -(-h // divisor) * divisor)
    pw = max(tile, -(-w // divisor) * divisor)
    x = np.pad(img, ((0, 0), (0, ph - h), (0, pw - w)), mode="reflect")
    stride = tile - overlap

    def starts(n):
        s = list(range(0, n - tile + 1, stride))
        return s if s[-1] == n - tile else s + [n - tile]

    pos = [(y, z) for y in starts(ph) for z in starts(pw)]
    win = torch.from_numpy(np.clip(np.outer(np.hanning(tile), np.hanning(tile)), 1e-3, None)).float().to(device)
    out = None  # (C_out, ph, pw); C_out = 4 per task (S01 v2), allocated from the first prediction
    wsum = torch.zeros(1, ph, pw, device=device)
    att = getattr(model, "has_attention", False) and return_attention and not tta
    prof = torch.zeros(len(ORGANELLES), c, device=device) if att else None
    keep = att and getattr(model, "keep_maps", False)
    ez = torch.zeros(len(ORGANELLES), ph, pw, device=device) if keep else None
    zt = None if zpos is None else torch.as_tensor(zpos, dtype=torch.float32, device=device)
    mt = None if meta is None else torch.as_tensor(meta, dtype=torch.float32, device=device)
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        xb = torch.from_numpy(np.stack([x[:, y:y + tile, z:z + tile] for y, z in chunk])).float().to(device)
        n = len(chunk)
        zb, mb = (None if zt is None else zt.expand(n, -1)), (None if mt is None else mt.expand(n, -1))
        if tta:  # S01 v2: mean over the 8 dihedral transforms (Moshkov et al. 2020), each mapped back
            pb = sum(_undihedral(model(_dihedral(xb, kk, fl), zb, mb).float(), kk, fl)
                     for kk in range(4) for fl in (False, True)) / 8.0
        else:
            pb = model(xb, zb, mb).float()
        if out is None:
            out = torch.zeros(pb.shape[1], ph, pw, device=device)
        if att:
            prof += model.aux["attn_mean"].float().sum(0)
            if keep:
                zi = torch.arange(c, device=device, dtype=torch.float32)[None, :, None, None, None]
                e = (model.last_maps * zi).sum(1)  # (n, 4, h', w') expected plane index
                if e.shape[-1] != tile:
                    e = torch.nn.functional.interpolate(e, size=(tile, tile), mode="bilinear", align_corners=False)
        for j, ((y, z), p) in enumerate(zip(chunk, pb)):
            out[:, y:y + tile, z:z + tile] += p * win
            wsum[:, y:y + tile, z:z + tile] += win
            if keep:
                ez[:, y:y + tile, z:z + tile] += e[j] * win
    pred = (out / wsum)[:, :h, :w].cpu().numpy()
    if not return_attention:
        return pred
    profile = (prof / len(pos)).cpu().numpy() if att else None
    ezmap = (ez / wsum)[:, :h, :w].cpu().numpy() if keep else None
    return pred, profile, ezmap


def _dihedral(t, k, flip):
    t = torch.rot90(t, k, (2, 3))
    return torch.flip(t, (3,)) if flip else t


def _undihedral(t, k, flip):
    if flip:
        t = torch.flip(t, (3,))
    return torch.rot90(t, -k, (2, 3))


def _score_image(meta, pred, y, m, extra, tasks=("best",), tgt_tasks=None, t2_baseline=False):
    """Official metrics for one image (runs in a worker process; meta is a plain dict).

    S01 v2: pred holds 4 channels per predicted task (`tasks` order), y/m 4 per target task (`tgt_tasks` order).
    Every record carries `task` ('best' = Task 1, 'mip' = Task 2). t2_baseline adds 'mip_from_best': the Task 1
    prediction scored against the MIP ground truth (Task 2 baseline without any MIP training)."""
    tasks = list(tasks)
    tgt_tasks = list(tgt_tasks or tasks)
    pairs = [(t, tasks.index(t), tgt_tasks.index(t)) for t in tasks]  # (name, pred block, GT block)
    if t2_baseline and "best" in tasks and "mip" in tgt_tasks and "mip" not in tasks:
        pairs.append(("mip_from_best", tasks.index("best"), tgt_tasks.index("mip")))
    recs = []
    for name, pi, gi in pairs:
        for c, org in enumerate(ORGANELLES):
            if m[4 * gi + c] > 0:
                ex = extra.get(org, {}) if name == "best" else {}
                recs.append(dict(**meta, task=name, organelle=org,
                                 **official_metrics(org, y[4 * gi + c], pred[4 * pi + c]), **ex))
    return recs


def _task_summary(res):
    """LMC-score summary of the records of ONE task (study macro, image mean, without S20, breakdown)."""
    s = lmc_scores(res)
    s["breakdown"] = breakdown(res)
    sub = res[~res.study.isin(SECONDARY_EXCLUDE)]
    s["lmc_score_excl"] = lmc_scores(sub)["lmc_score"] if len(sub) else float("nan")
    s["excluded_studies"] = list(SECONDARY_EXCLUDE)
    return s


def attention_errors(r, profile, k):
    """Per organelle: attention peak (argmax over z) vs resampled best_z, in planes and um."""
    out = {}
    if profile is None:
        return out
    spacing = float(r.dz_um) * (float(r.nZ) - 1.0) / (k - 1.0)
    for c, org in enumerate(ORGANELLES):
        bz = getattr(r, f"best_zr_{org}")
        if getattr(r, f"has_{org}") == 1 and np.isfinite(bz):
            peak = int(np.argmax(profile[c]))
            out[org] = dict(att_peak=peak, att_err_planes=abs(peak - bz), att_err_um=abs(peak - bz) * spacing)
    return out


def evaluate_rows(model, rows, cache_root, mode, k, device, tile=512, overlap=128, batch=2,
                  save_dir=None, log=print, metric_workers=4, max_inflight=6, tasks=("best",), t2_baseline=False,
                  tta=False):
    """Evaluate on full images. Returns (records DataFrame, summary dict).

    Pipeline: a thread pre-loads the next image from disk while the GPU predicts the current one;
    official metrics run in `metric_workers` processes (scikit-image SSIM holds the GIL), with at
    most `max_inflight` images queued to bound memory.

    S01 v2: `tasks` = what the model predicts (['best'] or ['best', 'mip']); the summary's top-level scores are
    Task 1 ('best', used for best.pt) when predicted, and summary['tasks'][name] holds the other tasks
    ('mip' = Task 2; 'mip_from_best' when t2_baseline). tta = 8x dihedral test-time augmentation.
    """
    model.eval()
    divisor = getattr(model, "divisor", 32)
    t0 = time.time()
    items = list(rows.itertuples(index=False))
    tasks = list(tasks)
    tgt_tasks = tasks + (["mip"] if t2_baseline and "mip" not in tasks else [])
    # every queued image holds 4 prediction + 4 target channels per task (~270 MB at 2048^2 for two tasks):
    # the first E1e validation ran out of RAM (16 GB laptop) with 6 two-task images in flight
    max_inflight = max(2, max_inflight // len(tgt_tasks))

    def load(r):
        x = load_input(r, cache_root, mode, k)
        y, m = load_targets(r, cache_root, tgt_tasks)
        return x, y, m

    futures, records = [], []
    with ProcessPoolExecutor(metric_workers) as pool, ThreadPoolExecutor(1) as io:
        nxt = io.submit(load, items[0]) if items else None
        for j, r in enumerate(items):
            x, y, m = nxt.result()
            nxt = io.submit(load, items[j + 1]) if j + 1 < len(items) else None
            pred, profile, _ = predict_full(model, x, device, tile, overlap, batch, divisor,
                                            plane_positions(r, mode, k), meta_vector(r), return_attention=True,
                                            tta=tta)
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                for ti, task in enumerate(tasks):
                    suffix = "" if task == "best" else "_MIP"
                    for c, org in enumerate(ORGANELLES):
                        tifffile.imwrite(save_dir / f"S{r.study:02d}_image_{r.image_id}_{r.tl}_{org}{suffix}.ome.tiff",
                                         pred[4 * ti + c].astype(np.float32), ome=True, metadata={"axes": "YX"})
            meta = dict(study=int(r.study), image_id=int(r.image_id), tl=str(r.tl))
            futures.append(pool.submit(_score_image, meta, pred, y, m, attention_errors(r, profile, k),
                                       tasks, tgt_tasks, t2_baseline))
            while sum(not f.done() for f in futures) >= max_inflight:
                time.sleep(0.05)
        records = [rec for f in futures for rec in f.result()]
    res = pd.DataFrame(records)
    main = "best" if "best" in tasks else tasks[0]
    summary = _task_summary(res[res.task == main])
    summary["task"] = main
    summary["tasks"] = {t: _task_summary(res[res.task == t]) for t in res.task.unique() if t != main}
    if "att_err_planes" in res:
        g = res[res.task == "best"].dropna(subset=["att_err_planes"]).groupby("organelle")
        summary["attention"] = {o: {"err_planes": round(float(v.att_err_planes.mean()), 3),
                                    "err_um": round(float(v.att_err_um.mean()), 3),
                                    "within_1_plane": round(float((v.att_err_planes <= 1).mean()), 3), "n": int(len(v))}
                                for o, v in g}
    summary["seconds"] = round(time.time() - t0, 1)
    summary["n_samples"] = int(len(rows))
    return res, summary
