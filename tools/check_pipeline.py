"""Non-training checks for E0 (PLAN.md section 6.3). Prints PASS/FAIL per check.

  python tools/check_pipeline.py                 # all checks
  python tools/check_pipeline.py --only metrics,resample,cache,split,oracle,loader,models
  --loader-samples N   number of training patches to time (default 48)
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmc import CACHE_ROOT, META_DIR, ORGANELLES, PROJECT_ROOT, TL_MODALITIES  # noqa: E402
from lmc.metrics import lmc_scores, official_metrics  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def t_metrics():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:256, :256]
    gt = np.exp(-((yy - 100) ** 2 + (xx - 140) ** 2) / 800.0) * 1000 + rng.normal(0, 5, (256, 256)) + 200
    m = official_metrics("Nucleus", gt, gt)
    check("metric: identical images -> SSIM 1, PCC 1", abs(m["off_ssim"] - 1) < 1e-9 and abs(m["off_pcc"] - 1) < 1e-9)
    m2 = official_metrics("Nucleus", gt, 3.7 * gt + 55)
    check("metric: invariant to affine rescaling of the prediction",
          abs(m2["off_ssim"] - 1) < 1e-6 and abs(m2["off_pcc"] - 1) < 1e-9, f"{m2['off_ssim']:.6f}/{m2['off_pcc']:.6f}")
    m3 = official_metrics("Actin", gt, rng.normal(size=gt.shape))
    check("metric: random prediction -> PCC ~ 0, low SSIM", abs(m3["off_pcc"]) < 0.05 and m3["off_ssim"] < 0.2,
          f"SSIM {m3['off_ssim']:.3f} PCC {m3['off_pcc']:.3f}")
    check("metric: MAE/Euclid/cosine only for Nucleus & Mitochondria", "off_mae" in m and "off_mae" not in m3)
    # LMC-score: study macro. Nucleus: study 1 images (1.0, 0.0) -> 0.5; study 2 (1.0) -> 1.0 => 0.75
    res = pd.DataFrame([dict(organelle="Nucleus", study=1, off_ssim=1, off_pcc=1),
                        dict(organelle="Nucleus", study=1, off_ssim=0, off_pcc=0),
                        dict(organelle="Nucleus", study=2, off_ssim=1, off_pcc=1),
                        dict(organelle="Actin", study=1, off_ssim=0.5, off_pcc=0.3)])
    s = lmc_scores(res)
    check("LMC-score: study-macro then organelle mean", abs(s["lmc_score"] - (0.75 + 0.4) / 2) < 1e-12
          and abs(s["lmc_score_image"] - ((2 / 3) + 0.4) / 2) < 1e-12, f"{s['lmc_score']:.4f} / {s['lmc_score_image']:.4f}")


def t_resample():
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from build_cache import interp_plane, resample_positions

    for nz in (4, 9, 16, 21, 37):
        planes = [np.full((2, 2), 100.0 * z) for z in range(nz)]  # intensity linear in z
        pos = resample_positions(nz, 16)
        got = np.array([interp_plane(planes, p)[0, 0] for p in pos])
        ok = np.allclose(got, 100.0 * pos) and pos[0] == 0 and abs(pos[-1] - (nz - 1)) < 1e-9
        check(f"resample: K=16 from nZ={nz} is exact linear interpolation, ends included", ok)


def t_cache(cache):
    from lmc.tsv import read_all_files, read_best_focus

    idx = pd.read_csv(cache / "index.csv")
    expected = []
    for sid in sorted(idx.study.unique()):
        files = read_all_files(META_DIR / f"Study_{sid:02d}_all_files.tsv")
        best = read_best_focus(META_DIR / f"Study_{sid:02d}_best_focus.tsv")
        for i, ch in files.items():
            nz = max(len(z) for z in ch.values())
            orgs = [o for o in ORGANELLES if o in ch and best.get(i, {}).get(o) is not None]
            for tl in TL_MODALITIES:
                if tl in ch and nz > 1 and orgs:
                    expected.append((sid, i, tl))
    got = set(zip(idx.study, idx.image_id, idx.tl))
    # rows whose only labels were duplicates are legitimately absent
    check("cache: every usable (image, TL) of the TSVs is in the index", set(expected) - got == set(),
          f"{len(got)} rows, {idx[['study', 'image_id']].drop_duplicates().shape[0]} images, "
          f"{idx.study.nunique()} studies; missing {sorted(set(expected) - got)[:5]}")
    check("cache: 28 studies", idx.study.nunique() == 28, str(sorted(idx.study.unique())))
    s03 = idx[(idx.study == 3) & idx.image_id.isin([53, 55, 57])]
    check("cache: S03 duplicate Nucleus/Actin labels dropped", len(s03) == 3 and s03.has_Nucleus.sum() == 0
          and s03.has_Actin.sum() == 0 and s03.has_Mitochondria.sum() == 3)
    bz = np.concatenate([idx.loc[idx[f"has_{o}"] == 1, f"best_zr_{o}"].to_numpy() for o in ORGANELLES])
    check("cache: resampled best_z within [0, K-1]", (bz >= 0).all() and (bz <= idx.K.max() - 1 + 1e-9).all(),
          f"min {bz.min():.2f} max {bz.max():.2f}")
    check("cache: lo < hi for every TL stack and label",
          (idx.tl_hi > idx.tl_lo).all() and all((idx.loc[idx[f'has_{o}'] == 1, f'hi_{o}'] >
                                                  idx.loc[idx[f'has_{o}'] == 1, f'lo_{o}']).all() for o in ORGANELLES))
    rng = np.random.default_rng(0)
    bad = []
    for r in idx.iloc[rng.choice(len(idx), min(25, len(idx)), replace=False)].itertuples():
        d = cache / r.dir
        a = np.load(d / f"{r.tl}_k{r.K}.npy", mmap_mode="r")
        g = np.load(d / "gt.npy", mmap_mode="r")
        c = np.load(d / f"{r.tl}_center.npy", mmap_mode="r")
        if a.shape != (r.K, r.H, r.W) or g.shape != (4, r.H, r.W) or c.shape != (r.H, r.W) or a.dtype != np.uint16:
            bad.append(r.dir)
        for k, o in enumerate(ORGANELLES):
            if (getattr(r, f"has_{o}") == 1) != bool(g[k].any()):
                bad.append(f"{r.dir}:{o}")
    check("cache: shapes/dtypes and GT-vs-mask consistent (25 random rows)", not bad, str(bad[:5]))


def t_split(cache):
    import json

    sp = json.loads((PROJECT_ROOT / "splits/split_v1.json").read_text())
    st = sp["studies"]
    allst = st["train"] + st["val"] + st["test"]
    check("split: studies disjoint and = 28", len(allst) == len(set(allst)) == 28)
    idx = pd.read_csv(cache / "index.csv")
    n = {k: len(v) for k, v in sp["samples"].items()}
    check("split: samples cover the cache index exactly", sum(n.values()) == len(idx), str(n))
    for k, v in sp["samples"].items():
        check(f"split: {k} samples only from {k} studies", {s for s, _, _ in v} <= set(st[k]))


def t_oracle():
    """The TL best_z from the TSV must never reach a model input."""
    hits = []
    for f in [PROJECT_ROOT / "train.py", PROJECT_ROOT / "evaluate.py"] + list((PROJECT_ROOT / "lmc").glob("*.py")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"tl_best_z|best_z_oracle|BF_best_z|PC_best_z|DIC_best_z", line):
                hits.append(f"{f.name}:{n}")
    check("no-oracle: TL best_z not referenced by data/model/train/eval code", not hits, str(hits))


def t_loader(cache, n_samples):
    from torch.utils.data import DataLoader

    from lmc.data import PatchDataset, load_split, split_rows, study_weights

    idx = pd.read_csv(cache / "index.csv")
    tr = split_rows(idx, load_split(PROJECT_ROOT / "splits/split_v1.json"), "train")
    w = study_weights(tr, "sqrt")
    share = pd.Series(w).groupby(tr.study.values).sum() / w.sum()
    check("sampler: sqrt study weights", True, f"S25 share {share.get(25, 0):.1%}, smallest study {share.min():.1%}")
    ds = PatchDataset(tr, cache, "stack", 16, 512, True)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w), n_samples, replacement=True,
                                                     generator=torch.Generator().manual_seed(0))
    dl = DataLoader(ds, 2, sampler=sampler, num_workers=3, persistent_workers=False)
    t0 = time.time()
    ok, first = True, None
    for b in dl:
        if first is None:
            first = time.time() - t0
            print(f"    batch shapes x {tuple(b['x'].shape)} y {tuple(b['y'].shape)} m {tuple(b['m'].shape)}")
        ok &= bool(torch.isfinite(b["x"]).all() and torch.isfinite(b["y"]).all())
        ok &= bool(((b["y"].abs().sum((2, 3)) > 0).float() >= b["m"]).all())  # labelled channels not empty
    dt_ = time.time() - t0 - first
    rate = (n_samples - 2) / max(dt_, 1e-9)
    check("loader: finite tensors, masks match targets", ok)
    check("loader: throughput (3 workers, 512^2 x 16 planes)", True,
          f"{rate:.1f} patches/s after warm-up ({first:.1f}s first batch); training needs ~8 patches/s "
          f"(2 per 0.25 s micro-step)")


def t_models():
    from lmc.evaluation import predict_full
    from lmc.models import build_model, count_params

    dev = torch.device("cuda")
    # S01 v2 adds the ConvNeXt-T encoder and the Task 1 + Task 2 (8-channel) head
    for bb, tasks in (("unet", ["best"]), ("smp_resnet34", ["best"]), ("runet", ["best"]),
                      ("smp_convnext_tiny", ["best"]), ("smp_convnext_tiny", ["best", "mip"])):
        cfg = {"data": {"in_channels": 16, "tasks": tasks},
               "model": {"backbone": bb, "base": 32, "depth": 4, "encoder_weights": "imagenet"}}
        m = build_model(cfg).to(dev).eval()
        c = 4 * len(tasks)
        with torch.no_grad():
            y = m(torch.randn(1, 16, 512, 512, device=dev))
            p = predict_full(m, np.random.rand(16, 700, 1100).astype(np.float32), dev, divisor=m.divisor)
        check(f"model {bb} {tasks}: (1,16,512,512)->{tuple(y.shape)}; full image 700x1100 -> {p.shape}",
              tuple(y.shape) == (1, c, 512, 512) and p.shape == (c, 700, 1100) and np.isfinite(p).all(),
              f"{count_params(m) / 1e6:.2f}M params")
        del m
        torch.cuda.empty_cache()


def t_mip(cache):
    """S01 v2: Task 2 targets built by tools/build_mip_cache.py."""
    f = cache / "mip_index.csv"
    if not f.exists():
        check("mip: mip_index.csv exists", False, "run tools/build_mip_cache.py")
        return
    from lmc.data import load_index

    idx = pd.read_csv(cache / "index.csv")
    mip = pd.read_csv(f)
    imgs = idx[["study", "image_id"]].drop_duplicates()
    check("mip: one row per cached image", len(mip) == len(imgs) and not mip.duplicated(["study", "image_id"]).any(),
          f"{len(mip)} rows, {len(imgs)} images")
    j = load_index(cache)
    same = all(((j[f"has_{o}"] == 1) == (j[f"has_mip_{o}"] == 1)).all() for o in ORGANELLES)
    check("mip: a MIP label exactly where a best-focus label exists", same,
          str({o: int(j[f"has_mip_{o}"].sum()) for o in ORGANELLES}))
    rng = np.random.default_rng(0)
    bad = []
    for r in j.iloc[rng.choice(len(j), min(20, len(j)), replace=False)].itertuples():
        g = np.load(cache / r.dir / "gt.npy", mmap_mode="r")
        mp = np.load(cache / r.dir / "mip.npy", mmap_mode="r")
        if mp.shape != g.shape or mp.dtype != np.uint16:
            bad.append(f"{r.dir}: shape/dtype")
        for k, o in enumerate(ORGANELLES):
            if getattr(r, f"has_mip_{o}") == 1 and not (np.asarray(mp[k]) >= np.asarray(g[k])).all():
                bad.append(f"{r.dir}:{o} mip<gt")
    check("mip: shapes, dtype, MIP >= best-focus plane (20 random rows)", not bad, str(bad[:5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="metrics,resample,cache,split,oracle,loader,models,mip")
    ap.add_argument("--cache", default=str(CACHE_ROOT))
    ap.add_argument("--loader-samples", type=int, default=48)
    a = ap.parse_args()
    cache = PROJECT_ROOT / a.cache
    todo = a.only.split(",")
    for name, fn in (("metrics", t_metrics), ("resample", t_resample), ("cache", lambda: t_cache(cache)),
                     ("split", lambda: t_split(cache)), ("oracle", t_oracle),
                     ("loader", lambda: t_loader(cache, a.loader_samples)), ("models", t_models),
                     ("mip", lambda: t_mip(cache))):
        if name in todo:
            print(f"--- {name}")
            fn()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
