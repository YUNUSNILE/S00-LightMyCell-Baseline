"""Build the Task 2 (MIP) targets inside the K=16 cache (S01 v2). Resumable: finished images are skipped.

Task 2 of Light My Cells 2 asks for the fluorescence maximum-intensity projection (MIP) of each organelle from the
transmitted-light z-stack (course brief, slides 4-5: "We'll evaluate only the MIP prediction"). For every image in
<cache>/index.csv and every organelle labelled in its meta.json (has == 1, so the S03 Nucleus == Actin duplicates stay
dropped), this reads ALL fluorescence z-planes of that organelle from the raw dataset and takes the pixel-wise maximum.

  <cache>/S<study>/<image_id>/mip.npy    uint16 (4, H, W), zeros where the organelle has no label
  <cache>/S<study>/<image_id>/mip.json   written LAST (atomic rename) = completion marker: has / lo / hi / n_planes
  <cache>/mip_index.csv                  one row per image: has_mip_<org>, lo_mip_<org>, hi_mip_<org>, nz_mip_<org>

lo / hi are the 1 / 99.8 percentiles, as for the best-focus targets (lmc.tsv.robust_range). index.csv, meta.json and
the cache version are NOT touched, so the K=16 cache and every run made on it stay valid. Test studies are built
(arrays only) but nothing here evaluates them.

Built-in check: the best-focus plane is one of the z-planes, so mip >= gt pixel-wise wherever a label exists.

Usage:
  python tools/build_mip_cache.py              # every image of the cache index, paths from configs/paths.yaml
  python tools/build_mip_cache.py --studies 72,81 --workers 2
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmc import CACHE_ROOT, DATASET_ROOT, META_DIR, ORGANELLES  # noqa: E402
from lmc.tsv import read_all_files, read_plane, robust_range  # noqa: E402

MIP_VERSION = 1


def atomic_save(path, arr):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def process_image(cache, raw_dir, sid, img_id, rel, files):
    d = cache / rel
    marker = d / "mip.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        if info.get("mip_version") == MIP_VERSION:
            return info, "cached"
    meta = json.loads((d / "meta.json").read_text())
    h, w = int(meta["H"]), int(meta["W"])
    gt = np.load(d / "gt.npy", mmap_mode="r")
    mip = np.zeros((len(ORGANELLES), h, w), np.uint16)
    out = {"mip_version": MIP_VERSION, "study": sid, "image_id": img_id, "organelles": {}}
    for c, org in enumerate(ORGANELLES):
        o = dict(has=0, lo=float("nan"), hi=float("nan"), n_planes=0)
        if meta["organelles"][org]["has"] == 1:
            zs = files[img_id][org]
            acc = None
            for z in sorted(zs):
                a = read_plane(raw_dir / zs[z])
                if a.shape != (h, w):
                    raise ValueError(f"S{sid:02d} {img_id} {org} z{z}: shape {a.shape} != {(h, w)}")
                acc = a.copy() if acc is None else np.maximum(acc, a, out=acc)
            if acc.dtype != np.uint16:
                acc = np.clip(acc, 0, 65535).astype(np.uint16)
            if not (acc >= gt[c]).all():
                raise ValueError(f"S{sid:02d} {img_id} {org}: MIP below the best-focus plane somewhere")
            mip[c] = acc
            lo, hi = robust_range(acc)
            o.update(has=1, lo=lo, hi=hi, n_planes=len(zs))
        out["organelles"][org] = o
    atomic_save(d / "mip.npy", mip)
    tmp = d / "mip.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, marker)
    return out, "built"


def index_row(info):
    r = {"study": info["study"], "image_id": info["image_id"]}
    for org, o in info["organelles"].items():
        r.update({f"has_mip_{org}": o["has"], f"lo_mip_{org}": o["lo"], f"hi_mip_{org}": o["hi"],
                  f"nz_mip_{org}": o["n_planes"]})
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(CACHE_ROOT))
    p.add_argument("--raw", default=str(DATASET_ROOT))
    p.add_argument("--meta", default=str(META_DIR))
    p.add_argument("--studies", default="", help="comma-separated ids; default: every study in the cache index")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    cache = Path(a.cache)
    idx = pd.read_csv(cache / "index.csv")[["study", "image_id", "dir"]].drop_duplicates(["study", "image_id"])
    wanted = [int(s) for s in a.studies.split(",") if s.strip()] or sorted(idx.study.unique())
    t_all, stat = time.time(), {}
    for sid in wanted:
        sid = int(sid)  # numpy int64 from the index is not JSON serialisable
        t0 = time.time()
        files = read_all_files(Path(a.meta) / f"Study_{sid:02d}_all_files.tsv")
        sub = idx[idx.study == sid]
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda r: process_image(cache, Path(a.raw) / f"Study_{sid:02d}", sid, int(r.image_id),
                                                      r.dir, files), sub.itertuples(index=False)))
        s = {}
        for _, st in res:
            s[st] = s.get(st, 0) + 1
            stat[st] = stat.get(st, 0) + 1
        print(f"S{sid:02d}: {len(sub)} images {s}  {time.time() - t0:.0f}s", flush=True)

    rows = []
    for r in idx.itertuples(index=False):
        f = cache / r.dir / "mip.json"
        if f.exists():
            rows.append(index_row(json.loads(f.read_text())))
    out = pd.DataFrame(rows).sort_values(["study", "image_id"])
    out.to_csv(cache / "mip_index.csv", index=False)
    print(f"\nmip_index.csv: {len(out)} images of {len(idx)} in the cache index "
          f"({(time.time() - t_all) / 60:.1f} min, {stat}) -> {cache}")
    print("MIP labels:", {o: int(out[f"has_mip_{o}"].sum()) for o in ORGANELLES})


if __name__ == "__main__":
    main()
