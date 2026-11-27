"""Build the training/eval cache (PLAN.md section 3). Resumable: finished images are skipped.

For every usable image (TL stack with nZ > 1 and at least one organelle best_z) of every study on disk:

  <out>/S<study>/<image_id>/
    gt.npy             uint16 (4, H, W)  fluorescence plane at each organelle's own best_z
                                         (zeros where the organelle is missing or dropped)
    <TL>_k<K>.npy      uint16 (K, H, W)  TL stack resampled to K planes, linear interpolation along z,
                                         positions linspace(0, nZ-1, K) (uniform over the whole stack)
    <TL>_center.npy    uint16 (H, W)     TL plane linearly interpolated at the stack centre (nZ-1)/2
                                         (E2 single-plane input, no oracle)
    meta.json          written LAST (atomic rename) = completion marker

  <out>/index.csv      one row per (image, TL modality), rebuilt from every meta.json

Normalisation is NOT applied in the cache: raw uint16 values are stored, together with the
1-99.8 percentile range (lo, hi) of the whole resampled TL stack (one pair per stack, so intensity
changes across defocus are kept), of the centre plane, and of every GT plane.

Organelle best_z is stored as the original index and mapped to the resampled index
best_zr = best_z * (K-1) / (nZ-1) (float). The TL best_z from the TSV is stored as `tl_best_z_oracle`
for analysis only: it must never be used as a model input (PLAN.md section 1).

Labels with pixel-identical planes between two organelles (S03: Nucleus == Actin) are dropped for both.

Usage:
  python tools/build_cache.py                    # all studies on disk, K=16, paths from configs/paths.yaml
  python tools/build_cache.py --studies 3,72     # subset
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmc import CACHE_ROOT, DATASET_ROOT, META_DIR, ORGANELLES, PROJECT_ROOT, TL_MODALITIES  # noqa: E402
from lmc.tsv import read_all_files, read_best_focus, read_ome_meta, read_plane, robust_range  # noqa: E402

CACHE_VERSION = 1


def study_dirs(raw_root):
    out = {}
    for d in Path(raw_root).iterdir():
        m = re.fullmatch(r"Study_(\d+)", d.name)
        if m and d.is_dir():
            out[int(m.group(1))] = d
    return out


def resample_positions(nz, k):
    return np.linspace(0.0, nz - 1.0, k)


def interp_plane(planes, pos):
    """Linear interpolation between integer planes; planes: list/array indexable by z."""
    i0 = int(np.floor(pos))
    i1 = min(i0 + 1, len(planes) - 1)
    w = pos - i0
    if w < 1e-6 or i1 == i0:
        return np.asarray(planes[i0], np.float32)
    return (1.0 - w) * np.asarray(planes[i0], np.float32) + w * np.asarray(planes[i1], np.float32)


def to_u16(a):
    return np.clip(np.rint(a), 0, 65535).astype(np.uint16)


def atomic_save(path, arr):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def process_image(sid, img_id, files, best, src, out_root, k):
    img_dir = out_root / f"S{sid:02d}" / str(img_id)
    marker = img_dir / "meta.json"
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta.get("K") == k and meta.get("cache_version") == CACHE_VERSION:
            return meta, "cached"
    tls = [t for t in TL_MODALITIES if t in files]
    nz = max(len(z) for z in files.values())
    orgs = [o for o in ORGANELLES if o in files and best.get(o) is not None and best[o] in files[o]]
    if not tls:
        return None, "no TL"
    if nz <= 1:
        return None, "nZ=1"
    if not orgs:
        return None, "no organelle best_z"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ground truth planes
    gt, info, hashes, shape = None, {}, {}, None
    for c, org in enumerate(ORGANELLES):
        info[org] = dict(has=0, best_z=-1, best_zr=float("nan"), lo=float("nan"), hi=float("nan"), dropped="")
        if org not in orgs:
            continue
        a = read_plane(src / files[org][best[org]])
        if gt is None:
            shape = a.shape
            gt = np.zeros((len(ORGANELLES),) + shape, np.uint16)
        gt[c] = a
        lo, hi = robust_range(a)
        info[org].update(has=1, best_z=int(best[org]), best_zr=best[org] * (k - 1) / (nz - 1), lo=lo, hi=hi)
        hashes[org] = hashlib.md5(a.tobytes()).hexdigest()
    groups = {}
    for org, h in hashes.items():
        groups.setdefault(h, []).append(org)
    for same in groups.values():
        if len(same) > 1:
            for org in same:
                info[org].update(has=0, dropped="identical pixels: " + "==".join(same))
                gt[ORGANELLES.index(org)] = 0
    if not any(info[o]["has"] for o in ORGANELLES):
        return None, "all labels dropped"
    atomic_save(img_dir / "gt.npy", gt)

    # TL stacks
    tl_meta = {}
    for tl in tls:
        zs = files[tl]
        if sorted(zs) != list(range(len(zs))) or len(zs) != nz:
            raise ValueError(f"S{sid:02d} {img_id} {tl}: non-contiguous z {sorted(zs)[:5]}.. (nZ {nz})")
        planes = [read_plane(src / zs[z]) for z in range(nz)]
        if planes[0].shape != shape:
            raise ValueError(f"S{sid:02d} {img_id} {tl}: shape {planes[0].shape} != GT {shape}")
        stack = to_u16(np.stack([interp_plane(planes, p) for p in resample_positions(nz, k)]))
        center = to_u16(interp_plane(planes, (nz - 1) / 2.0))
        atomic_save(img_dir / f"{tl}_k{k}.npy", stack)
        atomic_save(img_dir / f"{tl}_center.npy", center)
        lo, hi = robust_range(stack)
        clo, chi = robust_range(center)
        ome = read_ome_meta(src / zs[0])
        tl_meta[tl] = dict(lo=lo, hi=hi, center_lo=clo, center_hi=chi, tl_best_z_oracle=best.get(tl, -1), **ome)

    meta = dict(cache_version=CACHE_VERSION, K=k, study=sid, image_id=img_id, dir=f"S{sid:02d}/{img_id}",
                H=int(shape[0]), W=int(shape[1]), nZ=int(nz), organelles=info, tl=tl_meta)
    tmp = img_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, marker)
    return meta, "built"


def index_rows(meta):
    rows = []
    for tl, t in meta["tl"].items():
        r = dict(study=meta["study"], image_id=meta["image_id"], tl=tl, dir=meta["dir"], H=meta["H"], W=meta["W"],
                 nZ=meta["nZ"], K=meta["K"], px_um=t["px_um"], dz_um=t["dz_um"], mag=t["mag"], na=t["na"],
                 tl_lo=t["lo"], tl_hi=t["hi"], center_lo=t["center_lo"], center_hi=t["center_hi"],
                 tl_best_z_oracle=t["tl_best_z_oracle"])
        for org, o in meta["organelles"].items():
            r.update({f"has_{org}": o["has"], f"best_z_{org}": o["best_z"], f"best_zr_{org}": o["best_zr"],
                      f"lo_{org}": o["lo"], f"hi_{org}": o["hi"]})
        rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=str(DATASET_ROOT))
    p.add_argument("--meta", default=str(META_DIR))
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--out", default=None, help="default: cache_root of configs/paths.yaml (for K=16)")
    p.add_argument("--studies", default="", help="comma-separated ids; default: every study on disk")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if not args.out and args.k != 16:
        raise SystemExit("--k other than 16: give --out explicitly (cache_root in configs/paths.yaml is the K=16 cache)")
    out_root = Path(args.out) if args.out else CACHE_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    dirs = study_dirs(args.raw)
    wanted = [int(s) for s in args.studies.split(",") if s.strip()] or sorted(dirs)
    t_all = time.time()
    skipped = []
    for sid in wanted:
        if sid not in dirs:
            print(f"S{sid:02d}: not on disk, skipped")
            continue
        t0 = time.time()
        files = read_all_files(Path(args.meta) / f"Study_{sid:02d}_all_files.tsv")
        best = read_best_focus(Path(args.meta) / f"Study_{sid:02d}_best_focus.tsv")
        ids = sorted(files)
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda i: process_image(sid, i, files[i], best.get(i, {}), dirs[sid], out_root, args.k),
                              ids))
        stat = {}
        for i, (meta, s) in zip(ids, res):
            stat[s] = stat.get(s, 0) + 1
            if meta is None:
                skipped.append((sid, i, s))
        print(f"S{sid:02d}: {len(ids)} images {stat}  {time.time() - t0:.0f}s", flush=True)

    rows = []
    for f in sorted(out_root.glob("S*/*/meta.json")):
        meta = json.loads(f.read_text())
        if meta.get("K") == args.k:
            rows.extend(index_rows(meta))
    df = pd.DataFrame(rows).sort_values(["study", "image_id", "tl"])
    df.to_csv(out_root / "index.csv", index=False)
    (out_root / "skipped.json").write_text(json.dumps(skipped, indent=0))
    print(f"\nindex.csv: {len(df)} rows ({df[['study', 'image_id']].drop_duplicates().shape[0]} images, "
          f"{df.study.nunique()} studies) in {(time.time() - t_all) / 60:.1f} min -> {out_root}")
    print(f"skipped images: {len(skipped)} (see skipped.json)")


if __name__ == "__main__":
    main()
