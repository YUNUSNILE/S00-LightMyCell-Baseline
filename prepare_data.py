"""Convert raw Light My Cells 2 studies into .npy arrays plus one index CSV.

Per image it writes to <out>/S<study>/<image_id>/:
  <TL>.npy             transmitted-light plane at best focus (BF / PC / DIC)
  <TL>_stack.npy       TL planes within +-K of best focus (--tl-neighbors K)
  <Organelle>.npy      fluorescence plane at best focus       (Task 1 target)
  <Organelle>_mip.npy  max-intensity projection over all z    (Task 2 target)
  rows.json            index rows for this image (lets reruns skip it)

<out>/index.csv gets one row per (image, TL modality), rebuilt from every
rows.json under <out>, so studies can be added one at a time.

Example:
  python prepare_data.py --raw Dataset --studies 1,2,3
"""
import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from lmc import ORGANELLES, TL_MODALITIES
from lmc.io import read_all_files, read_best_focus, read_plane, read_plane_and_meta, robust_range

DEFAULT_META = "Dataset/Study_info_filenames_and_optimal_focus_V3/Study_info_filenames_and_optimal focus"


def find_study_dirs(raw_root):
    """study id -> folder that directly contains its .ome.tiff files."""
    found = {}
    for d in Path(raw_root).rglob("Study_*"):
        m = re.fullmatch(r"Study_(\d+)", d.name)
        if m and d.is_dir() and next(d.glob("*.ome.tiff"), None) is not None:
            found[int(m.group(1))] = d
    return found


def pick_z(best_z, zs):
    if len(zs) == 1:
        return next(iter(zs))
    if best_z is None:
        return None
    return best_z if best_z in zs else min(zs, key=lambda z: abs(z - best_z))


def process_image(sid, img_id, files, best, src, out_root, k_neighbors, do_mip, overwrite):
    img_dir = out_root / f"S{sid:02d}" / str(img_id)
    marker = img_dir / "rows.json"
    if marker.exists() and not overwrite:
        return json.loads(marker.read_text())
    img_dir.mkdir(parents=True, exist_ok=True)
    rel = f"S{sid:02d}/{img_id}"

    fluo, shape = {}, None
    plane_hash, mip_hash = {}, {}
    for org in ORGANELLES:
        zs = files.get(org, {})
        info = dict(has=0, z=-1, lo=np.nan, hi=np.nan, has_mip=0, lo_mip=np.nan, hi_mip=np.nan, nz=len(zs))
        try:
            z = pick_z(best.get(org), zs) if zs else None
            if z is not None:
                a = read_plane(src / zs[z])
                np.save(img_dir / f"{org}.npy", a)
                lo, hi = robust_range(a)
                info.update(has=1, z=z, lo=lo, hi=hi)
                shape = shape or a.shape
                plane_hash[org] = hashlib.md5(a.tobytes()).hexdigest()
            if do_mip and len(zs) > 1:
                mip = None
                for zz in sorted(zs):
                    a = read_plane(src / zs[zz])
                    mip = a if mip is None else np.maximum(mip, a)
                np.save(img_dir / f"{org}_mip.npy", mip)
                lo, hi = robust_range(mip)
                info.update(has_mip=1, lo_mip=lo, hi_mip=hi)
                shape = shape or mip.shape
                mip_hash[org] = hashlib.md5(mip.tobytes()).hexdigest()
        except FileNotFoundError as e:
            print(f"  [warn] S{sid:02d} image {img_id} {org}: missing file {e.filename}")
            info.update(has=0, has_mip=0)
        fluo[org] = info

    # Some released files hold identical pixels under two organelle names
    # (Study 3: Nucleus == Actin for 4 images). Such labels can't be trusted: drop both.
    for hashes, key in ((plane_hash, "has"), (mip_hash, "has_mip")):
        groups = {}
        for org, digest in hashes.items():
            groups.setdefault(digest, []).append(org)
        for orgs in groups.values():
            if len(orgs) > 1:
                print(f"  [warn] S{sid:02d} image {img_id}: identical pixels for {' & '.join(orgs)} "
                      f"({'MIP' if key == 'has_mip' else 'best focus'}), dropping these labels")
                for org in orgs:
                    fluo[org][key] = 0

    fluo_best =[i["z"] for i in fluo.values() if i["has"]]
    rows = []
    for tl in TL_MODALITIES:
        zs = files.get(tl, {})
        if not zs:
            continue
        zc = pick_z(best.get(tl), zs)
        if zc is None:  # no TL best focus given: use the fluorescence one, else the middle plane
            ref = int(np.median(fluo_best)) if fluo_best else sorted(zs)[len(zs) // 2]
            zc = min(zs, key=lambda z: abs(z - ref))
        try:
            a, meta = read_plane_and_meta(src / zs[zc])
        except FileNotFoundError as e:
            print(f"  [warn] S{sid:02d} image {img_id} {tl}: missing file {e.filename}")
            continue
        if shape is not None and a.shape != shape:
            print(f"  [warn] S{sid:02d} image {img_id}: {tl} shape {a.shape} != fluorescence {shape}, skipped")
            continue
        np.save(img_dir / f"{tl}.npy", a)
        lo, hi = robust_range(a)
        stack, center = 0, 0
        if k_neighbors > 0 and len(zs) > 1:
            zz = [z for z in sorted(zs) if abs(z - zc) <= k_neighbors]
            np.save(img_dir / f"{tl}_stack.npy", np.stack([read_plane(src / zs[z]) for z in zz]))
            stack, center = 1, zz.index(zc)
        row = dict(study=sid, image_id=img_id, tl=tl, dir=rel, tl_z=zc, tl_nz=len(zs),
                   height=a.shape[0], width=a.shape[1], tl_lo=lo, tl_hi=hi,
                   tl_stack=stack, tl_stack_center=center, **meta)
        for org, info in fluo.items():
            for key, val in info.items():
                row[f"{key}_{org}"] = val
        rows.append(row)

    marker.write_text(json.dumps(rows, default=float))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="Dataset", help="folder searched recursively for Study_XX folders")
    p.add_argument("--meta", default=DEFAULT_META, help="folder with Study_XX_all_files.tsv / _best_focus.tsv")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--studies", default="", help="comma-separated ids, default: every study found")
    p.add_argument("--tl-neighbors", type=int, default=2, help="TL planes kept on each side of best focus")
    p.add_argument("--no-mip", action="store_true", help="skip fluorescence MIPs (faster, Task 1 only)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out)
    meta_dir = Path(args.meta)
    dirs = find_study_dirs(args.raw)
    wanted = [int(s) for s in args.studies.split(",") if s.strip()] or sorted(dirs)
    for sid in wanted:
        if sid not in dirs:
            print(f"Study {sid}: not found under {args.raw} (download/extract it first)")
            continue
        t0 = time.time()
        files = read_all_files(meta_dir / f"Study_{sid:02d}_all_files.tsv")
        best = read_best_focus(meta_dir / f"Study_{sid:02d}_best_focus.tsv")
        ids = sorted(files)
        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(lambda i: process_image(sid, i, files[i], best.get(i, {}), dirs[sid], out_root,
                                                          args.tl_neighbors, not args.no_mip, args.overwrite), ids))
        n_rows = sum(len(r) for r in results)
        print(f"Study {sid:02d}: {len(ids)} images -> {n_rows} samples in {time.time() - t0:.0f}s")

    rows = [r for f in sorted(out_root.glob("S*/*/rows.json")) for r in json.loads(f.read_text())]
    if not rows:
        print("No processed images yet.")
        return
    df = pd.DataFrame(rows).sort_values(["study", "image_id", "tl"])
    df.to_csv(out_root / "index.csv", index=False)
    print(f"\nindex.csv: {len(df)} samples from {df.study.nunique()} studies")
    print(df.groupby("tl").size().rename("samples").to_string())
    print("Ground truth available (samples):")
    print(df[[f"has_{o}" for o in ORGANELLES]].sum().to_string())


if __name__ == "__main__":
    main()
