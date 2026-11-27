"""Copy the TRAIN part of the cache to a fast disk (SSD) for random patch reads. Resumable.

Random 512^2 x 16-plane crops from the HDD (E:) give ~1 patch/s; from the NVMe SSD (C:) ~38/s,
while training needs ~8/s. Val/Test stay on the HDD (whole images, read sequentially).

  set train_cache_root in configs/paths.yaml (e.g. C:/lmc_cache/k16), then
  python tools/mirror_cache.py              (source = cache_root, destination = train_cache_root)

Files already present with the same size are skipped; index.csv is always refreshed.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmc import CACHE_ROOT, PATHS, PROJECT_ROOT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(CACHE_ROOT))
    p.add_argument("--dest", default=None if PATHS["train_cache_root"] is None else str(PATHS["train_cache_root"]))
    p.add_argument("--split", default="splits/split_v1.json")
    p.add_argument("--which", default="train", help="comma-separated split names to mirror")
    a = p.parse_args()
    if not a.dest:
        raise SystemExit("set train_cache_root in configs/paths.yaml (or pass --dest)")
    src, dest = PROJECT_ROOT / a.src, Path(a.dest)
    sp = json.loads((PROJECT_ROOT / a.split).read_text())
    studies = sorted({s for w in a.which.split(",") for s in sp["studies"][w]})
    need = sum(f.stat().st_size for s in studies for f in (src / f"S{s:02d}").rglob("*") if f.is_file())
    free = shutil.disk_usage(dest.anchor).free
    print(f"mirroring studies {studies}: {need / 2**30:.1f} GB to {dest} (free on {dest.anchor}: {free / 2**30:.1f} GB)")
    if need > free - 5 * 2**30:
        raise SystemExit("not enough free space (keeping a 5 GB margin)")
    t0, copied = time.time(), 0
    for s in studies:
        for f in sorted((src / f"S{s:02d}").rglob("*")):
            if not f.is_file() or f.name.endswith(".tmp.npy"):
                continue
            t = dest / f.relative_to(src)
            if t.exists() and t.stat().st_size == f.stat().st_size:
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t.with_name(t.name + ".part"))
            t.with_name(t.name + ".part").replace(t)
            copied += f.stat().st_size
        print(f"  S{s:02d} done ({copied / 2**30:.1f} GB copied, {time.time() - t0:.0f}s)", flush=True)
    shutil.copy2(src / "index.csv", dest / "index.csv")
    print("done")


if __name__ == "__main__":
    main()
