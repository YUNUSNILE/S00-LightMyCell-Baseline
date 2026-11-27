"""Datasets over the cache built by tools/build_cache.py.

Input modes (cfg.data.input):
  "stack"   the K resampled TL planes as channels  -> x (K, P, P)
  "center"  the stack-centre TL plane (E2)          -> x (1, P, P)
Targets: the 4 organelle best-focus planes y (4, P, P), mask m (4,), resampled best-z indices bz (4,)
(NaN when missing; used only by the E4 auxiliary loss, never as input).

Normalisation (teammate's recipe, baseline_teammate/lmc/data.py `_norm`): (a - lo) / (hi - lo) with
lo/hi = 1 / 99.8 percentiles; one pair per TL stack, one per GT plane.
Patch cropping, dihedral augmentation and per-study sampling weights are adapted from the same file.

Test-time side information (allowed at inference; from the OME metadata, never from the TSV best_z):
  zpos (C,)  position of each input plane in um relative to the stack centre:
             (linspace(0, nZ-1, K) - (nZ-1)/2) * dz ; [0] for the centre-plane input
  meta (5,)  one-hot modality (BF, PC, DIC), log(px / 0.1 um), log(dz / 0.5 um)
Focal-plane augmentation (E6, train only, stack input), cfg train.zaug = {shift, plane_dropout, zflip}:
  shift s ~ U{-S..S}: the stack content moves by s planes (edge planes replicated); zpos stays the grid of
  the (new) stack, so the network cannot rely on absolute plane positions; bz moves with the content
  (clamped to [0, K-1]). zflip (p=0.5): plane order reversed, bz -> K-1-bz. plane_dropout: each plane
  set to 0 with probability p (at least one plane kept).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lmc import ORGANELLES, TL_MODALITIES


def norm(a, lo, hi):
    return (np.asarray(a, np.float32) - np.float32(lo)) / np.float32(max(float(hi) - float(lo), 1e-6))


def load_index(cache_root):
    """Cache index; S01 v2: the MIP columns of mip_index.csv (tools/build_mip_cache.py) are joined in when present
    (has_mip_<org> = 0 for images without a MIP)."""
    idx = pd.read_csv(Path(cache_root) / "index.csv")
    mp = Path(cache_root) / "mip_index.csv"
    if mp.exists():
        mip = pd.read_csv(mp)
        idx = idx.merge(mip, on=["study", "image_id"], how="left", validate="many_to_one")
        for org in ORGANELLES:
            idx[f"has_mip_{org}"] = idx[f"has_mip_{org}"].fillna(0).astype(int)
    return idx


TASK_FILES = {"best": ("gt.npy", ""), "mip": ("mip.npy", "mip_")}  # task -> (cache file, column prefix)


def _task_targets(arr, row, task):
    """Normalised targets (4, ...) and mask (4,) of one task from the (4, ...) uint16 array `arr`."""
    pre = TASK_FILES[task][1]
    y = np.zeros(arr.shape, np.float32)
    m = np.zeros(len(ORGANELLES), np.float32)
    for c, org in enumerate(ORGANELLES):
        if _get(row, f"has_{pre}{org}") == 1:
            y[c] = norm(arr[c], _get(row, f"lo_{pre}{org}"), _get(row, f"hi_{pre}{org}"))
            m[c] = 1.0
    return y, m


def _get(row, key):
    return row[key] if isinstance(row, pd.Series) else getattr(row, key, 0)


def load_split(split_path):
    return json.loads(Path(split_path).read_text())


def split_rows(index, split, name):
    """Rows of `index` listed in split['samples'][name] (study, image_id, tl)."""
    keys = {tuple(s) for s in split["samples"][name]}
    sel = [(s, i, t) in keys for s, i, t in zip(index.study, index.image_id, index.tl)]
    out = index[sel].reset_index(drop=True)
    if len(out) != len(keys):
        raise RuntimeError(f"split '{name}': {len(keys)} samples listed, {len(out)} found in cache index")
    return out


def study_weights(df, mode="sqrt"):
    """Per-row sampling weight. 'sqrt': each study's total weight is proportional to sqrt(n_rows);
    'equal': equal per study (teammate); 'uniform': plain per-row."""
    n = df.study.map(df.study.value_counts()).to_numpy(np.float64)
    if mode == "sqrt":
        return np.sqrt(n) / n
    if mode == "equal":
        return 1.0 / n
    if mode == "uniform":
        return np.ones(len(df))
    raise ValueError(mode)


def _crop(arr, y, x, p):
    """Crop (..., p, p) at (y, x); reflect-pad if the image is smaller than p."""
    c = np.asarray(arr[..., y:y + p, x:x + p])
    ph, pw = p - c.shape[-2], p - c.shape[-1]
    if ph or pw:
        c = np.pad(c, [(0, 0)] * (c.ndim - 2) + [(0, ph), (0, pw)], mode="reflect")
    return c


def plane_positions(row, mode, k):
    """zpos (C,) in um relative to the stack centre."""
    if mode == "center":
        return np.zeros(1, np.float32)
    nz = float(row.nZ)
    return ((np.linspace(0.0, nz - 1.0, k) - (nz - 1.0) / 2.0) * float(row.dz_um)).astype(np.float32)


def meta_vector(row):
    v = np.zeros(5, np.float32)
    v[TL_MODALITIES.index(row.tl)] = 1.0
    v[3] = np.log(float(row.px_um) / 0.1)
    v[4] = np.log(float(row.dz_um) / 0.5)
    return v


def z_augment(x, bz, rng, shift=0, plane_dropout=0.0, zflip=False):
    """Focal-plane augmentation of a (K, P, P) stack and its resampled best-z indices bz (4,)."""
    k = x.shape[0]
    bz = bz.copy()
    if shift:
        s = int(rng.integers(-shift, shift + 1))
        if s:
            x = x[np.clip(np.arange(k) - s, 0, k - 1)]
            bz = np.clip(bz + s, 0, k - 1)
    if zflip and rng.random() < 0.5:
        x = x[::-1]
        bz = (k - 1) - bz
    if plane_dropout > 0:
        drop = rng.random(k) < plane_dropout
        if drop.all():
            drop[int(rng.integers(k))] = False
        if drop.any():
            x = x.copy()
            x[drop] = 0.0
    return np.ascontiguousarray(x), bz


def load_input(row, cache_root, mode, k):
    """Full-image normalised input (C, H, W) float32."""
    d = Path(cache_root) / row.dir
    if mode == "stack":
        return norm(np.load(d / f"{row.tl}_k{k}.npy", mmap_mode="r"), row.tl_lo, row.tl_hi)
    if mode == "center":
        return norm(np.load(d / f"{row.tl}_center.npy", mmap_mode="r"), row.center_lo, row.center_hi)[None]
    raise ValueError(mode)


def load_targets(row, cache_root, tasks=("best",)):
    """Full-image normalised GT (4T, H, W) and mask (4T,) for the T tasks, in `tasks` order:
    'best' = organelle best-focus planes (Task 1), 'mip' = maximum-intensity projections (Task 2)."""
    ys, ms = [], []
    for task in tasks:
        f = Path(cache_root) / row.dir / TASK_FILES[task][0]
        if task == "mip" and not f.exists():  # image without any MIP label
            gt = np.load(Path(cache_root) / row.dir / "gt.npy", mmap_mode="r")
            ys.append(np.zeros(gt.shape, np.float32))
            ms.append(np.zeros(len(ORGANELLES), np.float32))
            continue
        y, m = _task_targets(np.load(f, mmap_mode="r"), row, task)
        ys.append(y)
        ms.append(m)
    return np.concatenate(ys), np.concatenate(ms)


class PatchDataset(Dataset):
    """Random P x P training patches (index i -> row i; the sampler picks rows)."""

    def __init__(self, df, cache_root, mode="stack", k=16, patch=512, augment=True, zaug=None, tasks=("best",)):
        self.df = df.reset_index(drop=True)
        self.root = Path(cache_root)
        self.mode, self.k, self.p, self.augment = mode, k, patch, augment
        self.zaug = {k_: v for k_, v in (zaug or {}).items() if v} if mode == "stack" else {}
        self.tasks = tuple(tasks)  # S01 v2: y/m hold 4 channels per task, in this order (same crop and flips)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        r = self.df.iloc[i]
        d = self.root / r.dir
        y0 = int(rng.integers(0, max(r.H - self.p, 0) + 1))
        x0 = int(rng.integers(0, max(r.W - self.p, 0) + 1))
        if self.mode == "stack":
            x = norm(_crop(np.load(d / f"{r.tl}_k{self.k}.npy", mmap_mode="r"), y0, x0, self.p), r.tl_lo, r.tl_hi)
        else:
            x = norm(_crop(np.load(d / f"{r.tl}_center.npy", mmap_mode="r"), y0, x0, self.p),
                     r.center_lo, r.center_hi)[None]
        bz = np.full(len(ORGANELLES), np.nan, np.float32)
        for c, org in enumerate(ORGANELLES):
            if r[f"has_{org}"] == 1:
                bz[c] = r[f"best_zr_{org}"]
        ys, ms = [], []
        for task in self.tasks:
            f = d / TASK_FILES[task][0]
            if task == "mip" and not f.exists():  # image without any MIP label: masked out
                ys.append(np.zeros((len(ORGANELLES), self.p, self.p), np.float32))
                ms.append(np.zeros(len(ORGANELLES), np.float32))
                continue
            yt, mt = _task_targets(_crop(np.load(f, mmap_mode="r"), y0, x0, self.p), r, task)
            ys.append(yt)
            ms.append(mt)
        y, m = np.concatenate(ys), np.concatenate(ms)
        if self.zaug:
            x, bz = z_augment(x, bz, rng, **self.zaug)
        if self.augment:  # the 8 dihedral transforms
            kk = int(rng.integers(4))
            x, y = np.rot90(x, kk, (1, 2)), np.rot90(y, kk, (1, 2))
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
        return {"x": torch.from_numpy(np.ascontiguousarray(x)), "y": torch.from_numpy(np.ascontiguousarray(y)),
                "m": torch.from_numpy(m), "bz": torch.from_numpy(bz.astype(np.float32)),
                "zpos": torch.from_numpy(plane_positions(r, self.mode, self.k)),
                "meta": torch.from_numpy(meta_vector(r)), "modality": TL_MODALITIES.index(r.tl)}
