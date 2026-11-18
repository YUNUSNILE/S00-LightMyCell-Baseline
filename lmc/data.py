"""PyTorch datasets over the arrays written by prepare_data.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lmc import ORGANELLES, TL_MODALITIES


def load_index(data_root):
    return pd.read_csv(Path(data_root) / "index.csv")


def split_index(df, val_studies=(), val_frac=0.0, seed=0):
    """Hold out whole studies (the real test set comes from unseen studies).

    val_frac gives an image-level split instead; only use it for smoke tests
    on a single study, because neighbouring images leak into validation.
    """
    if val_studies:
        is_val = df.study.isin(val_studies)
    elif val_frac > 0:
        rng = np.random.default_rng(seed)
        keys = df[["study", "image_id"]].drop_duplicates()
        val_keys = keys.sample(frac=val_frac, random_state=int(rng.integers(1 << 31)))
        is_val = df.set_index(["study", "image_id"]).index.isin(pd.MultiIndex.from_frame(val_keys))
    else:
        raise ValueError("give val studies or a val fraction")
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def study_balanced_weights(df):
    """Each study gets the same total sampling weight, whatever its size."""
    counts = df.study.map(df.study.value_counts())
    return (1.0 / counts).to_numpy()


def _norm(a, lo, hi):
    # float() keeps float32: a NumPy float64 scalar would promote the whole array.
    return (a.astype(np.float32) - float(lo)) / max(float(hi) - float(lo), 1e-6)


def _crop(arr, y, x, p):
    """Crop p x p at (y, x), reflect-padding when the image is smaller than p."""
    c = np.asarray(arr[y:y + p, x:x + p])
    if c.shape != (p, p):
        c = np.pad(c, ((0, p - c.shape[0]), (0, p - c.shape[1])), mode="reflect")
    return c


class LMCPatches(Dataset):
    """Random (train) or fixed-grid (val) patches.

    Returns x (1,P,P) TL input, y (4,P,P) targets, m (4,) mask of the
    organelles that have ground truth, plus the TL modality index.
    target="best" -> best-focus plane (Task 1), "mip" -> MIP (Task 2).
    """

    def __init__(self, df, data_root, patch=512, train=True, target="best", focus_aug=False, val_crops=4):
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
        self.p = patch
        self.train = train
        self.suffix = "" if target == "best" else "_mip"
        self.has = "has_" if target == "best" else "has_mip_"
        self.lo = "lo_" if target == "best" else "lo_mip_"
        self.hi = "hi_" if target == "best" else "hi_mip_"
        self.focus_aug = focus_aug
        if not train:  # fixed crop positions so val numbers are comparable between epochs
            g = int(np.ceil(np.sqrt(val_crops)))
            self.items = []
            for i, r in self.df.iterrows():
                ys = np.linspace(0, max(r.height - patch, 0), g).astype(int)
                xs = np.linspace(0, max(r.width - patch, 0), g).astype(int)
                pos = [(y, x) for y in ys for x in xs][:val_crops]
                self.items += [(i, y, x) for y, x in pos]

    def __len__(self):
        return len(self.df) if self.train else len(self.items)

    def _tl_plane(self, r, rng):
        d = self.root / r.dir
        if self.focus_aug and r.tl_stack:
            stack = np.load(d / f"{r.tl}_stack.npy", mmap_mode="r")
            return stack[rng.integers(len(stack))]
        return np.load(d / f"{r.tl}.npy", mmap_mode="r")

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        if self.train:
            r = self.df.iloc[i]
            y0 = int(rng.integers(0, max(r.height - self.p, 0) + 1))
            x0 = int(rng.integers(0, max(r.width - self.p, 0) + 1))
        else:
            idx, y0, x0 = self.items[i]
            r = self.df.iloc[idx]
        d = self.root / r.dir

        x = _norm(_crop(self._tl_plane(r, rng), y0, x0, self.p), r.tl_lo, r.tl_hi)[None]
        y = np.zeros((len(ORGANELLES), self.p, self.p), np.float32)
        m = np.zeros(len(ORGANELLES), np.float32)
        for c, org in enumerate(ORGANELLES):
            if r[f"{self.has}{org}"] == 1:
                a = np.load(d / f"{org}{self.suffix}.npy", mmap_mode="r")
                y[c] = _norm(_crop(a, y0, x0, self.p), r[f"{self.lo}{org}"], r[f"{self.hi}{org}"])
                m[c] = 1.0

        if self.train:  # the 8 flips/rotations are all valid for microscopy
            k = int(rng.integers(4))
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "m": torch.from_numpy(m),
            "modality": TL_MODALITIES.index(r.tl),
        }
