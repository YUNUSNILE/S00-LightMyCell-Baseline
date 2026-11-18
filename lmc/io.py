"""Readers for the Light My Cells 2 metadata TSVs and OME-TIFF planes."""
import csv
import re

import numpy as np
import tifffile

# Study 59 names its phase-contrast channel "Phase" instead of "PC".
CHANNEL_ALIASES = {"Phase": "PC"}

_OME_KEYS = {
    "pixel_size_um": r'PhysicalSizeX="([0-9.eE+-]+)"',
    "magnification": r'NominalMagnification="([0-9.eE+-]+)"',
    "na": r'LensNA="([0-9.eE+-]+)"',
}


def canon(channel):
    return CHANNEL_ALIASES.get(channel, channel)


def read_all_files(tsv_path):
    """{image_id: {channel: {z: filename}}} from Study_XX_all_files.tsv."""
    out = {}
    with open(tsv_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            img = int(row["image_id"])
            ch = canon(row["channel"])
            z = int(float(row["z"]))
            out.setdefault(img, {}).setdefault(ch, {})[z] = row["filename"]
    return out


def read_best_focus(tsv_path):
    """{image_id: {channel: best_z}} from Study_XX_best_focus.tsv.

    Column order and layout differ between studies (some use `nZ`, some
    `<channel>_nZ`, values can be "19.0" or empty), so only the `*_best_z`
    columns are used. The real number of planes comes from the file list.
    """
    out = {}
    with open(tsv_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            d = {}
            for k, v in row.items():
                if k and k.endswith("_best_z") and v not in (None, ""):
                    d[canon(k[: -len("_best_z")])] = int(float(v))
            out[int(row["image_id"])] = d
    return out


def read_plane(path):
    with tifffile.TiffFile(path) as t:
        a = np.squeeze(t.asarray())
    if a.ndim != 2:
        raise ValueError(f"expected a 2D plane in {path}, got shape {a.shape}")
    return a


def read_plane_and_meta(path):
    with tifffile.TiffFile(path) as t:
        a = np.squeeze(t.asarray())
        ome = t.ome_metadata or ""
    meta = {}
    for key, pat in _OME_KEYS.items():
        m = re.search(pat, ome)
        meta[key] = float(m.group(1)) if m else float("nan")
    return a, meta


def robust_range(a, lo=1.0, hi=99.8):
    """Percentiles used for intensity normalisation (computed on a 4x subsample)."""
    p = np.percentile(a[::4, ::4], [lo, hi])
    return float(p[0]), float(p[1])
