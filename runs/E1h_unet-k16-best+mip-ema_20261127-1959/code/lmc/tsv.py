"""TSV readers, OME metadata and percentile ranges.

Adapted from the teammate's baseline: baseline_teammate/lmc/io.py
(github.com/YUNUSNILE/S00-LightMyCell-Baseline, commit fc2ab9a). Changes: PhysicalSizeZ added to the
OME keys; tifffile reads the whole OME header once per call.
"""
import csv
import re

import numpy as np
import tifffile

# Study 59 names its phase-contrast channel "Phase" instead of "PC" (not on disk, kept for safety).
CHANNEL_ALIASES = {"Phase": "PC"}

_OME_KEYS = {
    "px_um": r'PhysicalSizeX="([0-9.eE+-]+)"',
    "dz_um": r'PhysicalSizeZ="([0-9.eE+-]+)"',
    "mag": r'NominalMagnification="([0-9.eE+-]+)"',
    "na": r'LensNA="([0-9.eE+-]+)"',
}


def canon(channel):
    return CHANNEL_ALIASES.get(channel, channel)


def read_all_files(tsv_path):
    """{image_id: {channel: {z: filename}}} from Study_XX_all_files.tsv."""
    out = {}
    with open(tsv_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out.setdefault(int(row["image_id"]), {}).setdefault(canon(row["channel"]), {})[
                int(float(row["z"]))] = row["filename"]
    return out


def read_best_focus(tsv_path):
    """{image_id: {channel: best_z}} from Study_XX_best_focus.tsv (only the *_best_z columns)."""
    out = {}
    with open(tsv_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            d = {}
            for k, v in row.items():
                if k and k.endswith("_best_z") and v not in (None, ""):
                    d[canon(k[: -len("_best_z")])] = int(float(v))
            out[int(float(row["image_id"]))] = d
    return out


def read_plane(path):
    a = np.squeeze(tifffile.imread(path))
    if a.ndim != 2:
        raise ValueError(f"expected a 2D plane in {path}, got shape {a.shape}")
    return a


def read_ome_meta(path):
    with tifffile.TiffFile(path) as t:
        ome = t.ome_metadata or ""
    meta = {}
    for key, pat in _OME_KEYS.items():
        m = re.search(pat, ome)
        meta[key] = float(m.group(1)) if m else float("nan")
    return meta


def robust_range(a, lo=1.0, hi=99.8):
    """Percentiles used for intensity normalisation, on a 4x subsample of the last two axes."""
    p = np.percentile(np.asarray(a)[..., ::4, ::4], [lo, hi])
    return float(p[0]), float(p[1])
