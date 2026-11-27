"""Official LMC metrics on full images and the LMC-score.

`official_metrics` and `lmc_scores` are copied from the teammate's baseline:
baseline_teammate/lmc/metrics.py (github.com/YUNUSNILE/S00-LightMyCell-Baseline, commit fc2ab9a),
itself a port of the official Light My Cells evaluation
(codeberg.org/FranceBioImaging/LightMyCellsChallenge). Both GT and prediction are normalised by their
own 2-99.8 percentiles; SSIM uses data_range = max(range_gt, range_pred); PCC is plain Pearson.
`breakdown` (per organelle x modality x study tables) is new.
"""
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from lmc import ORGANELLES


def pcc(gt, pred):
    g = np.asarray(gt, np.float64).ravel()
    p = np.asarray(pred, np.float64).ravel()
    g = g - g.mean()
    p = p - p.mean()
    d = np.sqrt((g * g).sum() * (p * p).sum())
    return float((g * p).sum() / d) if d > 0 else float("nan")


def _official_norm(a, pmin=2.0, pmax=99.8):
    lo, hi = np.percentile(a, [pmin, pmax])
    return a if lo == hi else (a - lo) / (hi - lo)


def official_metrics(organelle, gt, pred):
    """Official LMC metrics for one image: off_ssim, off_pcc (+ off_mae, off_edist, off_cdist for
    Nucleus / Mitochondria)."""
    g = _official_norm(np.asarray(gt, np.float64))
    p = _official_norm(np.asarray(pred, np.float64))
    r = pcc(g, p)
    rng = max(float(p.max() - p.min()), float(g.max() - g.min())) or 1.0
    out = {"off_ssim": float(structural_similarity(g, p, data_range=rng)),
           "off_pcc": 0.0 if not np.isfinite(r) else r}
    if organelle in ("Nucleus", "Mitochondria"):
        d = g - p
        den = np.linalg.norm(g) * np.linalg.norm(p)
        out["off_mae"] = float(np.abs(d).mean())
        out["off_edist"] = float(np.sqrt((d * d).sum()))
        out["off_cdist"] = float(abs(1.0 - (g * p).sum() / den)) if den > 0 else float("nan")
    return out


def lmc_scores(res):
    """LMC-score from per-image records (columns organelle, study, off_ssim, off_pcc).

    Per organelle: mean (SSIM+PCC)/2 per image -> mean per study -> mean over studies;
    lmc_score = mean over organelles. lmc_score_image skips the study step.
    """
    r = res.assign(score=(res.off_ssim + res.off_pcc) / 2)
    per_study = r.groupby(["organelle", "study"]).score.mean()
    org_macro = per_study.groupby("organelle").mean()
    org_image = r.groupby("organelle").score.mean()
    return {
        "lmc_score": float(org_macro.mean()),
        "lmc_score_image": float(org_image.mean()),
        "per_organelle_study_macro": {k: round(float(v), 4) for k, v in org_macro.items()},
        "per_organelle_image_mean": {k: round(float(v), 4) for k, v in org_image.items()},
        "n_images": {k: int(v) for k, v in r.groupby("organelle").size().items()},
    }


def breakdown(res):
    """Nested dict of mean SSIM / PCC / score per organelle x modality x study (+ counts)."""
    r = res.assign(score=(res.off_ssim + res.off_pcc) / 2)
    out = {}
    for keys, name in ((["organelle"], "by_organelle"), (["organelle", "tl"], "by_organelle_modality"),
                       (["organelle", "tl", "study"], "by_organelle_modality_study")):
        g = r.groupby(keys).agg(ssim=("off_ssim", "mean"), pcc=("off_pcc", "mean"), score=("score", "mean"),
                                n=("score", "size"))
        out[name] = [dict(zip(keys, k if isinstance(k, tuple) else (k,)), **{c: (round(float(v), 4) if c != "n" else int(v))
                                                                             for c, v in row.items()})
                     for k, row in g.iterrows()]
    return out


def organelle_order(d):
    return {o: d[o] for o in ORGANELLES if o in d}


def to_frame(records):
    return pd.DataFrame(records)
