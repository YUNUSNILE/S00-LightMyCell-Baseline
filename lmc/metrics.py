"""Full-image metrics (the challenge scores SSIM and Pearson per organelle).

`ssim` / `pcc`: our original metrics (GT normalised by its own 1-99.8 percentiles,
prediction used as is).
`official_metrics`: port of the official Light My Cells evaluation
(references/official_eval/lmc2024_metrics.py). Both images are normalised by their own
2-99.8 percentiles, and SSIM uses data_range = max(range_gt, range_pred). MAE,
Euclidean and cosine distance are added for nucleus and mitochondria. Percentile
normalisation cancels any earlier affine scaling, so it can be applied to our
already-normalised arrays.
`lmc_scores`: our single selection metric (project_goal.md section 2.2).
"""
import numpy as np
from skimage.metrics import structural_similarity


def ssim(gt, pred):
    rng = float(gt.max() - gt.min()) or 1.0
    return float(structural_similarity(gt, pred, data_range=rng))


def pcc(gt, pred):
    g = gt.ravel().astype(np.float64)
    p = pred.ravel().astype(np.float64)
    g -= g.mean()
    p -= p.mean()
    d = np.sqrt((g * g).sum() * (p * p).sum())
    return float((g * p).sum() / d) if d > 0 else float("nan")


def _official_norm(a, pmin=2.0, pmax=99.8):
    lo, hi = np.percentile(a, [pmin, pmax])
    return a if lo == hi else (a - lo) / (hi - lo)


def official_metrics(organelle, gt, pred):
    """Official LMC metrics for one image: dict with off_ssim, off_pcc (+ off_mae,
    off_edist, off_cdist for Nucleus / Mitochondria)."""
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

    per organelle: mean(SSIM, PCC) per image -> mean per study -> mean over studies;
    lmc_score = mean over organelles. lmc_score_image skips the study step (plain mean
    over images, closer to a leaderboard average).
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
