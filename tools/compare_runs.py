"""Compare finished runs on Val (best checkpoint of each run): tables, paired bootstrap, curves, timing.

  python tools/compare_runs.py --runs runs/E1a_* runs/E1b_* runs/E1c_* --out reports/E1_compare

Uses each run's val_metrics.json / val_per_image.csv (written at its best Val epoch), val_log.csv,
train_log.csv, stdout.log. Writes summary.md, by_organelle.csv, by_modality.csv, by_study.csv,
bootstrap.csv and curves_compare.png to --out.

Paired bootstrap: the LMC-score difference between two runs, resampling images WITH replacement
inside each (study, organelle) cell (the study structure of the LMC-score is kept), 2000 draws.
Caveat: best epochs were selected on this same Val set, so all scores are slightly optimistic.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmc import ORGANELLES, PROJECT_ROOT  # noqa: E402
from lmc.metrics import lmc_scores  # noqa: E402

EXCLUDE = (20,)  # secondary LMC-score without these studies (selection always uses the full Val set)


def load_run(d):
    d = Path(d)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    vm = json.loads((d / "val_metrics.json").read_text())
    per_all = pd.read_csv(d / "val_per_image.csv")
    # S01 v2: records carry a `task` column; Task 1 ('best') is what every table and bootstrap below compares
    per = per_all[per_all.task == "best"].copy() if "task" in per_all else per_all
    vl, tl = pd.read_csv(d / "val_log.csv"), pd.read_csv(d / "train_log.csv")
    log = (d / "stdout.log").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\(([\d.]+)M params\)", log)
    tot = re.findall(r"total time\s+(\S+)", log)
    return dict(dir=d, id=cfg["run"]["id"], name=f"{cfg['run']['id']} {cfg['model']['backbone']}", cfg=cfg, vm=vm,
                per=per, per_all=per_all, vl=vl, tl=tl, params=float(m.group(1)) if m else np.nan,
                total=tot[-1] if tot else "?")


def boot_diff_study(a, b, n=2000, seed=0):
    """LMC(a) - LMC(b) resampling STUDIES with replacement inside each organelle (paired; added for S01 v2).

    The LMC-score is a mean over organelles of a mean over studies, so it is linear in the per-study paired
    differences. With 2-5 Val studies per organelle this interval is much wider than the image-level one: it is
    the honest uncertainty for unseen studies (cluster bootstrap; Davison & Hinkley 1997, ch. 3.8)."""
    key = ["study", "image_id", "tl", "organelle"]
    m = a[key + ["off_ssim", "off_pcc"]].merge(b[key + ["off_ssim", "off_pcc"]], on=key, suffixes=("_a", "_b"))
    m["d"] = (m.off_ssim_a + m.off_pcc_a) / 2 - (m.off_ssim_b + m.off_pcc_b) / 2
    per = m.groupby(["organelle", "study"]).d.mean()
    orgs = [per.loc[o].to_numpy() for o in per.index.get_level_values(0).unique()]
    point = float(np.mean([v.mean() for v in orgs]))
    rng = np.random.default_rng(seed)
    draws = np.array([np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in orgs]) for _ in range(n)])
    return point, np.percentile(draws, 2.5), np.percentile(draws, 97.5), float((draws > 0).mean())


def boot_diff(a, b, n=2000, seed=0):
    """LMC(a) - LMC(b) with paired resampling of images inside each (organelle, study) cell."""
    key = ["study", "image_id", "tl", "organelle"]
    m = a[key + ["off_ssim", "off_pcc"]].merge(b[key + ["off_ssim", "off_pcc"]], on=key, suffixes=("_a", "_b"))
    m["sa"] = (m.off_ssim_a + m.off_pcc_a) / 2
    m["sb"] = (m.off_ssim_b + m.off_pcc_b) / 2
    cells = [g[["sa", "sb"]].to_numpy() for _, g in m.groupby(["organelle", "study"])]
    org_of = [o for (o, _), _ in m.groupby(["organelle", "study"])]
    rng = np.random.default_rng(seed)

    def lmc(sel):
        per_org = {}
        for o, c, idx in zip(org_of, cells, sel):
            per_org.setdefault(o, []).append(c[idx].mean(0))
        return np.mean([np.mean(v, 0) for v in per_org.values()], 0)

    point = lmc([np.arange(len(c)) for c in cells])
    draws = np.array([lmc([rng.integers(0, len(c), len(c)) for c in cells]) for _ in range(n)])
    diff = draws[:, 0] - draws[:, 1]
    return point[0] - point[1], np.percentile(diff, 2.5), np.percentile(diff, 97.5), float((diff > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirs = sorted({p for r in a.runs for p in glob.glob(r)})
    runs = [load_run(d) for d in dirs]
    out = PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    md = []

    # ---- headline
    rows = []
    for r in runs:
        s = r["vm"]
        tl = r["tl"]
        excl = r["per"][~r["per"].study.isin(EXCLUDE)]
        att = s.get("attention", {})
        rows.append({"run": r["name"], "params_M": r["params"], "best_epoch": s.get("epoch"),
                     "LMC (study macro)": round(s["lmc_score"], 4), "LMC (image mean)": round(s["lmc_score_image"], 4),
                     f"LMC w/o S{'/S'.join(map(str, EXCLUDE))}": round(lmc_scores(excl)["lmc_score"], 4),
                     **{o: s["per_organelle_study_macro"].get(o) for o in ORGANELLES},
                     "last-epoch LMC": round(r["vl"].lmc_score.iloc[-1], 4),
                     "s/step": round(tl.sec_per_step.median(), 2), "val pass (s)": int(r["vl"].seconds.median()),
                     "total time": r["total"], "peak GPU GB": round(tl.gpu_peak_gb.max(), 2),
                     **({f"att err {o[:4]} (planes)": att[o]["err_planes"] for o in ORGANELLES if o in att})})
    head = pd.DataFrame(rows)
    head.to_csv(out / "by_organelle.csv", index=False)
    md += ["## Headline (Val, best checkpoint; organelle columns = study-macro (SSIM+PCC)/2)", "", head.to_markdown(index=False)
           if hasattr(head, "to_markdown") and _has_tabulate() else _md(head), ""]

    # ---- by organelle x modality (SSIM / PCC / score)
    mods = []
    for r in runs:
        for b in r["vm"]["breakdown"]["by_organelle_modality"]:
            mods.append({"run": r["id"], **b})
    mods = pd.DataFrame(mods)
    mods.to_csv(out / "by_modality.csv", index=False)
    piv = mods.pivot_table(index=["organelle", "tl", "n"], columns="run", values="score").reset_index()
    md += ["## (SSIM+PCC)/2 per organelle x modality (image mean)", "", _md(piv.round(3)), ""]
    piv2 = mods.pivot_table(index=["organelle", "tl"], columns="run", values=["ssim", "pcc"]).round(3)
    piv2.columns = [f"{m}_{r}" for m, r in piv2.columns]
    md += ["## SSIM and PCC per organelle x modality", "", _md(piv2.reset_index()), ""]

    # ---- by study
    st = []
    for r in runs:
        p = r["per"].assign(score=lambda x: (x.off_ssim + x.off_pcc) / 2)
        g = p.groupby(["organelle", "study", "tl"]).score.agg(["mean", "size"]).reset_index()
        g["run"] = r["id"]
        st.append(g)
    st = pd.concat(st)
    pst = st.pivot_table(index=["organelle", "study", "tl", "size"], columns="run", values="mean").reset_index().round(3)
    pst.to_csv(out / "by_study.csv", index=False)
    md += ["## (SSIM+PCC)/2 per organelle x study", "", _md(pst), ""]

    # ---- paired bootstrap
    bs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            d, lo, hi, p = boot_diff(runs[i]["per"], runs[j]["per"])
            bs.append({"A": runs[i]["id"], "B": runs[j]["id"], "LMC(A)-LMC(B)": round(d, 4), "95% CI low": round(lo, 4),
                       "95% CI high": round(hi, 4), "P(A>B)": round(p, 3)})
            d2, lo2, hi2, p2 = boot_diff(runs[i]["per"][~runs[i]["per"].study.isin(EXCLUDE)],
                                         runs[j]["per"][~runs[j]["per"].study.isin(EXCLUDE)])
            bs[-1].update({f"w/o S{'/S'.join(map(str, EXCLUDE))}: diff": round(d2, 4), "w/o: CI low": round(lo2, 4),
                           "w/o: CI high": round(hi2, 4)})
            _, lo3, hi3, p3 = boot_diff_study(runs[i]["per"], runs[j]["per"])
            bs[-1].update({"study-level CI low": round(lo3, 4), "study-level CI high": round(hi3, 4),
                           "study-level P(A>B)": round(p3, 3)})
    bs = pd.DataFrame(bs)
    bs.to_csv(out / "bootstrap.csv", index=False)
    md += ["## Paired bootstrap of the LMC-score difference (images resampled within study x organelle, 2000 draws;"
           " study-level columns resample the Val studies within each organelle instead)", "", _md(bs), ""]

    # ---- Task 2 (S01 v2): MIP head and the Task 1 head scored against the MIP, where a run has them
    t2 = []
    for r in runs:
        for name, s in r["vm"].get("tasks", {}).items():
            t2.append({"run": r["name"], "task": name, "LMC (study macro)": round(s["lmc_score"], 4),
                       f"LMC w/o S{'/S'.join(map(str, EXCLUDE))}": round(s["lmc_score_excl"], 4),
                       **{o: s["per_organelle_study_macro"].get(o) for o in ORGANELLES}})
    if t2:
        t2 = pd.DataFrame(t2)
        t2.to_csv(out / "task2.csv", index=False)
        md += ["## Task 2 (MIP) on Val, at each run's best Task 1 epoch", "", _md(t2), ""]

    # ---- curves
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    for r in runs:
        tl, vl = r["tl"], r["vl"]
        spe = r["cfg"]["train"].get("samples_per_epoch") or 803
        ep = tl.step / (spe // (r["cfg"]["train"]["batch"] * r["cfg"]["train"]["accum"]))
        ax[0, 0].plot(ep, tl.loss.rolling(10, min_periods=1).mean(), label=r["name"])
        ax[0, 1].plot(vl.epoch, vl.lmc_score, "o-", label=r["name"])
        ax[0, 2].plot(vl.epoch, vl.lmc_score_image, "o-", label=r["name"])
        for k, o in enumerate(ORGANELLES):
            axx = ax[1, k] if k < 3 else None
            if axx is None:
                continue
            axx.plot(vl.epoch, vl[f"score_{o}"], "o-", label=r["name"])
        ax[1, 2].plot(vl.epoch, vl["score_Actin"], "s--", alpha=0.6, label=f"{r['id']} Actin")
    ax[0, 0].set_title("Train loss (smoothed)")
    ax[0, 1].set_title("Val LMC-score (study macro)")
    ax[0, 2].set_title("Val LMC-score (image mean)")
    ax[1, 0].set_title("Val Nucleus (SSIM+PCC)/2, image mean")
    ax[1, 1].set_title("Val Mitochondria (SSIM+PCC)/2, image mean")
    ax[1, 2].set_title("Val Tubulin (o) and Actin (--), image mean")
    for x in ax.ravel():
        x.set_xlabel("epoch")
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves_compare.png", dpi=110)
    plt.close(fig)

    md = ["# Val comparison: " + ", ".join(r["dir"].name for r in runs), "",
          "Scores at each run's best Val epoch (selected on this same Val set: slightly optimistic for all runs).",
          f"Secondary number: LMC-score without Study {', '.join(map(str, EXCLUDE))} (reporting only; selection uses all of Val).",
          ""] + md
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _md(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
