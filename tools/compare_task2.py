"""Task 2 (MIP) comparison on Val (added for S00 v2): tables, paired bootstraps and Task 1 / Task 2 curves.

  python tools/compare_task2.py --runs runs/E1h_* ../S01_v2/runs/E1g_* \
      --t2base ../S01_v2/runs/E1d_* --out reports/S00_compare

tools/compare_runs.py compares Task 1 only (and lists the Task 2 LMC-scores). This script does the same for Task 2:
  - `--runs`: dual-head runs; their Task 2 rows (task == "mip" in val_per_image.csv, written at the best epoch)
  - `--t2base`: Task 1 runs scored against the MIP target by `evaluate.py --t2-baseline`
    (eval_val_best_t2base_per_image.csv, task == "mip_from_best"), labelled "<id> T1->MIP"
Bootstraps reuse compare_runs.boot_diff (images resampled within study x organelle) and boot_diff_study (studies
resampled within organelle), so Task 1 and Task 2 intervals are computed the same way.
Writes task2_summary.md, task2_by_modality.csv, task2_by_study.csv, task2_bootstrap.csv, curves_task1_task2.png.
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmc import ORGANELLES, PROJECT_ROOT  # noqa: E402
from lmc.metrics import lmc_scores  # noqa: E402
from tools.compare_runs import EXCLUDE, _md, boot_diff, boot_diff_study, load_run  # noqa: E402


def entries(run_globs, base_globs):
    out = []
    for d in sorted({p for g in run_globs for p in glob.glob(g)}):
        r = load_run(d)
        per = r["per_all"]
        if "task" not in per or not (per.task == "mip").any():
            raise SystemExit(f"{d}: no Task 2 (mip) rows in val_per_image.csv")
        out.append(dict(id=r["id"], label=r["name"], per=per[per.task == "mip"].copy(), run=r))
    for d in sorted({p for g in base_globs for p in glob.glob(g)}):
        r = load_run(d)
        f = Path(d) / "eval_val_best_t2base_per_image.csv"
        if not f.exists():
            raise SystemExit(f"{f} missing: run evaluate.py --run {d} --t2-baseline first")
        per = pd.read_csv(f)
        out.append(dict(id=f"{r['id']} T1->MIP", label=f"{r['name']} (Task 1 head scored as MIP)",
                        per=per[per.task == "mip_from_best"].copy(), run=None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--t2base", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    es = entries(a.runs, a.t2base)
    out = PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    wo = f"w/o S{'/S'.join(map(str, EXCLUDE))}"
    md = [f"# Task 2 (MIP) Val comparison: {', '.join(e['id'] for e in es)}", "",
          "Scores at each run's best Val checkpoint (selected on this same Val set: slightly optimistic).", ""]

    rows = []
    for e in es:
        s = lmc_scores(e["per"])
        excl = lmc_scores(e["per"][~e["per"].study.isin(EXCLUDE)])
        rows.append({"run": e["label"], "LMC (study macro)": round(s["lmc_score"], 4),
                     "LMC (image mean)": round(s["lmc_score_image"], 4), f"LMC {wo}": round(excl["lmc_score"], 4),
                     **{o: s["per_organelle_study_macro"].get(o) for o in ORGANELLES}})
    md += ["## Headline (organelle columns = study-macro (SSIM+PCC)/2)", "", _md(pd.DataFrame(rows)), ""]

    mods, st = [], []
    for e in es:
        p = e["per"].assign(score=lambda x: (x.off_ssim + x.off_pcc) / 2)
        g = p.groupby(["organelle", "tl"]).agg(n=("score", "size"), score=("score", "mean")).reset_index()
        mods.append(g.assign(run=e["id"]))
        g = p.groupby(["organelle", "study", "tl"]).score.agg(["mean", "size"]).reset_index()
        st.append(g.assign(run=e["id"]))
    piv = pd.concat(mods).pivot_table(index=["organelle", "tl", "n"], columns="run", values="score").reset_index().round(3)
    piv.to_csv(out / "task2_by_modality.csv", index=False)
    md += ["## (SSIM+PCC)/2 per organelle x modality (image mean)", "", _md(piv), ""]
    pst = pd.concat(st).pivot_table(index=["organelle", "study", "tl", "size"], columns="run",
                                    values="mean").reset_index().round(3)
    pst.to_csv(out / "task2_by_study.csv", index=False)
    md += ["## (SSIM+PCC)/2 per organelle x study", "", _md(pst), ""]

    bs = []
    for i in range(len(es)):
        for j in range(i + 1, len(es)):
            A, B = es[i]["per"], es[j]["per"]
            d, lo, hi, p = boot_diff(A, B)
            d2, lo2, hi2, _ = boot_diff(A[~A.study.isin(EXCLUDE)], B[~B.study.isin(EXCLUDE)])
            _, lo3, hi3, p3 = boot_diff_study(A, B)
            bs.append({"A": es[i]["id"], "B": es[j]["id"], "LMC(A)-LMC(B)": round(d, 4), "95% CI low": round(lo, 4),
                       "95% CI high": round(hi, 4), "P(A>B)": round(p, 3), f"{wo}: diff": round(d2, 4),
                       "w/o: CI low": round(lo2, 4), "w/o: CI high": round(hi2, 4),
                       "study-level CI low": round(lo3, 4), "study-level CI high": round(hi3, 4),
                       "study-level P(A>B)": round(p3, 3)})
    bs = pd.DataFrame(bs)
    bs.to_csv(out / "task2_bootstrap.csv", index=False)
    md += ["## Paired bootstrap of the Task 2 LMC-score difference (images resampled within study x organelle, 2000 "
           "draws; study-level columns resample the Val studies within each organelle)", "", _md(bs), ""]

    # curves: Task 1 and Task 2 Val LMC-score per validation (dual-head runs only)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for e in es:
        if e["run"] is None:
            continue
        vl = e["run"]["vl"]
        ax[0].plot(vl.epoch, vl.lmc_score, "o-", label=e["label"])
        ax[1].plot(vl.epoch, vl.lmc_score_mip, "o-", label=e["label"])
    for x, t in zip(ax, ("Task 1 (best focus) Val LMC-score", "Task 2 (MIP) Val LMC-score")):
        x.set_title(t)
        x.set_xlabel("epoch")
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves_task1_task2.png", dpi=120)
    md += ["![Task 1 and Task 2 curves](curves_task1_task2.png)", ""]
    (out / "task2_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
