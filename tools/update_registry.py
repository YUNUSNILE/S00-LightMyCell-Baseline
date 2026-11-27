"""Rebuild runs/REGISTRY.csv from the run folders (PLAN.md section 7).

One row per real run (runs/_smoke/ is ignored). Notes typed by hand in the existing REGISTRY.csv
(column `notes`) are kept, matched by run_dir.

  python tools/update_registry.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmc import ORGANELLES  # noqa: E402
from lmc.metrics import lmc_scores  # noqa: E402
from lmc.runs import RUNS_ROOT, existing_runs  # noqa: E402
import pandas as pd  # noqa: E402

FIELDS = (["id", "run_dir", "date", "backbone", "input", "K", "patch", "batch_x_accum", "epochs", "steps_per_epoch", "total_steps", "lr", "seed",
           "sampler", "config_hash", "status", "last_epoch", "last_step", "best_epoch", "best_step", "val_lmc", "val_lmc_image", "val_lmc_wo_S20"]
          + [f"val_{o}" for o in ORGANELLES] + ["val_att_err_planes", "val_att_err_um", "test_lmc", "test_date", "notes"])


def main():
    reg = RUNS_ROOT / "REGISTRY.csv"
    notes = {}
    if reg.exists():
        with open(reg, newline="") as fh:
            notes = {r["run_dir"]: r.get("notes", "") for r in csv.DictReader(fh)}
    rows = []
    for d, cfg, st in existing_runs():
        dd, m, t, pv = cfg["data"], cfg["model"], cfg["train"], cfg.get("provenance", {})
        r = {"id": cfg["run"]["id"], "run_dir": d.name, "date": pv.get("created", ""), "backbone": m["backbone"],
             "input": dd["input"], "K": dd["k"] if dd["input"] == "stack" else 1, "patch": t["patch"],
             "batch_x_accum": f"{t['batch']}x{t['accum']}", "epochs": t.get("epochs", ""), "steps_per_epoch": st.get("steps_per_epoch", ""),
             "total_steps": st.get("total_steps", t.get("steps", "")), "lr": t["lr"], "seed": t["seed"],
             "sampler": t["sampler"], "config_hash": pv.get("config_hash", ""), "status": st.get("status", "?"),
             "last_epoch": st.get("epoch", ""), "last_step": st.get("step", ""),
             "best_epoch": st.get("best_epoch", ""), "best_step": st.get("best_step", ""), "notes": notes.get(d.name, "")}
        vm = d / "val_metrics.json"
        if vm.exists():
            v = json.loads(vm.read_text())
            att = v.get("attention") or {}
            if att:
                n = sum(a["n"] for a in att.values())
                r.update(val_att_err_planes=round(sum(a["err_planes"] * a["n"] for a in att.values()) / n, 3),
                         val_att_err_um=round(sum(a["err_um"] * a["n"] for a in att.values()) / n, 3))
            if v.get("lmc_score_excl") is not None:
                r["val_lmc_wo_S20"] = round(v["lmc_score_excl"], 4)
            elif (d / "val_per_image.csv").exists():  # runs older than the field: recompute from per-image scores
                pi = pd.read_csv(d / "val_per_image.csv")
                pi = pi[pi.task == "best"] if "task" in pi else pi  # S01 v2: Task 1 rows only
                r["val_lmc_wo_S20"] = round(lmc_scores(pi[pi.study != 20])["lmc_score"], 4)
            r.update(val_lmc=round(v["lmc_score"], 4), val_lmc_image=round(v["lmc_score_image"], 4),
                     **{f"val_{o}": v["per_organelle_study_macro"].get(o, "") for o in ORGANELLES})
        tm = d / "test_metrics.json"
        if tm.exists():
            tv = json.loads(tm.read_text())
            r.update(test_lmc=round(tv["lmc_score"], 4), test_date=tv.get("date", ""))
        rows.append(r)
    RUNS_ROOT.mkdir(exist_ok=True)
    with open(reg, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{reg}: {len(rows)} runs")
    for r in rows:
        print(f"  {r['id']:6s} {r['status']:11s} best_epoch {str(r['best_epoch']):>3s}  val_lmc {r.get('val_lmc', '')}  {r['run_dir']}")


if __name__ == "__main__":
    main()
