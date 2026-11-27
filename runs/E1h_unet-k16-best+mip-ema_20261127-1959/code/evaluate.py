"""Evaluate a run's checkpoint on full images with the official LMC metric.

Writes per organelle x modality x study tables and the LMC-score (study macro).

  python evaluate.py --run runs/<run_dir>                       # Val, best.pt -> <run>/eval_val_best.json
  python evaluate.py --run runs/<run_dir> --split test --confirm-test-once
                                                              # Test ONCE -> <run>/test_metrics.json
  options: --ckpt last.pt  --save-tiff (OME-TIFF predictions in <run>/pred_<split>/)  --limit N
           --t2-baseline  also score the Task 1 head against the MIP ground truth (Task 2 baseline; S01 v2)
           --tta          8x dihedral test-time augmentation (S01 v2)
The predicted tasks (Task 1 best focus, Task 2 MIP) come from the run's config (data.tasks).

The Test split is refused unless --confirm-test-once is given, and refused again if
test_metrics.json already exists (PLAN.md: Test is evaluated once, on the final model only).
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import torch
import yaml

from lmc import CACHE_ROOT, PROJECT_ROOT
from lmc.data import load_index, load_split, split_rows
from lmc.evaluation import evaluate_rows
from lmc.models import build_model, task_list


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--confirm-test-once", action="store_true")
    p.add_argument("--save-tiff", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="first N samples only (debugging)")
    p.add_argument("--t2-baseline", action="store_true", help="score the Task 1 head against the MIP target too")
    p.add_argument("--tta", action="store_true", help="8x dihedral test-time augmentation")
    args = p.parse_args()

    run = Path(args.run)
    run = run if run.is_absolute() else PROJECT_ROOT / run
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if args.split == "test":
        if not args.confirm_test_once:
            raise SystemExit("Test split: add --confirm-test-once (PLAN.md: evaluated ONCE, final model only)")
        if (run / "test_metrics.json").exists():
            raise SystemExit(f"{run / 'test_metrics.json'} already exists: the test set was already used")
        if cfg.get("provenance", {}).get("smoke"):
            raise SystemExit("refusing to evaluate Test with a smoke run")

    d, ev = cfg["data"], cfg["eval"]
    cache_root = CACHE_ROOT  # configs/paths.yaml
    rows = split_rows(load_index(cache_root), load_split(PROJECT_ROOT / d["split"]), args.split)
    if args.limit:
        rows = rows.head(args.limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, pretrained=False).to(device)
    ck = torch.load(run / args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"{run.name}: {args.ckpt} (step {ck['step']}), {args.split}: {len(rows)} samples")

    tasks = task_list(cfg)
    save_dir = run / (f"pred_{args.split}" + ("_tta" if args.tta else "")) if args.save_tiff else None
    res, summ = evaluate_rows(model, rows, cache_root, d["input"], d["k"], device, ev["tile"], ev["overlap"],
                              ev["batch"], save_dir, tasks=tasks, t2_baseline=args.t2_baseline, tta=args.tta)
    summ.update(run=run.name, ckpt=args.ckpt, step=ck["step"], split=args.split, tta=args.tta,
                date=dt.datetime.now().isoformat(timespec="seconds"))
    stem = "test_metrics" if args.split == "test" else f"eval_{args.split}_{Path(args.ckpt).stem}"
    if args.tta and args.split != "test":
        stem += "_tta"
    if args.t2_baseline and args.split != "test":
        stem += "_t2base"
    if args.limit:
        stem += f"_limit{args.limit}"
    (run / f"{stem}.json").write_text(json.dumps(summ, indent=1))
    res.to_csv(run / f"{stem}_per_image.csv", index=False)
    print(f"LMC-score {summ['lmc_score']:.4f} (image-mean {summ['lmc_score_image']:.4f})  "
          f"per organelle {summ['per_organelle_study_macro']}  n={summ['n_images']}  {summ['seconds']:.0f}s")
    for b in summ["breakdown"]["by_organelle_modality"]:
        print(f"  {b['organelle']:12s} {b['tl']:3s}  SSIM {b['ssim']:.3f}  PCC {b['pcc']:.3f}  n={b['n']}")
    for name, s in summ.get("tasks", {}).items():
        print(f"[{name}] LMC-score {s['lmc_score']:.4f} (without S20 {s['lmc_score_excl']:.4f})  "
              f"per organelle {s['per_organelle_study_macro']}  n={s['n_images']}")
    print(f"-> {run / (stem + '.json')}")


if __name__ == "__main__":
    main()
