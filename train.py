"""Train a model: TL stack (K planes) -> 4 organelle best-focus planes (PLAN.md sections 4, 7).

New run:      python train.py --config configs/E1a_unet.yaml
Resume:       python train.py --resume runs/<run_dir>          (works mid-epoch)
Paths:        dataset / cache / SSD train cache / runs folder are set in configs/paths.yaml
Overrides:    --set train.epochs=2 --set eval.limit=3   (recorded in config.yaml, change the config hash)
Smoke test:   --smoke  (output to runs/_smoke/, never in the registry)   --stop-at N  (stop after step N)

Epochs: one epoch = train.samples_per_epoch patches (null = number of training samples, 803) drawn by
the study-balanced sampler, i.e. steps_per_epoch = samples_per_epoch // (batch x accum) optimizer steps
(803 // 8 = 100). Total steps = epochs x steps_per_epoch; OneCycle runs over the total.
Validation (official full-image LMC-score on Val) every train.val_every_epochs epochs and at the end;
best.pt = best Val LMC-score. last.pt is written at the end of every epoch (and on Ctrl+C).
Each epoch's patch draws come from a generator seeded by (seed, epoch), so a resumed run replays
exactly the remaining draws of the interrupted epoch.

Console: a tqdm bar per epoch (console only); every printed line is mirrored to stdout.log.
The non-finite guards and grad accumulation follow baseline_teammate/train.py
(github.com/YUNUSNILE/S00-LightMyCell-Baseline, commit fc2ab9a).
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lmc import CACHE_ROOT, ORGANELLES, PROJECT_ROOT, TRAIN_CACHE_ROOT  # noqa: E402
from lmc import runs  # noqa: E402
from lmc.data import PatchDataset, load_index, load_split, split_rows, study_weights  # noqa: E402
from lmc.evaluation import evaluate_rows  # noqa: E402
from lmc.losses import focus_kl, masked_loss  # noqa: E402
from lmc.models import build_model, count_params, param_groups, task_list  # noqa: E402

TRAIN_FIELDS = ["epoch", "step", "lr", "loss", "l1", "ssim", "pcc", "focus", "sec_per_step", "data_wait_frac", "gpu_peak_gb",
                "skipped"]
VAL_FIELDS = (["epoch", "step", "lmc_score", "lmc_score_image", "lmc_score_excl"]
              + [f"att_err_planes_{o}" for o in ORGANELLES] + [f"att_err_um_{o}" for o in ORGANELLES]
              + [f"{m}_{o}" for o in ORGANELLES for m in ("ssim", "pcc", "score")]
              # S01 v2: Task 2 (MIP head) and the Task 2 baseline (Task 1 head scored against the MIP)
              + ["lmc_score_mip", "lmc_score_mip_excl", "lmc_score_mip_from_best"]
              + [f"score_mip_{o}" for o in ORGANELLES]
              + ["n_samples", "seconds", "is_best", "best_lmc", "best_epoch"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=2")
    p.add_argument("--resume", help="run folder to resume (uses its config.yaml and last.pt)")
    p.add_argument("--smoke", action="store_true", help="write to runs/_smoke/ (never registered)")
    p.add_argument("--stop-at", type=int, default=0, help="stop after this optimizer step as if interrupted")
    p.add_argument("--allow-duplicate", action="store_true")
    return p.parse_args()


def hms(sec):
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def append_csv(path, fields, row):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


class EpochSampler(Sampler):
    """Study-balanced draws, epoch after epoch. Epoch e uses a generator seeded by (seed, e), so the
    draws of any epoch are reproducible; `skip` drops the draws already consumed (mid-epoch resume)."""

    def __init__(self, weights, per_epoch, first_epoch, last_epoch, seed, skip=0):
        self.w = torch.as_tensor(weights, dtype=torch.double)
        self.per_epoch, self.first, self.last, self.seed, self.skip = per_epoch, first_epoch, last_epoch, seed, skip

    def __len__(self):
        return (self.last - self.first + 1) * self.per_epoch - self.skip

    def __iter__(self):
        for e in range(self.first, self.last + 1):
            g = torch.Generator().manual_seed(self.seed * 100_003 + e)
            idx = torch.multinomial(self.w, self.per_epoch, replacement=True, generator=g).tolist()
            yield from (idx[self.skip:] if e == self.first else idx)


def plot_curves(run_dir, steps_per_epoch):
    tl, vl = run_dir / "train_log.csv", run_dir / "val_log.csv"
    if not tl.exists():
        return
    import pandas as pd

    t = pd.read_csv(tl)
    ep = t.step / steps_per_epoch
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(ep, t.loss, alpha=0.35, label="train loss (per log interval)")
    ax[0].plot(ep, t.loss.rolling(10, min_periods=1).mean(), label="train loss (smoothed)")
    ax[0].set_title("Training loss (masked L1 + 0.5(1-SSIM) + 0.5(1-PCC))")
    if vl.exists():
        v = pd.read_csv(vl)
        ax[1].plot(v.epoch, v.lmc_score, "o-", label="Val LMC-score (study macro)")
        ax[1].plot(v.epoch, v.lmc_score_image, "s--", alpha=0.6, label="Val LMC-score (image mean)")
        for col, lab in (("lmc_score_mip", "Task 2 (MIP head) LMC-score"),
                         ("lmc_score_mip_from_best", "Task 2 baseline (Task 1 head vs MIP)")):
            if col in v and v[col].notna().any():
                ax[1].plot(v.epoch, v[col], "^-", label=lab)
        ax[1].set_title("Validation LMC-score (official metric, full images)")
        for o in ORGANELLES:
            if v[f"score_{o}"].notna().any():
                ax[2].plot(v.epoch, v[f"score_{o}"], "o-", label=o)
        ax[2].set_title("Validation (SSIM+PCC)/2 per organelle")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=110)
    plt.close(fig)


def print_val_table(epoch, epochs, step, summ, best, best_epoch, updated):
    by = {b["organelle"]: b for b in summ["breakdown"]["by_organelle"]}
    print(f"\n--- validation after epoch {epoch}/{epochs} (step {step}) -- {summ['n_samples']} images, "
          f"{hms(summ['seconds'])} ---")
    att = summ.get("attention", {})
    head = f"  {'organelle':<13}{'SSIM':>7}{'PCC':>8}{'(S+P)/2':>9}{'macro':>8}{'n':>5}"
    print(head + ("   attn-peak err (planes / um, <=1 plane)" if att else ""))
    for o in ORGANELLES:
        if o in by:
            b = by[o]
            a = att.get(o)
            print(f"  {o:<13}{b['ssim']:>7.3f}{b['pcc']:>8.3f}{b['score']:>9.3f}"
                  f"{summ['per_organelle_study_macro'][o]:>8.3f}{b['n']:>5d}"
                  + (f"   {a['err_planes']:.2f} / {a['err_um']:.2f}, {100 * a['within_1_plane']:.0f}%" if a else ""))
    print(f"  LMC-score {summ['lmc_score']:.4f} (image-mean {summ['lmc_score_image']:.4f}; "
          f"without S{'/S'.join(str(x) for x in summ.get('excluded_studies', []))} {summ.get('lmc_score_excl', float('nan')):.4f})   "
          f"best {best:.4f} @ epoch {best_epoch}   best.pt {'UPDATED' if updated else 'unchanged'}", flush=True)
    for name, s in summ.get("tasks", {}).items():
        po = "  ".join(f"{o[:4]} {s['per_organelle_study_macro'].get(o, float('nan')):.3f}" for o in ORGANELLES)
        print(f"  [{name}] LMC-score {s['lmc_score']:.4f} (without S20 {s['lmc_score_excl']:.4f})   {po}", flush=True)
    print(flush=True)


def main():
    args = parse_args()
    if bool(args.config) == bool(args.resume):
        raise SystemExit("give exactly one of --config or --resume")

    if args.resume:
        run_dir = Path(args.resume)
        run_dir = run_dir if run_dir.is_absolute() else PROJECT_ROOT / run_dir
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        if args.set:
            raise SystemExit("--set is not allowed with --resume (the config of a run is frozen)")
        if runs.read_status(run_dir).get("status") == "done":
            raise SystemExit(f"{run_dir} is already done; nothing to resume")
        if cfg["provenance"]["code_sha256"] != runs.code_sha256():
            print(f"WARNING: code changed since this run started ({cfg['provenance']['code_sha256']} -> "
                  f"{runs.code_sha256()}); the original code is in {run_dir / 'code'}")
    else:
        cfg = runs.resolve(runs.apply_overrides(runs.load_config(args.config), args.set))
        if not args.smoke:
            runs.check_not_done(cfg, args.allow_duplicate)
        run_dir = runs.new_run_dir(cfg, smoke=args.smoke)
        cfg["provenance"] = runs.provenance(cfg, sys.argv)
        cfg["provenance"]["smoke"] = bool(args.smoke)
        runs.save_config(run_dir, cfg)
        runs.snapshot_code(run_dir)
    runs.tee_output(run_dir)
    t_start = time.time()

    d, t, ev, lw = cfg["data"], cfg["train"], cfg["eval"], cfg["loss"]
    tasks = task_list(cfg)  # S01 v2: ['best'] (default), ['best', 'mip'] or ['mip']
    seed = int(t["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    # S01 v2: configurable (default True as before). The RTX 3050 laptop hit its GPU/driver fault (TDR, "illegal
    # memory access") with autotuned kernels at step ~170 of the first E1d smoke test; S01 v1 ran ConvNeXt-T at
    # batch 4 x 2 for 5000 steps cleanly with the default (non-autotuned) cuDNN algorithms.
    torch.backends.cudnn.benchmark = bool(t.get("cudnn_benchmark", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = CACHE_ROOT  # configs/paths.yaml

    # ---- data
    index = load_index(cache_root)
    split = load_split(PROJECT_ROOT / d["split"])
    if cfg["provenance"]["split_sha256"] != runs.file_sha256(PROJECT_ROOT / d["split"]):
        raise SystemExit("split file changed since the run started: refusing")
    if d.get("overfit_samples"):  # E0 overfit check: train and validate on the same few images
        keys = {tuple(s) for s in d["overfit_samples"]}
        train_df = index[[(s, i, tl) in keys for s, i, tl in zip(index.study, index.image_id, index.tl)]]
        train_df = train_df.reset_index(drop=True)
        val_df = train_df.copy()
    else:
        train_df, val_df = split_rows(index, split, "train"), split_rows(index, split, "val")
    if ev.get("limit"):
        val_df = val_df.groupby("study").head(1).head(int(ev["limit"])).reset_index(drop=True)
    train_root = TRAIN_CACHE_ROOT  # configs/paths.yaml (SSD mirror, or cache_root when null)
    missing = [r for r in train_df.dir.unique() if not (train_root / r / "meta.json").exists()]
    if missing:
        raise SystemExit(f"train cache {train_root} lacks {len(missing)} images (e.g. {missing[0]}); "
                         "run tools/mirror_cache.py or set train_cache_root: null in configs/paths.yaml")

    per_step = t["batch"] * t["accum"]
    samples_per_epoch = int(t.get("samples_per_epoch") or len(train_df))
    spe = samples_per_epoch // per_step
    if spe < 1:
        raise SystemExit(f"samples_per_epoch {samples_per_epoch} < batch x accum {per_step}")
    epochs, val_every = int(t["epochs"]), int(t["val_every_epochs"])
    total = epochs * spe

    # ---- model / optimiser
    model = build_model(cfg).to(device)
    # S01 v2: encoder lr multiplier (default 1.0 = one group, as before) and bf16 autocast (default fp32)
    enc_mult = float(t.get("enc_lr_mult", 1.0))
    groups = param_groups(model, t["lr"], enc_mult)
    opt = torch.optim.AdamW(groups, lr=t["lr"], weight_decay=t["wd"])
    max_lr = [g["lr"] for g in groups] if len(groups) > 1 else t["lr"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total, pct_start=t["pct_start"])
    precision = t.get("precision", "fp32")
    use_bf16 = precision == "bf16" and device.type == "cuda"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("train.precision=bf16 but this GPU does not support bf16")
    bn_bufs = [b for n_, b in model.named_buffers() if "running" in n_]
    sync_steps = bool(t.get("sync_steps", False)) and device.type == "cuda"
    # S01 v2: exponential moving average of the weights (Polyak averaging), used for validation and saved as best.pt.
    # E1e's Val score moved by +-0.02-0.04 between checkpoints; an average over ~1/(1-decay) steps smooths that.
    ema_decay = float(t.get("ema_decay", 0.0))
    ema = None
    if ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
    # checkpoint selection: 'task1' (default) = Task 1 LMC-score; 'mean' = mean of the Task 1 and Task 2 LMC-scores
    select_on = t.get("select_on", "task1")
    if select_on not in ("task1", "mean"):
        raise SystemExit(f"train.select_on must be task1 or mean, got {select_on!r}")

    step, best, best_step, best_epoch = 0, -math.inf, 0, 0
    if args.resume:
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        step, best, best_step, best_epoch = ck["step"], ck["best"], ck["best_step"], ck.get("best_epoch", 0)
        torch.set_rng_state(ck["rng_cpu"])

    bar = "=" * 78
    print(f"\n{bar}\n{'RESUME' if args.resume else 'START'}  {runs.rel(run_dir)}   "
          f"({time.strftime('%Y-%m-%d %H:%M:%S')})\n{bar}")
    print(f"  backbone      {cfg['model']['backbone']} ({count_params(model) / 1e6:.2f}M params), "
          f"{'bf16 autocast' if use_bf16 else 'fp32'}, {device}"
          + (f", encoder lr x{enc_mult:g}" if len(groups) > 1 else ""))
    if getattr(model, "n_depthwise_hooked", 0):
        print(f"  depthwise     {model.n_depthwise_hooked} depthwise convs get NCHW inputs (bf16 GPU workaround)")
    print(f"  input         {d['input']}, K={d['in_channels']} channel(s)")
    print(f"  data          train {len(train_df)} samples / {train_df.study.nunique()} studies, "
          f"val {len(val_df)} samples / {val_df.study.nunique()} studies")
    print(f"  epochs        {epochs} x {spe} steps/epoch = {total} steps "
          f"({samples_per_epoch} samples/epoch, batch {t['batch']} x accum {t['accum']}, patch {t['patch']})")
    print(f"  optimiser     AdamW lr {t['lr']:g} wd {t['wd']:g}, OneCycle over {total} steps, clip {t['clip']}")
    print(f"  validation    every {val_every} epochs ({val_every * spe} steps) + at the end; best.pt on Val LMC-score")
    if args.resume:
        bs = f"best {best:.4f} @ epoch {best_epoch}" if best > -math.inf else "no validation yet"
        print(f"  resumed at    step {step} (epoch {step // spe + 1}, step {step % spe} of it); {bs}")
    print(bar, flush=True)
    runs.write_status(run_dir, status="running", step=step, epoch=step // spe, epochs=epochs, steps_per_epoch=spe,
                      total_steps=total, best_step=best_step, best_epoch=best_epoch,
                      best_val_lmc=None if best == -math.inf else best)

    w = study_weights(train_df, t["sampler"])
    first_epoch = step // spe + 1
    sampler = EpochSampler(w, spe * per_step, first_epoch, epochs, seed, skip=(step % spe) * per_step)
    ds = PatchDataset(train_df, train_root, d["input"], d["k"], t["patch"], t["augment"], t.get("zaug"), tasks)
    if "mip" in tasks and "has_mip_Nucleus" not in train_df:
        raise SystemExit("data.tasks has 'mip' but the cache has no mip_index.csv: run tools/build_mip_cache.py")
    lam = float((lw.get("focus") or {}).get("weight", 0.0))
    sigma = float((lw.get("focus") or {}).get("sigma", 1.0))
    if lam > 0 and not getattr(model, "has_attention", False):
        raise SystemExit("loss.focus.weight > 0 needs a model with learned z-attention (model.zattn.mode: softmax)")
    loader = DataLoader(ds, t["batch"], sampler=sampler, drop_last=True, num_workers=t["workers"],
                        pin_memory=True, persistent_workers=t["workers"] > 0)
    it = iter(loader)

    def save_ckpt(name):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
              "best": best, "best_step": best_step, "best_epoch": best_epoch,
              "rng_cpu": torch.get_rng_state(), "config": cfg}
        if ema is not None:  # last.pt keeps the raw weights + EMA state (resume); best.pt holds the EMA weights
            ck["ema"] = ema.state_dict()
            if name == "best.pt":
                ck["raw_model"], ck["model"] = ck["model"], ema.module.state_dict()
        torch.save(ck, run_dir / name)

    val_secs = [2.4 * len(val_df)]  # prior guess until the first validation is measured

    def validate(epoch):
        nonlocal best, best_step, best_epoch
        res, summ = evaluate_rows(ema.module if ema is not None else model, val_df, cache_root, d["input"], d["k"], device, ev["tile"], ev["overlap"],
                                  ev["batch"], tasks=tasks, t2_baseline=bool(ev.get("t2_baseline", False)),
                                  tta=bool(ev.get("tta", False)))
        model.train()
        val_secs.append(summ["seconds"])
        sel = summ["lmc_score"]
        if select_on == "mean" and "mip" in summ.get("tasks", {}):
            sel = (summ["lmc_score"] + summ["tasks"]["mip"]["lmc_score"]) / 2
        summ["selection_score"] = sel
        updated = sel > best
        if updated:
            best, best_step, best_epoch = sel, step, epoch
            save_ckpt("best.pt")
            summ.update(step=step, epoch=epoch, split="val", run=run_dir.name)
            (run_dir / "val_metrics.json").write_text(json.dumps(summ, indent=1))
            res.to_csv(run_dir / "val_per_image.csv", index=False)
        row = {"epoch": epoch, "step": step, "lmc_score": summ["lmc_score"], "lmc_score_image": summ["lmc_score_image"],
               "lmc_score_excl": summ.get("lmc_score_excl"),
               **{f"att_err_planes_{o}": v["err_planes"] for o, v in summ.get("attention", {}).items()},
               **{f"att_err_um_{o}": v["err_um"] for o, v in summ.get("attention", {}).items()},
               "n_samples": summ["n_samples"], "seconds": summ["seconds"], "is_best": int(updated),
               "best_lmc": best, "best_epoch": best_epoch}
        for b in summ["breakdown"]["by_organelle"]:
            row.update({f"ssim_{b['organelle']}": b["ssim"], f"pcc_{b['organelle']}": b["pcc"],
                        f"score_{b['organelle']}": b["score"]})
        for name, s in summ.get("tasks", {}).items():
            row[f"lmc_score_{name}"] = s["lmc_score"]
            if name == "mip":
                row["lmc_score_mip_excl"] = s["lmc_score_excl"]
                row.update({f"score_mip_{o}": v for o, v in s["per_organelle_study_macro"].items()})
        append_csv(run_dir / "val_log.csv", VAL_FIELDS, row)
        print_val_table(epoch, epochs, step, summ, best, best_epoch, updated)
        runs.write_status(run_dir, status="running", step=step, epoch=epoch, epochs=epochs, best_step=best_step,
                          best_epoch=best_epoch, best_val_lmc=best)
        plot_curves(run_dir, spe)

    # A stop exactly at the end of a validation epoch (or during its validation) leaves that validation
    # undone: run it before continuing.
    if args.resume and step % spe == 0 and step > 0:
        done_ep = step // spe
        logged = set()
        if (run_dir / "val_log.csv").exists():
            with open(run_dir / "val_log.csv", newline="") as fh:
                logged = {int(r["epoch"]) for r in csv.DictReader(fh)}
        if (done_ep % val_every == 0 or done_ep == epochs) and done_ep not in logged:
            print(f"validation of epoch {done_ep} was not completed before the stop: running it now")
            validate(done_ep)
            save_ckpt("last.pt")

    model.train()
    recent = deque(maxlen=30)  # recent optimizer-step durations (median -> s/step and ETAs)
    session_steps = 0  # the first steps after a (re)start include worker spawn / cuDNN tuning: not timed
    t_run = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(first_epoch, epochs + 1):
            start_in_epoch = step % spe if epoch == first_epoch else 0
            pbar = tqdm(total=spe, initial=start_in_epoch, desc=f"epoch {epoch:>2d}/{epochs}", file=sys.__stdout__,
                        dynamic_ncols=True, ascii=" =", leave=True, mininterval=1.0,
                        bar_format="{desc} |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
            ep_loss, ep_n, t_ep = 0.0, 0, time.time()
            acc = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "pcc": 0.0, "focus": 0.0, "n": 0, "wait": 0.0, "skipped": 0}
            t_log = time.time()
            for _ in range(start_in_epoch, spe):
                t_step = time.time()
                opt.zero_grad(set_to_none=True)
                bad = False
                for _ in range(t["accum"]):
                    tw = time.time()
                    b = next(it)
                    acc["wait"] += time.time() - tw
                    x, y, m = (b["x"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True),
                               b["m"].to(device))
                    bn_backup = [b_.clone() for b_ in bn_bufs]
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        pred = model(x, b["zpos"].to(device), b["meta"].to(device))
                    loss, parts, _, _ = masked_loss(pred, y, m, lw["w_l1"], lw["w_ssim"], lw["w_pcc"])
                    total_loss = loss
                    if lam > 0:  # E4: best_z used as a TRAINING signal only
                        fl = focus_kl(model.aux["attn_mean"], b["bz"].to(device), m, sigma)
                        total_loss = loss + lam * fl
                        acc["focus"] += fl.item()
                    if not torch.isfinite(total_loss):
                        bad = True
                        for b_, s_ in zip(bn_bufs, bn_backup):  # a non-finite forward also polluted BN statistics
                            b_.copy_(s_)
                        continue
                    (total_loss / t["accum"]).backward()
                    if sync_steps:  # S01 v2: bounds the GPU queue (laptop fault mitigation, train.sync_steps)
                        torch.cuda.synchronize()
                    acc["loss"] += loss.item()
                    acc["n"] += 1
                    for k_, v_ in parts.items():
                        acc[k_] += v_
                    ep_loss += loss.item()
                    ep_n += 1
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
                if bad or not torch.isfinite(gnorm):
                    acc["skipped"] += 1  # never apply a non-finite gradient
                else:
                    opt.step()
                    if ema is not None:
                        ema.update_parameters(model)
                sched.step()
                step += 1
                session_steps += 1
                if session_steps > 2:
                    recent.append(time.time() - t_step)
                sps = float(np.median(recent)) if recent else None

                # ETA for the whole run: remaining steps + remaining validations
                n_val_left = len([e for e in range(epoch, epochs + 1) if e % val_every == 0 or e == epochs])
                eta = hms((total - step) * sps + n_val_left * val_secs[-1]) if sps else "..."
                pbar.set_postfix_str(f"loss {ep_loss / max(ep_n, 1):.4f}, lr {sched.get_last_lr()[0]:.2e}, "
                                     f"{f'{sps:.2f}' if sps else '--'}s/step, run ETA {eta}", refresh=False)
                pbar.update(1)

                if step % t["log_every"] == 0 or step == total:
                    n = max(acc["n"], 1)
                    el = time.time() - t_log
                    nsteps = t["log_every"] if step % t["log_every"] == 0 else step % t["log_every"]
                    append_csv(run_dir / "train_log.csv", TRAIN_FIELDS, {
                        "epoch": epoch, "step": step, "lr": sched.get_last_lr()[0], "loss": acc["loss"] / n,
                        "l1": acc["l1"] / n, "ssim": acc["ssim"] / n, "pcc": acc["pcc"] / n,
                        "focus": acc["focus"] / n if lam > 0 else "",
                        "sec_per_step": el / max(nsteps, 1), "data_wait_frac": acc["wait"] / max(el, 1e-9),
                        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                        "skipped": acc["skipped"]})
                    acc = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "pcc": 0.0, "focus": 0.0, "n": 0, "wait": 0.0,
                           "skipped": 0}
                    t_log = time.time()

                if args.stop_at and step >= args.stop_at and step < total:
                    pbar.close()
                    save_ckpt("last.pt")
                    runs.write_status(run_dir, status="interrupted", step=step, epoch=epoch - 1, epochs=epochs,
                                      best_step=best_step, best_epoch=best_epoch, best_val_lmc=best)
                    print(f"--stop-at {args.stop_at}: stopped at step {step} (epoch {epoch}, step "
                          f"{step - (epoch - 1) * spe}/{spe}); "
                          f"resume with: python train.py --resume {runs.rel(run_dir)}")
                    return
            pbar.close()
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            print(f"epoch {epoch:>2d}/{epochs}  loss {ep_loss / max(ep_n, 1):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - t_ep) / max(spe - start_in_epoch, 1):.2f}s/step  epoch {hms(time.time() - t_ep)}  "
                  f"peak GPU {peak:.2f}GB  run ETA {eta}",
                  flush=True)
            if epoch % val_every == 0 or epoch == epochs:
                validate(epoch)
            save_ckpt("last.pt")
            runs.write_status(run_dir, status="running", step=step, epoch=epoch, epochs=epochs, best_step=best_step,
                              best_epoch=best_epoch, best_val_lmc=best)
    except KeyboardInterrupt:
        save_ckpt("last.pt")
        runs.write_status(run_dir, status="interrupted", step=step, epoch=step // spe, epochs=epochs,
                          best_step=best_step, best_epoch=best_epoch, best_val_lmc=best)
        print(f"\ninterrupted at step {step}; resume with: python train.py --resume {runs.rel(run_dir)}")
        return
    except Exception as e:
        runs.write_status(run_dir, status="failed", step=step, error=repr(e))
        raise
    runs.write_status(run_dir, status="done", step=step, epoch=epochs, epochs=epochs, best_step=best_step,
                      best_epoch=best_epoch, best_val_lmc=best)
    plot_curves(run_dir, spe)
    print(f"{bar}\nDONE  {run_dir.name}\n  total time    {hms(time.time() - t_start)} (this session; "
          f"training+validation {hms(time.time() - t_run)})\n  best          Val LMC-score {best:.4f} at epoch "
          f"{best_epoch} (step {best_step})\n  best.pt       {runs.rel(run_dir / 'best.pt')}\n  curves        "
          f"{runs.rel(run_dir / 'curves.png')}\n{bar}", flush=True)


if __name__ == "__main__":
    main()
