# S00-LightMyCell-Baseline, version 2: the plain U-Net control for LMC2 Task 1 and Task 2

Strategy S00 (control: plain 2D U-Net from scratch) of our Light My Cells 2 course project, rebuilt on the same data,
split and framework as the team's other models (S01 v2 and the teammate's T01 model, TS01 = E1b): the K = 16 cache
built by `tools/build_cache.py` and `splits/split_v1.json`.
File-by-file changes: [`CHANGES_S00.md`](CHANGES_S00.md). The teammate's original README of the framework:
[`README_E1b_package.md`](README_E1b_package.md).

**Versions.** v1, the original baseline (`prepare_data.py` pipeline, one TL plane per image, Val studies 3, 6, 21,
`runs/baseline_v1`), is kept as tag [`v1`](https://github.com/YUNUSNILE/S00-LightMyCell-Baseline/tree/v1) (commit `fc2ab9a`). v2 (this branch) replaces it; see
[Why S00 v2](#why-s00-v2).

**Credits.** The training and evaluation framework (`lmc/`, `train.py`, `evaluate.py`, `tools/`, `splits/`,
`reference_run/`, `docs/`) is the team's E1b package, written by a teammate on top of this repository's v1 code, plus
the S01 v2 additions (Task 2 MIP targets, dual head, EMA, mean selection). The plain U-Net class and the official-metric
port come from v1. RUNet is ported from github.com/MWod/LightMyCells (Apache-2.0); the metric follows the official
LightMyCells evaluation (codeberg.org/FranceBioImaging/LightMyCellsChallenge). Details are in the file headers.

> AI-assisted working note (Claude). The course forbids generative-AI text in the submitted report: use this as
> source material only.

| | |
|---|---|
| Input | The transmitted-light (TL) z-stack of one image (BF, PC or DIC), resampled to K = 16 planes. No best_z from the TSV is used as input (`tools/check_pipeline.py` no-oracle check) |
| Task 1 output | 4 fluorescence images (Nucleus, Mitochondria, Tubulin, Actin), each at that organelle's own best-focus plane |
| Task 2 output | The 4 organelles' maximum-intensity projections (MIP) over all fluorescence z-planes, predicted directly |
| Network | The S00 plain 2D U-Net (depth 4, base 32, BatchNorm, ReLU, random init), 7.77 M parameters; first conv takes the 16 planes; 8 output channels (4 per task) |
| Data / split | 28 studies on disk, `splits/split_v1.json`: train 17 studies (803 samples) / Val 5 (58) / **Test 6 (71, locked)** |
| Metric | Official LightMyCells SSIM + PCC on percentile-normalised images; LMC-score = organelle mean of the study-macro mean of (SSIM+PCC)/2 |

## Why S00 v2

S00 v1 (tag `v1`, `runs/baseline_v1`, LMC 0.4924 on Val studies 3, 6, 21) cannot be compared with the team's runs
and cannot be used in the final evaluation:

| | S00 v1 (`baseline_v1`, 7 Nov) | S00 v2 (E1h, this branch) |
|---|---|---|
| Data | `prepare_data.py`, 11 studies | team K = 16 cache, 28 studies, 932 samples |
| Split | train 1 2 4 5 7 20 42 57 / val 3 6 21 | split_v1: train 17 / Val 20 72 81 82 86 / Test 21 56 57 62 73 80 (locked) |
| Leakage vs team test | trained on S57 and selected on S21, both team Test studies | none: Test untouched |
| Input | 1 TL plane chosen with the TSV TL best_z (oracle, not allowed at test time) | the whole TL stack, 16 resampled planes |
| Tasks | Task 1 only | Task 1 + Task 2 (MIP) heads in one network |
| Checkpoint selection | crop-level score on 4 random crops | official full-image Val LMC-score, mean of Task 1 and Task 2 |
| Weights | raw | EMA of the weights (decay 0.999, BatchNorm statistics averaged too) |
| Sampler | study-balanced (`--balance study`) | study weight proportional to sqrt(n_samples) (team protocol) |
| Batch | 8 | 4 x accumulation 2 (effective 8, as E1g) |
| Same | U-Net class, per-image 1-99.8 percentile normalisation, loss L1 + 0.5(1-SSIM) + 0.5(1-PCC), AdamW lr 1e-3 wd 1e-4, OneCycle, 50 x 100 steps of 8 patches (512²), 8 dihedral flips, clip 1.0, bf16, seed 0 | |

The teammate's E1a is the same network in the team framework with the base protocol (fp32, batch 2 x 4, Task 1 only,
no EMA), run on their GTX 1650. E1h re-trains it with the recipe of E1g, the final S01 v2 model (U-Net with a
pretrained ConvNeXt-T encoder, in the S01 v2 package, not in this repository), so:

- **E1g vs E1h** changes one axis, the backbone (pretrained ConvNeXt-T U-Net vs plain U-Net): the S01 effect at a
  fixed recipe, for both tasks.
- **E1h vs E1a** keeps the network and changes the recipe (dual head + EMA + mean selection, bf16, batch 4 x 2).

## Runs

| Run | Config | What |
|---|---|---|
| **E1h** | `configs/E1h_unet_mip_ema.yaml` | S00 network + Task 1 and Task 2 heads + EMA, checkpoint on mean(Task 1, Task 2) = the E1g recipe: **the S00 v2 model** |

Reference runs (same data, split, metric): E1a, E1b (teammate; `reference_run/` holds E1b's logs and per-image Val
scores), E1d / T2-0 / E1e / E1g (S01 v2 package; their numbers are quoted below).

## Reproduce

Environment: Windows 11, RTX 3050 Laptop 6 GB, Python 3.13.5, torch 2.8.0+cu128, timm 1.0.31,
segmentation-models-pytorch 0.6.0, pandas 2.2.3 (`requirements.txt` holds the teammate's GTX 1650 pins). The code is
byte-identical to the S01 v2 package (same `code_sha256` `10085315143e` as E1g).

```bash
# 0. paths: edit configs/paths.yaml (dataset, TSV folder, cache; it holds the paths of our machine)
python tools/build_cache.py --workers 2          # K=16 TL stacks + best-focus targets (~35 GB; resumable)
python tools/build_mip_cache.py --workers 3      # Task 2 MIP targets
python tools/check_pipeline.py                   # 33/33 PASS (logs/check_pipeline.log)
# 1. smoke test (300 steps; the laptop GPU fault showed up at step ~170 once)
python train.py --config configs/E1h_unet_mip_ema.yaml --smoke --set train.epochs=1 --set train.samples_per_epoch=2400 --set eval.limit=3
# 2. run
python train.py --config configs/E1h_unet_mip_ema.yaml           # the S00 v2 model
python evaluate.py --run runs/E1h_<stamp>                        # re-score best.pt on Val (both tasks)
# 3. compare (Task 1: compare_runs.py; Task 2: compare_task2.py, added for S00 v2).
#    The E1g / E1d runs come from the S01 v2 package (S01_v2/ next to this folder in our project).
python tools/update_registry.py
python tools/compare_runs.py --runs "reference_run/E1b_*" "runs/E1h_*" "../S01_v2/runs/E1g_*" --out reports/S00_compare
python tools/compare_task2.py --runs "runs/E1h_*" "../S01_v2/runs/E1g_*" --t2base "../S01_v2/runs/E1d_*" --out reports/S00_compare
```

**Trained weights in this repository:** `runs/E1h_unet-k16-best+mip-ema_20261127-1959/best.pt` (31 MB) is a
weights-only export of the E1h checkpoint: the EMA weights at the best Val epoch plus the config, identical tensor
for tensor to the training run's `best.pt` (whose 119 MB also hold the optimizer, scheduler and raw weights).
`evaluate.py --run runs/E1h_unet-k16-best+mip-ema_20261127-1959` loads it as is (cache and `configs/paths.yaml` needed).

Interrupted? `python train.py --resume runs/<run_dir>`. **Do not run `evaluate.py --split test`**: Test is evaluated
once, on the team's final model (`--confirm-test-once`).

## Results (Val, 27 Nov 2026)

**S00 v2 model: E1h `best.pt` (EMA weights, epoch 50 = the last epoch): Task 1 LMC 0.5138, Task 2 LMC 0.5405.**
Val = studies 20, 72, 81, 82, 86 (58 samples); official metric. Test (6 studies) is untouched.
Training: 5000/5000 steps, 0 skipped, 0.68 s/step median, 2.19 GB peak GPU, 1h35m in total (11 dual-task validations
of ~150 s each). `evaluate.py` on `best.pt` reproduces both scores exactly.

### Task 1 (best focus)

| Run | Network | Recipe | Task 1 LMC | w/o S20 | Nuc | Mito | Tub | Act | Best epoch |
|---|---|---|---|---|---|---|---|---|---|
| E1a (teammate) | plain U-Net (S00) | base: fp32, 2 x 4, Task 1, raw weights | 0.5281 | 0.6161 | 0.558 | 0.451 | 0.519 | 0.585 | 45 |
| E1b (teammate, T01) | U-Net + ResNet-34 | base | 0.5328 | 0.6208 | 0.578 | 0.460 | 0.526 | 0.567 | 35 |
| **E1h (S00 v2)** | plain U-Net (S00) | E1g: bf16, 4 x 2, dual head, EMA, mean selection | **0.5138** | **0.5938** | 0.563 | 0.424 | 0.492 | 0.576 | 50 (last) |
| E1h raw weights (`last.pt`, not in the repository) | same run, no EMA | | 0.5192 | | 0.570 | 0.426 | 0.499 | 0.582 | 50 |
| E1g (S01 v2) | U-Net + ConvNeXt-T | E1g | 0.5782 | 0.6525 | 0.661 | 0.480 | 0.554 | 0.619 | 25 |

### Task 2 (MIP)

| Run | How Task 2 is made | Task 2 LMC | w/o S20 | Nuc | Mito | Tub | Act |
|---|---|---|---|---|---|---|---|
| **E1h (S00 v2)** | MIP head | **0.5405** | **0.6117** | 0.600 | 0.449 | 0.523 | 0.591 |
| E1h raw weights | MIP head, no EMA | 0.5440 | 0.6177 | 0.601 | 0.446 | 0.526 | 0.603 |
| T2-0 (S01 v2) | E1d's Task 1 head scored against the MIP | 0.5740 | 0.6531 | 0.683 | 0.438 | 0.538 | 0.638 |
| E1g (S01 v2) | MIP head | 0.5896 | 0.6655 | 0.660 | 0.491 | 0.565 | 0.642 |

### Paired bootstraps (2000 draws; image level = images resampled within study x organelle; study level = Val studies resampled within organelle)

| A - B | Task | Diff | Image-level 95% CI | Study-level 95% CI | Study-level P(A > B) |
|---|---|---|---|---|---|
| E1g - E1h | 1 | **+0.065** | [+0.051, +0.077] | [+0.042, +0.086] | 1.00 |
| E1g - E1h | 2 | **+0.049** | [+0.035, +0.062] | [+0.027, +0.069] | 1.00 |
| E1b - E1h | 1 | +0.019 | [+0.014, +0.024] | [+0.006, +0.034] | 1.00 |
| E1h - T2-0 | 2 | -0.034 | [-0.044, -0.022] | [-0.066, +0.000] | 0.03 |

(E1a's per-image scores are not available here, so E1h vs E1a has no bootstrap; its point difference is -0.014.)

### What it shows

- **S01 vs S00 at a fixed recipe:** the pretrained ConvNeXt-T encoder is worth +0.065 on Task 1 and +0.049 on Task 2,
  on all 4 organelles in both tasks, and both study-level CIs exclude 0. This is the cleanest S01 effect in the project:
  E1g and E1h differ only in the backbone (same code hash, data, split, loss, schedule, seed, EMA, selection).
- **The E1g recipe does not help the plain U-Net:** E1h Task 1 is 0.014 below E1a (0.009 with the raw weights), while
  the same recipe lifted ConvNeXt-T from 0.561 (E1d) to 0.578 (E1g). E1h vs E1a also changes hardware, precision and
  micro-batch, and E1a's per-image scores are not here, so this is a point estimate from one seed.
- **EMA lag:** with decay 0.999 the EMA keeps ~60% weight on the random initialisation after 500 steps, so E1h scores
  only 0.16 at epoch 5 (E1a: ~0.51) and rises monotonically to epoch 50; the raw weights at epoch 50 are 0.005 higher on
  Task 1 and 0.004 higher on Task 2. Its best checkpoint is the last one, so E1h's Val score carries no epoch-selection
  optimism, unlike E1a (epoch 45) and E1g (epoch 25). A pretrained encoder starts near a good solution, which is why
  the same EMA is harmless for E1g.
- **Task 2:** the plain U-Net's dedicated MIP head (0.541) is worse than ConvNeXt-T's *Task 1* head scored as MIP
  (T2-0, 0.574): for Task 2 the encoder matters more than having a MIP head.
- Per modality (Task 1, image mean): E1h is close to E1b on BF/PC Actin, BF Nucleus and DIC Tubulin, and lowest on BF
  Tubulin (0.570 vs 0.684 for E1b), PC Nucleus and DIC Mitochondria (S20). Tables: `reports/S00_compare/summary.md`.

Reports: `reports/S00_compare/summary.md` (Task 1: headline, organelle x modality, per study, bootstrap, curves
`curves_compare.png`) and `reports/S00_compare/task2_summary.md` (Task 2 tables, bootstrap, `curves_task1_task2.png`).
Run outputs: `runs/E1h_*/` (`best.pt` = weights-only export of the EMA weights at epoch 50, `config.yaml`, `train_log.csv`,
`val_log.csv`, `curves.png`, `stdout.log`, `val_metrics.json`, `val_per_image.csv`, `eval_val_best*.json`,
`eval_val_last*.json` = raw weights of epoch 50, `code/` = code snapshot). `last.pt` (119 MB) and the smoke runs are not
committed.
Console logs: `logs/`.

Role in the team's final decision: E1h is the control, not a candidate. E1g stays the S01 v2 / final candidate.
