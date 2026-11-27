# LMC2 – E1b baseline (U-Net + ImageNet ResNet-34 encoder)

Input: the transmitted-light (TL) z-stack of one image (BF, PC or DIC), resampled to K = 16 planes.
Output: 4 fluorescence images (Nucleus, Mitochondria, Tubulin, Actin), each at that organelle's own
best-focus plane. No best_z (oracle) is used at test time. Background, results and the list of
experiments already tried: `docs/TEAM_HANDOVER.md` (E1 tables and curves in `docs/E1_compare/`).

> **The Test split (studies 21 56 57 62 73 80) is locked.** Do not evaluate on it and do not look at it.
> Select everything on Val. `evaluate.py --split test` exists only for the final evaluation, which is done once.

## Contents

| Path | What |
|---|---|
| `configs/paths.yaml` | **The only file with paths**: dataset, TSV folder, cache, SSD train cache, runs folder |
| `configs/base.yaml`, `configs/E1b_smp_resnet34.yaml` | Training protocol and the E1b config (`inherit: base.yaml`) |
| `splits/split_v1.json` | Fixed split by study (train 17 / val 5 / test 6). Never change it. |
| `lmc/` | Data, models, losses, official metric, evaluation, run manager |
| `train.py`, `evaluate.py` | Training (with resume) and Val evaluation |
| `tools/build_cache.py`, `tools/mirror_cache.py` | Build the K=16 cache; copy its train part to an SSD |
| `tools/update_registry.py`, `tools/compare_runs.py` | `runs/REGISTRY.csv`; Val comparison tables, bootstrap, curves |
| `tools/check_pipeline.py`, `tools/test_experiments.py` | Unit and pipeline checks |
| `docs/` | Team handover note and the E1 backbone comparison (tables, bootstrap, curves) |
| `reference_run/E1b_.../` | Logs, curves and Val metrics of our E1b run (no checkpoint: `best.pt` is 281 MB) |

## 1. Install (Python 3.12; tested on Windows 10, GTX 1650 4 GB)

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt                          # torch lines are already satisfied
```
On first use, segmentation-models-pytorch downloads the ImageNet ResNet-34 weights (~85 MB) into the Hugging Face cache.

## 2. Set the paths

Edit `configs/paths.yaml` (relative paths are relative to this folder):

```yaml
dataset_root: D:/LMC2/Dataset            # contains Study_02 ... Study_86
meta_dir: D:/LMC2/Dataset/Study_info_filenames_and_optimal_focus_V3/Study_info_filenames_and_optimal focus
cache_root: D:/LMC2/cache/k16            # ~35 GB
train_cache_root: C:/lmc_cache/k16       # optional SSD copy of the train studies (~21 GB); null = use cache_root
runs_root: runs
```

## 3. Build the cache (once, ~22 min, 35 GB)

```bash
python tools/build_cache.py          # resumable; 932 samples (924 images, 28 studies); skips 24 unusable images
python tools/mirror_cache.py         # only if train_cache_root is set (~8 min, 21 GB); needed if cache_root is on an HDD
python tools/check_pipeline.py       # cache / split / metric / no-oracle / loader / model checks: all must PASS
python tools/test_experiments.py     # model unit tests (uses the GPU briefly)
```

## 4. Train E1b

```bash
python train.py --config configs/E1b_smp_resnet34.yaml
```
- 50 epochs × 100 steps = 5,000 steps, Val (official full-image LMC-score) every 5 epochs; `best.pt` = best Val.
- GTX 1650: 0.86 s/step, ~98 s per Val pass, **1h33m in total, peak GPU memory 2.1 GB** (fp32).
- Our run: best Val LMC-score **0.5328 at epoch 35** (see `reference_run/`). A repeat with the same seed should
  land close to that, but it will not match exactly (cuDNN autotuning is non-deterministic).
- Interrupted (Ctrl+C or crash)? `python train.py --resume runs/<run_dir>` continues mid-epoch.
- The run folder `runs/<ID>_<name>_<date>/` holds `config.yaml`, `train_log.csv`, `val_log.csv`, `curves.png`,
  `best.pt`, `last.pt`, `val_metrics.json`, `val_per_image.csv`, `stdout.log`. Existing folders are never overwritten,
  and a config that already finished is refused (`--allow-duplicate` to override).
- Change a setting without a new file: `--set train.lr=5e-4` (it is recorded and changes the config hash).
  For a real experiment, prefer a new yaml with `inherit: E1b_smp_resnet34.yaml` and a new `run.id`.
- Smoke test (a few steps, output in `runs/_smoke/`, never registered):
  `python train.py --config configs/E1b_smp_resnet34.yaml --smoke --set train.epochs=1 --set train.samples_per_epoch=320 --set eval.limit=3`
  (keep at least ~40 steps: with fewer, the 5 % OneCycle warm-up rounds to 1 step and PyTorch divides by zero)

## 5. Evaluate on Val, registry, comparison

```bash
python evaluate.py --run runs/<run_dir>                # best.pt on Val -> <run>/eval_val_best.json (+ per image csv)
python evaluate.py --run runs/<run_dir> --save-tiff    # also writes OME-TIFF predictions
python tools/update_registry.py                         # rebuilds runs/REGISTRY.csv from the run folders
python tools/compare_runs.py --runs reference_run/E1b_* runs/<your_run> --out reports/my_compare
```
`compare_runs.py` writes the headline table (full Val, without S20, per organelle), per organelle × modality and
per study tables, a paired bootstrap of the LMC-score difference, and overlaid curves.

## Notes

- Metric: official LightMyCells SSIM and PCC per image and organelle; **LMC-score = mean over organelles of the
  study-macro mean of (SSIM+PCC)/2**. Checkpoint selection uses the full Val set (S20 included).
- Run configs written before this package (e.g. `reference_run/`) still list `cache_root`; the code now ignores
  that key and uses `configs/paths.yaml`. Paths are therefore not part of the config hash any more, so new runs
  get a different hash from the reference run.
- Code credits are in the file headers (teammate's baseline github.com/YUNUSNILE/S00-LightMyCell-Baseline;
  RUNet from github.com/MWod/LightMyCells; official metric from the LightMyCells challenge).
