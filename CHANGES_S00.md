# CHANGES_S00: what S00 v2 adds to the S01 v2 / E1b package

Paths like `S01_v2/` and `../S01_v2/` refer to our project folder, where this repository's v2 sits next to the
S01 v2 package (not in this repository).

Base: `S01_v2/` (the teammate's E1b package + the S01 v2 additions: Task 2 MIP targets, dual head, EMA, mean
selection; see `../S01_v2/CHANGES_S01.md`). Copied on 27 Nov 2026 **without any code change**: `lmc/`, `train.py`,
`evaluate.py`, `splits/` and every copied file in `tools/` are byte-identical to `S01_v2/`, so E1h has the same
`code_sha256` as E1g (`10085315143e`). The only new code is an analysis script (`tools/compare_task2.py`, below). Regenerate the file list with:

```bash
diff -rq S01_v2 S00_v2 -x runs -x logs -x reports -x __pycache__
```

The S00 network needed no new code: the team framework's `unet` backbone (`lmc/models.py`, class `UNet`) is the
S00 v1 class (`S00-LightMyCell Baseline/lmc/model.py`, github.com/YUNUSNILE/S00-LightMyCell-Baseline, commit
fc2ab9a) with `in_ch` = 16 TL planes; only the input and output channel counts differ.

## New files

| File | What |
|---|---|
| `configs/E1h_unet_mip_ema.yaml` | E1h = S00 v2: the plain U-Net (7.77 M params) trained with the E1g recipe (Task 1 + Task 2 heads, EMA 0.999, checkpoint on mean(Task 1, Task 2), bf16, batch 4 x 2, cuDNN autotuning off, 5 workers) |
| `tools/compare_task2.py` | Task 2 (MIP) comparison: headline, per organelle x modality and per study tables, paired image- and study-level bootstraps (the `compare_runs.py` functions), Task 1 / Task 2 curves. `--t2base` adds a Task 1 run scored as MIP (`evaluate.py --t2-baseline`). Not part of `code_sha256` (an analysis tool); checked against the logged S01 v2 result: E1g - T2-0 = +0.0156, study CI [-0.005, +0.036] |
| `README.md`, `CHANGES_S00.md` | S00 v2 README and this file (the teammate's README is kept as `README_E1b_package.md`) |
| `logs/` | Console logs: pipeline check, smoke test, E1h training, evaluation, comparison |

## Modified files

| File | Change |
|---|---|
| `configs/paths.yaml` | Comments only (S00 v2; runs folder relative to `S00_v2/`). Same dataset, TSV folder and cache as S01_v2 |

## Not copied

S01-only configs (E1d, E1e, E1f, E1g), S01 runs, logs and reports. They stay in `../S01_v2/`; the comparison
reads E1g from there (`--runs ../S01_v2/runs/E1g_*`).

## Repository notes (GitHub, v2)

- v1 of this repository is tag `v1` (commit `fc2ab9a`); v2 replaces its files.
- `runs/E1h_*/best.pt` is a 31 MB weights-only export (EMA weights + config, identical tensors); the full 119 MB
  checkpoint and `last.pt` exceed GitHub's 100 MB file limit and are not committed, nor are `runs/_smoke/`.
- `tools/plot_report_curves.py` draws Figure 5.1 of the report from the E1g, E1h and E1b logs (it expects the S01 v2
  package next to this folder).
