"""lmc: Light My Cells 2 course project (TL z-stack -> per-organelle best-focus fluorescence).

Constants shared by every module. Organelle order is fixed everywhere (model output channels,
masks, logs).

All machine-specific locations (dataset, TSV folder, cache, SSD train cache, runs folder) are read
from configs/paths.yaml. Nothing else in the code hard-codes a data path.
"""
from pathlib import Path

import yaml

ORGANELLES = ("Nucleus", "Mitochondria", "Tubulin", "Actin")
TL_MODALITIES = ("BF", "PC", "DIC")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = PROJECT_ROOT / "configs" / "paths.yaml"


def _load_paths():
    if not PATHS_FILE.exists():
        raise SystemExit(f"missing {PATHS_FILE}: copy it from the package and set your paths")
    raw = yaml.safe_load(PATHS_FILE.read_text(encoding="utf-8")) or {}
    need = ("dataset_root", "meta_dir", "cache_root", "train_cache_root", "runs_root")
    miss = [k for k in need if k not in raw]
    if miss:
        raise SystemExit(f"{PATHS_FILE}: missing keys {miss}")

    def res(v):
        if v in (None, ""):
            return None
        p = Path(str(v)).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    return {k: res(raw[k]) for k in need}


PATHS = _load_paths()
DATASET_ROOT = PATHS["dataset_root"]
META_DIR = PATHS["meta_dir"]
CACHE_ROOT = PATHS["cache_root"]
TRAIN_CACHE_ROOT = PATHS["train_cache_root"] or CACHE_ROOT  # SSD mirror of the train studies, else the cache
RUNS_ROOT = PATHS["runs_root"]
