"""Run management (PLAN.md section 7): one folder per run, never overwrite, never redo a finished run.

Run folder: runs/<ID>_<name>_<YYYYMMDD-HHMM>/ (smoke runs: runs/_smoke/...), containing
  config.yaml      resolved config + split sha256 + code sha256 + config hash + command + environment
  code/            snapshot of lmc/ and the entry scripts at launch time
  status.json      status (running / done / failed / interrupted), step, best step and score
  train_log.csv, val_log.csv, curves.png, best.pt, last.pt, val_metrics.json, stdout.log

Config files are YAML; `inherit: base.yaml` merges a parent first; `--set a.b=value` overrides.
"""
import copy
import datetime as dt
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import yaml

from lmc import PATHS, PROJECT_ROOT, RUNS_ROOT

SMOKE_ROOT = RUNS_ROOT / "_smoke"
# keys that do not change what is trained (excluded from the config hash)
_NON_IDENTITY = {("run", "id"), ("run", "name"), ("run", "notes")}


# ---------------------------------------------------------------- config
def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return cfg


def apply_overrides(cfg, sets):
    """sets: ['train.steps=30', ...]; values parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for s in sets or []:
        key, val = s.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve(cfg):
    """Fill derived fields (input channels) and validate."""
    cfg = copy.deepcopy(cfg)
    d = cfg["data"]
    d["in_channels"] = int(d["k"]) if d["input"] == "stack" else 1
    for sec in ("run", "data", "model", "train", "loss", "eval"):
        if sec not in cfg:
            raise ValueError(f"config section '{sec}' missing")
    return cfg


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(cfg):
    c = copy.deepcopy(cfg)
    for sec, key in _NON_IDENTITY:
        c.get(sec, {}).pop(key, None)
    c.pop("provenance", None)
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]


CODE_FILES = ["lmc/*.py", "train.py", "evaluate.py", "tools/build_cache.py", "tools/make_split.py",
              "tools/build_mip_cache.py"]  # S01 v2: the MIP target builder is part of what a run depends on


def code_sha256():
    h = hashlib.sha256()
    for pat in CODE_FILES:
        for f in sorted(PROJECT_ROOT.glob(pat)):
            h.update(f.relative_to(PROJECT_ROOT).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def snapshot_code(run_dir):
    dst = run_dir / "code"
    for pat in CODE_FILES:
        for f in PROJECT_ROOT.glob(pat):
            t = dst / f.relative_to(PROJECT_ROOT)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)


# ---------------------------------------------------------------- run folders
def read_status(run_dir):
    p = Path(run_dir) / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_status(run_dir, **kw):
    st = read_status(run_dir)
    st.update(kw, updated=dt.datetime.now().isoformat(timespec="seconds"))
    tmp = Path(run_dir) / "status.json.tmp"
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(Path(run_dir) / "status.json")


def existing_runs(root=None):
    """(run_dir, config dict, status dict) for every real run folder (smoke runs excluded)."""
    root = RUNS_ROOT if root is None else root
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and (d / "config.yaml").exists():
            out.append((d, yaml.safe_load((d / "config.yaml").read_text()), read_status(d)))
    return out


def rel(p):
    """Path relative to the project if possible (for messages)."""
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def check_not_done(cfg, allow_duplicate=False):
    """Refuse to start a run whose config hash already exists (PLAN.md section 7)."""
    h = config_hash(cfg)
    for d, c, st in existing_runs():
        if c.get("provenance", {}).get("config_hash") != h:
            continue
        s = st.get("status", "?")
        if s == "done" and not allow_duplicate:
            raise SystemExit(f"Refusing: an identical config already finished in {d} (status done). "
                             "Use --allow-duplicate only for a deliberate repeat (e.g. another seed changes the hash anyway).")
        if s in ("running", "interrupted", "failed") and not allow_duplicate:
            raise SystemExit(f"Refusing: an identical config exists in {d} (status {s}). "
                             f"Resume it with: python train.py --resume {rel(d)}")
    return h


def new_run_dir(cfg, smoke=False):
    root = SMOKE_ROOT if smoke else RUNS_ROOT
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    d = root / f"{cfg['run']['id']}_{cfg['run']['name']}_{stamp}"
    if d.exists():
        raise SystemExit(f"Refusing to overwrite existing run folder {d}")
    d.mkdir(parents=True)
    return d


def provenance(cfg, argv):
    import torch

    split = PROJECT_ROOT / cfg["data"]["split"]
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return {
        "config_hash": config_hash(cfg),
        "split_sha256": file_sha256(split) if split.exists() else None,
        "code_sha256": code_sha256(),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(argv),
        "python": platform.python_version(), "torch": str(torch.__version__), "gpu": gpu,
        # configs/paths.yaml at launch (not part of the config hash: paths do not change what is trained)
        "paths": {k: (None if v is None else str(v)) for k, v in PATHS.items()},
    }


def save_config(run_dir, cfg):
    (Path(run_dir) / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


class Tee:
    """Mirror stdout/stderr to stdout.log (append, so resumed runs keep their history)."""

    def __init__(self, path, stream):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


def tee_output(run_dir):
    sys.stdout = Tee(Path(run_dir) / "stdout.log", sys.__stdout__)
    sys.stderr = Tee(Path(run_dir) / "stdout.log", sys.__stderr__)
