"""Unit tests for the model code (z-attention, focus loss, FiLM, z-augmentation, no-oracle inference,
E1b checkpoint compatibility). lmc/ still contains the E3-E6 modules tried on top of E1b, so they are
tested too; their configs are not shipped, so the tests build them from the E1b config + the overrides below.

  python tools/test_experiments.py                       (uses the GPU briefly: check no run is active first)
  python tools/test_experiments.py --e1b-ckpt <best.pt>  (also checks that a trained E1b checkpoint loads strictly)
Needs the cache (cache_root in configs/paths.yaml).
"""
import copy
import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmc import CACHE_ROOT, ORGANELLES, PROJECT_ROOT  # noqa: E402
from lmc import runs  # noqa: E402
from lmc.data import meta_vector, plane_positions, z_augment  # noqa: E402
from lmc.evaluation import predict_full  # noqa: E402
from lmc.losses import focus_kl, focus_target  # noqa: E402
from lmc.models import ZAttention, build_model  # noqa: E402

RES = []
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


# Experiment configs tried on top of E1b (copied from the project's configs/E3-E5 yaml files).
_ZATTN = {"model": {"zattn": {"mode": "softmax", "features": 16, "pe": True, "checkpoint": False, "scale": 2}}}
_FOCUS = {"loss": {"focus": {"weight": 0.1, "sigma": 1.0}}}
_FILM = {"model": {"film": {"enabled": True, "hidden": 32}}}
_VARIANTS = {"E3_zattn.yaml": [_ZATTN], "E4_zattn_focus.yaml": [_ZATTN, _FOCUS],
             "E5_zattn_focus_film.yaml": [_ZATTN, _FOCUS, _FILM]}


def cfg_of(name):
    p = PROJECT_ROOT / "configs" / name
    if p.exists():
        return runs.resolve(runs.load_config(p))
    cfg = runs.load_config(PROJECT_ROOT / "configs" / "E1b_smp_resnet34.yaml")
    for ov in _VARIANTS[name]:
        cfg = runs._merge(cfg, ov)
    return runs.resolve(cfg)


def t_attention():
    torch.manual_seed(0)
    za = ZAttention(16, "softmax", True).to(DEV)
    with torch.no_grad():  # random scorer so the attention is not uniform
        za.scorer.weight.normal_(0, 0.5)
    x = torch.rand(2, 16, 64, 64, device=DEV)
    zpos = torch.linspace(-3, 3, 16, device=DEV).expand(2, -1)
    out, a_mean, maps = za(x, zpos, keep_maps=True)
    s = maps.sum(1)
    check("E3 softmax: attention sums to 1 over z at every pixel and organelle",
          torch.allclose(s, torch.ones_like(s), atol=1e-5), f"max |sum-1| {float((s - 1).abs().max()):.1e}")
    check("E3 softmax: weights not uniform after random scorer (attention is learnable)",
          float(maps.std(1).mean()) > 1e-3)
    check("E3: output = 4 organelle maps x F features", tuple(out.shape) == (2, 64, 64, 64))
    check("E3: pixel-mean attention (B, 4, K) sums to 1", torch.allclose(a_mean.sum(-1), torch.ones(2, 4, device=DEV), atol=1e-5))
    # organelle o's map uses only its own weights
    o = 2
    manual = (maps[:, :, o:o + 1] * (za.stem(x.reshape(32, 1, 64, 64)).reshape(2, 16, 16, 64, 64)
                                     + za.pe(__import__("lmc.models", fromlist=["fourier_z"]).fourier_z(zpos))[..., None, None])).sum(1)
    check("E3: organelle map = sum_z softmax_o(z) * features(z)", torch.allclose(manual, out[:, o * 16:(o + 1) * 16], atol=1e-4))
    o1, a1, m1 = za(x, zpos, keep_maps=True, per_plane=False)
    o2, a2_, m2 = za(x, zpos, keep_maps=True, per_plane=True)
    check("E3: plane-by-plane path (scale 1) == vectorised path", torch.allclose(o1, o2, atol=1e-5)
          and torch.allclose(a1, a2_, atol=1e-6) and torch.allclose(m1, m2, atol=1e-6))
    zm = ZAttention(16, "mean", True).to(DEV)
    out2, a2, _ = zm(x, zpos)
    check("E3m: mean over z -> uniform weights 1/K and identical maps per organelle",
          torch.allclose(a2, torch.full_like(a2, 1 / 16)) and torch.allclose(out2[:, :16], out2[:, 48:]))
    # zero-init scorer -> uniform start
    z0 = ZAttention(16, "softmax", True).to(DEV)
    _, a0, _ = z0(x, zpos)
    check("E3: zero-initialised scorer starts from uniform attention", torch.allclose(a0, torch.full_like(a0, 1 / 16), atol=1e-6))


def t_checkpoint():
    cfg = cfg_of("E3_zattn.yaml")
    m = build_model(cfg, pretrained=False).to(DEV)
    x = torch.rand(1, 16, 128, 128, device=DEV)
    zp = torch.linspace(-2, 2, 16, device=DEV)[None]
    m.train()
    m.ckpt = True
    torch.manual_seed(0)
    y1 = m(x, zp)
    g1 = torch.autograd.grad(y1.square().mean(), m.zattn.stem[0].weight)[0]
    m.ckpt = False
    torch.manual_seed(0)
    y2 = m(x, zp)
    g2 = torch.autograd.grad(y2.square().mean(), m.zattn.stem[0].weight)[0]
    check("E3: gradient checkpointing gives the same output and gradients", torch.allclose(y1, y2, atol=1e-5)
          and torch.allclose(g1, g2, atol=1e-5, rtol=1e-4))


def t_focus():
    k = 16
    bz = torch.tensor([[7.5, 0.0, 15.0, float("nan")]])
    q = focus_target(bz[:, :3], k)
    zi = torch.arange(k).float()
    check("E4 target: Gaussian centred on the resampled best_z (mean 7.5 / edge 0 / edge 15)",
          abs(float((q[0, 0] * zi).sum()) - 7.5) < 1e-4 and int(q[0, 1].argmax()) == 0 and int(q[0, 2].argmax()) == 15
          and torch.allclose(q.sum(-1), torch.ones(1, 3)))
    # resampling: best_zr = best_z * (K-1)/(nZ-1) for every cached row, and the resampled plane there IS best_z
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from build_cache import resample_positions

    idx = pd.read_csv(CACHE_ROOT / "index.csv")
    err = 0.0
    for o in ORGANELLES:
        h = idx[idx[f"has_{o}"] == 1]
        err = max(err, float((h[f"best_zr_{o}"] - h[f"best_z_{o}"] * (k - 1) / (h.nZ - 1)).abs().max()))
    pos_ok = all(abs(np.interp(bzr, np.arange(k), resample_positions(nz, k)) - b) < 1e-6
                 for nz, b in ((21, 10), (4, 3), (37, 0), (9, 5)) for bzr in [b * (k - 1) / (nz - 1)])
    check("E4 target index: cache best_zr = best_z*(K-1)/(nZ-1) for all rows; resampled index maps back to best_z",
          err < 1e-9 and pos_ok, f"max dev {err:.1e}")
    # masks
    att = torch.softmax(torch.randn(2, 4, k, requires_grad=True), -1)
    bz2 = torch.tensor([[3.0, 5.0, float("nan"), 9.0], [2.0, 2.0, 2.0, 2.0]])
    m = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    logits = torch.randn(2, 4, k, requires_grad=True)
    loss = focus_kl(torch.softmax(logits, -1), bz2, m)
    loss.backward()
    g = logits.grad.abs().sum(-1)
    valid = (m > 0) & torch.isfinite(bz2)
    check("E4 loss masked: zero gradient for missing organelles / NaN best_z, non-zero for valid ones",
          bool((g[~valid] == 0).all() and (g[valid] > 0).all()) and torch.isfinite(loss), f"valid {valid.tolist()}")
    perfect = focus_kl(focus_target(torch.tensor([[4.0, 6.0, 8.0, 10.0]]), k), torch.tensor([[4.0, 6.0, 8.0, 10.0]]),
                       torch.ones(1, 4))
    check("E4 loss: KL = 0 when attention equals the target", float(perfect) < 1e-6, f"{float(perfect):.2e}")
    check("E4 loss with no valid organelle is 0 and finite", float(focus_kl(att, bz2, torch.zeros(2, 4))) == 0.0)


def t_no_oracle():
    for cls in ("ZAttnSmpUNet", "SmpUNet", "UNet", "RUNet"):
        sig = inspect.signature(getattr(__import__("lmc.models", fromlist=[cls]), cls).forward)
        if any("best" in p or p == "bz" for p in sig.parameters):
            check(f"no-oracle: {cls}.forward has no best_z argument", False)
    src = (PROJECT_ROOT / "lmc/models.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"', "No model")))
    check("no-oracle: model forwards take no best_z; models.py code never reads best_z",
          not re.search(r"best_z|best_zr|\bbz\b", code.split('"""', 2)[-1]))
    ev = (PROJECT_ROOT / "lmc/evaluation.py").read_text(encoding="utf-8")
    pf = ev[ev.index("def predict_full"):ev.index("def _score_image")]
    check("no-oracle: predict_full never reads best_z", "best_z" not in pf)
    # end to end: perturbing best_z / best_zr of a Val row changes nothing in the prediction
    cfg = cfg_of("E4_zattn_focus.yaml")
    m = build_model(cfg, pretrained=False).to(DEV).eval()
    idx = pd.read_csv(CACHE_ROOT / "index.csv")
    r = idx[(idx.study == 81)].iloc[0]
    r2 = r.copy()
    for o in ORGANELLES:
        r2[f"best_z_{o}"], r2[f"best_zr_{o}"] = 0, 0.0
    from lmc.data import load_input

    x = load_input(r, CACHE_ROOT, "stack", 16)[:, :512, :512].copy()
    outs = []
    for rr in (r, r2):
        p, prof, _ = predict_full(m, x, DEV, 512, 128, 2, 32, plane_positions(rr, "stack", 16), meta_vector(rr),
                                  return_attention=True)
        outs.append((p, prof))
    check("no-oracle: prediction and attention identical when the row's best_z is changed",
          np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1]))


def t_film():
    cfg5 = cfg_of("E5_zattn_focus_film.yaml")
    cfg4 = cfg_of("E4_zattn_focus.yaml")
    torch.manual_seed(0)
    m5 = build_model(cfg5, pretrained=False).to(DEV).eval()
    m4 = build_model(cfg4, pretrained=False).to(DEV).eval()
    m4.load_state_dict({k: v for k, v in m5.state_dict().items() if not k.startswith("film.")})
    x = torch.rand(1, 16, 128, 128, device=DEV)
    zp = torch.linspace(-2, 2, 16, device=DEV)[None]
    meta = torch.tensor([[0, 1, 0, -0.4, 0.6]], dtype=torch.float32, device=DEV)
    with torch.no_grad():
        same = torch.allclose(m5(x, zp, meta), m4(x, zp), atol=1e-6)
        m5.film.mlp[-1].bias.normal_(0, 0.1)  # simulate a trained FiLM layer (zero-init at the start)
        m5.film.mlp[-1].weight.normal_(0, 0.1)
        diff = not torch.allclose(m5(x, zp, meta), m4(x, zp), atol=1e-4)
        meta2 = meta.clone()
        meta2[0, :3] = torch.tensor([0.0, 0.0, 1.0])
        dep = not torch.allclose(m5(x, zp, meta), m5(x, zp, meta2), atol=1e-6)
    check("E5 FiLM: zero-init -> identical to E4 at start; after training it changes the output and depends on meta",
          same and diff and dep)
    r = pd.read_csv(CACHE_ROOT / "index.csv").query("study == 72").iloc[0]
    v = meta_vector(r)
    check("E5 meta vector: one-hot modality + log px + log dz", v[:3].tolist() == [0, 1, 0]
          and abs(v[3] - np.log(r.px_um / 0.1)) < 1e-6 and abs(v[4] - np.log(r.dz_um / 0.5)) < 1e-6, str(v.round(3)))


def t_zpos_zaug():
    r = pd.Series(dict(nZ=21, dz_um=0.5, tl="BF", px_um=0.1))
    zp = plane_positions(r, "stack", 16)
    check("zpos: centred, symmetric, spacing dz*(nZ-1)/(K-1)", abs(zp.mean()) < 1e-6 and abs(zp[0] + 5) < 1e-6
          and np.allclose(np.diff(zp), 0.5 * 20 / 15), f"{zp[0]:.2f}..{zp[-1]:.2f} um")
    k = 16
    x = np.broadcast_to(np.arange(k, dtype=np.float32)[:, None, None], (k, 4, 4)).copy()  # plane i has value i
    bz = np.array([7.0, 3.0, np.nan, 12.0], np.float32)
    ok_shift = ok_flip = True
    rng = np.random.default_rng(0)
    for _ in range(200):
        xs, bzs = z_augment(x, bz, rng, shift=2)
        for o in (0, 1, 3):
            j = int(round(bzs[o]))
            if 0 < bzs[o] < k - 1 and xs[j, 0, 0] != bz[o]:
                ok_shift = False
        xf, bzf = z_augment(x, bz, rng, zflip=True)
        for o in (0, 1, 3):
            if xf[int(round(bzf[o])), 0, 0] != bz[o]:
                ok_flip = False
    check("E6 z-shift: the plane at the new best_z index is the original best-focus plane", ok_shift)
    check("E6 z-flip: best_z -> K-1-best_z follows the content", ok_flip)
    drops = [int((z_augment(x + 1, bz, rng, plane_dropout=0.1)[0][:, 0, 0] == 0).sum()) for _ in range(2000)]
    check("E6 plane dropout: ~10% of planes zeroed, never all", 0.08 < np.mean(drops) / k < 0.12 and max(drops) < k,
          f"mean {np.mean(drops) / k:.3f}")
    xs, bzs = z_augment(x, np.array([0.5, 15.0, 1.0, 14.0], np.float32), np.random.default_rng(3), shift=2)
    check("E6 z-shift: best_z clamped to [0, K-1] at the stack edges", bool((bzs >= 0).all() and (bzs <= k - 1).all()))


def t_e1b_compat():
    cfg = runs.resolve(runs.load_config(PROJECT_ROOT / "configs/E1b_smp_resnet34.yaml"))
    ref = runs.load_config(next((PROJECT_ROOT / "reference_run").glob("E1b_*/config.yaml")))
    same = all(cfg[s] == {k: v for k, v in ref[s].items() if k not in ("cache_root", "train_cache_root")}
               for s in ("model", "train", "loss", "eval", "data"))
    check("E1b config == the reference E1b run config (paths aside)", same)
    ck_path = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--e1b-ckpt=")), None)
    if "--e1b-ckpt" in sys.argv:
        ck_path = sys.argv[sys.argv.index("--e1b-ckpt") + 1]
    if not ck_path:
        print("[SKIP] E1b checkpoint load (no --e1b-ckpt given; best.pt is not shipped)")
        return
    m = build_model(cfg, pretrained=False)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])  # strict
    check("E1b best.pt loads (strict) into SmpUNet", True, f"step {ck['step']}, best {ck['best']:.4f}")


def main():
    for f in (t_attention, t_checkpoint, t_focus, t_no_oracle, t_film, t_zpos_zaug, t_e1b_compat):
        print(f"--- {f.__name__[2:]}")
        f()
        torch.cuda.empty_cache()
    print(f"\n{sum(RES)}/{len(RES)} checks passed")
    sys.exit(0 if all(RES) else 1)


if __name__ == "__main__":
    main()
