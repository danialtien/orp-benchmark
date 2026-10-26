#!/usr/bin/env python3
"""Turn a release config (configs/train/*.yaml, configs/eval/*.yaml) into the
exact command line that produced the paper's numbers, and run it.

    # training (one model)
    python tools/run_config.py train configs/train/DA_s0.yaml --data-root ../Dataset --out-root runs --gpu 0
    python tools/run_config.py train configs/train/irra_init.yaml --out-root runs --irra-pth /path/IRRA/logs/CUHK-PEDES/best.pth

    # occluded galleries (once; shared by all models)
    python tools/run_config.py occlude configs/eval/orp_ground.yaml --data-root ../Dataset --out-root runs
    python tools/run_config.py occlude configs/eval/oof_mgcc.yaml   --data-root ../Dataset --out-root runs

    # scoring (clean + 15 ORP cells per benchmark, or OOF / NOMAD-val)
    python tools/run_config.py eval configs/eval/orp_ground.yaml --model DA_s0 --data-root ../Dataset --out-root runs
    python tools/run_config.py eval configs/eval/nomad_val.yaml  --model DA_nomad_s0 --data-root ../Dataset --out-root runs

Add --dry-run to print the commands without running them. Needs PyYAML.

Checkpoint resolution for --model / init_from NAME (first hit wins):
    1. --ckpt PATH (eval) or --init-ckpt PATH (train)
    2. <out-root>/<NAME>/<file the paper used>   (lora_ckpt.pt or lora_ckpt_last.pt, from the NAME config)
    3. checkpoints/<NAME>.pt                     (released checkpoint, see checkpoints/README.md)
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml")

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "configs" / "train"
GEOMS_DEFAULT = ("torso", "cutout", "foliage")


# --------------------------------------------------------------------------- #
def load(p) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flags(args: dict) -> list[str]:
    """{'img-size': [384, 128], 'aug': True, 'lr': '1e-5'} -> ['--img-size', '384', '128', '--aug', '--lr', '1e-5']"""
    out: list[str] = []
    for k, v in (args or {}).items():
        if v is None or v is False:
            continue
        if v is True:
            out.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            out.append(f"--{k}")
            out += [str(x) for x in v]
        else:
            out += [f"--{k}", str(v)]
    return out


def selected_file(name: str) -> str:
    cfg = TRAIN_DIR / f"{name}.yaml"
    if not cfg.exists():
        return "lora_ckpt.pt"
    sel = (load(cfg).get("checkpoint") or {}).get("selection")
    return "lora_ckpt_last.pt" if sel == "last" else "lora_ckpt.pt"


DRY = False


def resolve_ckpt(name: str, out_root: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    cands = [out_root / name / selected_file(name), REPO / "checkpoints" / f"{name}.pt"]
    for c in cands:
        if c.exists():
            return c
    if DRY:
        return Path(f"<missing: {cands[0]} or {cands[1]}>")
    raise SystemExit(f"no checkpoint for '{name}'. Tried:\n  " + "\n  ".join(map(str, cands)) +
                     "\nTrain it (configs/train), download the release (checkpoints/README.md) or pass --ckpt.")


def run(cmd: list[str], dry: bool, log: Path | None = None) -> int:
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return 0
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            return subprocess.call(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    return subprocess.call(cmd, cwd=REPO)


# --------------------------------------------------------------------------- #
def cmd_train(a) -> int:
    cfg = load(a.config)
    name, out_root = cfg["name"], Path(a.out_root).resolve()
    out = out_root / name
    if cfg["entry"].endswith("convert_irra_ckpt.py"):
        if not a.irra_pth:
            raise SystemExit("irra_init needs --irra-pth <IRRA>/logs/CUHK-PEDES/best.pth")
        ref = resolve_ckpt(cfg["args"]["reference_ckpt"], out_root, None)
        out.mkdir(parents=True, exist_ok=True)
        rc = run([sys.executable, str(REPO / cfg["entry"]), a.irra_pth, str(out / "lora_ckpt.pt"), str(ref)], a.dry_run)
        print(f"[gate] now score it: python tools/run_config.py eval configs/eval/orp_ground.yaml "
              f"--model irra_init --benchmarks cuhk --clean-only ... (expect R@1 = {cfg['gate']['expected']})")
        return rc
    args = dict(cfg["args"])
    if cfg.get("init_from"):
        args["init-ckpt"] = str(resolve_ckpt(cfg["init_from"], out_root, a.init_ckpt))
    args["data-root"] = a.data_root
    args["seed"] = cfg["seed"]
    if a.gpu is not None:
        args["gpu"] = a.gpu
    args["out"] = str(out)
    cmd = [sys.executable, "-u", "-m", cfg["entry"]] + flags(args)
    rc = run(cmd, a.dry_run, None if a.dry_run else out_root / "logs" / f"{name}.log")
    print(f"[train] {name}: the paper's checkpoint is {out / selected_file(name)} "
          f"(selection: {cfg['checkpoint']['selection']}, epoch {cfg['checkpoint']['selected_epoch']})")
    return rc


def cmd_occlude(a) -> int:
    cfg = load(a.config)
    out_root = Path(a.out_root).resolve()
    rc = 0
    if "galleries" in cfg:                                   # OOF
        common = dict(cfg["generate"].get("common_args") or {})
        if a.oil_root:
            common["oil-root"] = a.oil_root
        for g in cfg["galleries"]:
            args = {"benchmark": g["benchmark"], "data-root": a.data_root,
                    "evalset-cache": str(out_root / "evalsets"),
                    "out-root": str(out_root / "occluded_oof"), **common, **g["args"]}
            rc |= run([sys.executable, "-m", cfg["generate"]["entry"]] + flags(args), a.dry_run)
        return rc
    for b in (a.benchmarks or cfg["benchmarks"]):
        args = {"benchmark": b, "data-root": a.data_root, "evalset-cache": str(out_root / "evalsets"),
                "out-root": str(out_root / "occluded"), **cfg["generate"]["args"]}
        rc |= run([sys.executable, "-m", cfg["generate"]["entry"]] + flags(args), a.dry_run)
    print("[occlude] check the result against the paper: python scripts/verify_occlusion.py --data-root", a.data_root)
    return rc


def _model_args(cfg: dict, model: str, out_root: Path, ckpt: str | None) -> dict:
    base = dict(cfg["score"]["args"])
    if model == "irra_published":
        base.pop("full-finetune", None)
        base.update(cfg.get("irra_published", {}).get("args", {}))
        return base
    if model == "clip_zs":
        base.pop("full-finetune", None)
        return base
    base["ckpt"] = str(resolve_ckpt(model, out_root, ckpt))
    return base


def cmd_eval(a) -> int:
    cfg = load(a.config)
    out_root = Path(a.out_root).resolve()
    model, entry = a.model, cfg["score"]["entry"]
    margs = _model_args(cfg, model, out_root, a.ckpt)
    common = {"data-root": a.data_root, "tag": model, **({"gpu": a.gpu} if a.gpu is not None else {})}
    log = None if a.dry_run else out_root / "logs" / f"{model}.{cfg['name']}.eval.log"
    rc = 0

    def go(extra: dict, out: Path):
        nonlocal rc
        if (out / "metrics.json").exists() and not a.force:
            print(f"[skip] {out}/metrics.json exists")
            return
        rc |= run([sys.executable, "-m", entry] + flags({**margs, **common, **extra, "out": str(out)}), a.dry_run, log)

    if cfg["name"] == "nomad_val":
        go({}, out_root / model / "eval" / "nomad_val")
        return rc
    if cfg["name"] == "oof_mgcc":
        for g in cfg["galleries"]:
            if a.benchmarks and g["benchmark"] not in a.benchmarks:
                continue
            go({"manifest": str(out_root / "occluded_oof" / "manifests" / f"{g['tag']}.json")},
               out_root / model / "eval" / g["tag"])
        return rc
    geoms = cfg["grid"]["geometries"]
    levels = [int(round(100 * c)) for c in cfg["grid"]["coverages"]]
    for b in (a.benchmarks or cfg["benchmarks"]):
        go({"benchmark": b, "evalset-cache": str(out_root / "evalsets")}, out_root / model / "eval" / f"{b}_clean")
        if a.clean_only:
            continue
        for g in geoms:
            for l in levels:
                go({"manifest": str(out_root / "occluded" / "manifests" / f"{b}_{g}_l{l:02d}.json")},
                   out_root / model / "eval" / f"{b}_{g}_l{l:02d}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("train", "occlude", "eval"):
        p = sub.add_parser(n)
        p.add_argument("config")
        p.add_argument("--data-root", default=str(REPO.parent / "Dataset"))
        p.add_argument("--out-root", default=str(REPO / "runs"))
        p.add_argument("--gpu", default=None)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--benchmarks", nargs="+", default=None)
        if n == "train":
            p.add_argument("--init-ckpt", default=None)
            p.add_argument("--irra-pth", default=None)
        if n == "occlude":
            p.add_argument("--oil-root", default=None)
        if n == "eval":
            p.add_argument("--model", required=True,
                           help="config name (D, DA_s0, ...), clip_zs, irra_published or irra_init")
            p.add_argument("--ckpt", default=None)
            p.add_argument("--clean-only", action="store_true")
            p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    global DRY
    DRY = a.dry_run
    return {"train": cmd_train, "occlude": cmd_occlude, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
