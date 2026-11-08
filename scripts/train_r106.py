"""Train sensitivity controls under the Pressure21 protocol."""
from __future__ import annotations

import argparse
import json
import msvcrt
import os
import shutil
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_il_pressure21 import autoregressive, digest, load_data

REFERENCE_SAMPLES = 150174
STRATA_COUNTS = {10: (24, 50), 20: (3, 5), 30: (1, 2), 40: (1, 2), 50: (1, 1)}
WIDTHS = {"w64": 64, "w256": 256, "n30": 128, "n60": 128}


def write_json(path: Path, data: object) -> None:
    """Atomically write a JSON document."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf8")
    os.replace(temp, path)


def save_torch(path: Path, data: object) -> None:
    """Atomically write a PyTorch checkpoint."""
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temp)
    os.replace(temp, path)


def select_families() -> dict:
    """Select nested, stratified families using fixed independent seeds."""
    source = ROOT / "artifacts/runs/ideal10h_v1/experts/manifest.json"
    entries = [e for e in json.loads(source.read_text()) if e["split"] == "train"]
    families = sorted({(e["seed"], e["vehicles"]) for e in entries})
    assert len(families) == 120
    assert all(sum((e["seed"], e["vehicles"]) == f for e in entries) == 3 for f in families)
    selected = {30: [], 60: []}
    strata = {}
    for vehicles, (small, large) in STRATA_COUNTS.items():
        pool = [f for f in families if f[1] == vehicles]
        seed = 20261107 + vehicles
        order = np.random.default_rng(seed).permutation(len(pool))
        shuffled = [pool[i] for i in order]
        strata[str(vehicles)] = {"pool": len(pool), "seed": seed, "n30": small, "n60": large}
        selected[30].extend(shuffled[:small])
        selected[60].extend(shuffled[:large])
    assert len(selected[30]) == 30 and len(selected[60]) == 60
    assert set(selected[30]) <= set(selected[60])
    return {"selection_seed_rule": "20261107 + vehicles", "strata": strata,
            "families": {str(k): [list(v) for v in sorted(vals)] for k, vals in selected.items()},
            "source_manifest_sha256": digest(source)}


def subset_data(data: dict, families: list) -> dict:
    """Retain all three capacities for each selected family and reindex groups."""
    cache = ROOT / "artifacts/runs/pressure21_v1/expert_cache/train.npz"
    with np.load(cache) as archive:
        sources = archive["group_source"].copy()
    allowed = {(int(a), int(b)) for a, b in families}
    mask = np.array([(int(s.split("_")[1]), int(s.split("_")[2])) in allowed for s in sources])
    selected_sources = set(sources[mask])
    assert len(selected_sources) == 3 * len(allowed)
    index = data["index"][mask].copy()
    rows = np.sort(index[index >= 0])
    mapping = np.full(len(data["x"]), -1, dtype=np.int64)
    mapping[rows] = np.arange(len(rows))
    valid = index >= 0
    index[valid] = mapping[index[valid]]
    assert np.array_equal(np.sort(index[index >= 0]), np.arange(len(rows)))
    return {"x": data["x"][rows], "y": data["y"][rows], "index": index,
            "power": data["power"][mask], "limit": data["limit"][mask]}


def make_model(width: int) -> nn.Sequential:
    """Construct the two-hidden-layer bounded policy."""
    return nn.Sequential(nn.Linear(21, width), nn.GELU(), nn.Linear(width, width),
                         nn.GELU(), nn.Linear(width, 1), nn.Sigmoid())


def reference_generator(seed: int) -> torch.Generator:
    """Match the original teacher RNG after 128-wide network initialization."""
    previous = torch.get_rng_state()
    torch.manual_seed(seed)
    make_model(128)
    reference_state = torch.get_rng_state()
    torch.set_rng_state(previous)
    generator = torch.Generator()
    generator.set_state(reference_state)
    return generator


def cycle_indices(size: int, generator: torch.Generator) -> torch.Tensor:
    """Draw a full reference epoch by reshuffling and cycling the subset."""
    remaining = REFERENCE_SAMPLES
    parts = []
    while remaining:
        perm = torch.randperm(size, generator=generator)
        take = min(size, remaining)
        parts.append(perm[:take])
        remaining -= take
    return torch.cat(parts)


def prepare(run_dir: Path) -> None:
    """Freeze input identities, protocol and executable sources for a new run."""
    run_dir.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / "scripts/train_il_pressure21.py",
                    ROOT / "configs/pressure21_il.json", ROOT / "src/evcharging/models.py",
                    ROOT / "artifacts/runs/pressure21_v1/expert_cache/manifest.json"]
    archive = run_dir / "source_archive"
    archive.mkdir()
    hashes = {}
    for source in source_paths:
        relative = source.relative_to(ROOT)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = digest(source)
    write_json(run_dir / "subsets.json", select_families())
    write_json(run_dir / "manifest.json", {"schema": "r106-training-v1", "source_sha256": hashes,
        "subsets_sha256": digest(run_dir / "subsets.json"), "reference_samples": REFERENCE_SAMPLES,
        "widths": WIDTHS, "seeds": [11002, 11003, 11004],
        "teacher_permutation": "original 128-wide initialization RNG for each seed",
        "continuation_permutation": "fresh torch generator seeded by training seed"})


def verify_sources(run_dir: Path) -> None:
    """Reject execution if any frozen source or selection has changed."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if digest(ROOT / relative) != expected or digest(run_dir / "source_archive" / relative) != expected:
            raise RuntimeError(f"Source changed: {relative}")
    if digest(run_dir / "subsets.json") != manifest["subsets_sha256"]:
        raise RuntimeError("Subset manifest changed")


def train(group: str, seed: int, run_dir: Path) -> None:
    """Train or resume a single independent teacher and continuation run."""
    torch.set_num_threads(1)
    verify_sources(run_dir)
    cfg = json.loads((ROOT / "configs/pressure21_il.json").read_text(encoding="utf-8-sig"))
    out = run_dir / group / f"il_{seed}" / "il_model"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.lock", "a+b") as lock:
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("This training run is already active") from exc
        train_locked(group, seed, run_dir, out, cfg)


def train_locked(group: str, seed: int, run_dir: Path, out: Path, cfg: dict) -> None:
    """Perform training while holding the run-specific operating-system lock."""
    if (out / "summary.json").exists():
        print("already complete", group, seed, flush=True)
        return
    train_data = load_data("train", "pressure")
    val = load_data("validation", "pressure")
    subsets = json.loads((run_dir / "subsets.json").read_text())
    if group.startswith("n"):
        train_data = subset_data(train_data, subsets["families"][group[1:]])
    assert len(val["x"]) == 49308
    width = WIDTHS[group]
    torch.manual_seed(seed)
    model = make_model(width)
    generator = reference_generator(seed)
    teacher_cfg = cfg["teacher"]
    continuation_cfg = cfg["continuation"]
    opt = torch.optim.Adam(model.parameters(), lr=teacher_cfg["learning_rate"])
    x = torch.from_numpy(train_data["x"])
    y = torch.from_numpy(train_data["y"][:, None])
    vx = torch.from_numpy(val["x"])
    vy = torch.from_numpy(val["y"][:, None])
    state_path = out / "resume.pt"
    curves, continuation = [], []
    stage, epoch, best, stale, initial = "teacher", 0, float("inf"), 0, None
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        stage, epoch, best, stale = (state[k] for k in ("stage", "epoch", "best", "stale"))
        curves, continuation, initial = (state[k] for k in ("curves", "continuation", "initial"))
        generator.set_state(state["generator"])
        if stage == "continuation":
            opt = torch.optim.Adam(model.parameters(), lr=continuation_cfg["learning_rate"])
        opt.load_state_dict(state["optimizer"])

    def save_state() -> None:
        save_torch(state_path, {"model": model.state_dict(), "optimizer": opt.state_dict(),
            "stage": stage, "epoch": epoch, "best": best, "stale": stale, "curves": curves,
            "continuation": continuation, "initial": initial, "generator": generator.get_state()})

    def checkpoint(current_stage: str, current_epoch: int, score: float) -> dict:
        return {"model": model.state_dict(), "variant": "pressure", "state_schema": "pressure21_v1",
                "seed": seed, "width": width, "group": group, "stage": current_stage,
                "epoch": current_epoch, "validation_mse": score,
                "teacher_epoch": None if initial is None else initial["epoch"]}

    if stage == "teacher":
        while epoch < teacher_cfg["epochs"] and stale < teacher_cfg["patience"]:
            epoch += 1
            perm = cycle_indices(len(x), generator)
            total = 0.0
            model.train()
            for ii in perm.split(teacher_cfg["batch_size"]):
                loss = ((model(x[ii]) - y[ii]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(ii)
            model.eval()
            with torch.no_grad():
                score = float(((model(vx) - vy) ** 2).mean())
            curves.append({"epoch": epoch, "train_mse": total / len(perm), "validation_mse": score,
                           "samples": len(perm), "updates": len(perm.split(teacher_cfg["batch_size"]))})
            if score < best:
                best, stale = score, 0
                save_torch(out / "teacher_best.pt", checkpoint("teacher", epoch, score))
            else:
                stale += 1
            save_state()
            print(group, seed, "teacher", epoch, "val", score, flush=True)
        initial = torch.load(out / "teacher_best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(initial["model"])
        opt = torch.optim.Adam(model.parameters(), lr=continuation_cfg["learning_rate"])
        generator = torch.Generator().manual_seed(seed)
        score = float(np.mean((autoregressive(model, val, "pressure") - val["y"]) ** 2))
        best, continuation, stage, epoch, stale = score, [{"epoch": 0, "autoregressive_validation_mse": score}], "continuation", 0, 0
        save_torch(out / "continuation_best.pt", checkpoint(stage, epoch, score))
        save_state()
    while epoch < continuation_cfg["epochs"]:
        epoch += 1
        train_x = torch.from_numpy(autoregressive(model, train_data, "pressure", inputs=True))
        perm = cycle_indices(len(train_x), generator)
        total = 0.0
        model.train()
        for ii in perm.split(continuation_cfg["batch_size"]):
            loss = ((model(train_x[ii]) - y[ii]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ii)
        score = float(np.mean((autoregressive(model, val, "pressure") - val["y"]) ** 2))
        continuation.append({"epoch": epoch, "train_mse": total / len(perm), "autoregressive_validation_mse": score,
            "samples": len(perm), "updates": len(perm.split(continuation_cfg["batch_size"]))})
        if score < best:
            best = score
            save_torch(out / "continuation_best.pt", checkpoint(stage, epoch, score))
        save_state()
        print(group, seed, "continuation", epoch, "val", score, flush=True)
    selected = torch.load(out / "continuation_best.pt", map_location="cpu", weights_only=True)
    save_torch(out / "best.pt", selected)
    write_json(out / "teacher_curve.json", curves)
    write_json(out / "continuation_curve.json", continuation)
    verify_sources(run_dir)
    summary = {"group": group, "seed": seed, "width": width, "train_samples": len(x),
        "validation_samples": len(vx), "samples_per_epoch": REFERENCE_SAMPLES, "updates_per_epoch": 294,
        "teacher_stop_epoch": len(curves), "teacher_best_epoch": initial["epoch"],
        "teacher_validation_mse": initial["validation_mse"], "continuation_best_epoch": selected["epoch"],
        "continuation_validation_mse": selected["validation_mse"],
        "continuation_epoch0_mse": continuation[0]["autoregressive_validation_mse"],
        "best_sha256": digest(out / "best.pt"), "manifest_sha256": digest(run_dir / "manifest.json")}
    write_json(out / "summary.json", summary)
    print("COMPLETE", json.dumps(summary), flush=True)


def main() -> None:
    """Parse the isolated-run command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", choices=list(WIDTHS))
    parser.add_argument("--seed", type=int, choices=[11002, 11003, 11004])
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("run-id must be a simple directory name")
    run_dir = ROOT / "artifacts/runs" / args.run_id
    if args.prepare:
        prepare(run_dir)
    else:
        if args.group is None or args.seed is None:
            parser.error("group and seed are required")
        train(args.group, args.seed, run_dir)


if __name__ == "__main__":
    main()
