"""Train one refined policy and matched self-A continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "artifacts/runs/r102_distributional_v1_reproduce"
sys.path.insert(0, str(ROOT / "src"))
from torch import nn


def imitation_network():
    return nn.Sequential(
        nn.Linear(27, 128),
        nn.GELU(),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )


def digest(path: Path) -> str:
    """Hash a file.

    :param path: File.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(split: str, variant: str) -> dict:
    """Load one frozen split and select its representation.

    :param split: Train or validation.
    :param variant: The selected Pressure policy.
    :return: Dataset arrays and grouped sequence metadata.
    """
    path = HERE / "expert_cache" / f"{split}.npz"
    manifest = json.loads((HERE / "expert_cache/manifest.json").read_text())
    if digest(path) != manifest[split]["sha256"]:
        raise RuntimeError("Cache digest mismatch")
    with np.load(path) as z:
        x = z["x_distribution"].copy()
        result = {
            "x": x,
            "y": z["y"].copy(),
            "index": z["group_index"].copy(),
            "power": z["group_power"].copy(),
            "limit": z["group_limit"].copy(),
        }
    index = result["index"]
    observed = index[index >= 0]
    if len(observed) != len(x) or not np.array_equal(
        np.sort(observed), np.arange(len(x))
    ):
        raise RuntimeError("Decision groups must partition samples")
    return result


def autoregressive(
    model: torch.nn.Module, data: dict, variant: str, inputs: bool = False
) -> np.ndarray:
    """Rebuild raw preceding-power feature from the current policy.

    :param model: Frozen policy for one reconstruction.
    :param data: Fixed expert SoCs, labels, and grouped decisions.
    :param variant: Model representation.
    :param inputs: Return reconstructed inputs instead of predictions.
    :return: Reconstructed inputs or predictions in expert-record order.
    """
    out = data["x"].copy()
    model.eval()
    index, powers, limits = data["index"], data["power"], data["limit"]
    allocation = np.zeros(len(index), dtype=np.float64)
    prediction = np.empty(len(out), dtype=np.float32)
    with torch.no_grad():
        for position in range(index.shape[1]):
            groups = np.flatnonzero(index[:, position] >= 0)
            rows = index[groups, position]
            out[rows, 7] = 1 - 2 * allocation[groups] / limits[groups]
            y = model(torch.from_numpy(out[rows])).numpy().ravel()
            prediction[rows] = y
            allocation[groups] += y * powers[groups, position]
    return out if inputs else prediction


def run(variant: str, seed: int, run_id: str) -> None:
    """Train teacher stage and matched continuation with frozen checkpoint selection.

    :param variant: The selected Pressure policy.
    :param seed: Matched network initialization and permutation seed.
    :param run_id: Independent run directory under artifacts/runs.
    :return: None.
    """
    global HERE
    if Path(run_id).name != run_id:
        raise ValueError("Run ID must be a single directory name")
    HERE = ROOT / "artifacts/runs" / run_id
    torch.set_num_threads(1)
    cfg = json.loads(
        (ROOT / "configs/pressure21_il.json").read_text(encoding="utf-8-sig")
    )
    if (
        seed not in cfg["pilot_seeds"] + cfg["confirmation_seeds"]
        or variant not in cfg["models"]
    ):
        raise ValueError("Unregistered run")
    out = ROOT / "artifacts/runs" / run_id / f"il_{seed}" / "il_model"
    out.mkdir(parents=True, exist_ok=False)
    train = load_data("train", variant)
    val = load_data("validation", variant)
    torch.manual_seed(seed)
    model = imitation_network()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["teacher"]["learning_rate"])
    x = torch.from_numpy(train["x"])
    y = torch.from_numpy(train["y"][:, None])
    vx = torch.from_numpy(val["x"])
    vy = torch.from_numpy(val["y"][:, None])
    best = float("inf")
    stale = 0
    curves = []
    for epoch in range(1, cfg["teacher"]["epochs"] + 1):
        perm = torch.randperm(len(x))
        total = 0.0
        model.train()
        for ii in perm.split(cfg["teacher"]["batch_size"]):
            loss = ((model(x[ii]) - y[ii]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ii)
        model.eval()
        with torch.no_grad():
            score = float(((model(vx) - vy) ** 2).mean())
        curves.append(
            {"epoch": epoch, "train_mse": total / len(x), "validation_mse": score}
        )
        checkpoint = {
            "model": model.state_dict(),
            "variant": variant,
            "state_schema": "distribution27_v1",
            "seed": seed,
            "stage": "teacher",
            "epoch": epoch,
            "validation_mse": score,
        }
        if score < best:
            best = score
            stale = 0
            torch.save(checkpoint, out / "teacher_best.pt")
        else:
            stale += 1
        print(variant, seed, "teacher", epoch, "val", score, flush=True)
        if stale >= cfg["teacher"]["patience"]:
            break
    torch.save(checkpoint, out / "teacher_last.pt")
    (out / "teacher_curve.json").write_text(json.dumps(curves, indent=2))
    initial = torch.load(out / "teacher_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(initial["model"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["continuation"]["learning_rate"])
    gen = torch.Generator().manual_seed(seed)
    score = float(np.mean((autoregressive(model, val, variant) - val["y"]) ** 2))
    best = score
    continuation = [{"epoch": 0, "autoregressive_validation_mse": score}]
    torch.save(
        {
            "model": model.state_dict(),
            "variant": variant,
            "state_schema": "distribution27_v1",
            "seed": seed,
            "stage": "continuation",
            "epoch": 0,
            "validation_mse": score,
            "teacher_epoch": initial["epoch"],
        },
        out / "continuation_best.pt",
    )
    for epoch in range(1, cfg["continuation"]["epochs"] + 1):
        trainx = torch.from_numpy(autoregressive(model, train, variant, inputs=True))
        perm = torch.randperm(len(trainx), generator=gen)
        total = 0.0
        model.train()
        for ii in perm.split(cfg["continuation"]["batch_size"]):
            loss = ((model(trainx[ii]) - y[ii]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ii)
        score = float(np.mean((autoregressive(model, val, variant) - val["y"]) ** 2))
        continuation.append(
            {
                "epoch": epoch,
                "train_mse": total / len(trainx),
                "autoregressive_validation_mse": score,
            }
        )
        checkpoint = {
            "model": model.state_dict(),
            "variant": variant,
            "state_schema": "distribution27_v1",
            "seed": seed,
            "stage": "continuation",
            "epoch": epoch,
            "validation_mse": score,
            "teacher_epoch": initial["epoch"],
        }
        if score < best:
            best = score
            torch.save(checkpoint, out / "continuation_best.pt")
        print(variant, seed, "continuation", epoch, "val", score, flush=True)
    torch.save(checkpoint, out / "continuation_last.pt")
    (out / "continuation_curve.json").write_text(json.dumps(continuation, indent=2))
    selected = torch.load(
        out / "continuation_best.pt", map_location="cpu", weights_only=True
    )
    shutil.copyfile(out / "continuation_best.pt", out / "best.pt")
    files = [
        Path(__file__),
        ROOT / "src/evcharging/r102_distribution.py",
        ROOT / "configs/pressure21_il.json",
        HERE / "expert_cache/manifest.json",
        ROOT / "src/evcharging/models.py",
    ]
    summary = {
        "variant": variant,
        "seed": seed,
        "teacher_best_epoch": initial["epoch"],
        "teacher_validation_mse": initial["validation_mse"],
        "teacher_stop_epoch": len(curves),
        "continuation_best_epoch": selected["epoch"],
        "continuation_validation_mse": selected["validation_mse"],
        "continuation_epoch0_mse": continuation[0]["autoregressive_validation_mse"],
        "teacher_sha256": digest(out / "teacher_best.pt"),
        "continuation_sha256": digest(out / "continuation_best.pt"),
        "source_sha256": {str(p): digest(p) for p in files},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["pressure"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default="r102_distributional_v1_reproduce")
    args = parser.parse_args()
    run(args.variant, args.seed, args.run_id)
