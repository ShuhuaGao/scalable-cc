"""Evaluate frozen Pressure policies on existing paper inputs in a new run."""

from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evcharging.core import generate, load_scenario
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


from evcharging.r102_distribution import simulate as simulate_public
import evcharging.simulation_mpc10h_timing as timing
from evcharging.state_schema import build_state

CONFIG = ROOT / "configs/pressure21_evaluation.json"
DEFAULT_RUN_ID = "r102_distributional_v1_reproduce"
ARRAYS = ("soc", "raw", "clip", "scale", "final", "cost", "unfair")


def digest(path: Path) -> str:
    """Hash a file.

    :param path: Input path.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(dataset: str, seed: int, n: int, cfg: dict):
    """Load an existing formal, small or scale scenario and verify its identity.

    :param dataset: Input collection.
    :param seed: Day seed.
    :param n: Fleet size.
    :param cfg: Evaluation settings.
    :return: Scenario, source file and SHA-256 digest.
    """
    if dataset in ("formal", "development"):
        path = (
            ROOT
            / cfg[
                "formal_input_dir" if dataset == "formal" else "development_input_dir"
            ]
            / f"{seed}.npz"
        )
        s = load_scenario(str(path))
    elif dataset == "small":
        path = ROOT / cfg["small_expert_dir"] / f"seed_{seed}.npz"
        s = generate(seed, n, fixed_day=True)
        with np.load(path) as data:
            if any(
                not np.array_equal(data[k], getattr(s, k))
                for k in ("ta", "td", "arrival", "target", "capacity", "power", "price")
            ):
                raise RuntimeError(f"Small expert input mismatch: {path}")
    else:
        path = ROOT / cfg["scale_input_dir"] / f"input_{n}_{seed}.npz"
        s = load_scenario(str(path))
        expected = generate(seed, n, fixed_day=True)
        if any(
            not np.array_equal(np.asarray(getattr(s, k)), np.asarray(v))
            for k, v in vars(expected).items()
        ):
            raise RuntimeError(f"Scale input mismatch: {path}")
    if len(s.ta) != n or s.beta != 0.3:
        raise RuntimeError(f"Scenario identity mismatch: {path}")
    return s, path, digest(path)


def evaluate_one(task: tuple[str, int, int, int, int, str, str]) -> dict:
    """Re-evaluate one frozen model/scenario/capacity combination.

    :param task: Dataset, model seed, day seed, fleet size, limit, output path and run ID.
    :return: Audited scenario metrics.
    """
    dataset, train_seed, seed, n, limit, out, run_id = task
    torch.set_num_threads(1)
    cfg = json.loads(CONFIG.read_text())
    s, input_path, input_hash = load_case(dataset, seed, n, cfg)
    model_path = ROOT / "artifacts/runs" / run_id / f"il_{train_seed}/il_model/best.pt"
    summary = json.loads((model_path.parent / "summary.json").read_text())
    if digest(model_path) != summary["continuation_sha256"]:
        raise RuntimeError("Model digest mismatch")
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    if (ck.get("variant"), ck.get("seed"), ck.get("stage"), ck.get("state_schema")) != (
        "pressure",
        train_seed,
        "continuation",
        "distribution27_v1",
    ):
        raise RuntimeError("Checkpoint identity mismatch")
    model = imitation_network()
    model.load_state_dict(ck["model"])
    model.eval()
    builder = lambda scenario, t, k, soc, proposed, capacity: build_state(
        scenario, t, k, soc, proposed, capacity, "pressure21_v1"
    )
    timing.causal_state = builder

    def policy(x: np.ndarray, k: int, urgency: np.ndarray) -> float:
        """Return the frozen Pressure policy action.

        :param x: Current causal Pressure state.
        :param k: Vehicle index.
        :param urgency: Current fleet urgency.
        :return: Normalized action.
        """
        del k, urgency
        with torch.no_grad():
            return float(model(torch.tensor(x, dtype=torch.float32)[None, :])[0, 0])

    if dataset == "timing":
        r = timing.simulate(s, limit, policy)
        ref = simulate_public(
            s, limit, policy, state_schema="distribution27_v1", pressure_reference=True
        )
        for key in (*ARRAYS, "states", "records"):
            if not np.array_equal(r[key], ref[key]):
                raise RuntimeError(f"Timing differs from standard {n} {seed} {key}")
    else:
        r = simulate_public(s, limit, policy, state_schema="distribution27_v1")
    a, soc = r["final"], r["soc"]
    t = np.arange(len(s.price))[:, None]
    active = (s.ta[None, :] <= t) & (t < s.td[None, :])
    rem = np.maximum(s.td - t, 1)
    urgency = np.where(
        active,
        np.maximum(s.target - soc[:-1], 0)
        * s.capacity
        / (s.efficiency * s.power * s.dt * rem),
        0,
    )
    shares = np.divide(
        urgency,
        urgency.sum(axis=1, keepdims=True),
        out=np.zeros_like(urgency),
        where=urgency.sum(axis=1, keepdims=True) > 0,
    )
    cost = a * s.power * s.price[:, None] * s.dt
    unfair = (a - a.sum(axis=1, keepdims=True) * shares) ** 2
    checks = {
        "cost": np.abs(cost - r["cost"]).max(),
        "unfair": np.abs(unfair - r["unfair"]).max(),
        "dynamics": np.abs(
            soc[1:] - soc[:-1] - a * s.power * s.efficiency * s.dt / s.capacity
        ).max(),
        "station": np.maximum(a @ s.power - limit, 0).max(),
        "lower_action": np.maximum(-a, 0).max(),
        "upper_action": np.maximum(a - 1, 0).max(),
        "disconnected": np.abs(a[~active]).max(initial=0),
        "initial": np.abs(soc[0] - s.arrival).max(),
        "lower_soc": np.maximum(0.2 - soc, 0).max(),
        "upper_soc": np.maximum(soc - 1, 0).max(),
        "overcharge": np.maximum(soc - s.target, 0).max(),
    }
    residual = float(max(checks.values()))
    if not np.isfinite(residual) or residual > cfg["physical_tolerance"]:
        raise RuntimeError(f"Physical audit failed {task}: {checks}")
    gap = np.maximum(s.target - soc[s.td, np.arange(n)], 0)
    failed = int((gap > cfg["soc_tolerance"]).sum())
    energy = float((a * s.power * s.dt).sum())
    c = float(cost.sum())
    d = float(unfair.sum())
    path = Path(out) / f"{train_seed}_{n}_{seed}_{limit}.npz"
    saved = {key: r[key] for key in ARRAYS}
    if dataset == "timing":
        saved["slot_seconds"] = r["slot_seconds"]
    np.savez_compressed(path, **saved)
    result = {
        "dataset": dataset,
        "method": "IL",
        "run_id": f"{run_id}/il_{train_seed}",
        "training_seed": train_seed,
        "seed": seed,
        "vehicles": n,
        "capacity_kw": limit,
        "cost": c,
        "unfairness": d,
        "weighted_objective": 0.7 * c + 0.3 * d,
        "all_met": int(failed == 0),
        "failed_vehicles": failed,
        "vehicle_met_percent": 100 * (1 - failed / n),
        "gap_kwh": float(gap @ s.capacity),
        "purchased_kwh": energy,
        "total_spp_l1": float(np.abs(r["raw"] - a).sum()),
        "physical_max_residual": residual,
        "input_path": str(input_path),
        "input_sha256": input_hash,
        "model_sha256": digest(model_path),
        "trajectory_path": str(path),
        "trajectory_sha256": digest(path),
    }
    if dataset == "timing":
        result.update(
            seconds_per_step=float(np.mean(r["slot_seconds"])),
            max_step_seconds=float(np.max(r["slot_seconds"])),
        )
    (path.with_suffix(".json")).write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    return result


def main() -> None:
    """Run explicitly selected work while keeping previous archives untouched.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["formal"], required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or (args.dataset == "timing" and args.workers != 1):
        raise ValueError("Use 1-4 workers, and exactly 1 for timing")
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Run ID must be a single directory name")
    cfg = json.loads(CONFIG.read_text())
    out = ROOT / "artifacts/runs" / args.run_id / "evaluation" / args.dataset
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source_paths = [
        CONFIG,
        Path(__file__),
        ROOT / "src/evcharging/r102_distribution.py",
        ROOT / "src/evcharging/core.py",
    ]
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "formal_and_scale",
                "sources_sha256": {str(p): digest(p) for p in source_paths},
                "workers": args.workers,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    tasks = [
        ("formal", train, seed, 200, cap, str(out), args.run_id)
        for train in cfg["training_seeds"]
        for seed in cfg["formal_seeds"]
        for cap in cfg["formal_capacities_kw"]
    ]
    tasks += [
        ("scale", train, 81000 + n * 10 + i, n, n * 4, str(out), args.run_id)
        for train in cfg["training_seeds"]
        for n in (1000, 3000)
        for i in range(3)
    ]
    rows = []
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            rows.append(evaluate_one(task))
            print(args.dataset, i, "/", len(tasks), flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(evaluate_one, tasks), 1):
                rows.append(row)
                if i % 10 == 0:
                    print(args.dataset, i, "/", len(tasks), flush=True)
    if len(rows) != len(tasks):
        raise RuntimeError("Incomplete evaluation")
    with (out / "scenarios.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    (out / "complete.json").write_text(
        json.dumps(
            {"records": len(rows), "scenarios_sha256": digest(out / "scenarios.csv")},
            indent=2,
        ),
        encoding="utf8",
    )
    print(args.dataset, "complete", len(rows), flush=True)


if __name__ == "__main__":
    main()
