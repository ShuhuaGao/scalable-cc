"""Evaluate validation-frozen LR winners against matched original models."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_pressure21 import CONFIG, load_case

from evcharging.models import network
from evcharging.simulation_schema import simulate


def sha(p: Path) -> str:
    """Hash a file.

    :param p: Input file.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(task: tuple[str, str, str, str, int, int]) -> dict:
    """Evaluate and audit one frozen model and formal scenario.

    :param task: Method, comparison label, model path and hash, day and capacity.
    :return: Scenario metrics and identity hashes.
    """
    method, label, path, expected, seed, cap = task
    torch.set_num_threads(1)
    cfg = json.loads(CONFIG.read_text())
    s, _input_path, input_hash = load_case("formal", seed, 200, cfg)
    path = Path(path)
    if sha(path) != expected:
        raise RuntimeError("Frozen model changed")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = network(6 if method == "ddqn" else 2)
    model.load_state_dict(ck["model"])
    model.eval()

    def policy(x: np.ndarray, k: int, urgency: np.ndarray) -> float:
        """Evaluate the deterministic checkpoint policy.

        :param x: Policy input.
        :param k: Vehicle index.
        :param urgency: Fleet urgency.
        :return: Normalized action.
        """
        with torch.no_grad():
            z = model(torch.tensor(x, dtype=torch.float32)[None, :])
            if method == "ddqn":
                return float(z.argmax()) / 5
            ab = torch.nn.functional.softplus(z) + 1
            return float(ab[0, 0] / ab.sum())

    r = simulate(s, cap, policy, state_schema="pressure21_v1")
    a, soc = (r["final"], r["soc"])
    t = np.arange(len(s.price))[:, None]
    active = (s.ta[None, :] <= t) & (t < s.td[None, :])
    rem = np.maximum(s.td - t, 1)
    urg = np.where(
        active,
        np.maximum(s.target - soc[:-1], 0)
        * s.capacity
        / (s.efficiency * s.power * s.dt * rem),
        0,
    )
    shares = np.divide(
        urg,
        urg.sum(axis=1, keepdims=True),
        out=np.zeros_like(urg),
        where=urg.sum(axis=1, keepdims=True) > 0,
    )
    cost = a * s.power * s.price[:, None] * s.dt
    unfair = (a - a.sum(axis=1, keepdims=True) * shares) ** 2
    checks = [
        np.abs(cost - r["cost"]).max(),
        np.abs(unfair - r["unfair"]).max(),
        np.abs(
            soc[1:] - soc[:-1] - a * s.power * s.efficiency * s.dt / s.capacity
        ).max(),
        np.maximum(a @ s.power - cap, 0).max(),
        np.maximum(-a, 0).max(),
        np.maximum(a - 1, 0).max(),
        np.abs(a[~active]).max(initial=0),
        np.abs(soc[0] - s.arrival).max(),
        np.maximum(0.2 - soc, 0).max(),
        np.maximum(soc - 1, 0).max(),
        np.maximum(soc - s.target, 0).max(),
    ]
    res = float(max(checks))
    if not np.isfinite(res) or res > cfg["physical_tolerance"]:
        raise RuntimeError(f"Physical audit failed {task}: {checks}")
    gap = np.maximum(s.target - soc[s.td, np.arange(200)], 0)
    failed = int((gap > cfg["soc_tolerance"]).sum())
    c = float(cost.sum())
    u = float(unfair.sum())
    return {
        "method": method,
        "label": label,
        "seed": seed,
        "capacity_kw": cap,
        "cost": c,
        "unfairness": u,
        "weighted_objective": 0.7 * c + 0.3 * u,
        "failed_vehicles": failed,
        "all_met": int(failed == 0),
        "gap_kwh": float(gap @ s.capacity),
        "purchased_kwh": float((a * s.power * s.dt).sum()),
        "physical_max_residual": res,
        "input_sha256": input_hash,
        "model_sha256": expected,
    }


def main() -> None:
    """Compare each changed validation winner with its matched baseline.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    selection_path = out / "selection.json"
    sel = json.loads(selection_path.read_text())
    dest = out / "formal"
    if sel["formal_test_used_for_selection"]:
        raise RuntimeError("Invalid selection")
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir()
    cfg = json.loads(CONFIG.read_text())
    tasks = []
    for method in ("ddqn", "ppo"):
        winner = sel["winners"][method]
        base = sel["baseline"][method]
        if winner["sha256"] == base["sha256"]:
            continue
        for label, item in [("original", base), ("winner", winner)]:
            for cap in cfg["formal_capacities_kw"]:
                for seed in cfg["formal_seeds"]:
                    root = ROOT if item["path_base"] == "root" else out
                    tasks.append(
                        (
                            method,
                            label,
                            str(root / item["relative_path"]),
                            item["sha256"],
                            seed,
                            cap,
                        )
                    )
    source = [
        Path(__file__),
        CONFIG,
        selection_path,
        ROOT / "src/evcharging/core.py",
        ROOT / "src/evcharging/models.py",
        ROOT / "src/evcharging/state_schema.py",
        ROOT / "src/evcharging/simulation_schema.py",
    ]
    (dest / "manifest.json").write_text(
        json.dumps(
            {
                "selection_sha256": sha(selection_path),
                "sources_sha256": {str(p): sha(p) for p in source},
                "tasks": len(tasks),
                "workers": args.workers,
                "note": "Both baseline and winner use the same current simulator and archived formal inputs; validation selection was frozen before this evaluation.",
            },
            indent=2,
        )
    )
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(evaluate, tasks), 1):
            rows.append(row)
            if i % 20 == 0:
                print("formal", i, "/", len(tasks), flush=True)
    if rows:
        with (dest / "scenarios.csv").open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
        summary = []
        for method in sorted({r["method"] for r in rows}):
            for cap in cfg["formal_capacities_kw"]:
                for label in ("original", "winner"):
                    group = [
                        r
                        for r in rows
                        if r["method"] == method
                        and r["capacity_kw"] == cap
                        and (r["label"] == label)
                    ]
                    if len(group) != 50:
                        raise RuntimeError("Incomplete group")
                    summary.append(
                        {
                            "method": method,
                            "capacity_kw": cap,
                            "label": label,
                            **{
                                k: float(np.mean([r[k] for r in group]))
                                for k in (
                                    "cost",
                                    "unfairness",
                                    "weighted_objective",
                                    "all_met",
                                    "failed_vehicles",
                                    "gap_kwh",
                                    "purchased_kwh",
                                )
                            },
                        }
                    )
    else:
        summary = []
    (dest / "summary.json").write_text(
        json.dumps({"count": len(rows), "groups": summary}, indent=2)
    )
    (dest / "complete.json").write_text(
        json.dumps(
            {"records": len(rows), "summary_sha256": sha(dest / "summary.json")},
            indent=2,
        )
    )
    print("formal complete", len(rows), flush=True)


if __name__ == "__main__":
    main()
