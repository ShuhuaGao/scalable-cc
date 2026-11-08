"""Replay and audit the validation-selected PPO charging intensity."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from statistics import mean

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import torch
from evaluate_pressure21 import CONFIG, load_case
from summarize_r106_charging_intensity import (
    INPUT,
    digest,
    local_source,
    read_rows,
    write_rows,
)

from evcharging.models import network
from evcharging.simulation_schema import simulate

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/r106_v1/charging_intensity"
COMPARATORS = ("IL", "UC", "DDQN", "UPA", "RDDPG")


def replay(task: tuple[str, str, int, int]) -> dict:
    """Replay one selected policy case and audit its physical residuals.

    :param task: Model path and hash, scenario seed, and station capacity.
    :return: Daily intensity and audit metrics.
    """
    model_path, model_hash, seed, cap = task
    torch.set_num_threads(1)
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    s, _, input_hash = load_case("formal", seed, 200, cfg)
    if digest(Path(model_path)) != model_hash:
        raise ValueError("Selected model changed")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = network(2)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    def policy(x: np.ndarray, k: int, urgency: np.ndarray) -> float:
        """Return the frozen PPO actor's deterministic beta mean.

        :param x: Current policy state.
        :param k: Vehicle index.
        :param urgency: Current fleet urgency.
        :return: Normalized charging action.
        """
        with torch.no_grad():
            z = model(torch.tensor(x, dtype=torch.float32)[None, :])
            ab = torch.nn.functional.softplus(z) + 1
            return float(ab[0, 0] / ab.sum())

    r = simulate(s, cap, policy, state_schema="pressure21_v1")
    a, soc = r["final"], r["soc"]
    power = a * s.power
    battery = power * s.efficiency * s.dt
    intensity = power * s.efficiency / s.capacity
    t = np.arange(len(s.price))[:, None]
    active = (s.ta <= t) & (t < s.td)
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
        "cost": float(np.abs(cost - r["cost"]).max()),
        "unfairness": float(np.abs(unfair - r["unfair"]).max()),
        "dynamics_soc": float(
            np.abs(
                soc[1:] - soc[:-1] - a * s.power * s.efficiency * s.dt / s.capacity
            ).max()
        ),
        "station_capacity_kw": float(np.maximum(a @ s.power - cap, 0).max()),
        "action_lower": float(np.maximum(-a, 0).max()),
        "action_upper": float(np.maximum(a - 1, 0).max()),
        "disconnected_action": float(np.abs(a[~active]).max(initial=0)),
        "initial_soc": float(np.abs(soc[0] - s.arrival).max()),
        "soc_lower": float(np.maximum(0.2 - soc, 0).max()),
        "soc_upper": float(np.maximum(soc - 1, 0).max()),
        "target_overshoot": float(np.maximum(soc - s.target, 0).max()),
        "energy_balance_kwh": float(
            np.abs((soc[1:] - soc[:-1]) * s.capacity - battery).max()
        ),
        "disconnected_power_kw": float(np.abs(power[~active]).max(initial=0)),
    }
    if any(not np.isfinite(v) or v >= 1e-8 for v in checks.values()):
        raise ValueError(f"Physical audit failed for {seed}, {cap}: {checks}")
    total = float(battery.sum())
    if not np.isfinite(total) or total <= 0 or not np.isfinite(intensity).all():
        raise ValueError("Invalid charging intensity")
    gap = np.maximum(s.target - soc[s.td, np.arange(200)], 0)
    failed = int((gap > cfg["soc_tolerance"]).sum())
    c, u = float(cost.sum()), float(unfair.sum())
    original_check_names = list(checks)[:11]
    return {
        "method": "ppo",
        "label": "winner",
        "seed": seed,
        "capacity_kw": cap,
        "cost": c,
        "unfairness": u,
        "weighted_objective": 0.7 * c + 0.3 * u,
        "failed_vehicles": failed,
        "all_met": int(failed == 0),
        "gap_kwh": float(gap @ s.capacity),
        "purchased_kwh": float((a * s.power * s.dt).sum()),
        "physical_max_residual": max(checks[k] for k in original_check_names),
        "input_sha256": input_hash,
        "model_sha256": model_hash,
        "battery_energy_kwh": total,
        "energy_weighted_intensity_per_hour": float(
            (battery * intensity).sum() / total
        ),
        "daily_peak_intensity_per_hour": float(intensity.max()),
        "maximum_departure_soc_gap": float(gap.max()),
        **{f"residual_{k}": v for k, v in checks.items()},
    }


def verified_comparators() -> list[dict]:
    """Verify the original archive and return its five comparator summaries.

    :return: Ten capacity-method summary rows.
    """
    manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf8"))
    for name in ("daily.csv", "summary.csv", "paired_differences.csv"):
        if digest(OLD / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Original output hash mismatch: {name}")
    for recorded, expected in manifest["source_table_sha256"].items():
        if digest(local_source(recorded)) != expected:
            raise ValueError(f"Original source table hash mismatch: {recorded}")
    daily = [r for r in read_rows(OLD / "daily.csv") if r["method"] in COMPARATORS]
    expected = {
        (m, str(seed), str(cap))
        for m in COMPARATORS
        for seed in range(65000, 65050)
        for cap in (600, 800)
    }
    actual = [(r["method"], r["seed"], r["capacity_kw"]) for r in daily]
    if len(actual) != 500 or set(actual) != expected:
        raise ValueError("Original comparator daily cohort mismatch")
    for row in daily:
        if digest(INPUT / f"{row['seed']}.npz") != row["input_sha256"]:
            raise ValueError("Original comparator input hash mismatch")
        if digest(local_source(row["trajectory_path"])) != row["trajectory_sha256"]:
            raise ValueError("Original comparator trajectory hash mismatch")
    other = [r for r in read_rows(OLD / "summary.csv") if r["method"] in COMPARATORS]
    pairs = [(r["method"], r["capacity_kw"]) for r in other]
    if len(pairs) != 10 or set(pairs) != {
        (m, str(cap)) for m in COMPARATORS for cap in (600, 800)
    }:
        raise ValueError("Original comparator summary cohort mismatch")
    return other


def main() -> None:
    """Recompute selected PPO and preserve the original intensity archive.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Learning-rate search archive")
    parser.add_argument(
        "--output", help="New output directory; defaults to a directory under --run"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    out = (
        Path(args.output).resolve()
        if args.output
        else run / "charging_intensity_selected_ppo"
    )
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads((run / "selection.json").read_text(encoding="utf8"))
    if selection["formal_test_used_for_selection"]:
        raise ValueError("Formal test used for selection")
    winner = selection["winners"]["ppo"]
    if winner["episode"] != 140 or winner["label"] != "half":
        raise ValueError("Unexpected selected PPO")
    model = (ROOT if winner["path_base"] == "root" else run) / winner["relative_path"]
    if digest(model) != winner["sha256"]:
        raise ValueError("Selected model hash mismatch")
    formal = [
        r
        for r in read_rows(run / "formal/scenarios.csv")
        if r["method"] == "ppo" and r["label"] == "winner"
    ]
    old = {(int(r["seed"]), int(r["capacity_kw"])): r for r in formal}
    keys = [(seed, cap) for cap in (600, 800) for seed in range(65000, 65050)]
    if len(formal) != 100 or set(old) != set(keys):
        raise ValueError("Unexpected formal winner cohort")
    other = verified_comparators()
    tasks = [(str(model), winner["sha256"], seed, cap) for seed, cap in keys]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(replay, tasks))
    numeric = (
        "cost",
        "unfairness",
        "weighted_objective",
        "failed_vehicles",
        "all_met",
        "gap_kwh",
        "purchased_kwh",
        "physical_max_residual",
    )
    maxima = {k: 0.0 for k in numeric}
    for row in rows:
        previous = old[row["seed"], row["capacity_kw"]]
        for k in ("input_sha256", "model_sha256"):
            if row[k] != previous[k]:
                raise ValueError(f"Identity mismatch: {k}")
        for k in numeric:
            difference = abs(float(row[k]) - float(previous[k]))
            if not np.isfinite(difference):
                raise ValueError(f"Nonfinite difference: {k}")
            maxima[k] = max(maxima[k], difference)
    if any(maxima.values()):
        raise ValueError(f"Formal replay differs: {maxima}")
    summary = []
    for cap in (600, 800):
        group = [r for r in rows if r["capacity_kw"] == cap]
        summary.append(
            {
                "capacity_kw": cap,
                "method": "PPO_selected_lr5e-5",
                "days": len(group),
                "mean_daily_energy_weighted_intensity_per_hour": mean(
                    r["energy_weighted_intensity_per_hour"] for r in group
                ),
                "maximum_daily_peak_intensity_per_hour": max(
                    r["daily_peak_intensity_per_hour"] for r in group
                ),
                "mean_daily_battery_energy_kwh": mean(
                    r["battery_energy_kwh"] for r in group
                ),
            }
        )
    out.mkdir(parents=True)
    write_rows(out / "daily.csv", rows)
    write_rows(out / "summary.csv", summary)
    write_rows(out / "combined_summary.csv", other + summary)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/evaluate_pressure21.py",
        ROOT / "scripts/summarize_r106_charging_intensity.py",
        CONFIG,
        ROOT / "src/evcharging/core.py",
        ROOT / "src/evcharging/models.py",
        ROOT / "src/evcharging/state_schema.py",
        ROOT / "src/evcharging/simulation_schema.py",
        run / "selection.json",
        run / "formal/scenarios.csv",
        OLD / "manifest.json",
        OLD / "daily.csv",
        OLD / "summary.csv",
    ]
    manifest = {
        "selected_checkpoint_sha256": winner["sha256"],
        "records": len(rows),
        "formal_max_absolute_differences": maxima,
        "maximum_residuals": {
            k: max(r[k] for r in rows) for k in rows[0] if k.startswith("residual_")
        },
        "residual_strict_tolerance": 1e-8,
        "verified_comparator_trajectories": 500,
        "failed_vehicles_total": sum(r["failed_vehicles"] for r in rows),
        "days_all_met": sum(r["all_met"] for r in rows),
        "maximum_departure_soc_gap": max(r["maximum_departure_soc_gap"] for r in rows),
        "sources_sha256": {str(p): digest(p) for p in sources},
        "output_sha256": {
            p: digest(out / p)
            for p in ("daily.csv", "summary.csv", "combined_summary.csv")
        },
        "definition": "Battery-side efficiency * grid power / battery capacity; daily mean weighted by delivered battery energy; peak is maximum slot-vehicle value over 50 days.",
        "note": "Selected PPO is separate from the original six-method archive; other methods are reused from that archive.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    print(f"Audited {len(rows)} selected PPO trajectories: {out}")


if __name__ == "__main__":
    main()
