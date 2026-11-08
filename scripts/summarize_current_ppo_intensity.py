"""Summarize battery-side intensity for the frozen main PPO controller."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from evaluate_pressure21 import CONFIG, ROOT, digest, load_case
from pressure21_current_rl import PPO_RUN, current_rl, local_path, selection
from summarize_r106_selected_ppo_intensity import verified_comparators, write_rows


def main() -> None:
    """Check and aggregate the 100 main PPO formal trajectories.

    :return: None.
    """
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    chosen = selection()["ppo_main"]
    data = [r for r in current_rl("formal") if r["method"] == "PPO" and int(r["training_seed"]) == int(chosen["seed"])]
    expected = {(int(seed), int(cap)) for seed in cfg["formal_seeds"] for cap in cfg["formal_capacities_kw"]}
    if len(data) != 100 or {(int(r["seed"]), int(r["capacity_kw"])) for r in data} != expected:
        raise RuntimeError("Incomplete main PPO formal cohort")
    rows = []
    for row in data:
        seed, cap = int(row["seed"]), int(row["capacity_kw"])
        scenario, _, input_hash = load_case("formal", seed, 200, cfg)
        path = local_path(row["trajectory_path"])
        if digest(path) != row["trajectory_sha256"] or input_hash != row["input_sha256"] or row["model_sha256"] != chosen["sha256"]:
            raise RuntimeError("PPO trajectory identity mismatch")
        with np.load(path) as trajectory:
            action, soc = trajectory["final"], trajectory["soc"]
            power = action * scenario.power
            battery = power * scenario.efficiency * scenario.dt
            intensity = power * scenario.efficiency / scenario.capacity
            residual = np.abs(soc[1:] - soc[:-1] - battery / scenario.capacity).max()
            if not np.isfinite(residual) or residual > 1e-8:
                raise RuntimeError("SoC dynamics mismatch")
            if not np.isclose((power * scenario.dt).sum(), float(row["purchased_kwh"]), rtol=1e-10, atol=1e-10):
                raise RuntimeError("Purchased energy mismatch")
            total = float(battery.sum())
            rows.append({"method": "PPO", "seed": seed, "capacity_kw": cap, "model_sha256": chosen["sha256"], "trajectory_sha256": row["trajectory_sha256"], "battery_energy_kwh": total, "energy_weighted_intensity_per_hour": float((battery * intensity).sum() / total), "daily_peak_intensity_per_hour": float(intensity.max()), "soc_dynamics_max_residual": float(residual)})
    summary = []
    for cap in cfg["formal_capacities_kw"]:
        group = [r for r in rows if r["capacity_kw"] == cap]
        summary.append({"capacity_kw": cap, "method": "PPO", "days": len(group), "mean_daily_energy_weighted_intensity_per_hour": mean(r["energy_weighted_intensity_per_hour"] for r in group), "maximum_daily_peak_intensity_per_hour": max(r["daily_peak_intensity_per_hour"] for r in group), "mean_daily_battery_energy_kwh": mean(r["battery_energy_kwh"] for r in group)})
    out = ROOT / "results/pressure21_v1/charging_intensity"
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "daily.csv", rows)
    write_rows(out / "summary.csv", summary)
    write_rows(out / "combined_summary.csv", verified_comparators() + summary)
    sources = [Path(__file__), PPO_RUN / "selection.json", PPO_RUN / "evaluation/formal/scenarios.csv"]
    manifest = {"main_model_sha256": chosen["sha256"], "records": len(rows), "source_sha256": {str(p): digest(p) for p in sources}, "output_sha256": {n: digest(out / n) for n in ("daily.csv", "summary.csv", "combined_summary.csv")}, "maximum_soc_dynamics_residual": max(r["soc_dynamics_max_residual"] for r in rows), "definition": "Battery-side efficiency times grid power divided by battery capacity; daily means weighted by delivered battery energy; peak is the maximum over slot, EV, and day."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print("Summarized and checked 100 main PPO trajectories")


if __name__ == "__main__":
    main()
