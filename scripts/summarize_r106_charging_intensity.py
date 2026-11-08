"""Summarize battery-side charging intensity from the main formal trajectories."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evcharging.core import load_scenario  # noqa: E402

EVAL = ROOT / "artifacts/runs/pressure21_v1/evaluation"
INPUT = ROOT / "artifacts/runs/ideal10h_v1/formal/inputs"
OUTPUT = ROOT / "outputs/charging_intensity"
METHODS = ("IL", "UC", "DDQN", "PPO", "UPA", "RDDPG")


def digest(path: Path) -> str:
    """Return the SHA-256 digest of a source file.

    :param path: Source path.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_source(recorded_path: str) -> Path:
    """Resolve a recorded source path in the current copy of the repository.

    :param recorded_path: Original absolute or repository-relative path.
    :return: Path under the current repository root.
    """
    parts = recorded_path.replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        if part in ("artifacts", "results"):
            return ROOT.joinpath(*parts[index:])
    raise ValueError(f"Cannot resolve recorded source path: {recorded_path}")


def read_rows(path: Path) -> list[dict]:
    """Read a UTF-8 CSV file.

    :param path: Source CSV path.
    :return: Row dictionaries.
    """
    with path.open(newline="", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write homogeneous rows to CSV.

    :param path: Destination path.
    :param rows: Row dictionaries.
    :return: None.
    """
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Audit and summarize 600 paired, previously saved full-day runs.

    :return: None.
    """
    sources = {
        EVAL / "formal/scenarios.csv": {"IL"},
        EVAL / "rl/formal/scenarios.csv": {"DDQN"},
        ROOT / "artifacts/runs/ppo_half_formal_20261107/evaluation/formal/scenarios.csv": {"PPO"},
        ROOT / "results/pressure21_v1/formal_references.csv": {"UC", "UPA", "RDDPG"},
    }
    selected = {"IL": 11002, "DDQN": 21002, "PPO": 24003}
    lookup = {}
    for path, methods in sources.items():
        for row in read_rows(path):
            method = row["method"]
            if method not in methods or (method in selected and int(row["training_seed"]) != selected[method]):
                continue
            key = (method, int(row["seed"]), int(row["capacity_kw"]))
            if key in lookup:
                raise ValueError(f"Duplicate case: {key}")
            lookup[key] = row
    expected = {(m, seed, cap) for m in METHODS for seed in range(65000, 65050) for cap in (600, 800)}
    if set(lookup) != expected:
        raise ValueError(f"Incomplete cohort: missing {expected - set(lookup)}, extra {set(lookup) - expected}")

    rows = []
    for method, seed, cap in sorted(expected, key=lambda x: (x[2], x[1], METHODS.index(x[0]))):
        row = lookup[method, seed, cap]
        source = INPUT / f"{seed}.npz"
        if digest(source) != row["input_sha256"]:
            raise ValueError(f"Input hash mismatch: {source}")
        scenario = load_scenario(str(source))
        path = local_source(row["trajectory_path"])
        if digest(path) != row["trajectory_sha256"]:
            raise ValueError(f"Trajectory hash mismatch: {path}")
        with np.load(path) as data:
            soc = data["soc"].copy()
            power = data["power"].copy() if method == "RDDPG" else data["final"].copy() * scenario.power
        dt = float(scenario.dt)
        efficiency = np.asarray(scenario.efficiency)
        if power.shape != (len(scenario.price), 200) or soc.shape != (len(scenario.price) + 1, 200):
            raise ValueError(f"Unexpected shape: {path}")
        if not (np.isfinite(power).all() and np.isfinite(soc).all() and np.all(scenario.capacity > 0) and np.all(efficiency > 0) and dt > 0):
            raise ValueError(f"Invalid values: {path}")
        active = (scenario.ta <= np.arange(len(scenario.price))[:, None]) & (np.arange(len(scenario.price))[:, None] < scenario.td)
        battery_kwh = efficiency * power * dt
        intensity = efficiency * power / scenario.capacity
        residual = float(np.max(np.abs((soc[1:] - soc[:-1]) * scenario.capacity - battery_kwh)))
        inactive = float(np.max(np.abs(power[~active]), initial=0))
        purchased = float((power * dt).sum())
        if (residual > 1e-8 or inactive > 1e-8 or power.min() < -1e-8
                or np.max(power - scenario.power) > 1e-8 or np.max(power.sum(axis=1) - cap) > 1e-8
                or not np.isclose(purchased, float(row["purchased_kwh"]), rtol=1e-10, atol=1e-8)
                or not np.allclose(soc[0], scenario.arrival, rtol=0, atol=1e-8)):
            raise ValueError(f"Energy or physical audit failed: {path}")
        total = float(battery_kwh.sum())
        if total <= 0:
            raise ValueError(f"No delivered energy: {path}")
        rows.append(dict(method=method, seed=seed, capacity_kw=cap,
                         battery_energy_kwh=total, purchased_energy_kwh=purchased,
                         energy_weighted_intensity_per_hour=float((battery_kwh * intensity).sum() / total),
                         daily_peak_intensity_per_hour=float(intensity.max()),
                         max_slot_vehicle_energy_residual_kwh=residual,
                         max_disconnected_power_kw=inactive,
                         input_sha256=digest(source), trajectory_sha256=digest(path),
                         trajectory_path=path.relative_to(ROOT).as_posix()))

    summary, paired = [], []
    for cap in (600, 800):
        for method in METHODS:
            group = [r for r in rows if r["capacity_kw"] == cap and r["method"] == method]
            summary.append(dict(capacity_kw=cap, method=method, days=len(group),
                                mean_daily_energy_weighted_intensity_per_hour=mean(r["energy_weighted_intensity_per_hour"] for r in group),
                                maximum_daily_peak_intensity_per_hour=max(r["daily_peak_intensity_per_hour"] for r in group),
                                mean_daily_battery_energy_kwh=mean(r["battery_energy_kwh"] for r in group)))
            ours = [r for r in rows if r["capacity_kw"] == cap and r["method"] == "IL"]
            if [r["seed"] for r in group] != [r["seed"] for r in ours]:
                raise ValueError("Scenario pairing mismatch")
            if method != "IL":
                paired.append(dict(capacity_kw=cap, comparator=method, days=len(group),
                                   mean_paired_intensity_difference_ours_minus_comparator_per_hour=mean(a["energy_weighted_intensity_per_hour"] - b["energy_weighted_intensity_per_hour"] for a, b in zip(ours, group))))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT / "daily.csv", rows)
    write_rows(OUTPUT / "summary.csv", summary)
    write_rows(OUTPUT / "paired_differences.csv", paired)
    metadata = dict(trajectory_count=len(rows), days_per_capacity=50, methods=METHODS,
                    definition="Battery-side intensity = efficiency * executed grid power (kW) / battery capacity (kWh), in h^-1. Daily mean is weighted by delivered battery-side energy efficiency * power * slot hours. Reported peak is the largest slot/vehicle intensity across 50 daily peaks.",
                    interpretation="Energy-normalized power exposure; not measured current C-rate, temperature, aging or lifetime.",
                    max_slot_vehicle_energy_residual_kwh=max(r["max_slot_vehicle_energy_residual_kwh"] for r in rows),
                    max_disconnected_power_kw=max(r["max_disconnected_power_kw"] for r in rows),
                    script_sha256=digest(Path(__file__).resolve()),
                    source_table_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
                    output_sha256={name: digest(OUTPUT / name) for name in ("daily.csv", "summary.csv", "paired_differences.csv")})
    (OUTPUT / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(f"Audited {len(rows)} trajectories; saved {OUTPUT}")


if __name__ == "__main__":
    main()
