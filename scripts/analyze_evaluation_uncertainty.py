"""Analyze scenario uncertainty for the frozen ideal10h evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file.

    :param path: Input file.
    :return: Rows as dictionaries.
    """
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def analyze(rows: list[dict[str, str]], group_fields: tuple[str, ...], metrics: dict[str, str], comparisons: tuple[str, ...], count: int, seed: int) -> list[dict[str, object]]:
    """Summarize groups and paired differences with scenario bootstrap intervals.

    :param rows: Scenario records.
    :param group_fields: Fields defining independent analysis groups.
    :param metrics: Output names and source column names.
    :param comparisons: Baseline methods compared with IL.
    :param count: Number of bootstrap resamples.
    :param seed: Random generator seed.
    :return: Summary records.
    """
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    groups = sorted({tuple(r[k] for k in group_fields) for r in rows})
    for group in groups:
        selected = [r for r in rows if tuple(r[k] for k in group_fields) == group]
        methods = sorted({r["method"] for r in selected})
        indexed = {m: {r["seed"]: r for r in selected if r["method"] == m} for m in methods}
        for method in methods:
            nrows = sum(r["method"] == method for r in selected)
            if len(indexed[method]) != nrows:
                raise ValueError(f"Duplicate scenario in {group}, {method}")
        if any(set(indexed[m]) != set(indexed[methods[0]]) for m in methods):
            raise ValueError(f"Unmatched scenarios in {group}")
        ids = sorted(indexed[methods[0]])
        n = len(ids)
        draws = rng.integers(0, n, size=(count, n))
        for metric, field in metrics.items():
            def values(method: str) -> np.ndarray:
                """Extract one method’s values.

                :param method: Method name.
                :return: Values in scenario order.
                """
                return np.array([float(indexed[method][i][field]) for i in ids])

            def estimate(method: str, samples: np.ndarray | None = None) -> np.ndarray:
                """Compute the requested statistic.

                :param method: Method name.
                :param samples: Optional resampling indices.
                :return: Statistic for original or resampled scenes.
                """
                x = values(method)
                x = x if samples is None else x[samples]
                if metric == "cost_per_purchased_kwh":
                    energy = values_energy(method)
                    energy = energy if samples is None else energy[samples]
                    return np.sum(x, axis=-1) / np.sum(energy, axis=-1)
                if metric == "scenario_failure_rate":
                    x = (x > 0).astype(float) * 100
                elif metric == "vehicle_failure_rate":
                    x = x / 200 * 100
                if metric == "total_shortfall_kwh":
                    return np.sum(x, axis=-1)
                return np.mean(x, axis=-1)

            def values_energy(method: str) -> np.ndarray:
                """Extract purchased energy.

                :param method: Method name.
                :return: Energy in scenario order.
                """
                return np.array([float(indexed[method][i]["final_energy_kwh"]) for i in ids])

            for method in methods:
                x = values(method)
                if metric == "cost_per_purchased_kwh":
                    x = x / values_energy(method)
                elif metric == "scenario_failure_rate":
                    x = (x > 0).astype(float) * 100
                elif metric == "vehicle_failure_rate":
                    x = x / 200 * 100
                boot = estimate(method, draws) if group_fields != ("vehicles",) else None
                lo, hi = np.quantile(boot, [0.025, 0.975]) if boot is not None else (None, None)
                output.append({**dict(zip(group_fields, group)), "metric": metric, "method": method, "comparison": "", "n_scenarios": n, "estimate": float(estimate(method)), "sd_scenarios": float(np.std(x, ddof=1)), "ci_low": float(lo) if lo is not None else None, "ci_high": float(hi) if hi is not None else None, "min_scenario": float(x.min()), "max_scenario": float(x.max())})
            if "IL" in methods:
                for baseline in comparisons:
                    if baseline not in methods:
                        continue
                    diff = estimate("IL", draws) - estimate(baseline, draws) if group_fields != ("vehicles",) else None
                    lo, hi = np.quantile(diff, [0.025, 0.975]) if diff is not None else (None, None)
                    raw_diff = values("IL") - values(baseline)
                    if metric == "scenario_failure_rate":
                        raw_diff = ((values("IL") > 0).astype(float) - (values(baseline) > 0).astype(float)) * 100
                    elif metric == "vehicle_failure_rate":
                        raw_diff = raw_diff / 200 * 100
                    output.append({**dict(zip(group_fields, group)), "metric": metric, "method": "IL", "comparison": baseline, "n_scenarios": n, "estimate": float(estimate("IL") - estimate(baseline)), "sd_scenarios": None, "ci_low": float(lo) if lo is not None else None, "ci_high": float(hi) if hi is not None else None, "min_scenario": float(raw_diff.min()) if metric != "cost_per_purchased_kwh" else None, "max_scenario": float(raw_diff.max()) if metric != "cost_per_purchased_kwh" else None})
    return output



def analyze_runs(rows: list[dict[str, str]], formal: list[dict[str, str]]) -> list[dict[str, object]]:
    """Average scenarios within each training run, then summarize across runs.

    :param rows: Complete independently trained model evaluations.
    :param formal: Frozen-model records used to verify scenario identities.
    :return: Per-run values and across-run mean and sample standard deviation.
    """
    if len(rows) != 900 or {r["method"] for r in rows} != {"IL", "DDQN", "PPO"} or {r["capacity_kw"] for r in rows} != {"600", "800"}:
        raise ValueError("Expected 900 rows for IL, DDQN, and PPO at both capacities")
    metrics = {"cost": "cost", "unfairness": "unfairness", "weighted_objective": "weighted_objective", "cost_per_purchased_kwh": "cost", "shortfall_kwh": "gap_kwh", "scenario_failure_rate": "failed_vehicles", "vehicle_failure_rate": "failed_vehicles", "all_met_rate": "all_met"}
    records: list[dict[str, object]] = []
    for method in sorted({r["method"] for r in rows}):
        for capacity in sorted({r["capacity_kw"] for r in rows}):
            subset = [r for r in rows if r["method"] == method and r["capacity_kw"] == capacity]
            runs = sorted({r["run_id"] for r in subset})
            if len(runs) != 3:
                raise ValueError(f"Exactly three runs are required: {method}, {capacity}")
            groups = {run: [r for r in subset if r["run_id"] == run] for run in runs}
            seed_sets = [{r["seed"] for r in group} for group in groups.values()]
            if any(len(group) != 50 or len(seeds) != 50 for group, seeds in zip(groups.values(), seed_sets)) or any(seeds != seed_sets[0] for seeds in seed_sets):
                raise ValueError(f"Incomplete or mismatched evaluation scenes: {method}, {capacity}")
            training_seeds = [set(r["training_seed"] for r in groups[run]) for run in runs]
            if any(len(x) != 1 for x in training_seeds) or len({next(iter(x)) for x in training_seeds}) != len(runs):
                raise ValueError(f"Training seeds must be distinct: {method}, {capacity}")
            expected = {r["seed"] for r in formal if r["capacity_kw"] == capacity and r["feasibility"] == "feasible"}
            if seed_sets[0] != expected:
                raise ValueError(f"Training-repeat scenes differ from formal evaluation: {method}, {capacity}")
            for metric, field in metrics.items():
                values = []
                for run in runs:
                    x = np.array([float(r[field]) for r in groups[run]])
                    if metric == "scenario_failure_rate":
                        x = (x > 0).astype(float) * 100
                    elif metric == "vehicle_failure_rate":
                        x = x / 200 * 100
                    elif metric == "all_met_rate":
                        x = x * 100
                    if metric == "cost_per_purchased_kwh":
                        energy = np.array([float(r["purchased_kwh"]) for r in groups[run]])
                        values.append(float(x.sum() / energy.sum()))
                    else:
                        values.append(float(x.mean()))
                records.append({"method": method, "capacity_kw": capacity, "metric": metric, "n_runs": len(runs), "run_ids": runs, "training_seeds": [next(iter(x)) for x in training_seeds], "run_means": values, "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))})
    return records

def main() -> None:
    """Run the analysis and save results locally.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/ideal10h_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ideal10h_v1")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--training-repeats", type=Path, help="Complete multi-run scenarios CSV")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    paths = {name: args.input_dir / name for name in ("physical_audit.csv", "small_evaluation_scenarios.csv", "scalability_scenarios.csv")}
    formal = read_csv(paths["physical_audit.csv"])
    small = read_csv(paths["small_evaluation_scenarios.csv"])
    scale = read_csv(paths["scalability_scenarios.csv"])
    if len(formal) != 500 or len(small) != 100 or len(scale) != 42:
        raise ValueError("Unexpected scenario record count")
    if any(r["feasibility"] != "feasible" for r in formal):
        raise ValueError("Formal records include non-feasible initial scenarios")
    result = {
        "formal": analyze(formal, ("capacity_kw",), {"cost": "cost", "unfairness": "unfairness", "cost_per_purchased_kwh": "cost", "scenario_failure_rate": "failed_vehicles", "vehicle_failure_rate": "failed_vehicles", "shortfall_kwh": "gap_kwh", "total_shortfall_kwh": "gap_kwh"}, ("PPO", "DDQN", "UPA"), args.bootstrap, args.seed),
        "small": analyze(small, (), {"cost": "cost", "unfairness": "unfairness", "scenario_failure_rate": "failed", "shortfall_kwh": "gap_kwh"}, ("IDEAL", "PPO", "DDQN", "UC"), args.bootstrap, args.seed + 1),
        "scalability": analyze(scale, ("vehicles",), {"seconds_per_step": "seconds_per_step", "max_step_seconds": "max_step_seconds"}, ("UC",), args.bootstrap, args.seed + 2),
    }
    if args.training_repeats:
        paths["training_repeats"] = args.training_repeats
        result["training_repeats"] = analyze_runs(read_csv(args.training_repeats), formal)
    payload = {"bootstrap_resamples": args.bootstrap, "random_seed": args.seed, "inputs_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}, "results": result}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "uncertainty_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = [{"dataset": dataset, **row} for dataset, group in result.items() if dataset != "training_repeats" for row in group]
    fields = ["dataset", "capacity_kw", "vehicles", "metric", "method", "comparison", "n_scenarios", "estimate", "sd_scenarios", "ci_low", "ci_high", "min_scenario", "max_scenario"]
    with (args.output_dir / "uncertainty_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {args.output_dir}")


if __name__ == "__main__":
    main()
