"""Analyze paired PPO learning-rate results and the original IL comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_evaluation_uncertainty import analyze

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read scenario metrics.

    :param path: Input CSV.
    :return: Scenario rows.
    """
    with path.open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    """Hash a source file.

    :param path: Input file.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Summarize paired scenarios using the established bootstrap implementation.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    ppo_path = out / "formal/scenarios.csv"
    il_path = ROOT / "artifacts/runs/pressure21_v1/evaluation/formal/scenarios.csv"
    ppo_rows = read_csv(ppo_path)
    il_rows = [r for r in read_csv(il_path) if r["training_seed"] == "11002"]
    matched = []
    relative = []
    for capacity in ("600", "800"):
        il = {r["seed"]: r for r in il_rows if r["capacity_kw"] == capacity}
        win = {
            r["seed"]: r
            for r in ppo_rows
            if r["capacity_kw"] == capacity and r["label"] == "winner"
        }
        base = {
            r["seed"]: r
            for r in ppo_rows
            if r["capacity_kw"] == capacity and r["label"] == "original"
        }
        if len(il) != 50 or set(il) != set(win) or set(il) != set(base):
            raise RuntimeError("Incomplete paired cohort")
        for seed in sorted(il):
            if (
                il[seed]["input_sha256"] != win[seed]["input_sha256"]
                or il[seed]["input_sha256"] != base[seed]["input_sha256"]
            ):
                raise RuntimeError("Input mismatch")
            matched.append({**il[seed], "method": "IL"})
            matched.append({**win[seed], "method": "PPO"})
        item = {"capacity_kw": int(capacity)}
        for metric in ("cost", "unfairness", "weighted_objective"):
            original = float(np.mean([float(r[metric]) for r in base.values()]))
            winner = float(np.mean([float(r[metric]) for r in win.values()]))
            item[metric] = {
                "original_mean": original,
                "winner_mean": winner,
                "winner_minus_original_percent": 100 * (winner - original) / original,
            }
        item["original_all_met_days"] = sum(int(r["all_met"]) for r in base.values())
        item["winner_all_met_days"] = sum(int(r["all_met"]) for r in win.values())
        relative.append(item)
    analysis = analyze(
        matched,
        ("capacity_kw",),
        {"cost": "cost", "unfairness": "unfairness"},
        ("PPO",),
        10000,
        20261101,
    )
    comparisons = []
    for capacity in ("600", "800"):
        for metric in ("cost", "unfairness"):
            delta = next(
                r
                for r in analysis
                if r["capacity_kw"] == capacity
                and r["metric"] == metric
                and r["comparison"] == "PPO"
            )
            il_mean = next(
                r["estimate"]
                for r in analysis
                if r["capacity_kw"] == capacity
                and r["metric"] == metric
                and r["method"] == "IL"
                and not r["comparison"]
            )
            ppo_mean = next(
                r["estimate"]
                for r in analysis
                if r["capacity_kw"] == capacity
                and r["metric"] == metric
                and r["method"] == "PPO"
                and not r["comparison"]
            )
            comparisons.append(
                {
                    "capacity_kw": int(capacity),
                    "metric": metric,
                    "il_mean": il_mean,
                    "ppo_mean": ppo_mean,
                    "ppo_minus_il_mean": -delta["estimate"],
                    "paired_bootstrap_95_ci": [-delta["ci_high"], -delta["ci_low"]],
                    "il_reduction_relative_to_ppo_percent": 100
                    * (ppo_mean - il_mean)
                    / ppo_mean,
                }
            )
    sources = [
        Path(__file__),
        ROOT / "scripts/analyze_evaluation_uncertainty.py",
        ppo_path,
        il_path,
    ]
    result = {
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261101,
        "unit": "paired day, separately per capacity",
        "ppo_minus_il": comparisons,
        "winner_vs_original": relative,
        "sources_sha256": {str(p): digest(p) for p in sources},
    }
    (out / "paired_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
