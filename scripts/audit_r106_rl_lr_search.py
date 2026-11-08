"""Audit the learning-rate search against the original validation and formal records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts/runs/pressure21_v1"


def digest(path: Path) -> str:
    """Hash a source file.

    :param path: Input file.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    """Read a CSV record set.

    :param path: CSV file.
    :return: Scenario rows.
    """
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    """Check baseline validation and paired formal identities.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    selection_path = out / "selection.json"
    selected = json.loads(selection_path.read_text())
    old_selection_path = ARCHIVE / "rl_selection.json"
    old_selection = json.loads(old_selection_path.read_text())
    old_ddqn_path = ARCHIVE / "ddqn_21002/ddqn_model/validation.json"
    old_ddqn = json.loads(old_ddqn_path.read_text())
    candidates = selected["candidates"]
    if len(candidates) != 60:
        raise RuntimeError("Expected 60 candidates")
    maxima = {}
    for method in ("ddqn", "ppo"):
        rows = [
            r for r in candidates if r["method"] == method and r["label"] == "original"
        ]
        if len(rows) != 10:
            raise RuntimeError("Expected ten original checkpoints")
        fields = (
            ("failed", "gap", "objective")
            if method == "ddqn"
            else ("failed", "gap_kwh", "objective")
        )
        differences = []
        for row in rows:
            if method == "ddqn":
                previous = next(r for r in old_ddqn if r["episode"] == row["episode"])
            else:
                previous = next(
                    r
                    for r in old_selection["ppo_candidates"]
                    if r["seed"] == 24003 and r["episode"] == row["episode"]
                )
                if previous["sha256"] != row["sha256"]:
                    raise RuntimeError("Original PPO checkpoint differs from archive")
            differences += [abs(row["metrics"][key] - previous[key]) for key in fields]
        maxima[method] = max(differences)
        if maxima[method] != 0:
            raise RuntimeError(f"Original validation differs: {method}")
    orig_ddqn = next(r for r in old_selection["ddqn"] if r["seed"] == 21002)
    if selected["baseline"]["ddqn"]["sha256"] != orig_ddqn["sha256"]:
        raise RuntimeError("Original selected DDQN differs from archive")
    current_path = out / "formal/scenarios.csv"
    old_formal_path = ARCHIVE / "evaluation/rl/formal/scenarios.csv"
    rows = read_csv(current_path)
    previous_rows = read_csv(old_formal_path)
    previous = {
        (r["seed"], r["capacity_kw"]): r
        for r in previous_rows
        if r["method"] == "PPO" and r["training_seed"] == "24003"
    }
    if len(rows) != 200 or len(previous) != 100:
        raise RuntimeError("Unexpected formal cohort size")
    fields = (
        "cost",
        "unfairness",
        "weighted_objective",
        "gap_kwh",
        "purchased_kwh",
        "failed_vehicles",
    )
    differences = {key: [] for key in fields}
    for row in rows:
        prior = previous[(row["seed"], row["capacity_kw"])]
        if row["input_sha256"] != prior["input_sha256"]:
            raise RuntimeError("Formal input identity mismatch")
        if row["label"] == "original":
            if row["model_sha256"] != prior["model_sha256"]:
                raise RuntimeError("Formal baseline model identity mismatch")
            for key in fields:
                differences[key].append(abs(float(row[key]) - float(prior[key])))
    if any(len(values) != 100 or max(values) != 0 for values in differences.values()):
        raise RuntimeError("Formal baseline metrics differ")
    source_paths = (
        selection_path,
        old_selection_path,
        old_ddqn_path,
        current_path,
        old_formal_path,
        Path(__file__),
    )
    result = {
        "baseline_checkpoint_counts": {"ddqn": 10, "ppo": 10},
        "validation_max_absolute_difference": maxima,
        "formal_input_hashes_matched": len(rows),
        "formal_baseline_records_matched": 100,
        "formal_max_absolute_difference": {
            key: max(values) for key, values in differences.items()
        },
        "sources_sha256": {str(path): digest(path) for path in source_paths},
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print("Validation and formal baseline audit passed")


if __name__ == "__main__":
    main()
