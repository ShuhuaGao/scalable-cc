"""Load the current DDQN and half-rate PPO evaluation archives."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluate_pressure21 import ROOT, OUT, digest

PPO_RUN = ROOT / "artifacts/runs/ppo_half_formal_20261107"



def local_path(recorded: str) -> Path:
    """Resolve an archived path after moving the repository.

    :param recorded: Absolute or repository-relative recorded path.
    :return: Existing local asset path.
    """
    path = Path(recorded)
    parts = recorded.replace("\\", "/").split("/")
    for marker in ("artifacts", "results", "DATASET"):
        if marker in parts:
            candidate = ROOT.joinpath(*parts[parts.index(marker):])
            if candidate.exists():
                return candidate
    if path.exists():
        return path
    raise FileNotFoundError(recorded)


def selection() -> dict:
    """Load the PPO selection frozen before formal evaluation.

    :return: Selection record.
    """
    result = json.loads((PPO_RUN / "selection.json").read_text(encoding="utf8"))
    if result.get("formal_test_used_for_selection") is not False or result.get("state_schema") != "pressure21_v1":
        raise RuntimeError("Invalid PPO selection provenance")
    for choice in result["ppo_per_run"]:
        if choice.get("state_schema") != "pressure21_v1" or digest(local_path(choice["path"])) != choice["sha256"]:
            raise RuntimeError("Selected PPO checkpoint identity mismatch")
    return result


def current_rl(dataset: str) -> list[dict]:
    """Combine original DDQN and current PPO without altering historical data.

    :param dataset: Formal or small evaluation set.
    :return: Audited DDQN and PPO scenario rows.
    """
    if dataset not in ("formal", "small"):
        raise ValueError(dataset)
    chosen = selection()
    folders = (OUT / "rl" / dataset, PPO_RUN / "evaluation" / dataset)
    rows = []
    for folder, method in zip(folders, ("DDQN", "PPO")):
        path = folder / "scenarios.csv"
        complete = json.loads((folder / "complete.json").read_text(encoding="utf8"))
        if digest(path) != complete["sha256"]:
            raise RuntimeError(f"Incomplete evaluation: {path}")
        with path.open(newline="", encoding="utf8") as stream:
            data = list(csv.DictReader(stream))
        if len(data) != complete["records"]:
            raise RuntimeError(f"Wrong evaluation count: {path}")
        rows.extend(row for row in data if row["method"] == method)
    main = chosen["ppo_main"]
    expected = {int(x["seed"]): x["sha256"] for x in chosen["ppo_per_run"]}
    if expected.get(int(main["seed"])) != main["sha256"]:
        raise RuntimeError("Main PPO differs from its per-run selection")
    for row in rows:
        if row["method"] == "PPO" and row["model_sha256"] != expected[int(row["training_seed"])]:
            raise RuntimeError("PPO evaluation model identity mismatch")
    return rows
