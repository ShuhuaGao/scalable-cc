"""Run the selected charging policy on the ten-vehicle example day."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evcharging.core import generate
from evcharging.models import imitation_network
from evcharging.simulation_schema import simulate


def main() -> None:
    """Evaluate one day and compare its metrics with the published trajectory.

    :return: None.
    """
    config = json.loads((ROOT / "configs/pressure21_v1.json").read_text())
    checkpoint = ROOT / config["model_paths"]["IL"]
    scenarios = ROOT / "artifacts/runs/pressure21_v1/evaluation/small/scenarios.csv"
    with scenarios.open(encoding="utf-8", newline="") as handle:
        record = next(row for row in csv.DictReader(handle) if row["seed"] == "76000")

    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["model_sha256"]:
        raise ValueError("Checkpoint does not match the recorded example day.")
    torch.set_num_threads(1)
    model = imitation_network()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights["model"])
    model.eval()

    def policy(state: np.ndarray, vehicle: int, urgency: np.ndarray) -> float:
        """Evaluate the policy for the current vehicle.

        :param state: Pressure21 features for the current decision.
        :param vehicle: Vehicle index, supplied by the simulator.
        :param urgency: Fleet urgency, already represented in the state.
        :return: Proposed normalized charging action.
        """
        return model(torch.as_tensor(state, dtype=torch.float32)[None, :]).item()

    day = generate(76000, 10, fixed_day=True)
    with torch.inference_mode():
        result = simulate(day, 40, policy, state_schema=config["state_schema"])
    cost = float(result["cost"].sum())
    unfairness = float(result["unfair"].sum())
    failed = int(np.count_nonzero(day.target - result["soc"][day.td, np.arange(10)] > 1e-4))
    np.testing.assert_allclose([cost, unfairness],
                               [float(record["cost"]), float(record["unfairness"])],
                               rtol=1e-6, atol=1e-7)
    if failed != int(record["failed_vehicles"]):
        raise ValueError("Departure service differs from the archived example.")
    print(f"Day 76000 | 10 EVs | 40 kW | {len(day.price)} slots")
    print(f"Cost {cost:.6f} | Unfairness {unfairness:.6f} | Targets met {10 - failed}/10")


if __name__ == "__main__":
    main()
