"""Replay the six frozen policies on one common 200-EV test day."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_r24_baselines import direct_upa
from evaluate_pressure21 import ROOT
from evcharging.core import Scenario, load_scenario
from evcharging.li2023 import actor, simulate as simulate_rddpg
from evcharging.models import imitation_network, network
from evcharging.simulation_schema import simulate
from run_li2023_r105 import numpy_policy

FORMAL = ROOT / "artifacts/runs/pressure21_v1/evaluation"
PPO = ROOT / "artifacts/runs/ppo_half_formal_20261107"
SEEDS = {"IL": "11002", "DDQN": "21002", "PPO": "24003"}


def records(seed: int, limit: int) -> dict[str, dict]:
    """Read the six archived records for the selected operating conditions.

    :param seed: Formal test day seed.
    :param limit: Station capacity in kW.
    :return: Method-keyed reference records.
    """
    sources = [(FORMAL / "formal/scenarios.csv", {"IL"}),
               (FORMAL / "rl/formal/scenarios.csv", {"DDQN"}),
               (PPO / "evaluation/formal/scenarios.csv", {"PPO"}),
               (ROOT / "results/pressure21_v1/formal_references.csv", {"UC", "UPA", "RDDPG"})]
    matched = {}
    for path, methods in sources:
        with path.open(encoding="utf8", newline="") as stream:
            for row in csv.DictReader(stream):
                method = row["method"]
                if method not in methods or (int(row["seed"]), int(row["capacity_kw"])) != (seed, limit):
                    continue
                if method in SEEDS and row["training_seed"] != SEEDS[method]:
                    continue
                if method in matched:
                    raise ValueError(f"Duplicate reference for {method}.")
                matched[method] = row
    if len(matched) != 6:
        raise ValueError("Expected six main-model records.")
    return matched


def replay(method: str, day: Scenario, limit: int, paths: dict, expected_hash: str) -> tuple[np.ndarray, np.ndarray]:
    """Run one controller and return its SoC and grid-power trajectories.

    :param method: Method name in the evaluation archive.
    :param day: Common fleet and tariff.
    :param limit: Shared station capacity in kW.
    :param paths: Repository-relative checkpoint paths.
    :param expected_hash: Archived checkpoint digest for a learned policy.
    :return: SoC and grid power for each vehicle.
    """
    if method == "UPA":
        result = direct_upa(day, limit)
    elif method == "UC":
        result = simulate(day, limit, lambda x, k, u: 1.0, state_schema="pressure21_v1")
    else:
        path = ROOT / paths[method]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"{method} checkpoint digest mismatch.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=(method != "DDQN"))
        if method == "RDDPG":
            model = actor()
            model.load_state_dict(checkpoint["model"])
            result = simulate_rddpg(day, limit, numpy_policy(model.eval()))
            return result["soc"], result["power"]

        model = imitation_network() if method == "IL" else network(6 if method == "DDQN" else 2)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        def policy(state: np.ndarray, vehicle: int, urgency: np.ndarray) -> float:
            """Decode the deterministic action for this network.

            :param state: Current Pressure21 features.
            :param vehicle: Current vehicle index.
            :param urgency: Current fleet urgency.
            :return: Proposed normalized charging action.
            """
            output = model(torch.as_tensor(state, dtype=torch.float32)[None, :])
            if method == "IL":
                return output.item()
            if method == "DDQN":
                return output.argmax().item() / 5
            shapes = torch.nn.functional.softplus(output) + 1
            return (shapes[0, 0] / shapes.sum()).item()

        result = simulate(day, limit, policy, state_schema="pressure21_v1")
    return result["soc"], result["final"] * day.power


def main() -> None:
    """Check one common-day replay against cost, unfairness and service records.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(65000, 65050), default=65000, metavar="65000..65049")
    parser.add_argument("--capacity", type=int, choices=(600, 800), default=600)
    args = parser.parse_args()
    reference = records(args.seed, args.capacity)
    path = ROOT / "artifacts/runs/ideal10h_v1/formal/inputs" / f"{args.seed}.npz"
    if {r["input_sha256"] for r in reference.values()} != {hashlib.sha256(path.read_bytes()).hexdigest()}:
        raise ValueError("Scenario digest differs between methods.")
    day = load_scenario(str(path))
    paths = json.loads((ROOT / "configs/pressure21_v1.json").read_text())["model_paths"]
    paths["RDDPG"] = "artifacts/runs/li2023_r105/33001/episode_6000.pt"
    rddpg_hash = json.loads((ROOT / "results/li2023_r105/manifest.json").read_text())["models_sha256"]["33001"]
    slots = np.arange(len(day.price))[:, None]
    connected = (day.ta <= slots) & (slots < day.td)
    torch.set_num_threads(1)
    print(f"Day {args.seed} | {len(day.ta)} EVs | {args.capacity} kW")
    print("Method        Cost    Unfairness    Targets met")
    with torch.inference_mode():
        for method, row in reference.items():
            expected = rddpg_hash if method == "RDDPG" else row.get("model_sha256", "")
            soc, power = replay(method, day, args.capacity, paths, expected)
            actions = power / day.power
            need = np.maximum(day.target - soc[:-1], 0) * day.capacity
            urgency = np.where(connected, need / (day.efficiency * day.power * day.dt * np.maximum(day.td-slots, 1)), 0)
            total = urgency.sum(axis=1, keepdims=True)
            share = np.divide(urgency, total, out=np.zeros_like(urgency), where=total > 0)
            cost = float((power * day.price[:, None] * day.dt).sum())
            unfair = float(np.square(actions - actions.sum(axis=1, keepdims=True) * share).sum())
            misses = int(np.count_nonzero(day.target - soc[day.td, np.arange(len(day.ta))] > 1e-4))
            np.testing.assert_allclose([cost, unfair], [float(row["cost"]), float(row["unfairness"])],
                                       rtol=1e-6, atol=1e-7, err_msg=method)
            if misses != int(row["failed_vehicles"]):
                raise ValueError(f"{method} departure service mismatch.")
            print(f"{'Ours' if method == 'IL' else method:6} {cost:11.4f} {unfair:13.4f} {len(day.ta)-misses:9d}/{len(day.ta)}")


if __name__ == "__main__":
    main()
