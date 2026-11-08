"""Verify frozen inputs and selected energy-delivery trajectories."""

from pathlib import Path
from typing import Any
from dataclasses import replace
import csv, hashlib, json, sys
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from evcharging.core import load_scenario, urgency_allocation
from evcharging.simulation_schema import simulate as public_simulate
from run_r26_diagnostics import model, predict


def sha(p: Path) -> str:
    """Hash a file.

    :param p: File to hash.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    """Verify provenance, draws and representative trajectories.

    :return: None.
    """
    art = R / "artifacts/runs/pressure21_v1/r44_efficiency"
    out = R / "results/pressure21_v1/r44"
    old = R / "artifacts/runs/pressure21_v1/r26_diagnostics"
    previous = {key.replace("\\", "/"): value for key, value in json.loads((old / "manifest.json").read_text(encoding="utf8"))["sha256"].items()}
    checked = 0
    checks = []
    for seed in range(65000, 65050):
        source = R / f"artifacts/runs/ideal10h_v1/formal/inputs/{seed}.npz"
        files = (
            [source]
            + [
                R
                / f"artifacts/runs/pressure21_v1/evaluation/formal/11002_200_{seed}_{cap}.npz"
                for cap in (600, 800)
            ]
            + [old / f"d_nominal_{seed}_{cap}_upa.npz" for cap in (600, 800)]
        )
        for p in files:
            key = p.relative_to(R).as_posix()
            if sha(p) != previous[key]:
                raise RuntimeError(f"R26 archived hash mismatch: {p}")
            checked += 1
        original = load_scenario(str(source))
        s = load_scenario(str(art / f"input_{seed}.npz"))
        for key in (
            "ta",
            "td",
            "arrival",
            "target",
            "capacity",
            "power",
            "price",
            "dt",
            "beta",
        ):
            np.testing.assert_array_equal(getattr(s, key), getattr(original, key))
        np.testing.assert_array_equal(
            s.efficiency,
            np.random.default_rng(np.random.SeedSequence([seed, 4401])).uniform(
                0.92, 0.98, 200
            ),
        )
        demand = (s.target - s.arrival) * s.capacity
        possible = (s.td - s.ta) * s.efficiency * s.power * s.dt
        checks.append(
            dict(
                seed=seed,
                individually_unreachable=int(np.sum(demand > possible + 1e-10)),
                max_excess_kwh=float(np.maximum(demand - possible, 0).max()),
            )
        )
    with (out / "individual_feasibility.csv").open(
        "w", newline="", encoding="utf8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=checks[0])
        w.writeheader()
        w.writerows(checks)
    s = load_scenario(str(art / "input_65000.npz"))
    net = model()
    for limit in (600, 800):
        for method in ("Full", "UPA"):
            cache = {"time": -1, "a": None}

            def policy(x: np.ndarray, k: int, u: np.ndarray) -> float:
                """Evaluate the selected frozen method.

                :param x: Current state.
                :param k: Vehicle index.
                :param u: Fleet urgency.
                :return: Proposed action.
                """
                if method == "Full":
                    return predict(net, x)
                t = int(round((1 - x[1]) * 56 / 2))
                if t != cache["time"]:
                    cache["time"] = t
                    cache["a"] = urgency_allocation(u, s.power, s.td - t, limit)
                return float(cache["a"][k])

            public = public_simulate(s, limit, policy, state_schema="pressure21_v1")
            with np.load(art / f"65000_{limit}_{method}.npz") as saved:
                for key in ("soc", "raw", "clip", "scale", "final", "cost", "unfair"):
                    np.testing.assert_allclose(
                        public[key], saved[key], rtol=0, atol=1e-12
                    )
    # Compare both historical scalar and equivalent vector inputs against the
    # original nominal Full and UPA archives with the public simulator.
    nominal = load_scenario(
        str(R / "artifacts/runs/ideal10h_v1/formal/inputs/65000.npz")
    )
    for scenario in (nominal, replace(nominal, efficiency=np.full(200, 0.95))):
        s = scenario
        for limit in (600, 800):
            for method in ("Full", "UPA"):
                cache = {"time": -1, "a": None}
                public = public_simulate(s, limit, policy, state_schema="pressure21_v1")
                path = (
                    R
                    / f"artifacts/runs/pressure21_v1/evaluation/formal/11002_200_65000_{limit}.npz"
                    if method == "Full"
                    else old / f"d_nominal_65000_{limit}_upa.npz"
                )
                with np.load(path) as saved:
                    for key in (
                        "soc",
                        "raw",
                        "clip",
                        "scale",
                        "final",
                        "cost",
                        "unfair",
                    ):
                        np.testing.assert_allclose(
                            public[key], saved[key], rtol=0, atol=1e-12
                        )
    report = dict(
        r26_source_hashes_verified=checked,
        inputs_and_random_draws_verified=50,
        public_simulator_heterogeneous_trajectories_verified=4,
        public_simulator_nominal_scalar_and_vector_trajectories_verified=8,
        individually_unreachable_vehicles=sum(
            x["individually_unreachable"] for x in checks
        ),
        days_with_individually_unreachable=sum(
            x["individually_unreachable"] > 0 for x in checks
        ),
        source_sha256=sha(Path(__file__)),
        individual_file_sha256=sha(out / "individual_feasibility.csv"),
        command="uv run python scripts/verify_r44_efficiency.py",
    )
    (out / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
