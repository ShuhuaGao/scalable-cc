"""Evaluate fixed, known per-EV efficiency against frozen nominal trajectories."""

from __future__ import annotations
import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
import argparse, concurrent.futures, csv, hashlib, json
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
from run_r26_diagnostics import INPUT, FORM, MODEL, R, active, model, simulate, totals
from evcharging.core import load_scenario, fixed_feasibility

A = R / "artifacts/runs/pressure21_v1/r44_efficiency"
O = R / "results/pressure21_v1/r44"
OLD = R / "artifacts/runs/pressure21_v1/r26_diagnostics"
METRICS = (
    "cost",
    "D",
    "met_ev_percent",
    "all_met",
    "gap_kwh",
    "gap_fraction_battery_demand",
    "grid_demand_kwh",
    "purchased_kwh",
)


def sha(path: Path) -> str:
    """Hash a file.

    :param path: File to hash.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(s: Any, limit: float, z: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Audit a trajectory and calculate daily metrics.

    :param s: Charging scenario.
    :param limit: Station limit in kW.
    :param z: Saved trajectory arrays.
    :return: Daily metrics and maximum physical residual.
    """
    a, soc = z["final"], z["soc"]
    mask = active(s)
    cost, unfair = totals(s, soc, a)
    residual = max(
        float(
            np.max(
                np.abs(
                    soc[1:] - soc[:-1] - a * s.power * s.efficiency * s.dt / s.capacity
                )
            )
        ),
        float(np.max(np.maximum(a @ s.power - limit, 0))),
        float(np.max(np.abs(a[~mask]), initial=0)),
        float(np.max(np.maximum(-a, 0))),
        float(np.max(np.maximum(a - 1, 0))),
        float(np.max(np.maximum(soc - s.target, 0))),
        float(np.max(np.abs(soc[0] - s.arrival))),
        float(np.max(np.abs(cost - z["cost"]))),
        float(np.max(np.abs(unfair - z["unfair"]))),
    )
    if residual > 1e-8:
        raise RuntimeError(f"Physical residual {residual}")
    gap = np.maximum(s.target - soc[s.td, np.arange(len(s.ta))], 0)
    demand = (s.target - s.arrival) * s.capacity
    battery_gap = float(gap @ s.capacity)
    return dict(
        cost=float(cost.sum()),
        D=float(unfair.sum()),
        met_ev_percent=float(100 * np.mean(gap <= 1e-4)),
        all_met=int(np.all(gap <= 1e-4)),
        gap_kwh=battery_gap,
        gap_fraction_battery_demand=float(battery_gap / demand.sum()),
        grid_demand_kwh=float(np.sum(demand / s.efficiency)),
        purchased_kwh=float((a * s.power * s.dt).sum()),
        physical_residual=residual,
    )


def one(seed: int) -> list[dict[str, Any]]:
    """Evaluate both methods and capacities for one paired day.

    :param seed: Original formal scenario seed.
    :return: Four daily records.
    """
    base = load_scenario(str(INPUT / f"{seed}.npz"))
    if len(base.ta) != 200 or base.efficiency != 0.95:
        raise RuntimeError("Unexpected source scenario")
    eta = np.random.default_rng(np.random.SeedSequence([seed, 4401])).uniform(
        0.92, 0.98, len(base.ta)
    )
    s = replace(base, efficiency=eta)
    ip = A / f"input_{seed}.npz"
    np.savez_compressed(ip, **vars(s))
    m = model()
    old_rows = {
        (int(row["limit"]), row["method"]): row
        for row in csv.DictReader(
            (R / "results/pressure21_v1/r26/d_daily.csv").open(encoding="utf8")
        )
        if row["distribution"] == "nominal" and int(row["seed"]) == seed
    }
    rows = []
    for limit in (600, 800):
        status, res = fixed_feasibility(s, limit)
        for method in ("Full", "UPA"):
            path = (
                FORM / f"11002_200_{seed}_{limit}.npz"
                if method == "Full"
                else OLD / f"d_nominal_{seed}_{limit}_upa.npz"
            )
            with np.load(path) as z:
                nominal = {k: z[k] for k in z.files}
            nom = metric(base, limit, nominal)
            record = old_rows[limit, method]
            for key, oldkey in [
                ("cost", "cost"),
                ("D", "D"),
                ("met_ev_percent", "met_ev_percent"),
                ("all_met", "all_met"),
                ("gap_kwh", "gap_kwh"),
                ("gap_fraction_battery_demand", "gap_fraction_battery_demand"),
                ("grid_demand_kwh", "demand_kwh"),
                ("purchased_kwh", "purchased_kwh"),
            ]:
                if not np.isclose(
                    nom[key], float(record[oldkey]), atol=1e-8, rtol=1e-12
                ):
                    raise RuntimeError(
                        f"Nominal mismatch {seed} {limit} {method} {key}"
                    )
            new = simulate(
                s,
                limit,
                m if method == "Full" else None,
                "full" if method == "Full" else "upa",
            )
            dest = A / f"{seed}_{limit}_{method}.npz"
            np.savez_compressed(dest, **new)
            rows.append(
                dict(
                    seed=seed,
                    limit=limit,
                    method=method,
                    feasibility=status,
                    feasibility_residual=res,
                    eta_mean=float(eta.mean()),
                    eta_min=float(eta.min()),
                    eta_max=float(eta.max()),
                    source_input_sha256=sha(INPUT / f"{seed}.npz"),
                    input_sha256=sha(ip),
                    nominal_trajectory_sha256=sha(path),
                    trajectory_sha256=sha(dest),
                    **{"nominal_" + k: v for k, v in nom.items()},
                    **{
                        "heterogeneous_" + k: v
                        for k, v in metric(s, limit, new).items()
                    },
                )
            )
    return rows


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a table.

    :param path: Output CSV path.
    :param rows: Records with common fields.
    :return: None.
    """
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize paired daily outcomes for each predefined cohort.

    :param rows: Daily records.
    :return: Mean, sample SD and paired changes.
    """
    out = []
    for limit in (600, 800):
        for subset in ("all", "feasible"):
            rr = [
                r
                for r in rows
                if r["limit"] == limit
                and (subset == "all" or r["feasibility"] == "feasible")
            ]
            for method in ("Full", "UPA", "Full_minus_UPA"):
                for key in METRICS:
                    if method == "Full_minus_UPA":
                        pairs = [
                            (
                                next(
                                    r
                                    for r in rr
                                    if r["seed"] == seed and r["method"] == "Full"
                                ),
                                next(
                                    r
                                    for r in rr
                                    if r["seed"] == seed and r["method"] == "UPA"
                                ),
                            )
                            for seed in sorted({r["seed"] for r in rr})
                        ]
                        nominal = np.array(
                            [
                                x["nominal_" + key] - y["nominal_" + key]
                                for x, y in pairs
                            ]
                        )
                        hetero = np.array(
                            [
                                x["heterogeneous_" + key] - y["heterogeneous_" + key]
                                for x, y in pairs
                            ]
                        )
                    else:
                        xx = [r for r in rr if r["method"] == method]
                        nominal = np.array([r["nominal_" + key] for r in xx])
                        hetero = np.array([r["heterogeneous_" + key] for r in xx])
                    delta = hetero - nominal
                    out.append(
                        dict(
                            limit=limit,
                            subset=subset,
                            method=method,
                            metric=key,
                            n=len(nominal),
                            nominal_mean=float(nominal.mean())
                            if len(nominal)
                            else None,
                            nominal_sd=float(nominal.std(ddof=1))
                            if len(nominal) > 1
                            else None,
                            heterogeneous_mean=float(hetero.mean())
                            if len(hetero)
                            else None,
                            heterogeneous_sd=float(hetero.std(ddof=1))
                            if len(hetero) > 1
                            else None,
                            paired_delta_mean=float(delta.mean())
                            if len(delta)
                            else None,
                            paired_delta_sd=float(delta.std(ddof=1))
                            if len(delta) > 1
                            else None,
                            relative_change_percent=float(
                                100 * delta.mean() / nominal.mean()
                            )
                            if len(nominal) and abs(nominal.mean()) > 1e-12
                            else None,
                        )
                    )
    return out


def main() -> None:
    """Run the preregistered evaluation into fresh directories.

    :return: None.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    A.mkdir(parents=True, exist_ok=False)
    O.mkdir(parents=True, exist_ok=False)
    selection = R / "artifacts/runs/pressure21_v1/il_selection.json"
    provenance = json.loads((OLD / "protocol_manifest.json").read_text(encoding="utf8"))
    source_hashes = {key.replace("\\", "/"): value for key, value in provenance["sources"].items()}
    if source_hashes["artifacts/runs/pressure21_v1/il_11002/il_model/best.pt"] != sha(MODEL):
        raise RuntimeError("Nominal model mismatch")
    sources = [
        Path(__file__),
        R / "docs/r44_efficiency_protocol.md",
        R / "src/evcharging/core.py",
        R / "src/evcharging/state_schema.py",
        R / "src/evcharging/simulation_schema.py",
        R / "scripts/run_r26_diagnostics.py",
        selection,
        MODEL,
        OLD / "protocol_manifest.json",
        R / "results/pressure21_v1/r26/d_daily.csv",
    ]
    manifest = dict(
        command=f"uv run python scripts/run_r44_efficiency.py --workers {args.workers}",
        sources_sha256={x.relative_to(R).as_posix(): sha(x) for x in sources},
        source_input_sha256={
            (INPUT / f"{i}.npz").relative_to(R).as_posix(): sha(INPUT / f"{i}.npz")
            for i in range(65000, 65050)
        },
    )
    (A / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, result in enumerate(pool.map(one, range(65000, 65050)), 1):
            rows.extend(result)
            print(f"{i}/50 days", flush=True)
    write(O / "daily.csv", rows)
    write(O / "summary.csv", summarize(rows))
    all_eta = np.concatenate(
        [np.load(A / f"input_{i}.npz")["efficiency"] for i in range(65000, 65050)]
    )
    counts = {
        str(cap): {
            status: sum(
                r["limit"] == cap
                and r["method"] == "Full"
                and r["feasibility"] == status
                for r in rows
            )
            for status in ("feasible", "infeasible", "unknown")
        }
        for cap in (600, 800)
    }
    final = dict(
        rows=len(rows),
        lp_counts=counts,
        efficiency_mean=float(all_eta.mean()),
        efficiency_min=float(all_eta.min()),
        efficiency_max=float(all_eta.max()),
        max_physical_residual=max(
            max(r["nominal_physical_residual"], r["heterogeneous_physical_residual"])
            for r in rows
        ),
        daily_sha256=sha(O / "daily.csv"),
        summary_sha256=sha(O / "summary.csv"),
    )
    (O / "complete.json").write_text(json.dumps(final, indent=2), encoding="utf8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
