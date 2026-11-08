"""Evaluate an existing result file without solving: feasibility, recomputed objective, comparison with stored values.

Usage: ``uv run python -m PYSOLVER.evaluate EVDATA.npz RESULT.npz [--atol 1e-3]``
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ev import EVData
from .nlp2a import (
    _max_by_type,
    load_config,
    model_quantities,
    objective_parts,
    primal_feasibility_report,
)


def evaluate(evdata_npz: str | Path, result_npz: str | Path, atol: float = 1e-3) -> dict[str, Any]:
    """Check the feasibility of a stored solution and recompute its objective value.

    :param evdata_npz: Scenario file (EV parameters, prices, ``delta_t``, ``beta``, ``Plim``).
    :param result_npz: Result file with the keys ``a, P, g, G, SoC, c_obj, f_obj, obj_value``.
    :param atol: Tolerance of the primal feasibility check.
    :return: A dict with the problem size, the number and the maximum per type of violated constraints,
        the stored objective, the objective recomputed by ``compute_objective`` (explicit ``if sum_k g > 0`` branch)
        and by the model formula (``G`` with ``+1e-8``), the cost and fairness terms, and the maximum absolute
        difference between each recomputed quantity and the stored one.
    """
    evs, cfg = EVData.from_npz(evdata_npz), load_config(evdata_npz)
    with np.load(result_npz) as f:
        res = {k: f[k] for k in f.files}
    dt, beta, rho, Plim = cfg["dt"], cfg["beta"], cfg["rho"], cfg["Plim"]
    a = res["a"]
    fr = primal_feasibility_report(a, evs, Plim, dt, atol)
    c_obj, f_obj = objective_parts(a, evs, rho, dt)
    q = model_quantities(a, evs, rho, dt)
    return {
        "K": a.shape[0],
        "T": a.shape[1],
        "n_violated": len(fr),
        "violations_max": _max_by_type(fr),
        "stored_obj": float(res["obj_value"]),
        "compute_objective": (1 - beta) * c_obj + beta * f_obj,  # explicit branch for sum_k g == 0
        "model_formula_objective": float(np.sum((1 - beta) * q["c_obj"] + beta * q["f_obj"])),  # G with +1e-8
        "cost": c_obj,
        "fairness": f_obj,
        "max_abs_diff_vs_stored": {k: float(np.max(np.abs(q[k] - res[k]))) for k in q if k in res},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a result file from the command line and print a summary.

    :param argv: Command-line arguments; ``sys.argv[1:]`` if None.
    :return: Process exit code, 0 if the stored solution is feasible.
    """
    p = argparse.ArgumentParser(description="Evaluate an existing result npz without solving.")
    p.add_argument("evdata_npz")
    p.add_argument("result_npz")
    p.add_argument("--atol", type=float, default=1e-3)
    args = p.parse_args(argv)
    r = evaluate(args.evdata_npz, args.result_npz, args.atol)
    print(f"K={r['K']} T={r['T']}  feasible: {r['n_violated'] == 0} ({r['n_violated']} violated, {r['violations_max']})")
    print(f"stored obj_value        = {r['stored_obj']:.10g}")
    print(f"compute_objective       = {r['compute_objective']:.10g}  (diff {r['compute_objective'] - r['stored_obj']:+.3e})")
    print(f"model-formula objective = {r['model_formula_objective']:.10g}  (diff {r['model_formula_objective'] - r['stored_obj']:+.3e})")
    print(f"cost = {r['cost']:.6g}  fairness = {r['fairness']:.6g}")
    print("max |recomputed - stored|:", {k: f"{v:.2e}" for k, v in r["max_abs_diff_vs_stored"].items()})
    return 0 if r["n_violated"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
