"""Centralized solution of the EV charging problem P1 with Knitro multi-start (port of ``JULIA/nlp2a.jl``).

Command line: ``uv run python -m PYSOLVER.nlp2a [options] EVDATA.npz``.

Indexing: the Julia code is 1-based, this port is 0-based.
The time axis is shifted such that ``min(ta)`` becomes step 0.
EV ``k`` is connected during ``[ta0_k, td0_k)`` (the departure step is excluded) and ``SoC`` has ``T + 1`` columns,
``SoC[k, t]`` being the state of charge at the beginning of step ``t``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .ev import EVData
from .knitro_env import checkout_license, knitro_release, prepare_license

if TYPE_CHECKING:
    import knitro as kn

# Violated constraints: maps (constraint type, indices...) to the size of the violation.
type Violations = dict[tuple[str | int, ...], float]

# build_model uses G = g / (sum_k g + G_EPS), whereas compute_objective uses an explicit "if sum_k g > 0" branch.
# Both are kept exactly as in the Julia code.
G_EPS = 1e-8
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Options passed to Knitro through AmplNLWriter in the Julia code, with the names of the Knitro 16 API
# (the old names alg and maxtime_real are deprecated there).
DEFAULT_KNITRO_OPTIONS: dict[str, int | float] = {
    "nlp_algorithm": 0,  # alg=0 (automatic)
    "maxit": 200000,
    "maxtime": 6000,  # maxtime_real=6000; in Knitro 16 this is the total real-time limit including multi-start
    "opttol": 5e-2,
    "opttol_abs": 10,
    "feastol": 1e-3,
    "feastol_abs": 1e-2,
    "outlev": 1,
    "ms_enable": 1,
    "ms_maxsolves": 64,
    "ms_outsub": 0,
    "ms_seed": 0,  # not set in the Julia code; the Knitro default is written out explicitly for the record
}
# Options that are read back and recorded after the solve, including those left at their defaults.
RECORDED_OPTIONS = ("numthreads", "ms_numthreads", "ms_terminate", "ms_startptrange", "presolve", "hessopt")


# --------------------------------------------------------------------------------------
# Objective, derived quantities and feasibility check (pure numpy, independent of the solver)
# --------------------------------------------------------------------------------------
def simulate_soc(a: np.ndarray, evs: EVData, dt: float) -> np.ndarray:
    """Simulate the SoC trajectories for given charging actions.

    ``SoC[:, 0] = SoCa`` and ``SoC[:, t + 1] = SoC[:, t] + P[:, t] * e * dt / C`` with ``P = a * Pmax``.

    :param a: Charging actions in [0, 1], shape (K, T).
    :param evs: EV parameters.
    :param dt: Length of a time step [h].
    :return: SoC at the beginning of every step, shape (K, T + 1).
    """
    dsoc = a * evs.Pmax[:, None] * evs.e[:, None] * dt / evs.C[:, None]
    return np.cumsum(np.hstack([evs.SoCa[:, None], dsoc]), axis=1)


def _urgency(soc: np.ndarray, evs: EVData, dt: float, T: int) -> np.ndarray:
    """Compute the urgency ``g[k, t]`` of every EV.

    ``g = C * (SoCd - SoC) / (Pmax * e * dt) / (td - tau)`` while the EV is connected (``ta <= tau < td``, where
    ``tau = t + min(ta)`` is the unshifted time) and 0 otherwise.

    :param soc: SoC trajectories, shape (K, T + 1).
    :param evs: EV parameters.
    :param dt: Length of a time step [h].
    :param T: Number of time steps.
    :return: Urgency, shape (K, T).
    """
    tau = int(evs.ta.min()) + np.arange(T)
    win = (evs.ta[:, None] <= tau) & (tau < evs.td[:, None])
    remain = np.where(win, evs.td[:, None] - tau, 1)  # use 1 outside the window to avoid a division by zero
    need = (evs.SoCd[:, None] - soc[:, :T]) * evs.C[:, None] / (evs.Pmax[:, None] * evs.e[:, None] * dt)
    return np.where(win, need / remain, 0.0)


def objective_parts(a: ArrayLike, evs: EVData, rho: ArrayLike, dt: float = 1.0) -> tuple[float, float]:
    """Compute the unweighted cost and fairness terms of the objective.

    The normalized urgency G is computed with the explicit ``if sum_k g > 0`` branch, as in the Julia function
    ``compute_objective``.

    :param a: Charging actions in [0, 1], shape (K, T).
    :param evs: EV parameters.
    :param rho: Electricity prices for the T steps.
    :param dt: Length of a time step [h].
    :return: The cost term ``sum(P * dt * rho)`` and the fairness term ``sum((a - sum_k a * G) ** 2)``.
    """
    a = np.asarray(a, dtype=np.float64)
    T = a.shape[1]
    soc = simulate_soc(a, evs, dt)
    c_obj = float(np.sum(a * evs.Pmax[:, None] * dt * np.asarray(rho, dtype=np.float64)[None, :T]))
    g = _urgency(soc, evs, dt, T)
    gt_total = g.sum(axis=0)
    G = np.divide(g, gt_total, out=np.zeros_like(g), where=gt_total > 0)  # explicit "if gt_total > 0" branch
    f_obj = float(np.sum((a - a.sum(axis=0) * G) ** 2))
    return c_obj, f_obj


def compute_objective(a: ArrayLike, evs: EVData, rho: ArrayLike, dt: float = 1.0, *, beta: float) -> float:
    """Compute the objective value ``(1 - beta) * c_obj + beta * f_obj`` of a decision matrix.

    :param a: Charging actions in [0, 1], shape (K, T).
    :param evs: EV parameters.
    :param rho: Electricity prices for the T steps.
    :param dt: Length of a time step [h].
    :param beta: Weight of the fairness term.
    :return: The objective value.
    """
    c_obj, f_obj = objective_parts(a, evs, rho, dt)
    return (1 - beta) * c_obj + beta * f_obj


def model_quantities(a: ArrayLike, evs: EVData, rho: ArrayLike, dt: float) -> dict[str, np.ndarray]:
    """Compute the quantities stored by the Julia code from the actions, using the formulas of ``build_model``.

    In contrast to :func:`compute_objective`, ``G = g / (sum_k g + 1e-8)`` here.

    :param a: Charging actions in [0, 1], shape (K, T).
    :param evs: EV parameters.
    :param rho: Electricity prices for the T steps.
    :param dt: Length of a time step [h].
    :return: Dict with ``a, P, g, G, c_obj, f_obj`` of shape (K, T) and ``SoC`` of shape (K, T + 1).
    """
    a = np.asarray(a, dtype=np.float64)
    T = a.shape[1]
    P = a * evs.Pmax[:, None]
    soc = simulate_soc(a, evs, dt)
    g = _urgency(soc, evs, dt, T)
    G = g / (g.sum(axis=0) + G_EPS)
    return {
        "a": a,
        "P": P,
        "g": g,
        "G": G,
        "SoC": soc,
        "c_obj": np.asarray(rho, dtype=np.float64)[None, :T] * P * dt,
        "f_obj": (a - a.sum(axis=0) * G) ** 2,
    }


def primal_feasibility_report(a: ArrayLike, evs: EVData, Plim: float, dt: float = 1.0, atol: float = 1e-3) -> Violations:
    """Report the violated constraints of a solution (counterpart of JuMP's ``primal_feasibility_report``).

    Checked constraints: ``0 <= a <= 1`` while an EV is connected and ``a == 0`` otherwise;
    ``SoCmin <= SoC <= SoCmax`` at all ``T + 1`` steps; ``SoC[k, td0_k] == SoCd_k``; ``sum_k P[k, t] <= Plim``.

    :param a: Charging actions, shape (K, T).
    :param evs: EV parameters.
    :param Plim: Station power limit [kW].
    :param dt: Length of a time step [h].
    :param atol: Violations up to this size are tolerated.
    :return: Maps ``(type, indices...)`` to the violation for every violation larger than ``atol``;
        the types are ``a_bound``, ``soc_range``, ``soc_final`` and ``power_limit``.
    """
    a = np.asarray(a, dtype=np.float64)
    K, T = a.shape
    ta_min = int(evs.ta.min())
    ta0, td0 = evs.ta - ta_min, evs.td - ta_min
    t_idx = np.arange(T)
    win = (ta0[:, None] <= t_idx) & (t_idx < td0[:, None])
    soc = simulate_soc(a, evs, dt)
    checks = {
        "a_bound": np.maximum(-a, a - np.where(win, 1.0, 0.0)),  # 0 <= a <= 1 inside the window, a == 0 outside
        "soc_range": np.maximum(evs.SoCmin[:, None] - soc, soc - evs.SoCmax[:, None]),
        "soc_final": np.abs(soc[np.arange(K), td0] - evs.SoCd),
        "power_limit": (a * evs.Pmax[:, None]).sum(axis=0) - Plim,
    }
    report: Violations = {}
    for name, viol in checks.items():
        for idx in zip(*np.nonzero(viol > atol)):
            report[(name, *map(int, idx))] = float(viol[idx])
    return report


def max_violation_by_type(violations: Violations) -> dict[str, float]:
    """Summarize a feasibility report by the largest violation of every constraint type.

    :param violations: Output of :func:`primal_feasibility_report`.
    :return: Maps the constraint type to its largest violation.
    """
    out: dict[str, float] = {}
    for key, viol in violations.items():
        out[str(key[0])] = max(out.get(str(key[0]), 0.0), viol)
    return out


# --------------------------------------------------------------------------------------
# Model building and solving (Knitro expression API; Knitro differentiates the expressions exactly)
# --------------------------------------------------------------------------------------
@dataclass
class Model:
    """Optimization model of one scenario and the data needed to map the solver output back to ``a``."""

    prob: kn.Problem  # Knitro problem
    evs: EVData  # EV parameters
    rho: np.ndarray  # electricity prices, shape (T,)
    dt: float  # length of a time step [h]
    beta: float  # weight of the fairness term
    Plim: float  # station power limit [kW]
    var_index: np.ndarray  # (K, T) Knitro variable index of a[k, t]; -1 where a is fixed to 0 (no variable)

    def a_from_x(self, x: ArrayLike) -> np.ndarray:
        """Map the Knitro solution vector to the decision matrix ``a``.

        :param x: Values of all Knitro variables.
        :return: The matrix ``a`` of shape (K, T), zero where an EV is not connected.
        """
        a = np.zeros(self.var_index.shape)
        free = self.var_index >= 0
        a[free] = np.asarray(x, dtype=np.float64)[self.var_index[free]]
        return a

    def close(self) -> None:
        """Free the underlying Knitro problem."""
        self.prob.clear()

    def __enter__(self) -> Model:
        """Enter the context manager.

        :return: The model itself.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Free the Knitro problem when leaving the context.

        :param exc: Exception information, ignored.
        """
        self.close()


def build_model(
    evs: EVData, rho: ArrayLike, dt: float = 1.0, *, beta: float, Plim: float, license: kn.License | None = None, terminal_reference: ArrayLike | None = None, initial_actions: ArrayLike | None = None, window_start: int | None = None
) -> Model:
    """Build the optimization model, equivalent to ``build_model`` in the Julia code.

    The model has the variables ``a[k, t]`` in [0, 1], the SoC dynamics, ``SoCmin <= SoC <= SoCmax``,
    ``SoC[k, td0_k] == SoCd_k``, the station power limit and the objective
    ``sum((1 - beta) * rho * P * dt + beta * (a - sum_k a * G) ** 2)`` with ``G = g / (sum_k g + 1e-8)``.
    The only difference to the Julia code: variables ``a[k, t]`` that are fixed to 0 outside the connection window
    are not created, so the SoC range constraints that are constant or duplicate the departure step are not added.
    The feasible set and the objective are unchanged.

    :param evs: EV parameters.
    :param rho: Electricity prices for steps ``[min(ta), max(td) - 1]``, length T.
    :param dt: Length of a time step [h].
    :param beta: Weight of the fairness term, in [0, 1].
    :param Plim: Station power limit [kW].
    :param license: A checked-out Knitro license token shared between problems; a new one is checked out if None.
    :param terminal_reference: Optional window-end SoC reference for vehicles departing after the window.
    :param initial_actions: Optional initial action matrix of shape (K, T).
    :param window_start: Optional absolute first slot; defaults to the earliest arrival.
    :return: The model.
    :raises ValueError: If the time ranges, ``rho``, ``beta``, ``Plim`` or the arrival SoC are invalid.
    """
    import knitro as kn

    if not np.all(evs.td > evs.ta):
        raise ValueError("Invalid time range")
    rho = np.asarray(rho, dtype=np.float64)
    ta_min = int(evs.ta.min()) if window_start is None else int(window_start)
    if ta_min > int(evs.ta.min()):
        raise ValueError("window_start must not follow an arrival")
    T = len(rho) if terminal_reference is not None else int(evs.td.max()) - ta_min  # td is excluded
    if len(rho) != T:
        raise ValueError(f"len(rho) = {len(rho)} but T = {T}")
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    if not Plim > 0:
        raise ValueError("Plim must be positive")
    if np.any(evs.SoCa < evs.SoCmin) or np.any(evs.SoCa > evs.SoCmax):
        raise ValueError("SoCa outside [SoCmin, SoCmax]: the SoC range constraint is violated by a constant")
    K = len(evs)
    ta0, td0 = (evs.ta - ta_min).tolist(), (evs.td - ta_min).tolist()
    end0 = [min(td, T) for td in td0]
    references = None if terminal_reference is None else np.asarray(terminal_reference, dtype=np.float64)
    if references is not None and references.shape != (K,):
        raise ValueError("terminal_reference must have one value per vehicle")
    Pmax, e, C = evs.Pmax.tolist(), evs.e.tolist(), evs.C.tolist()
    SoCa, SoCd, SoCmin, SoCmax = evs.SoCa.tolist(), evs.SoCd.tolist(), evs.SoCmin.tolist(), evs.SoCmax.tolist()
    initial = None if initial_actions is None else np.asarray(initial_actions, dtype=np.float64)
    if initial is not None and initial.shape != (K, T):
        raise ValueError("initial_actions must have shape (K, T)")
    rho_l = rho.tolist()

    prob = kn.Problem(license) if license is not None else kn.Problem()
    var_index = -np.ones((K, T), dtype=np.int64)
    a = [[None] * T for _ in range(K)]  # a[k][t]: a Variable inside the window, None (constant 0) outside
    for k in range(K):
        for t in range(ta0[k], end0[k]):
            a[k][t] = prob.add_variable(lb=0.0, ub=1.0, x0=None if initial is None else float(initial[k, t]))
            var_index[k, t] = a[k][t].idx

    # soc[k][t] is the SoC at the beginning of step t (ta0_k <= t <= td0_k): SoCa + c * sum_{s<t} a, c = Pmax * e * dt / C
    soc = [[None] * (T + 1) for _ in range(K)]
    for k in range(K):
        c = Pmax[k] * e[k] * dt / C[k]
        cum = None  # sum_{s<t} a[k][s]
        soc[k][ta0[k]] = SoCa[k]
        for t in range(ta0[k] + 1, end0[k] + 1):
            cum = a[k][t - 1] if cum is None else cum + a[k][t - 1]
            soc[k][t] = SoCa[k] + c * cum
            prob.add_constraint(soc[k][t], lb=SoCmin[k], ub=SoCmax[k])  # SoCmin <= SoC <= SoCmax
        if td0[k] <= T:
            prob.add_constraint(soc[k][end0[k]] == SoCd[k])
        else:
            prob.add_constraint(soc[k][end0[k]], lb=float(references[k]), ub=SoCd[k])

    # Station power limit: sum_k P[k, t] <= Plim
    for t in range(T):
        P_t = [Pmax[k] * a[k][t] for k in range(K) if a[k][t] is not None]
        if P_t:
            prob.add_constraint(prob.nsum(P_t) <= Plim)

    # Urgency g = C * (SoCd - SoC) / (Pmax * dt * e) / (td - t), G = g / (sum_k g + 1e-8), f = (a - sum_k a * G) ** 2
    g = [[None] * T for _ in range(K)]
    for k in range(K):
        for t in range(ta0[k], end0[k]):
            g[k][t] = C[k] * (SoCd[k] - soc[k][t]) / (Pmax[k] * dt * e[k]) / (td0[k] - t)
    fair, cost = [], []
    for t in range(T):
        ks = [k for k in range(K) if a[k][t] is not None]
        if not ks:
            continue
        a_total = prob.nsum([a[k][t] for k in ks])
        denom = prob.nsum([g[k][t] for k in ks]) + G_EPS
        for k in ks:
            fair.append((a[k][t] - a_total * (g[k][t] / denom)) ** 2)
            cost.append((rho_l[t] * Pmax[k] * dt) * a[k][t])  # rho_t * P * dt
    prob.add_objective((1 - beta) * prob.nsum(cost) + beta * prob.nsum(fair))
    return Model(prob, evs, rho, dt, beta, Plim, var_index)


def _rc_name(rc: int) -> str:
    """Look up the symbolic name of a Knitro return code.

    :param rc: The return code.
    :return: The name of the ``KN_RC_*`` constant, or the number as a string if there is none.
    """
    import knitro as kn

    names = sorted(n for n in dir(kn) if n.startswith("KN_RC_") and getattr(kn, n) == rc)
    return names[0] if names else str(rc)


def _get_param(prob: kn.Problem, name: str) -> int | float | None:
    """Read a Knitro option, returning None for options that cannot be read back (e.g. string options).

    :param prob: The Knitro problem.
    :param name: Option name.
    :return: The option value or None.
    """
    try:
        return prob.get_param(name)
    except (RuntimeError, TypeError):
        return None


def solve(model: Model, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Solve the model with Knitro and collect the solution and all settings.

    :param model: The model to solve.
    :param options: Knitro options that override :data:`DEFAULT_KNITRO_OPTIONS`.
    :return: Dict with ``status`` and ``status_name`` (Knitro return code), ``obj_value``, ``a`` (None if Knitro
        returned no point), ``abs_feas_error``, ``abs_opt_error``, ``iterations``, ``solve_time_wall``,
        ``options`` (all options set plus ``RECORDED_OPTIONS`` read back) and ``multistart_solves``
        (objective, errors and iterations of every local solve, sorted by objective).
    """
    import knitro as kn

    prob = model.prob
    opts = {**DEFAULT_KNITRO_OPTIONS, **(options or {})}
    for name, value in opts.items():
        prob.set_param(name, value)

    starts: list[dict[str, float]] = []  # result of every local solve of the multi-start

    def on_start_done(kc: Any, x: Any, lam: Any, user: Any) -> int:
        """Record the result of one multi-start local solve (Knitro callback).

        :param kc: Knitro context of the local solve.
        :param x: Primal solution of the local solve.
        :param lam: Dual solution of the local solve.
        :param user: User parameters, unused.
        :return: 0 to continue the multi-start.
        """
        starts.append(
            {
                "obj": kn.KN_get_obj_value(kc),
                "abs_feas_error": kn.KN_get_abs_feas_error(kc),
                "abs_opt_error": kn.KN_get_abs_opt_error(kc),
                "iters": kn.KN_get_number_iters(kc),
            }
        )
        return 0

    if opts.get("ms_enable"):
        kn.KN_set_ms_process_callback(prob.kc, on_start_done)
    t0 = time.perf_counter()
    rc = prob.solve()
    wall = time.perf_counter() - t0
    x = kn.KN_get_var_primal_values(prob.kc) if rc > -500 else None  # codes <= -500 are errors without a point
    return {
        "status": int(rc),
        "status_name": _rc_name(rc),
        "obj_value": kn.KN_get_obj_value(prob.kc) if x is not None else None,
        "a": model.a_from_x(x) if x is not None else None,
        "abs_feas_error": prob.get_attr(kn.KN_ATTR_ABS_FEAS_ERROR),
        "abs_opt_error": prob.get_attr(kn.KN_ATTR_ABS_OPT_ERROR),
        "iterations": prob.get_attr(kn.KN_ATTR_NUM_ITERS),
        "solve_time_wall": wall,
        "options": {n: _get_param(prob, n) for n in [*opts, *RECORDED_OPTIONS]},
        "multistart_solves": sorted(starts, key=lambda s: s["obj"]),
    }


# --------------------------------------------------------------------------------------
# Per-file workflow and command line
# --------------------------------------------------------------------------------------
def load_config(npz_file: str | Path) -> dict[str, Any]:
    """Read the scenario settings (everything except the EV parameters) from a scenario npz file.

    :param npz_file: Path of the scenario file.
    :return: Dict with ``dt`` (``delta_t``), ``beta``, ``rho`` (prices, array) and ``Plim``.
    """
    with np.load(npz_file) as cfg:
        return {
            "dt": float(cfg["delta_t"]),
            "beta": float(cfg["beta"]),
            "rho": np.asarray(cfg["rho"], dtype=np.float64),
            "Plim": float(cfg["Plim"]),
        }


def take_first(evs: EVData, rho: np.ndarray, k: int) -> tuple[EVData, np.ndarray]:
    """Keep only the first ``k`` EVs of a scenario (small-scale tests).

    :param evs: EV parameters of the full scenario.
    :param rho: Prices of the full scenario, starting at ``min(ta)`` of all EVs.
    :param k: Number of EVs to keep.
    :return: The EV subset and the prices cut to ``[min(ta), max(td) - 1]`` of the subset.
    """
    sub = evs.take(slice(0, k))
    start = int(sub.ta.min()) - int(evs.ta.min())
    return sub, rho[start : start + int(sub.td.max()) - int(sub.ta.min())]


def _jsonable(obj: Any) -> Any:
    """Convert numpy objects inside nested containers to JSON-serializable Python objects.

    :param obj: A (nested) object.
    :return: The converted object.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run(
    npz_file: str | Path,
    *,
    output: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    num_evs: int | None = None,
    plim: float | None = None,
    options: Mapping[str, Any] | None = None,
    atol: float = 1e-3,
    license: kn.License | None = None,
) -> dict[str, Any]:
    """Solve one scenario file and write the results.

    The result npz (keys as in the Julia code) is written only if the feasibility check passes;
    a json report with all settings and results is always written.

    :param npz_file: Scenario file.
    :param output: Path of the result npz; defaults to ``<output_dir>/<stem>-res-KNITRO.npz``.
    :param output_dir: Output directory, used if ``output`` is None.
    :param overwrite: Whether existing outputs may be overwritten.
    :param num_evs: If given, use only the first ``num_evs`` EVs and set ``Plim = 4 * num_evs`` kW.
    :param plim: If given, override the station power limit [kW].
    :param options: Knitro options that override :data:`DEFAULT_KNITRO_OPTIONS`.
    :param atol: Tolerance of the primal feasibility check.
    :param license: A checked-out Knitro license token shared between problems.
    :return: The report that is also written to the json file.
    :raises FileExistsError: If an output exists and ``overwrite`` is False.
    """
    npz_file = Path(npz_file)
    out_npz = Path(output) if output else Path(output_dir) / f"{npz_file.stem}-res-KNITRO.npz"
    out_json = out_npz.with_suffix(".json")
    if not overwrite and (out_npz.exists() or out_json.exists()):
        raise FileExistsError(f"{out_npz} or {out_json} already exists (use --overwrite)")

    evs, cfg = EVData.from_npz(npz_file), load_config(npz_file)
    rho, dt, beta, Plim = cfg["rho"], cfg["dt"], cfg["beta"], cfg["Plim"]
    if num_evs is not None:
        evs, rho = take_first(evs, rho, num_evs)
        Plim = 4.0 * num_evs  # setting of the paper: station power limit 4N kW
    if plim is not None:
        Plim = plim
    K, T = len(evs), int(evs.td.max() - evs.ta.min())
    print(f"[nlp2a] {npz_file}: K={K} T={T} dt={dt} beta={beta} Plim={Plim}", flush=True)

    report: dict[str, Any] = {
        "input": str(npz_file),
        "K": K,
        "T": T,
        "delta_t": dt,
        "beta": beta,
        "Plim": Plim,
        "num_evs": num_evs,
        "knitro_release": knitro_release(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "feasibility_atol": atol,
    }
    try:
        t0 = time.perf_counter()
        with build_model(evs, rho, dt, beta=beta, Plim=Plim, license=license) as model:
            report["build_time"] = time.perf_counter() - t0
            res = solve(model, options)
        a = res.pop("a")
        report.update(res)
        if a is None:
            raise RuntimeError(f"Knitro failed with status {res['status']} ({res['status_name']})")
        print(
            f"[nlp2a] optimization finished: status {res['status']} ({res['status_name']}), "
            f"{res['solve_time_wall']:.2f} s. Checking feasibility...",
            flush=True,
        )
        fr = primal_feasibility_report(a, evs, Plim, dt, atol)
        report["feasible"] = not fr
        report["n_violated"] = len(fr)
        report["violations_max"] = max_violation_by_type(fr)
        c_obj, f_obj = objective_parts(a, evs, rho, dt)
        eval_obj = (1 - beta) * c_obj + beta * f_obj
        report.update(eval_obj=eval_obj, eval_cost=c_obj, eval_fairness=f_obj, obj_diff=res["obj_value"] - eval_obj)
        if fr:
            print(f"[nlp2a] Infeasible. {len(fr)} constraints violated (max per type: {report['violations_max']}).", flush=True)
        else:
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            data = model_quantities(a, evs, rho, dt)
            data["obj_value"] = np.array(res["obj_value"])
            with open(out_npz, "wb" if overwrite else "xb") as f:
                np.savez(f, **data)
            print(f"[nlp2a] Feasible solution found and written to {out_npz}.", flush=True)
            print(f"[nlp2a] opt_obj / eval_obj = {res['obj_value']} / {eval_obj}. Difference = {report['obj_diff']}", flush=True)
        report["npz_written"] = not fr
    except Exception as err:  # as in the Julia code, a failure on one file does not affect the others
        print(f"[nlp2a] Unable to optimize {npz_file} ! Error: {err}", flush=True)
        report.update(error=repr(err), npz_written=False)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w" if overwrite else "x", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2)
    print(f"[nlp2a] report written to {out_json}", flush=True)
    return report


def _parse_value(text: str) -> int | float | str:
    """Parse the value of a ``--set NAME=VALUE`` option as int, float or string (in this order).

    :param text: The value as given on the command line.
    :return: The parsed value.
    """
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser; the defaults are the options of the Julia code.

    :return: The parser.
    """
    d = DEFAULT_KNITRO_OPTIONS
    p = argparse.ArgumentParser(
        description="Solve the centralized EV charging problem P1 with Knitro multi-start (port of nlp2a.jl).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("npz_files", nargs="+", help="scenario file(s) with keys ta, td, SoCa, ..., rho, delta_t, beta, Plim")
    p.add_argument("-o", "--output", help="result npz path (single input only); a .json report is written alongside")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="directory for <stem>-res-KNITRO.npz/.json")
    p.add_argument("--overwrite", action="store_true", help="allow overwriting existing outputs")
    p.add_argument("--num-evs", type=int, help="use only the first K EVs (small tests); sets Plim = 4K unless --plim is given")
    p.add_argument("--plim", type=float, help="override the station power limit Plim [kW]")
    p.add_argument("--ms-enable", type=int, choices=(0, 1), default=d["ms_enable"], help="Knitro multi-start on/off")
    p.add_argument("--ms-maxsolves", type=int, default=d["ms_maxsolves"], help="number of multi-start start points (N_start)")
    p.add_argument("--seed", type=int, default=d["ms_seed"], help="multi-start random seed (ms_seed)")
    p.add_argument("--maxtime", type=float, default=d["maxtime"], help="total real-time limit [s] (Julia: maxtime_real)")
    p.add_argument("--maxit", type=int, default=d["maxit"], help="max iterations per solve")
    p.add_argument("--opttol", type=float, default=d["opttol"], help="relative KKT tolerance")
    p.add_argument("--opttol-abs", type=float, default=d["opttol_abs"], help="absolute KKT tolerance")
    p.add_argument("--feastol", type=float, default=d["feastol"], help="relative feasibility tolerance")
    p.add_argument("--feastol-abs", type=float, default=d["feastol_abs"], help="absolute feasibility tolerance")
    p.add_argument("--nlp-algorithm", type=int, default=d["nlp_algorithm"], help="0 = auto (Julia: alg=0)")
    p.add_argument("--outlev", type=int, default=d["outlev"], help="Knitro output level")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="extra Knitro option (repeatable)")
    p.add_argument("--atol", type=float, default=1e-3, help="feasibility tolerance for the primal feasibility check")
    p.add_argument("--license-file", help="Knitro license file; sets ARTELYS_LICENSE for this process only")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Solve the scenario files given on the command line.

    :param argv: Command-line arguments; ``sys.argv[1:]`` if None.
    :return: Process exit code, 0 if a feasible solution was written for every file.
    """
    args = build_parser().parse_args(argv)
    if args.output and len(args.npz_files) != 1:
        raise SystemExit("--output needs exactly one input file; use --output-dir for several")
    options: dict[str, Any] = {
        "ms_enable": args.ms_enable,
        "ms_maxsolves": args.ms_maxsolves,
        "ms_seed": args.seed,
        "maxtime": args.maxtime,
        "maxit": args.maxit,
        "opttol": args.opttol,
        "opttol_abs": args.opttol_abs,
        "feastol": args.feastol,
        "feastol_abs": args.feastol_abs,
        "nlp_algorithm": args.nlp_algorithm,
        "outlev": args.outlev,
    }
    for item in args.set:
        name, _, value = item.partition("=")
        options[name.strip()] = _parse_value(value.strip())
    prepare_license(args.license_file)
    try:
        lic = checkout_license()
    except RuntimeError as err:
        raise SystemExit(f"[nlp2a] {err}")
    ok = True
    try:
        for f in args.npz_files:
            try:
                report = run(
                    f,
                    output=args.output,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                    num_evs=args.num_evs,
                    plim=args.plim,
                    options=options,
                    atol=args.atol,
                    license=lic,
                )
                ok &= bool(report.get("npz_written"))
            except FileExistsError as err:  # existing outputs are not overwritten by default
                print(f"[nlp2a] skipped {f}: {err}", flush=True)
                ok = False
    finally:
        lic.release()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
