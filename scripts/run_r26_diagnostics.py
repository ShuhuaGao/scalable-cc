"""Run scheduling and distribution diagnostics."""

import os

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "1"
import sys, json, csv, hashlib, concurrent.futures, shutil
from pathlib import Path
from dataclasses import replace
from typing import Any, Callable
import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from evcharging.core import (
    Scenario,
    load_scenario,
    generate,
    spp,
    fixed_feasibility,
    urgency_allocation,
)
from evcharging.models import imitation_network
from evcharging.state_schema import build_state
from evcharging.simulation_schema import simulate as original

A = R / "artifacts/runs/pressure21_v1/r26_diagnostics"
O = R / "results/pressure21_v1/r26"
MODEL = R / "artifacts/runs/pressure21_v1/il_11002/il_model/best.pt"
FORM = R / "artifacts/runs/pressure21_v1/evaluation/formal"
INPUT = R / "artifacts/runs/ideal10h_v1/formal/inputs"
SMALL = R / "artifacts/runs/ideal10h_v1/small_experts"
SMALLPOL = R / "artifacts/runs/pressure21_v1/evaluation/small"


def sha(p: Path) -> str:
    """Hash an input file.

    :param p: File to hash.
    :return: SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model() -> torch.nn.Module:
    """Load and verify the frozen policy.

    :return: Evaluation-mode frozen network.
    """
    torch.set_num_threads(1)
    selection = json.loads(
        (R / "artifacts/runs/pressure21_v1/il_selection.json").read_text(
            encoding="utf8"
        )
    )
    entry = next(x for x in selection["models"] if x["seed"] == 11002)
    assert entry["state_schema"] == "pressure21_v1" and sha(MODEL) == entry["sha256"]
    m = imitation_network()
    m.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True)["model"])
    m.eval()
    return m


def predict(m: torch.nn.Module, x: np.ndarray) -> float:
    """Evaluate one normalized state.

    :param m: Frozen policy.
    :param x: Pressure state vector.
    :return: Normalized raw action.
    """
    with torch.no_grad():
        return float(m(torch.tensor(x, dtype=torch.float32)[None, :])[0, 0])


def policy(m: torch.nn.Module) -> Callable[[np.ndarray, int, np.ndarray], float]:
    """Adapt the policy to the public simulation interface.

    :param m: Frozen policy.
    :return: Policy callable.
    """
    return lambda x, k, u: predict(m, x)


def active(s: Scenario) -> np.ndarray:
    """Construct the slot-by-vehicle connection mask.

    :param s: Scenario.
    :return: Boolean connection mask.
    """
    return (s.ta[None, :] <= np.arange(len(s.price))[:, None]) & (
        np.arange(len(s.price))[:, None] < s.td[None, :]
    )


def totals(
    s: Scenario, soc: np.ndarray, a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute same-state cost and unfairness.

    :param s: Scenario.
    :param soc: Slot-start and terminal SoC trajectory.
    :param a: Normalized actions.
    :return: Per-decision cost and unfairness arrays.
    """
    t = np.arange(len(s.price))[:, None]
    mask = active(s)
    rem = np.maximum(s.td[None, :] - t, 1)
    u = np.where(
        mask,
        np.maximum(s.target - soc[:-1], 0)
        * s.capacity
        / (s.efficiency * s.power * s.dt * rem),
        0,
    )
    shares = np.divide(
        u,
        u.sum(axis=1, keepdims=True),
        out=np.zeros_like(u),
        where=u.sum(axis=1, keepdims=True) > 0,
    )
    return (
        a * s.power * s.price[:, None] * s.dt,
        (a - a.sum(axis=1, keepdims=True) * shares) ** 2,
    )


def simulate(
    s: Scenario, limit: float, m: torch.nn.Module | None, mode: str
) -> dict[str, np.ndarray]:
    """Run one closed-loop policy and SPP configuration.

    :param s: Scenario.
    :param limit: Station limit in kW.
    :param m: Frozen policy, or None for UPA.
    :param mode: Full, no-scale, no-boost, or UPA mode.
    :return: SoC, actions, cost and unfairness arrays.
    """
    n = len(s.ta)
    h = len(s.price)
    soc = np.zeros((h + 1, n))
    soc[0] = s.arrival
    raw = np.zeros((h, n))
    clip = raw.copy()
    scale = raw.copy()
    final = raw.copy()
    for t in range(h):
        mask = (s.ta <= t) & (t < s.td)
        rem = s.td - t
        u = np.where(
            mask,
            np.maximum(s.target - soc[t], 0)
            * s.capacity
            / (s.efficiency * s.power * s.dt * np.maximum(rem, 1)),
            0,
        )
        if mode == "upa":
            raw[t] = urgency_allocation(u, s.power, rem, limit)
        else:
            for k in np.flatnonzero(mask):
                raw[t, k] = predict(
                    m, build_state(s, t, int(k), soc[t], raw[t], limit, "pressure21_v1")
                )
        c, sc, f = spp(
            raw[t],
            soc[t],
            s.target,
            s.capacity,
            s.power,
            mask,
            rem,
            limit,
            s.efficiency,
            s.dt,
        )
        if mode == "no_boost":
            f = sc.copy()
        elif mode == "no_scale":
            sc = c.copy()
            f = c.copy()
            total = float(f @ s.power)
            req = np.maximum(
                0, (s.target - soc[t]) * s.capacity / (s.efficiency * s.power * s.dt)
            )
            minimum = np.minimum(1, np.maximum(0, req - (rem - 1)))
            for k in np.flatnonzero(mask):
                add = min(
                    max(0, minimum[k] - f[k]),
                    max(0, limit - total) / s.power[k],
                    max(0, req[k] - f[k]),
                )
                f[k] += add
                total += add * s.power[k]
        clip[t], scale[t], final[t] = c, sc, f
        soc[t + 1] = soc[t] + f * s.power * s.efficiency * s.dt / s.capacity
    cost, unfair = totals(s, soc, final)
    return dict(
        soc=soc, raw=raw, clip=clip, scale=scale, final=final, cost=cost, unfair=unfair
    )


def metrics(
    s: Scenario, limit: float, r: dict[str, np.ndarray]
) -> dict[str, float | int]:
    """Audit a trajectory and calculate daily metrics.

    :param s: Scenario.
    :param limit: Station limit in kW.
    :param r: Trajectory arrays.
    :return: Physical, objective and overload metrics.
    """
    a = r["final"]
    soc = r["soc"]
    mask = active(s)
    cost, unfair = totals(s, soc, a)
    checks = [
        np.max(np.abs(cost - r["cost"])),
        np.max(np.abs(unfair - r["unfair"])),
        np.max(
            np.abs(soc[1:] - soc[:-1] - a * s.power * s.efficiency * s.dt / s.capacity)
        ),
        np.max(np.abs(a[~mask]), initial=0),
        np.max(np.maximum(-a, 0)),
        np.max(np.maximum(a - 1, 0)),
        np.max(np.maximum(soc - s.target, 0)),
        np.max(np.abs(soc[0] - s.arrival)),
    ]
    assert max(checks) < 1e-8, checks
    gap = np.maximum(s.target - soc[s.td, np.arange(len(s.ta))], 0)
    overload = np.maximum(a @ s.power - limit, 0)
    return dict(
        cost=float(cost.sum()),
        D=float(unfair.sum()),
        F=float(0.7 * cost.sum() + 0.3 * unfair.sum()),
        purchased_kwh=float((a * s.power * s.dt).sum()),
        met_ev_percent=float(100 * np.mean(gap <= 1e-4)),
        all_met=int(np.all(gap <= 1e-4)),
        gap_kwh=float(gap @ s.capacity),
        demand_kwh=float((s.target - s.arrival) @ s.capacity / s.efficiency),
        battery_demand_kwh=float((s.target - s.arrival) @ s.capacity),
        gap_fraction_battery_demand=float(
            (gap @ s.capacity) / ((s.target - s.arrival) @ s.capacity)
        ),
        overload_slot_fraction=float(np.mean(overload > 1e-8)),
        overload_active_slot_fraction=float(
            (overload > 1e-8).sum() / mask.any(axis=1).sum()
        ),
        overload_max_kw=float(overload.max()),
        overload_kwh=float(overload.sum() * s.dt),
    )


def stages(
    s: Scenario, limit: float, r: dict[str, np.ndarray], seed: int
) -> list[dict[str, Any]]:
    """Calculate same-state changes between correction stages.

    :param s: Scenario.
    :param limit: Station limit in kW.
    :param r: Saved trajectory arrays.
    :param seed: Scenario seed.
    :return: Daily stage statistics.
    """
    mask = active(s)
    rows = []
    savedcost, savedunfair = totals(s, r["soc"], r["final"])
    assert (
        np.max(abs(savedcost - r["cost"])) < 1e-8
        and np.max(abs(savedunfair - r["unfair"])) < 1e-8
    )
    cs = {}
    ds = {}
    cs_slot = {}
    ds_slot = {}
    for name in ("raw", "clip", "scale", "final"):
        c, d = totals(s, r["soc"], r[name])
        cs[name] = float(c.sum())
        ds[name] = float(d.sum())
        cs_slot[name] = c.sum(axis=1)
        ds_slot[name] = d.sum(axis=1)
    for t in range(len(s.price)):
        total = r["clip"][t] @ s.power
        alpha = min(1, limit / total) if total > 0 else 1
        _, d1 = totals(s, r["soc"], r["clip"])
        _, d2 = totals(s, r["soc"], r["scale"])
        assert abs(d2[t].sum() - alpha**2 * d1[t].sum()) < 1e-8
    for left, right in [
        ("raw", "clip"),
        ("clip", "scale"),
        ("scale", "final"),
        ("raw", "final"),
    ]:
        delta = r[right] - r[left]
        changed = (abs(delta) > 1e-6) & mask
        vals = abs(delta[changed])
        active_slots = mask.any(axis=1)
        dc = cs_slot[right] - cs_slot[left]
        dd = ds_slot[right] - ds_slot[left]
        nslots = int(mask.any(axis=1).sum())
        tol = 1e-10
        rows.append(
            dict(
                positive_D_slots=int((dd > tol).sum()),
                negative_D_slots=int((dd < -tol).sum()),
                positive_D_slot_rate=float((dd > tol).sum() / nslots),
                negative_D_slot_rate=float((dd < -tol).sum() / nslots),
                positive_D_sum=float(dd[dd > tol].sum()),
                negative_D_sum=float(dd[dd < -tol].sum()),
                positive_cost_sum=float(dc[dc > tol].sum()),
                negative_cost_sum=float(dc[dc < -tol].sum()),
                seed=seed,
                limit=limit,
                stage=left + "_to_" + right,
                connected_decisions=int(mask.sum()),
                decision_fraction=float(changed.sum() / mask.sum()),
                slot_fraction=float(np.any(changed, axis=1).sum() / active_slots.sum()),
                day_trigger=int(changed.any()),
                mean_abs=float(abs(delta[mask]).mean()),
                changed_mean=float(vals.mean()) if vals.size else None,
                changed_p95=float(np.percentile(vals, 95)) if vals.size else None,
                abs_kwh=float((abs(delta) * s.power * s.dt).sum()),
                signed_kwh=float((delta * s.power * s.dt).sum()),
                delta_cost=cs[right] - cs[left],
                delta_D=ds[right] - ds[left],
            )
        )
    return rows


def shifted(s: Scenario, seed: int, kind: str) -> tuple[Scenario, dict[str, Any]]:
    """Draw one reproducible distribution-shifted scenario.

    :param s: Nominal scenario.
    :param seed: Scenario seed.
    :param kind: Gaussian or uniform distribution.
    :return: Shifted scenario and draw diagnostics.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 2601, 1 if kind == "gaussian" else 2])
    )
    n = len(s.ta)
    clips = [0, 0, 0]
    draws = [0, 0, 0]

    def draw(
        mu: float, sd: float, lo: float, hi: float, j: int, rounding: bool = False
    ) -> float:
        """Draw and clip one perturbed value.

        :param mu: Distribution mean.
        :param sd: Distribution standard deviation.
        :param lo: Lower bound.
        :param hi: Upper bound.
        :param j: Clipping counter index.
        :param rounding: Whether to round before clipping.
        :return: Clipped value.
        """
        x = (
            rng.normal(mu, sd)
            if kind == "gaussian"
            else rng.uniform(mu - np.sqrt(3) * sd, mu + np.sqrt(3) * sd)
        )
        if rounding:
            x = np.rint(x)
        clips[j] += int(x < lo or x > hi)
        draws[j] += 1
        return np.clip(x, lo, hi)

    arrival = np.array([draw(0.4, 0.15, 0.2, 0.6 - 1e-6, 2) for _ in range(n)])
    ta = np.empty(n, dtype=int)
    td = ta.copy()
    attempts = 0
    for k in range(n):
        required = np.ceil(
            s.capacity[k]
            * (s.target[k] - arrival[k])
            / (s.efficiency * s.power[k] * s.dt)
        )
        for it in range(10000):
            a = int(draw(8, 6, 0, 16, 0, True))
            d = int(draw(40, 6, 32, 56, 1, True))
            if d - a >= required:
                break
            attempts += 1
        else:
            raise RuntimeError(f"Feasibility redraw cap {seed} {kind} {k}")
        ta[k] = a
        td[k] = d
    meta = dict(
        seed=seed,
        distribution=kind,
        rejections=attempts,
        **{
            f"{x}_clip_rate": clips[i] / draws[i]
            for i, x in enumerate(("ta", "td", "arrival"))
        },
    )
    for name, arr in [("ta", ta), ("td", td), ("arrival", arrival)]:
        meta[name + "_mean"] = float(arr.mean())
        meta[name + "_sd"] = float(arr.std(ddof=1))
    return replace(s, ta=ta, td=td, arrival=arrival), meta


def save_scenario(p: Path, s: Scenario) -> None:
    """Save a scenario archive.

    :param p: Output path.
    :param s: Scenario.
    :return: None.
    """
    np.savez_compressed(p, **vars(s))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save heterogeneous records with a union of fields.

    :param path: Output path.
    :param rows: Records to write.
    :return: None.
    """
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        w.writeheader()
        w.writerows(rows)


def one(
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate one formal day across the frozen conditions.

    :param seed: Scenario seed.
    :return: Stage, ablation, distribution and input records.
    """
    m = model()
    s = load_scenario(str(INPUT / f"{seed}.npz"))
    aa = []
    cc = []
    dd = []
    meta = []
    for limit in (600, 800):
        with np.load(FORM / f"11002_200_{seed}_{limit}.npz") as z:
            full = {k: z[k] for k in z.files}
        aa.extend(stages(s, limit, full, seed))
        base = metrics(s, limit, full)
        if seed == 65000:
            check = simulate(s, limit, m, "full")
            for key in ("soc", "raw", "clip", "scale", "final", "cost", "unfair"):
                assert np.allclose(check[key], full[key], atol=1e-12, rtol=0), (
                    key,
                    np.max(abs(check[key] - full[key])),
                )
        for mode in ("no_scale", "no_boost"):
            r = simulate(s, limit, m, mode)
            np.savez_compressed(A / f"c_{seed}_{limit}_{mode}.npz", **r)
            mm = metrics(s, limit, r)
            assert mode == "no_scale" or mm["overload_max_kw"] < 1e-8
            cc.append(
                dict(
                    seed=seed,
                    limit=limit,
                    mode=mode,
                    **mm,
                    **{
                        "delta_" + k: mm[k] - base[k]
                        for k in (
                            "cost",
                            "D",
                            "F",
                            "purchased_kwh",
                            "met_ev_percent",
                            "all_met",
                            "gap_kwh",
                        )
                    },
                )
            )
        status, res = fixed_feasibility(s, limit)
        dd.append(
            dict(
                seed=seed,
                limit=limit,
                distribution="nominal",
                method="Full",
                feasibility=status,
                feasibility_residual=res,
                **base,
            )
        )
        # Original UPA uses causal state only to obtain time; same urgency allocation and SPP.
        up = simulate(s, limit, None, "upa")
        um = metrics(s, limit, up)
        assert um["overload_max_kw"] < 1e-8
        np.savez_compressed(A / f"d_nominal_{seed}_{limit}_upa.npz", **up)
        dd.append(
            dict(
                seed=seed,
                limit=limit,
                distribution="nominal",
                method="UPA",
                feasibility=status,
                feasibility_residual=res,
                **um,
            )
        )
    for kind in ("gaussian", "uniform"):
        sh, info = shifted(s, seed, kind)
        ip = A / f"input_{kind}_{seed}.npz"
        save_scenario(ip, sh)
        info["input_sha256"] = sha(ip)
        meta.append(info)
        for limit in (600, 800):
            status, res = fixed_feasibility(sh, limit)
            for method in ("Full", "UPA"):
                r = simulate(
                    sh,
                    limit,
                    m if method == "Full" else None,
                    "full" if method == "Full" else "upa",
                )
                mm = metrics(sh, limit, r)
                assert mm["overload_max_kw"] < 1e-8
                np.savez_compressed(A / f"d_{kind}_{seed}_{limit}_{method}.npz", **r)
                dd.append(
                    dict(
                        seed=seed,
                        limit=limit,
                        distribution=kind,
                        method=method,
                        feasibility=status,
                        feasibility_residual=res,
                        **mm,
                    )
                )
    return aa, cc, dd, meta


def small(seed: int, m: torch.nn.Module) -> list[dict[str, Any]]:
    """Reconstruct expert and closed-loop small-fleet states.

    :param seed: Small-scenario seed.
    :param m: Frozen policy.
    :return: Per-scenario prediction and trajectory records.
    """
    s = generate(seed, 10, fixed_day=True)
    p = SMALL / f"seed_{seed}.npz"
    q = SMALLPOL / f"11002_10_{seed}_40.npz"
    with np.load(p) as e, np.load(q) as r:
        for k in ("ta", "td", "arrival", "target", "capacity", "power", "price"):
            assert np.array_equal(e[k], getattr(s, k))
        records = np.argwhere(active(s))
        assert np.array_equal(records, e["records"])
        assert np.allclose(
            e["y"], e["actions"][records[:, 0], records[:, 1]], atol=1e-8
        )
        errors = {"expert_prefix": [], "self_prefix": []}
        quarters = {
            "expert_prefix": [[] for _ in range(4)],
            "self_prefix": [[] for _ in range(4)],
        }
        deviations = [[] for _ in range(3)]
        corrections = [[] for _ in range(3)]
        for t in range(len(s.price)):
            ix = np.flatnonzero(active(s)[t])
            selfprefix = np.zeros(10)
            for rank, k in enumerate(ix):
                quart = min(3, 4 * rank // len(ix))
                y = e["actions"][t, k]
                for name, prefix in [
                    ("expert_prefix", e["actions"][t]),
                    ("self_prefix", selfprefix),
                ]:
                    pred = predict(
                        m,
                        build_state(
                            s, t, int(k), e["soc"][t], prefix, 40, "pressure21_v1"
                        ),
                    )
                    err = pred - y
                    errors[name].append(err)
                    quarters[name][quart].append(err)
                    if name == "self_prefix":
                        selfprefix[k] = pred
                st = build_state(
                    s, t, int(k), r["soc"][t], r["raw"][t], 40, "pressure21_v1"
                )
                assert abs(predict(m, st) - r["raw"][t, k]) < 1e-7
                ref = build_state(
                    s, t, int(k), e["soc"][t], r["raw"][t], 40, "pressure21_v1"
                )
                group = min(2, 3 * t // len(s.price))
                deviations[group].append(
                    [
                        100 * abs(r["soc"][t, k] - e["soc"][t, k]),
                        abs(st[12] - ref[12]) / 2,
                        abs(st[14] - ref[14]) / 2,
                        abs(st[6] - ref[6]) / 2,
                        abs(st[19] - ref[19]) / 2,
                    ]
                )
                corrections[group].append(abs(r["final"][t, k] - r["raw"][t, k]))
        rows = []
        for name in errors:
            for label, vals in [("all", errors[name])] + [
                (f"q{i+1}", x) for i, x in enumerate(quarters[name])
            ]:
                v = np.asarray(vals)
                rows.append(
                    dict(
                        seed=seed,
                        kind=name,
                        group=label,
                        n=len(v),
                        mae=float(abs(v).mean()),
                        mse=float((v * v).mean()),
                        soc_pp=None,
                        remaining_demand=None,
                        urgency=None,
                        pressure_before=None,
                        pressure_after=None,
                        correction=None,
                    )
                )
        for group in range(3):
            v = np.asarray(deviations[group])
            rows.append(
                dict(
                    seed=seed,
                    kind="trajectory_deviation",
                    group=("early", "middle", "late")[group],
                    n=len(v),
                    mae=None,
                    mse=None,
                    soc_pp=float(v[:, 0].mean()),
                    remaining_demand=float(v[:, 1].mean()),
                    urgency=float(v[:, 2].mean()),
                    pressure_before=float(v[:, 3].mean()),
                    pressure_after=float(v[:, 4].mean()),
                    correction=float(np.mean(corrections[group])),
                )
            )
        return rows


def summarize(rows: list[dict[str, Any]], groups: list[str]) -> list[dict[str, Any]]:
    """Aggregate daily records by group.

    :param rows: Daily records.
    :param groups: Grouping field names.
    :return: Across-day means, sample standard deviations and counts.
    """
    out = []
    for key in sorted({tuple(r[k] for k in groups) for r in rows}):
        subset = [r for r in rows if tuple(r[k] for k in groups) == key]
        row = dict(zip(groups, key))
        row["n_days"] = len(subset)
        for k in subset[0]:
            if k in groups or k in (
                "seed",
                "n",
                "connected_decisions",
                "feasibility",
                "feasibility_residual",
                "method",
                "distribution",
                "mode",
                "stage",
                "group",
                "kind",
                "limit",
            ):
                continue
            vals = [
                r[k]
                for r in subset
                if isinstance(r[k], (float, int)) and r[k] is not None
            ]
            if vals:
                row[k + "_mean"] = float(np.mean(vals))
                row[k + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        if "changed_mean" in subset[0]:
            row["changed_days_n"] = sum(x["changed_mean"] is not None for x in subset)
        if "overload_max_kw" in subset[0]:
            row["overload_max_kw_observed_max"] = max(
                float(x["overload_max_kw"]) for x in subset
            )
        out.append(row)
    return out


def supplemental_small(m: torch.nn.Module) -> None:
    """Write small-fleet slot and fixed-cohort diagnostics.

    :param m: Frozen policy.
    :return: None.
    """
    rows = []
    for seed in range(76000, 76020):
        s = generate(seed, 10, fixed_day=True)
        with np.load(SMALL / f"seed_{seed}.npz") as e, np.load(
            SMALLPOL / f"11002_10_{seed}_40.npz"
        ) as r:
            for t in range(len(s.price)):
                ix = np.flatnonzero(active(s)[t])
                vals = []
                for k in ix:
                    x = build_state(
                        s, t, int(k), r["soc"][t], r["raw"][t], 40, "pressure21_v1"
                    )
                    y = build_state(
                        s, t, int(k), e["soc"][t], r["raw"][t], 40, "pressure21_v1"
                    )
                    vals.append(
                        [
                            100 * abs(r["soc"][t, k] - e["soc"][t, k]),
                            abs(x[12] - y[12]) / 2,
                            abs(x[14] - y[14]) / 2,
                            abs(x[6] - y[6]) / 2,
                            abs(x[19] - y[19]) / 2,
                            abs(r["raw"][t, k] - r["final"][t, k]),
                        ]
                    )
                v = np.array(vals)
                departed = s.td == t + 1
                row = dict(
                    seed=seed,
                    slot=t,
                    active_count=len(ix),
                    cohort_soc_pp=float(100 * np.abs(r["soc"][t] - e["soc"][t]).mean()),
                    departure_count=int(departed.sum()),
                    terminal_gap_kwh=float(
                        (
                            np.maximum(
                                s.target[departed] - r["soc"][t + 1, departed], 0
                            )
                            * s.capacity[departed]
                        ).sum()
                    ),
                )
                for i, key in enumerate(
                    (
                        "active_soc_pp",
                        "remaining_demand",
                        "urgency",
                        "pressure_before",
                        "pressure_after",
                        "correction",
                    )
                ):
                    row[key] = float(v[:, i].mean()) if len(v) else None
                rows.append(row)
    save_csv(O / "b_slot_daily.csv", rows)
    save_csv(O / "b_slot_summary.csv", summarize(rows, ["slot"]))


def main() -> None:
    """Run and archive the frozen diagnostic protocol.

    :return: None.
    """
    if (A / "complete.json").exists():
        raise FileExistsError("Completed run already exists")
    A.mkdir(parents=True, exist_ok=True)
    O.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        R / "docs/r26_distribution_spp.md",
        R / "artifacts/runs/pressure21_v1/il_selection.json",
        MODEL,
        R / "src/evcharging/core.py",
        R / "src/evcharging/models.py",
        R / "src/evcharging/state_schema.py",
        R / "src/evcharging/simulation_schema.py",
    ]
    shutil.copy2(R / "docs/r26_distribution_spp.md", A / "frozen_protocol.md")
    (A / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "sources": {p.relative_to(R).as_posix(): sha(p) for p in sources},
                "seeds": list(range(65000, 65050)),
                "small_seeds": list(range(76000, 76020)),
                "limits": [600, 800],
                "workers": 3,
                "cpu_threads": 1,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    m = model()
    b = []
    for seed in range(76000, 76020):
        b.extend(small(seed, m))
    save_csv(O / "b_daily.csv", b)
    save_csv(O / "b_summary.csv", summarize(b, ["kind", "group"]))
    a = []
    c = []
    d = []
    meta = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for i, (aa, cc, dd, mm) in enumerate(pool.map(one, range(65000, 65050)), 1):
            a.extend(aa)
            c.extend(cc)
            d.extend(dd)
            meta.extend(mm)
            print("formal days completed", i, flush=True)
    for name, rows, groups in [
        ("a", a, ["limit", "stage"]),
        ("c", c, ["limit", "mode"]),
        ("d", d, ["distribution", "limit", "method"]),
    ]:
        save_csv(O / f"{name}_daily.csv", rows)
        save_csv(O / f"{name}_summary.csv", summarize(rows, groups))
    save_csv(O / "d_inputs.csv", meta)
    feasible = [r for r in d if r["feasibility"] == "feasible"]
    save_csv(
        O / "d_feasible_summary.csv",
        summarize(feasible, ["distribution", "limit", "method"]),
    )
    pairs = []
    for row in d:
        if row["method"] != "Full":
            continue
        u = next(
            x
            for x in d
            if all(x[k] == row[k] for k in ("seed", "limit", "distribution"))
            and x["method"] == "UPA"
        )
        pairs.append(
            dict(
                seed=row["seed"],
                limit=row["limit"],
                distribution=row["distribution"],
                feasibility=row["feasibility"],
                **{
                    k: row[k] - u[k]
                    for k in (
                        "cost",
                        "D",
                        "F",
                        "purchased_kwh",
                        "met_ev_percent",
                        "all_met",
                        "gap_kwh",
                    )
                },
            )
        )
    save_csv(O / "d_paired_daily.csv", pairs)
    save_csv(O / "d_paired_summary.csv", summarize(pairs, ["distribution", "limit"]))
    save_csv(
        O / "d_paired_feasible_summary.csv",
        summarize(
            [x for x in pairs if x["feasibility"] == "feasible"],
            ["distribution", "limit"],
        ),
    )
    supplemental_small(m)
    inputs = (
        list(INPUT.glob("65*.npz"))
        + [SMALL / f"seed_{x}.npz" for x in range(76000, 76020)]
        + [SMALLPOL / f"11002_10_{x}_40.npz" for x in range(76000, 76020)]
        + [
            FORM / f"11002_200_{x}_{l}.npz"
            for x in range(65000, 65050)
            for l in (600, 800)
        ]
    )
    files = (
        sources
        + [
            p
            for p in A.iterdir()
            if p.is_file() and p.name not in ("manifest.json", "complete.json")
        ]
        + list(O.glob("*.csv"))
        + inputs
    )
    (A / "manifest.json").write_text(
        json.dumps(
            {"sha256": {p.relative_to(R).as_posix(): sha(p) for p in files}}, indent=2
        ),
        encoding="utf8",
    )
    (A / "complete.json").write_text(
        json.dumps(
            {
                "formal_days": 50,
                "small_days": 20,
                "manifest_sha256": sha(A / "manifest.json"),
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
