"""Linear-time 27-feature distributional policy and online simulation."""

from collections.abc import Callable
import time
import numpy as np
from numpy.typing import NDArray
from .core import Scenario, spp


def schedule_distribution27(
    indices: NDArray[np.int64],
    soc: NDArray[np.float64],
    target: NDArray[np.float64],
    capacity: NDArray[np.float64],
    power: NDArray[np.float64],
    remaining: NDArray[np.float64],
    efficiency: float | NDArray[np.float64],
    price: float,
    t: int,
    dt: float,
    limit: float,
    policy: Callable[[NDArray[np.float64], int, NDArray[np.float64]], float],
) -> dict:
    """Schedule explicitly supplied connected vehicles in their supplied order.

    All vehicle arrays have one entry per connected vehicle; indices identify the
    original vehicles and need not be contiguous. The policy receives a compact rank and compact urgency vector. The caller supplies records in the intended processing order.

    :param indices: Ordered vehicle identifiers.
    :param soc: Start-of-slot SoC values.
    :param target: Target SoC values.
    :param capacity: Battery capacities.
    :param power: Rated powers.
    :param remaining: Remaining slots including the current slot.
    :param efficiency: Scalar or per-vehicle charging efficiency.
    :param price: Current price.
    :param t: Current slot index.
    :param dt: Slot duration.
    :param limit: Station power limit.
    :param policy: Callable receiving state, compact rank and compact urgency.
    :return: Compact states, proposals, correction stages and urgency.
    """
    ids = np.asarray(indices)
    n = len(ids)
    arrays = [
        np.asarray(x, dtype=float) for x in (soc, target, capacity, power, remaining)
    ]
    if ids.ndim != 1 or any(x.shape != (n,) for x in arrays):
        raise ValueError("Connected records must be one-dimensional and aligned")
    soc, target, capacity, power, remaining = arrays
    eff = np.asarray(efficiency, dtype=float)
    if eff.ndim == 0:
        eff = np.full(n, eff)
    if (
        eff.shape != (n,)
        or not np.isfinite(eff).all()
        or np.any(eff <= 0)
        or np.any(eff > 1)
        or not np.isfinite(limit)
        or limit <= 0
        or not np.isfinite(dt)
        or dt <= 0
        or any(not np.isfinite(x).all() for x in arrays)
        or np.any(capacity <= 0)
        or np.any(power <= 0)
        or np.any(remaining < 1)
    ):
        raise ValueError("Invalid connected vehicle or physical parameters")
    energy = np.maximum(target - soc, 0) * capacity
    urgency = energy / (eff * power * dt) / remaining
    bounded = urgency / (1 + urgency)
    values = np.column_stack((energy, power, bounded, remaining, urgency * power))
    prefix = np.vstack((np.zeros(5), np.cumsum(values, axis=0)))
    suffix = np.vstack((np.cumsum(values[::-1], axis=0)[::-1], np.zeros(5)))
    raw = np.zeros(n)
    states = np.empty((n, 27))
    mins = np.where(energy > 0, remaining, np.inf)
    pre_min = np.r_[np.inf, np.minimum.accumulate(mins)]
    suf_min = np.r_[np.minimum.accumulate(mins[::-1])[::-1], np.inf]
    pre_max = np.r_[0.0, np.maximum.accumulate(bounded)]
    suf_max = np.r_[np.maximum.accumulate(bounded[::-1])[::-1], 0.0]
    pre_sq = np.r_[0.0, np.cumsum(bounded**2)]
    suf_sq = np.r_[np.cumsum((bounded**2)[::-1])[::-1], 0.0]

    def extra(count, minimum, maximum, summation, square):
        if not count:
            return [0.0, 0.0, 0.0]
        variance = max(0.0, square / count - (summation / count) ** 2)
        return [(minimum / 40 if np.isfinite(minimum) else 0.0), maximum, variance]

    proposed_power = 0.0
    for j in range(n):

        def group(total: NDArray[np.float64], count: int) -> list[float]:
            """Normalize five sums for a group, including the empty case.

            :param total: Five aggregate sums.
            :param count: Number of vehicles.
            :return: Normalized group features.
            """
            if not count:
                return [0.0] * 5
            return [
                total[0] / count / 40,
                total[1] / count / 8,
                total[2] / count,
                total[3] / count / 40,
                total[4] / limit,
            ]

        a = group(prefix[j], j) + [proposed_power / limit, j / n]
        b = [
            j / n,
            capacity[j] / 40,
            power[j] / 8,
            energy[j] / 40,
            remaining[j] / 40,
            bounded[j],
        ]
        after = n - j - 1
        c = group(suffix[j + 1], after) + [after / n]
        x = np.asarray([price, 1 - 2 * t / 56] + [1 - 2 * v for v in a + b + c])
        a_extra = extra(j, pre_min[j], pre_max[j], prefix[j, 2], pre_sq[j])
        c_extra = extra(
            after, suf_min[j + 1], suf_max[j + 1], suffix[j + 1, 2], suf_sq[j + 1]
        )
        x = np.r_[x, 1 - 2 * np.asarray(a_extra + c_extra)]
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite state")
        states[j] = x
        raw[j] = np.clip(policy(x, j, urgency), 0.0, 1.0)
        proposed_power += raw[j] * power[j]
    clipped, scaled, final = spp(
        raw,
        soc,
        target,
        capacity,
        power,
        np.ones(n, dtype=bool),
        remaining,
        limit,
        eff,
        dt,
    )
    return dict(
        states=states, raw=raw, clip=clipped, scale=scaled, final=final, urgency=urgency
    )


def simulate(
    s: Scenario,
    limit: float,
    policy: Callable[[NDArray[np.float64], int, NDArray[np.float64]], float],
    *,
    state_schema: str,
    pressure_reference: bool = False,
    profile: dict | None = None,
) -> dict:
    """Simulate an entire scenario with decisions made in pile order.

    :param s: Scenario.
    :param limit: Station limit in kW.
    :param policy: Function taking the state, vehicle index and urgency.
    :param state_schema: Must be ``distribution27_v1``.
    :param pressure_reference: Reserved; reference implementation is unavailable.
    :param profile: Optional dictionary with a memory callback and timing results.
    :return: Trajectories and decision records.
    """
    if state_schema != "distribution27_v1" or pressure_reference:
        raise ValueError("Expected distribution27_v1 without pressure_reference")
    n, h = len(s.ta), len(s.price)
    soc = np.zeros((h + 1, n))
    soc[0] = s.arrival
    raw = np.zeros((h, n))
    clip = np.zeros_like(raw)
    scale = np.zeros_like(raw)
    final = np.zeros_like(raw)
    urgency = np.zeros_like(raw)
    states = []
    records = []
    start = time.perf_counter()
    if profile is not None:
        profile["preloop_memory"] = profile["memory_callback"]()
        profile["slot_seconds"] = []
    for t in range(h):
        if profile is not None:
            slot_start = time.perf_counter()
        active = (s.ta <= t) & (t < s.td)
        remaining = np.maximum(s.td - t, 1)
        urgency[t] = np.where(
            active,
            np.maximum(s.target - soc[t], 0)
            * s.capacity
            / (s.efficiency * s.power * s.dt * remaining),
            0.0,
        )
        indices = np.flatnonzero(active)
        eff = np.asarray(s.efficiency)
        eff = eff if eff.ndim == 0 else eff[indices]
        result = schedule_distribution27(
            indices=indices,
            soc=soc[t, indices],
            target=s.target[indices],
            capacity=s.capacity[indices],
            power=s.power[indices],
            remaining=s.td[indices] - t,
            efficiency=eff,
            price=s.price[t],
            t=t,
            dt=s.dt,
            limit=limit,
            policy=lambda x, j, _: policy(x, int(indices[j]), urgency[t]),
        )
        raw[t, indices] = result["raw"]
        clip[t, indices] = result["clip"]
        scale[t, indices] = result["scale"]
        final[t, indices] = result["final"]
        states.extend(result["states"])
        records.extend((t, int(k)) for k in indices)
        soc[t + 1] = soc[t] + final[t] * s.power * s.efficiency * s.dt / s.capacity
        if profile is not None:
            profile["slot_seconds"].append(time.perf_counter() - slot_start)
    if profile is not None:
        profile["postloop_memory"] = profile["memory_callback"]()
    total_urgency = urgency.sum(axis=1, keepdims=True)
    shares = np.divide(
        urgency, total_urgency, out=np.zeros_like(urgency), where=total_urgency > 0
    )
    unfair = (final - final.sum(axis=1, keepdims=True) * shares) ** 2
    costs = final * s.power[None, :] * s.price[:, None] * s.dt
    return {
        "soc": soc,
        "raw": raw,
        "clip": clip,
        "scale": scale,
        "final": final,
        "states": np.array(states),
        "records": np.array(records),
        "cost": costs,
        "unfair": unfair,
        "seconds": time.perf_counter() - start,
    }
