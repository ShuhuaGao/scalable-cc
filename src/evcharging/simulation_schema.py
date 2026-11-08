"""Online simulation with shared state construction and safety correction."""

from collections.abc import Callable
import time
import numpy as np
from numpy.typing import NDArray
from .core import Scenario, spp
from .state_schema import build_state


def schedule_pressure21(
    indices: NDArray[np.int64], soc: NDArray[np.float64], target: NDArray[np.float64],
    capacity: NDArray[np.float64], power: NDArray[np.float64],
    remaining: NDArray[np.float64], efficiency: float | NDArray[np.float64],
    price: float, t: int, dt: float, limit: float,
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
    arrays = [np.asarray(x, dtype=float) for x in (soc, target, capacity, power, remaining)]
    if ids.ndim != 1 or any(x.shape != (n,) for x in arrays):
        raise ValueError("Connected records must be one-dimensional and aligned")
    soc, target, capacity, power, remaining = arrays
    eff = np.asarray(efficiency, dtype=float)
    if eff.ndim == 0:
        eff = np.full(n, eff)
    if (eff.shape != (n,) or not np.isfinite(eff).all() or np.any(eff <= 0)
            or np.any(eff > 1) or not np.isfinite(limit) or limit <= 0
            or not np.isfinite(dt) or dt <= 0 or any(not np.isfinite(x).all() for x in arrays)
            or np.any(capacity <= 0) or np.any(power <= 0) or np.any(remaining < 1)):
        raise ValueError("Invalid connected vehicle or physical parameters")
    energy = np.maximum(target - soc, 0) * capacity
    urgency = energy / (eff * power * dt) / remaining
    bounded = urgency / (1 + urgency)
    values = np.column_stack((energy, power, bounded, remaining, urgency * power))
    prefix = np.vstack((np.zeros(5), np.cumsum(values, axis=0)))
    suffix = np.vstack((np.cumsum(values[::-1], axis=0)[::-1], np.zeros(5)))
    raw = np.zeros(n)
    states = np.empty((n, 21))
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
            return [total[0] / count / 40, total[1] / count / 8,
                    total[2] / count, total[3] / count / 40, total[4] / limit]
        a = group(prefix[j], j) + [proposed_power / limit, j / n]
        b = [j / n, capacity[j] / 40, power[j] / 8, energy[j] / 40,
             remaining[j] / 40, bounded[j]]
        after = n - j - 1
        c = group(suffix[j + 1], after) + [after / n]
        x = np.asarray([price, 1 - 2 * t / 56] + [1 - 2 * v for v in a + b + c])
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite state")
        states[j] = x
        raw[j] = np.clip(policy(x, j, urgency), 0.0, 1.0)
        proposed_power += raw[j] * power[j]
    clipped, scaled, final = spp(raw, soc, target, capacity, power,
                                np.ones(n, dtype=bool), remaining, limit, eff, dt)
    return dict(states=states, raw=raw, clip=clipped, scale=scaled, final=final, urgency=urgency)


def simulate(
    s: Scenario,
    limit: float,
    policy: Callable[[NDArray[np.float64], int, NDArray[np.float64]], float],
    *, state_schema: str, pressure_reference: bool = False,
    profile: dict | None = None,
) -> dict:
    """Simulate an entire scenario with decisions made in pile order.

    :param s: Scenario.
    :param limit: Station limit in kW.
    :param policy: Function taking the state, vehicle index and urgency.
    :param state_schema: Explicit versioned 21-feature state schema.
    :param pressure_reference: Retain repeated-aggregation Pressure implementation.
    :param profile: Optional dictionary with a memory callback and timing results.
    :return: Trajectories and decision records.
    """
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
        if state_schema == "pressure21_v1" and not pressure_reference:
            eff = np.asarray(s.efficiency)
            eff = eff if eff.ndim == 0 else eff[indices]
            result = schedule_pressure21(
                indices=indices, soc=soc[t, indices], target=s.target[indices],
                capacity=s.capacity[indices], power=s.power[indices],
                remaining=s.td[indices] - t, efficiency=eff, price=s.price[t],
                t=t, dt=s.dt, limit=limit,
                policy=lambda x, j, _: policy(x, int(indices[j]), urgency[t]),
            )
            raw[t, indices] = result["raw"]
            clip[t, indices] = result["clip"]
            scale[t, indices] = result["scale"]
            final[t, indices] = result["final"]
            states.extend(result["states"])
            records.extend((t, int(k)) for k in indices)
        else:
            for k in indices:
                x = build_state(s, t, int(k), soc[t], raw[t], limit, state_schema)
                raw[t, k] = np.clip(policy(x, int(k), urgency[t]), 0.0, 1.0)
                states.append(x)
                records.append((t, int(k)))
            clip[t], scale[t], final[t] = spp(
                raw[t],
                soc[t],
                s.target,
                s.capacity,
                s.power,
                active,
                s.td - t,
                limit,
                s.efficiency,
                s.dt,
            )
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
