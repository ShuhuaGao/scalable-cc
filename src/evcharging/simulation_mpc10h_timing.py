"""Online simulation with shared state construction and safety correction."""

from collections.abc import Callable
import time
import numpy as np
from numpy.typing import NDArray
from .core import Scenario, causal_state, spp


def simulate(
    s: Scenario,
    limit: float,
    policy: Callable[[NDArray[np.float64], int, NDArray[np.float64]], float],
) -> dict:
    """Simulate an entire scenario with decisions made in pile order.

    :param s: Scenario.
    :param limit: Station limit in kW.
    :param policy: Function taking the state, vehicle index and urgency.
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
    slot_seconds = []
    for t in range(h):
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
        for k in np.flatnonzero(active):
            x = causal_state(s, t, int(k), soc[t], raw[t], limit)
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
        slot_seconds.append(time.perf_counter() - slot_start)
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
        "slot_seconds": np.array(slot_seconds),
        "seconds": time.perf_counter() - start,
    }
