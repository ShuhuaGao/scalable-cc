"""Versioned 21-feature causal state representations."""
from __future__ import annotations

import numpy as np

from .core import Scenario, causal_state, efficiency_vector

LOCAL_INDICES = (0, 1, 10, 11, 12, 13, 14)


def refined_state(s: Scenario, t: int, k: int, soc: np.ndarray, proposed: np.ndarray, limit: float, variant: str) -> np.ndarray:
    """Construct one normalized state from the currently connected EVs only.

    :param s: Original fixed-day scenario.
    :param t: Current operating-day slot.
    :param k: Index of a currently connected vehicle.
    :param soc: Current start-of-slot SoCs.
    :param proposed: Sequential raw proposals for earlier EVs.
    :param limit: Positive station power limit.
    :param variant: ``pressure`` or ``tail`` group summary.
    :return: Finite 21-dimensional float64 state.
    """
    if variant not in ("pressure", "tail"):
        raise ValueError("Unknown feature variant")
    active = (s.ta <= t) & (t < s.td)
    indices = np.flatnonzero(active)
    if k < 0 or k >= len(active) or not active[k] or limit <= 0 or s.dt <= 0:
        raise ValueError("Invalid current vehicle or physical parameters")
    if np.any(s.power[indices] <= 0) or np.any(s.capacity[indices] <= 0):
        raise ValueError("Active vehicles require positive capacity and power")
    remaining = s.td[indices] - t
    if np.any(remaining < 1):
        raise ValueError("Active remaining time must be positive")
    energy = np.maximum(s.target[indices] - np.asarray(soc)[indices], 0) * s.capacity[indices]
    r = energy / (efficiency_vector(s)[indices] * s.power[indices] * s.dt)
    g = r / remaining
    bounded = g / (1 + g)
    rank = int(np.searchsorted(indices, k))
    n = len(indices)

    def aggregate(which: slice) -> list[float]:
        """Summarize a positional active-EV group.

        :param which: Slice into the active fleet arrays.
        :return: Five unnormalized values, or zeros for an empty group.
        """
        group = indices[which]
        if len(group) == 0:
            return [0.0] * 5
        e, p, rem, urgency, bounded_g, required = energy[which], s.power[group], remaining[which], g[which], bounded[which], r[which]
        pressure = float(urgency @ p / limit)
        if variant == "pressure":
            return [float(e.mean() / 40), float(p.mean() / 8), float(bounded_g.mean()), float(rem.mean() / 40), pressure]
        return [float(e.mean() / 40), float(rem.mean() / 40), float(bounded_g.max()), float(np.mean(rem - required <= 2)), pressure]

    before, after = indices[:rank], indices[rank + 1:]
    a = aggregate(slice(None, rank)) + [float(np.asarray(proposed)[before] @ s.power[before] / limit), len(before) / n]
    b = [rank / n, s.capacity[k] / 40, s.power[k] / 8, energy[rank] / 40, remaining[rank] / 40, bounded[rank]]
    c = aggregate(slice(rank + 1, None)) + [len(after) / n]
    state = np.asarray([s.price[t], 1 - 2 * t / 56] + [1 - 2 * x for x in a + b + c], dtype=np.float64)
    if state.shape != (21,) or not np.isfinite(state).all():
        raise ValueError("Nonfinite or invalid state")
    return state


def mask_local(state: np.ndarray) -> np.ndarray:
    """Remove fleet context and rank from a refined state or batch.

    :param state: Array with 21 features in the final dimension.
    :return: Independent masked array retaining the five common B features.
    """
    x = np.asarray(state)
    if x.ndim < 1 or x.shape[-1] != 21:
        raise ValueError("Expected 21 features")
    out = np.zeros_like(x)
    out[..., LOCAL_INDICES] = x[..., LOCAL_INDICES]
    return out


def build_state(s: Scenario, t: int, k: int, soc: np.ndarray, proposed: np.ndarray, limit: float, state_schema: str) -> np.ndarray:
    """Construct a causal state with an explicit, versioned schema.

    :param s: Fixed-day charging scenario.
    :param t: Current operating-day slot.
    :param k: Currently processed vehicle index.
    :param soc: Start-of-slot state of charge.
    :param proposed: Earlier raw proposals in this slot.
    :param limit: Station power limit.
    :param state_schema: ``causal21_v1`` or ``pressure21_v1``.
    :return: A 21-feature state.
    """
    if state_schema == "causal21_v1":
        return causal_state(s, t, k, soc, proposed, limit)
    if state_schema == "pressure21_v1":
        return refined_state(s, t, k, soc, proposed, limit, "pressure")
    raise ValueError(f"Unknown state schema: {state_schema}")
