"""Scenario generation, safety correction, and offline feasibility checks."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

Array = NDArray[np.float64]


@dataclass
class Scenario:
    """A charging scenario with slot indices measured from 07:00."""

    ta: NDArray[np.int64]
    td: NDArray[np.int64]
    arrival: Array
    target: Array
    capacity: Array
    power: Array
    price: Array
    efficiency: float = 0.95
    dt: float = 0.25
    beta: float = 0.3


def generate(seed: int, n: int = 200, *, fixed_day: bool = False) -> Scenario:
    """Generate independent EV and price draws using the notebook distributions.

    :param seed: Random seed.
    :param n: Number of vehicles.
    :param fixed_day: Keep the fixed 07:00 to 21:00 operational day.
    :return: Generated scenario.
    """
    rng = np.random.default_rng(seed)
    origin = np.array(
        [
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.7,
            0.8,
            1.0,
            0.9,
            0.8,
            0.6,
            0.5,
            1.0,
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
            0.8,
            0.6,
            0.4,
            0.35,
        ]
    )
    if fixed_day:
        nominal = origin[7 + np.arange(56) // 4]
        price = rng.normal(nominal, 0.1 * nominal)
    else:
        hourly = rng.normal(origin, 0.1 * origin)
        price = rng.normal(hourly[7 + np.arange(56) // 4], 0.1 * hourly[7 + np.arange(56) // 4])
    ta = np.clip(np.rint(rng.normal(8, 4, n)), 0, 16).astype(int)
    td = np.clip(np.rint(rng.normal(40 if fixed_day else 48, 4, n)), 32 if fixed_day else 40, 56).astype(int)
    first, last = (0, 56) if fixed_day else (int(ta.min()), int(td.max()))
    arrival = np.clip(rng.normal(0.4, 0.1, n), 0.2, 0.6 - 1e-6)
    target = np.clip(rng.normal(0.8, 0.1, n), 0.6 + 1e-6, 1)
    capacity = np.clip(rng.normal(30, 5, n), 20, 40)
    power = np.clip(rng.normal(6, 1, n), 4, 8)
    if fixed_day:
        required = np.ceil(capacity * (target - arrival) / (0.95 * power * 0.25)).astype(int)
        pending = td - ta < required
        while pending.any():
            count = int(pending.sum())
            ta[pending] = np.clip(np.rint(rng.normal(8, 4, count)), 0, 16).astype(int)
            td[pending] = np.clip(np.rint(rng.normal(40, 4, count)), 32, 56).astype(int)
            pending = td - ta < required
    return Scenario(ta - first, td - first, arrival, target, capacity, power, price[first:last])



def nominal_price(t: int, horizon: int = 40) -> Array:
    """Return a fixed public price forecast from the operational-day origin.

    :param t: Current slot on the fixed day grid.
    :param horizon: Forecast length in slots.
    :return: Nominal prices, extended at the final value after 21:00.
    """
    hourly = np.array([0.7, 0.8, 1.0, 0.9, 0.8, 0.6, 0.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.8])
    return hourly[np.minimum((t + np.arange(horizon)) // 4, len(hourly) - 1)]


def spp(
    raw: Array,
    soc: Array,
    target: Array,
    capacity: Array,
    power: Array,
    connected: NDArray[np.bool_],
    remaining: Array,
    limit: float,
    efficiency: float,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Apply overcharge clipping, station scaling, and ordered compensation.

    :param raw: Normalized proposed actions.
    :param soc: SoC at the start of the slot.
    :param target: Target SoC values.
    :param capacity: Battery capacities in kWh.
    :param power: Rated powers in kW.
    :param connected: Vehicles connected in this slot.
    :param remaining: Number of slots remaining including this slot.
    :param limit: Station power limit in kW.
    :param efficiency: Charging efficiency.
    :param dt: Slot duration in hours.
    :return: Actions after each of the three corrections.
    """
    r = np.maximum(0.0, (target - soc) * capacity / (efficiency * power * dt))
    clipped = np.where(connected, np.minimum(np.clip(raw, 0, 1), r), 0.0)
    total = float(clipped @ power)
    scaled = clipped * min(1.0, limit / total) if total > 0 else clipped.copy()
    final = scaled.copy()
    total = float(final @ power)
    minimum = np.minimum(1.0, np.maximum(0.0, r - (remaining - 1)))
    for k in np.flatnonzero(connected):
        add = min(
            max(0.0, minimum[k] - final[k]),
            max(0.0, limit - total) / power[k],
            max(0.0, r[k] - final[k]),
        )
        final[k] += add
        total += add * power[k]
    return clipped, scaled, final


def minimum_capacity(s: Scenario) -> tuple[str, float | None]:
    """Find the least constant station capacity that meets all targets.

    :param s: Scenario with known arrivals and departures.
    :return: Solver classification and minimum capacity in kW, if optimal.
    """
    n, horizon = len(s.ta), len(s.price)
    vehicle, slot = np.nonzero(
        (np.arange(horizon)[None, :] >= s.ta[:, None])
        & (np.arange(horizon)[None, :] < s.td[:, None])
    )
    m = len(vehicle)
    indices = np.arange(m)
    rows = np.r_[vehicle, n + slot, n + np.arange(horizon)]
    cols = np.r_[indices, indices, np.full(horizon, m)]
    data = np.r_[
        -s.efficiency * s.dt * s.power[vehicle], s.power[vehicle], -np.ones(horizon)
    ]
    mat = coo_matrix((data, (rows, cols)), shape=(n + horizon, m + 1)).tocsr()
    rhs = np.r_[-(s.target - s.arrival) * s.capacity, np.zeros(horizon)]
    objective = np.r_[np.zeros(m), 1.0]
    result = linprog(
        objective, A_ub=mat, b_ub=rhs, bounds=[(0, 1)] * m + [(0, None)], method="highs"
    )
    return (
        ("optimal", float(result.fun))
        if result.success
        else (("infeasible" if result.status == 2 else "unknown"), None)
    )


def state(
    s: Scenario, t: int, k: int, soc: Array, proposed: Array, limit: float
) -> Array:
    """Construct the notebook's 21-dimensional sequential decision state.

    :param s: Charging scenario.
    :param t: Slot index relative to the first arrival.
    :param k: Vehicle index in pile order.
    :param soc: SoC values at the start of the slot.
    :param proposed: Previously proposed actions in this slot.
    :param limit: Station limit in kW.
    :return: Normalized state vector.
    """
    n = len(s.ta)
    horizon = len(s.price) + 1
    active = (s.ta <= t) & (t < s.td)

    def aggregate(mask: NDArray[np.bool_]) -> list[float]:
        """Summarize the selected connected vehicles.

        :param mask: Vehicle selection.
        :return: Aggregate normalized features.
        """
        count = int(mask.sum())
        if not count:
            return [0.0] * 5
        cap = s.capacity[mask]
        return [
            float(cap.mean() / 40),
            float(s.power[mask].mean() / 8),
            float(soc[mask] @ cap / cap.sum()),
            float(s.target[mask] @ cap / cap.sum()),
            float(((s.td[mask] - t) / horizon).mean()),
        ]

    before = active & (np.arange(n) < k)
    after = active & (np.arange(n) > k)
    a = aggregate(before)
    c = aggregate(after)
    a.extend(
        [float(proposed[before] @ s.power[before] / limit), float(before.sum() / n)]
    )
    b = [
        k / n,
        s.capacity[k] / 40,
        s.power[k] / 8,
        soc[k],
        s.target[k],
        (s.td[k] - t) / horizon,
    ]
    c.append(float(after.sum() / n))
    return np.asarray(
        [s.price[t], 1 - 2 * t / horizon] + [1 - 2 * x for x in a + b + c],
        dtype=np.float64,
    )


def causal_state(
    s: Scenario, t: int, k: int, soc: Array, proposed: Array, limit: float
) -> Array:
    """Construct 21 features using only the currently connected fleet.

    :param s: Scenario on the fixed operational-day grid.
    :param t: Slot index on that grid.
    :param k: Current connected vehicle index.
    :param soc: Current state of charge.
    :param proposed: Earlier proposed actions in this slot.
    :param limit: Station power limit.
    :return: Causal normalized feature vector.
    """
    active = (s.ta <= t) & (t < s.td)
    indices = np.flatnonzero(active)
    if not active[k]:
        raise ValueError("Vehicle is not connected")
    rank = int(np.searchsorted(indices, k))
    count = len(indices)

    def aggregate(selected: NDArray[np.int64]) -> list[float]:
        """Summarize currently connected vehicles.

        :param selected: Selected vehicle indices.
        :return: Five normalized features.
        """
        if not len(selected):
            return [0.0] * 5
        cap = s.capacity[selected]
        return [float(cap.mean() / 40), float(s.power[selected].mean() / 8),
                float(soc[selected] @ cap / cap.sum()),
                float(s.target[selected] @ cap / cap.sum()),
                float(np.mean((s.td[selected] - t) / 40))]

    before, after = indices[:rank], indices[rank + 1:]
    a = aggregate(before) + [float(proposed[before] @ s.power[before] / limit), len(before) / count]
    b = [rank / count, s.capacity[k] / 40, s.power[k] / 8, soc[k], s.target[k], (s.td[k] - t) / 40]
    c = aggregate(after) + [len(after) / count]
    return np.asarray([s.price[t], 1 - 2 * t / 56] + [1 - 2 * x for x in a + b + c], dtype=np.float64)


def fixed_feasibility(s: Scenario, limit: float) -> tuple[str, float | None]:
    """Classify target feasibility and check the returned primal residual.

    :param s: Scenario.
    :param limit: Fixed station limit in kW.
    :return: Classification and maximum constraint residual for a feasible solution.
    """
    n, h = len(s.ta), len(s.price)
    vehicle, slot = np.nonzero(
        (np.arange(h)[None, :] >= s.ta[:, None])
        & (np.arange(h)[None, :] < s.td[:, None])
    )
    m = len(vehicle)
    ix = np.arange(m)
    mat = coo_matrix(
        (
            np.r_[-s.efficiency * s.dt * s.power[vehicle], s.power[vehicle]],
            (np.r_[vehicle, n + slot], np.r_[ix, ix]),
        ),
        shape=(n + h, m),
    ).tocsr()
    rhs = np.r_[-(s.target - s.arrival) * s.capacity, np.full(h, limit)]
    res = linprog(np.zeros(m), A_ub=mat, b_ub=rhs, bounds=(0, 1), method="highs")
    if res.success:
        residual = float(
            max(0.0, np.max(mat @ res.x - rhs), np.max(-res.x), np.max(res.x - 1))
        )
        return ("feasible" if residual <= 1e-6 else "unknown", residual)
    return ("infeasible", None) if res.status == 2 else ("unknown", None)


def urgency_allocation(
    urgency: Array, power: Array, remaining: Array, limit: float
) -> Array:
    """Allocate power by urgency with charger and remaining-energy caps.

    :param urgency: Per-vehicle urgency, zero for disconnected vehicles.
    :param power: Rated powers in kW.
    :param remaining: Remaining connected slots.
    :param limit: Available station power in kW.
    :return: Normalized charging actions.
    """
    cap = power * np.minimum(1.0, np.maximum(0.0, urgency * remaining))
    allocation = np.zeros_like(power)
    active = (urgency > 0) & (cap > 0)
    residual = limit
    while active.any() and residual > 1e-12:
        weights = np.where(active, urgency, 0.0)
        offer = residual * weights / weights.sum()
        room = cap - allocation
        saturated = active & (offer >= room - 1e-12)
        if not saturated.any():
            allocation += offer
            break
        allocation[saturated] += room[saturated]
        residual -= float(room[saturated].sum())
        active[saturated] = False
    return allocation / power


def load_scenario(path: str) -> Scenario:
    """Load a persisted generated scenario.

    :param path: Path to the scenario archive.
    :return: Loaded scenario.
    """
    with np.load(path) as d:
        return Scenario(
            d["ta"].copy(),
            d["td"].copy(),
            d["arrival"].copy(),
            d["target"].copy(),
            d["capacity"].copy(),
            d["power"].copy(),
            d["price"].copy(),
            float(d["efficiency"]),
            float(d["dt"]),
            float(d["beta"]),
        )
