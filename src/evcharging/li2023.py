"""Building blocks for the Li et al. (2023) shared sequential actor.

ReLU and tanh are implementation conventions; the paper does not specify them.
"""
from collections import deque
from collections.abc import Callable
from typing import Any
import time

import numpy as np
from torch import nn
from .core import Scenario


def _network(inputs: int, bounded: bool) -> nn.Sequential:
    """Build the scalar network.

    :param inputs: Number of input features.
    :param bounded: Whether to apply tanh to the output.
    :return: Neural network.
    """
    layers = []
    for left, right in zip((inputs, 64, 32), (64, 32, 16), strict=True):
        layers.extend((nn.Linear(left, right), nn.ReLU()))
    layers.append(nn.Linear(16, 1))
    if bounded:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def actor() -> nn.Sequential:
    """Return the shared actor.

    :return: Nine-input network with bounded scalar output.
    """
    return _network(9, True)


def critic() -> nn.Sequential:
    """Create the critic.

    :return: Ten-input scalar action-value network.
    """
    return _network(10, False)


class PreviousEpisodeD:
    """Hold complete time-by-pile D arrays from the preceding episodes."""

    def __init__(self, episodes: int = 5):
        self.history = deque(maxlen=episodes)

    def add(self, values: np.ndarray) -> None:
        """Store a copy of a completed episode's D values.

        :param values: Time-by-pile D array.
        :return: None.
        """
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("D must be a finite time-by-pile array")
        if self.history and array.shape != self.history[0].shape:
            raise ValueError("D episode shapes must agree")
        self.history.append(array.copy())

    def mean(self) -> np.ndarray:
        """Calculate the previous-episode mean.

        :return: Mean time-by-pile D array once history is full.
        """
        if len(self.history) != self.history.maxlen:
            raise ValueError("previous-episode history is incomplete")
        return np.mean(self.history, axis=0)


class Replay:
    """Finite FIFO transition memory."""

    def __init__(self, capacity: int = 60000):
        self.items = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: float, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Store a transition with copied nine-element state arrays.

        :param state: Current nine-element state.
        :param action: Selected normalized action.
        :param reward: Scalar reward.
        :param next_state: Next nine-element state.
        :param done: Whether the vehicle departed or the day ended.
        :return: None.
        """
        s = np.asarray(state, dtype=np.float32)
        ns = np.asarray(next_state, dtype=np.float32)
        if s.shape != (9,) or ns.shape != (9,):
            raise ValueError("states must each have nine features")
        self.items.append((s.copy(), float(action), float(reward), ns.copy(), bool(done)))

    def sample(self, size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        """Sample distinct transitions uniformly.

        :param size: Number of transitions.
        :param rng: Random generator.
        :return: Arrays of transition columns.
        """
        indices = rng.choice(len(self.items), size=size, replace=False)
        return tuple(np.asarray(c) for c in zip(*(self.items[int(i)] for i in indices), strict=True))

    def __len__(self) -> int:
        """Return the current number of transitions.

        :return: Replay size.
        """
        return len(self.items)


def simulate(s: Scenario, limit: float, policy: Callable[[np.ndarray], float], *,
             d_history: np.ndarray | None = None, profile: dict | None = None) -> dict[str, Any]:
    """Simulate the adapted single-direction DEB and sequential actor.

    :param s: Fixed-day scenario.
    :param limit: Station power limit in kW.
    :param policy: Callable accepting a nine-element state and returning an action.
    :param d_history: Prior-five-episode mean D, or None for evaluation.
    :param profile: Optional dictionary receiving per-slot wall times.
    :return: Physical trajectories, training records, and diagnostics.
    """
    configured_piles, horizon = len(s.ta), len(s.price)
    n = configured_piles
    soc = np.zeros((horizon + 1, n))
    soc[0] = s.arrival
    power = np.zeros((horizon, n))
    shares = np.zeros_like(power)
    ds = np.zeros_like(power)
    states = np.zeros((horizon, n, 9))
    actions = np.zeros_like(power)
    local_lo = np.zeros_like(power)
    local_hi = np.zeros_like(power)
    hs = np.zeros_like(power)
    boundary_trace = np.zeros_like(power)
    rate_trace = np.zeros_like(power)
    active_all = np.zeros_like(power, dtype=bool)
    risk_power = np.minimum(s.power, limit / configured_piles)
    risk_start = s.td - (s.target - s.arrival) * s.capacity / (s.efficiency * risk_power * s.dt)
    boundary = np.maximum(s.arrival, s.target - risk_power * (s.td - s.ta) * s.dt * s.efficiency / s.capacity)
    initial_deficit = boundary - s.arrival
    rates = risk_power.copy()
    empty_count, empty_gap = 0, 0.0
    if profile is not None:
        profile['slot_seconds'] = []
    for t in range(horizon):
        if profile is not None:
            slot_start = time.perf_counter()
        active = (s.ta <= t) & (t < s.td)
        active_all[t] = active
        count = int(active.sum())
        urgency = np.where(active, np.maximum(s.target - soc[t], 0) / np.maximum((s.td - t) * s.dt, s.dt), 0)
        shares[t] = urgency / urgency.sum() if urgency.sum() else 0
        overlap = np.maximum(0., np.minimum(t + 1., s.td) - np.maximum(t, risk_start))
        on_boundary = active & (overlap > 0)
        if count:
            rates[on_boundary] = np.maximum(rates[on_boundary], np.minimum(s.power[on_boundary], limit / count))
        boundary[active] = np.minimum(s.target[active], boundary[active] + overlap[active] * s.dt * s.efficiency * rates[active] / s.capacity[active])
        boundary_trace[t], rate_trace[t] = boundary, rates
        lo = np.where(active, np.maximum(0., (boundary - soc[t]) * s.capacity / (s.efficiency * s.dt)), 0.)
        hi = np.where(active, np.minimum(s.power, np.maximum(0., (s.target - soc[t]) * s.capacity / (s.efficiency * s.dt))), 0.)
        local_lo[t], local_hi[t] = lo, hi
        pending = np.cumsum(lo[::-1])[::-1] - lo
        used = 0.
        for k in range(n):
            ds[t, k] = used / limit
            hs[t, k] = pending[k] / limit
            if not active[k]:
                continue
            upper = max(0., min(hi[k], limit - used - pending[k]))
            lower = lo[k]
            if lower > upper:
                empty_count += 1
                empty_gap += lower - upper
                lower = upper
            state = np.array([(7 + t * s.dt) / 14, s.price[t], shares[t, k], hs[t, k], ds[t, k], (s.td[k] - t) * s.dt / 14, soc[t, k], upper / s.power[k], lower / s.power[k]], dtype=np.float32)
            action = float(np.clip(policy(state), -1, 1))
            p = lower + (action + 1) * 0.5 * (upper - lower)
            states[t, k], actions[t, k], power[t, k] = state, action, p
            used += p
        soc[t + 1] = soc[t] + power[t] * s.efficiency * s.dt / s.capacity
        if profile is not None:
            profile['slot_seconds'].append(time.perf_counter() - slot_start)
    cost = power * s.price[:, None] * s.dt
    rewards = -(1 - .4) * cost - .4 * np.abs(cost - cost.sum(axis=1, keepdims=True) * shares)
    transitions = []
    next_empty_count, next_empty_gap = 0, 0.0
    if d_history is not None:
        if d_history.shape != ds.shape:
            raise ValueError("D history has incorrect shape")
        for t, k in np.argwhere(active_all):
            done = t + 1 == horizon or not active_all[t + 1, k]
            nxt = np.zeros(9, dtype=np.float32)
            if not done:
                nxt[:] = states[t + 1, k]
                nxt[4] = d_history[t + 1, k]
                upper = max(0., min(local_hi[t + 1, k], limit * (1 - nxt[3] - nxt[4])))
                if local_lo[t + 1, k] > upper:
                    next_empty_count += 1
                    next_empty_gap += local_lo[t + 1, k] - upper
                lower = min(local_lo[t + 1, k], upper)
                nxt[7:9] = upper / s.power[k], lower / s.power[k]
            transitions.append((states[t, k], actions[t, k], rewards[t, k], nxt, done))
    return dict(soc=soc, power=power, d=ds, shares=shares, reward=rewards,
                transitions=transitions, empty_count=empty_count, empty_gap=empty_gap,
                boundary=boundary_trace, boundary_rate=rate_trace,
                next_empty_count=next_empty_count, next_empty_gap=next_empty_gap,
                initial_deficit_count=int(np.count_nonzero(initial_deficit > 1e-12)),
                initial_deficit_soc=float(initial_deficit.sum()))
