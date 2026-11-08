"""Solve one independent perfect-information 10-hour rolling expert scenario."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from evcharging.core import Scenario, causal_state, generate, nominal_price
from PYSOLVER.ev import EVData
from PYSOLVER.knitro_env import checkout_license
from PYSOLVER.nlp2a import build_model, solve

SOURCE_FILES = ("scripts/expert_ideal10h.py", "src/evcharging/core.py", "PYSOLVER/nlp2a.py", "PYSOLVER/ev.py")
SOURCE_SHA256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def input_digest(s: Scenario, limit: float) -> str:
    """Hash all numerical scenario inputs and the station limit canonically.

    :param s: Full fixed-day scenario.
    :param limit: Station power limit in kW.
    :return: SHA-256 digest of names, shapes, types and values.
    """
    digest = hashlib.sha256()
    values = {name: getattr(s, name) for name in ('ta', 'td', 'arrival', 'target', 'capacity', 'power', 'price', 'efficiency', 'dt', 'beta')}
    values['limit'] = limit
    for name, value in values.items():
        array = np.ascontiguousarray(value, dtype='<i8' if name in ('ta', 'td') else '<f8')
        digest.update(name.encode('ascii') + b'\0')
        digest.update(str(array.shape).encode('ascii') + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def extended_price(seed: int, s: Scenario) -> np.ndarray:
    """Extend the unchanged daily prices with an independent deterministic stream.

    :param seed: Scenario seed.
    :param s: Fixed-day scenario.
    :return: Realized prices for the day and the next 39 slots.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 104010]))
    nominal = nominal_price(56, 39)
    return np.concatenate((s.price, rng.normal(nominal, 0.1 * nominal)))


def window_inputs(s: Scenario, t: int, soc: np.ndarray, prices: np.ndarray) -> tuple[np.ndarray, EVData, np.ndarray, np.ndarray]:
    """Construct a perfect-information window with actual departures and prices.

    :param s: Fixed-day scenario.
    :param t: Current slot.
    :param soc: Current SoC for all vehicles.
    :param prices: Extended realized price series.
    :return: Window vehicle indices, data, prices, and terminal references.
    """
    idx = np.flatnonzero((s.ta < t + 40) & (s.td > t))
    ta = np.maximum(s.ta[idx] - t, 0)
    td = s.td[idx] - t
    initial = np.where(s.ta[idx] <= t, soc[idx], s.arrival[idx])
    evs = EVData(ta, td, initial, s.target[idx], s.capacity[idx], np.full(len(idx), s.efficiency), np.full(len(idx), 0.2), np.ones(len(idx)), s.power[idx])
    fraction = np.minimum((40 - ta) / (td - ta), 1.0)
    reference = initial + fraction * (s.target[idx] - initial)
    rho = prices[t:t + 40]
    if len(rho) != 40:
        raise ValueError('Price series does not cover the full window')
    return idx, evs, rho, reference


def run(seed: int, n: int, limit: float, output: Path, starts: int, seconds: float) -> None:
    """Solve and save a rolling expert trajectory and each window's status.

    :param seed: Scenario seed.
    :param n: Number of vehicles.
    :param limit: Station power limit in kW.
    :param output: New output path without extension.
    :param starts: Multistart count per window.
    :param seconds: Solver time budget per window.
    :return: None.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.with_suffix('.npz').exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'Existing expert output: {output}')
    s = generate(seed, n, fixed_day=True)
    prices = extended_price(seed, s)
    soc = np.zeros((57, n)); soc[0] = s.arrival
    actions = np.zeros((56, n)); xs = []; ys = []; records = []; reports = []
    report = {'seed': seed, 'n': n, 'limit': limit, 'horizon_slots': 40, 'starts': starts, 'seconds_per_window': seconds, 'complete': False, 'windows': reports, 'schema': 'ideal10h-perfect-information-v1', 'source_sha256': SOURCE_SHA256, 'input_sha256': input_digest(s, limit), 'extended_price_sha256': hashlib.sha256(prices.astype('<f8').tobytes()).hexdigest(), 'price_extension': 'independent seeded stream; nominal final daily tariff with 10 percent noise, no arrivals after operating day'}
    previous = np.zeros((n, 40))
    started = time.perf_counter()
    try:
        license = checkout_license()
        try:
            for t in range(56):
                idx, evs, rho, ref = window_inputs(s, t, soc[t], prices)
                if len(idx) == 0:
                    soc[t + 1] = soc[t]
                    continue
                remain = s.td[idx] - t
                with build_model(evs, rho, s.dt, beta=s.beta, Plim=limit, license=license, terminal_reference=ref, window_start=0, initial_actions=np.column_stack((previous[idx, 1:], np.zeros(len(idx))))) as model:
                    result = solve(model, {'ms_enable': int(starts > 1), 'ms_maxsolves': starts, 'maxtime': seconds, 'ms_numthreads': 1, 'numthreads': 1, 'outlev': 0, 'feastol': 1e-8, 'feastol_abs': 1e-8})
                a = result.pop('a')
                info = {key: result[key] for key in ('status', 'status_name', 'abs_feas_error', 'abs_opt_error', 'solve_time_wall', 'iterations')}
                info.update(slot=t, active=len(idx), multistart_completed=len(result['multistart_solves']), multistart_records=result['multistart_solves']); reports.append(info)
                if a is None or not np.isfinite(a).all():
                    raise RuntimeError(f'No finite solution at slot {t}')
                end = np.minimum(remain, 40)
                predicted = evs.SoCa[:, None] + np.cumsum(a * (s.power[idx] * s.efficiency * s.dt / s.capacity[idx])[:, None], axis=1)
                endpoint = predicted[np.arange(len(idx)), end - 1]
                error = max(float(np.max(np.maximum(-a, 0))), float(np.max(np.abs(a[np.arange(40)[None, :] < evs.ta[:, None]]), initial=0)), float(np.max(np.maximum(a - 1, 0))), float(np.max(np.maximum((a * s.power[idx, None]).sum(axis=0) - limit, 0))), float(np.max(np.maximum(predicted - s.target[idx, None], 0))), float(np.max(np.maximum(ref - endpoint, 0))), float(np.max(np.abs(endpoint[remain <= 40] - s.target[idx][remain <= 40]), initial=0)))
                info['checked_residual'] = error
                if error > 1e-4:
                    raise RuntimeError(f'Infeasible solution at slot {t}: residual {error}')
                previous.fill(0.0)
                previous[idx] = a
                active_idx = idx[s.ta[idx] <= t]
                actions[t, active_idx] = a[s.ta[idx] <= t, 0]
                for k in active_idx:
                    xs.append(causal_state(s, t, int(k), soc[t], actions[t], limit))
                    ys.append(actions[t, k]); records.append((t, int(k)))
                soc[t + 1] = soc[t] + actions[t] * s.power * s.efficiency * s.dt / s.capacity
                print(f'seed={seed} slot={t} active={len(idx)} solve={info["solve_time_wall"]:.2f}s residual={error:.2g}', flush=True)
            report['complete'] = True
        finally:
            license.release()
    except Exception as exc:
        report['error'] = repr(exc)
    report['elapsed_seconds'] = time.perf_counter() - started
    np.savez_compressed(output.with_suffix('.npz'), x=np.asarray(xs).reshape(-1, 21), y=np.asarray(ys), records=np.asarray(records, dtype=int).reshape(-1, 2), soc=soc, actions=actions, ta=s.ta, td=s.td, arrival=s.arrival, target=s.target, capacity=s.capacity, power=s.power, price=s.price, price_extended=prices, limit=limit, efficiency=s.efficiency, dt=s.dt, beta=s.beta)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    if not report['complete']:
        raise RuntimeError(report.get('error', 'Incomplete expert scenario'))


def main() -> None:
    """Parse arguments and solve one scenario.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=62000)
    parser.add_argument('--vehicles', type=int, default=10)
    parser.add_argument('--capacity-per-ev', type=float, default=4.0)
    parser.add_argument('--starts', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/runs/ideal10h_v1/pilot/seed_62000')
    args = parser.parse_args()
    run(args.seed, args.vehicles, args.vehicles * args.capacity_per_ev, args.output, args.starts, args.seconds)

if __name__ == '__main__':
    main()
