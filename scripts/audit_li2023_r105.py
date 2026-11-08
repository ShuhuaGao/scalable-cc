"""Audit recorded Li intervals and evaluate Li's criterion for both methods."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import generate, load_scenario
from run_li2023_r105 import OUT, RESULT


def metrics(s, soc: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calculate Li's cost and absolute cost deviation criterion.

    :param s: Scenario.
    :param soc: SoC trajectory.
    :param p: Charging power in kW.
    :return: Cost and total positive loss corresponding to negative reward.
    """
    t = np.arange(len(s.price))[:, None]
    active = (s.ta <= t) & (t < s.td)
    urgency = np.where(active, np.maximum(s.target - soc[:-1], 0) / np.maximum((s.td - t)*s.dt, s.dt), 0)
    denom = urgency.sum(axis=1, keepdims=True)
    share = np.divide(urgency, denom, out=np.zeros_like(urgency), where=denom > 0)
    cost = p * s.price[:, None] * s.dt
    loss = .6*cost + .4*np.abs(cost - cost.sum(axis=1, keepdims=True)*share)
    return float(cost.sum()), float(loss.sum())


def main() -> None:
    """Audit saved trajectories and aggregate run means and sample SD.

    :return: None.
    """
    archived = OUT / 'source_archive/scripts/audit_li2023_r105.py'
    if archived.exists() and archived.read_bytes() != Path(__file__).read_bytes():
        raise RuntimeError('Archived audit script differs')
    if not archived.exists():
        shutil.copy2(Path(__file__), archived)
    rows = []
    scale_inputs = {}
    for n, limits, scenario_seeds in ((200, (600, 800), range(65000, 65050)),
                                      (1000, (4000,), range(91000, 91003)),
                                      (3000, (12000,), range(111000, 111003))):
        for scenario_seed in scenario_seeds:
            s = load_scenario(str(ROOT / 'artifacts/runs/ideal10h_v1/formal/inputs' / f'{scenario_seed}.npz')) if n == 200 else generate(scenario_seed, n, fixed_day=True)
            if n != 200:
                source = ROOT / 'artifacts/runs/ideal10h_v1/scalability' / f'input_{n}_{scenario_seed}.npz'
                with np.load(source) as stored:
                    mismatches = [key for key, value in vars(s).items() if not np.array_equal(stored[key], np.asarray(value))]
                if mismatches:
                    raise RuntimeError(f'Scalability input mismatch: {source}: {mismatches}')
                scale_inputs[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
            t = np.arange(len(s.price))[:, None]
            active = (s.ta <= t) & (t < s.td)
            for limit in limits:
                for method, seeds in (('Li2023', (33001, 33002, 33003)), ('IL', (11002, 11003, 11004))):
                    for seed in seeds:
                        if method == 'Li2023':
                            path = OUT / 'evaluation' / str(seed) / f'{n}_{scenario_seed}_{limit}.npz'
                        elif n == 200:
                            path = ROOT / 'artifacts/runs/training_repeats/formal' / f'IL_{seed}_{scenario_seed}_{limit}.npz'
                        else:
                            path = OUT / 'evaluation' / f'IL_{seed}' / f'{n}_{scenario_seed}.npz'
                        with np.load(path) as f:
                            soc = f['soc']; p = f['power'] if method == 'Li2023' else f['final']*s.power
                            cost, loss = metrics(s, soc, p)
                            row = dict(method=method, training_seed=seed, vehicles=n, scenario_seed=scenario_seed,
                                       capacity_kw=limit, cost=cost, li_cost_fair_loss=loss,
                                       station_overload_kw=float(np.maximum(p.sum(axis=1)-limit, 0).max()),
                                       pile_overload_kw=float(np.maximum(p-s.power, 0).max()),
                                       negative_power_kw=float(np.maximum(-p, 0).max()),
                                       inactive_power_kw=float(np.abs(p[~active]).max(initial=0)),
                                       soc_over_target=float(np.maximum(soc[1:]-s.target, 0).max()),
                                       soc_below_arrival=float(np.maximum(s.arrival-soc, 0).max()))
                            if method == 'Li2023':
                                lo = np.where(active, np.maximum(0., (f['boundary']-soc[:-1])*s.capacity/(s.efficiency*s.dt)), 0.)
                                hi = np.where(active, np.minimum(s.power, np.maximum(0., (s.target-soc[:-1])*s.capacity/(s.efficiency*s.dt))), 0.)
                                pending = np.cumsum(lo[:, ::-1], axis=1)[:, ::-1]-lo
                                previous = np.cumsum(p, axis=1)-p
                                upper = np.maximum(0., np.minimum(hi, limit-previous-pending))
                                gap = np.where(active, np.maximum(lo-upper, 0.), 0.)
                                row.update(strict_empty=int((gap > 0).sum()), material_empty=int((gap > 1e-8).sum()),
                                           material_empty_kw=float(gap[gap > 1e-8].sum()), decisions=int(active.sum()))
                            else:
                                row.update(strict_empty='', material_empty='', material_empty_kw='', decisions=int(active.sum()))
                            rows.append(row)
    path = RESULT / 'audit_scenarios.csv'
    with path.open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = []
    for method in ('IL', 'Li2023'):
        for n, limit in ((200, 600), (200, 800), (1000, 4000), (3000, 12000)):
            group = [r for r in rows if r['method'] == method and r['vehicles'] == n and r['capacity_kw'] == limit]
            for metric in (('li_cost_fair_loss', 'material_empty', 'material_empty_rate') if method == 'Li2023' else ('li_cost_fair_loss',)):
                means = []
                for seed in sorted({r['training_seed'] for r in group}):
                    subset = [r for r in group if r['training_seed'] == seed]
                    vals = [(r['material_empty']/r['decisions'] if r['decisions'] else 0) if metric == 'material_empty_rate' else r[metric] for r in subset]
                    means.append(float(np.mean(vals)))
                summary.append(dict(method=method, vehicles=n, capacity_kw=limit, metric=metric,
                                    mean=float(np.mean(means)), run_sd=float(np.std(means, ddof=1))))
            for metric in ('station_overload_kw', 'pile_overload_kw', 'negative_power_kw',
                           'inactive_power_kw', 'soc_over_target', 'soc_below_arrival'):
                summary.append(dict(method=method, vehicles=n, capacity_kw=limit, metric='max_' + metric,
                                    mean=max(r[metric] for r in group), run_sd=''))
    with (RESULT / 'audit_summary.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    manifest_path = RESULT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest['sources_sha256'][str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    manifest['verified_scale_input_sha256'] = scale_inputs
    for filename in ('audit_scenarios.csv', 'audit_summary.csv'):
        manifest['results_sha256'][filename] = digest(RESULT / filename)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
