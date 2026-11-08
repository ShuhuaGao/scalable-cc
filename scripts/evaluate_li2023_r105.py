"""Evaluate frozen Li actors and reconstruct comparison summaries."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import generate, load_scenario
from evcharging.li2023 import actor, simulate
from run_li2023_r105 import OUT, RESULT, numpy_policy


def evaluate(seed: int, scenario_seed: int, n: int, limit: float) -> dict:
    """Evaluate one frozen actor and save its physical trajectory.

    :param seed: Training seed.
    :param scenario_seed: Evaluation scenario seed.
    :param n: Number of fixed charging piles.
    :param limit: Station limit in kW.
    :return: One scenario-level metrics record.
    """
    source = ROOT / 'artifacts/runs/ideal10h_v1/formal/inputs' / f'{scenario_seed}.npz'
    s = load_scenario(str(source)) if n == 200 else generate(scenario_seed, n, fixed_day=True)
    model_path = OUT / str(seed) / 'episode_6000.pt'
    model = actor()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if checkpoint['seed'] != seed or checkpoint['episode'] != 6000:
        raise ValueError('Unexpected actor checkpoint')
    model.load_state_dict(checkpoint['model'])
    r = simulate(s, limit, numpy_policy(model))
    p, soc = r['power'], r['soc']
    active = (s.ta[None, :] <= np.arange(len(s.price))[:, None]) & (np.arange(len(s.price))[:, None] < s.td[None, :])
    remaining = np.maximum(s.td[None, :] - np.arange(len(s.price))[:, None], 1)
    urgency = np.where(active, np.maximum(s.target - soc[:-1], 0) * s.capacity / (s.efficiency * s.power * s.dt * remaining), 0.)
    denom = urgency.sum(axis=1, keepdims=True)
    shares = np.divide(urgency, denom, out=np.zeros_like(urgency), where=denom > 0)
    final = p / s.power
    unfair = float(((final - final.sum(axis=1, keepdims=True) * shares) ** 2).sum())
    cost = float((p * s.price[:, None] * s.dt).sum())
    gap = np.maximum(s.target - soc[s.td, np.arange(n)], 0.)
    violation = float(max(np.maximum(p.sum(axis=1) - limit, 0).max(), np.maximum(-final, 0).max(),
                          np.maximum(final - 1, 0).max(), np.abs(final[~active]).max(initial=0),
                          np.maximum(.2 - soc, 0).max(), np.maximum(soc - 1, 0).max(),
                          np.maximum(soc[1:] - s.target, 0).max()))
    folder = OUT / 'evaluation' / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f'{n}_{scenario_seed}_{int(limit)}.npz'
    np.savez_compressed(dest, power=p, soc=soc, shares=r['shares'], reward=r['reward'],
                        boundary=r['boundary'], boundary_rate=r['boundary_rate'])
    return dict(method='Li2023', training_seed=seed, seed=scenario_seed, vehicles=n, capacity_kw=limit,
                cost=cost, unfairness=unfair, weighted_objective=.7 * cost + .3 * unfair,
                failed_vehicles=int((gap > 1e-4).sum()), gap_kwh=float(gap @ s.capacity),
                physical_max_violation=violation, empty_count=r['empty_count'], empty_gap=r['empty_gap'],
                initial_deficit_count=r['initial_deficit_count'], initial_deficit_soc=r['initial_deficit_soc'],
                li_reward=float(r['reward'].sum()), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())


def main() -> None:
    """Evaluate all fixed actors and summarize per-run means and between-run SD.

    :return: None.
    """
    torch.set_num_threads(1)
    RESULT.mkdir(parents=True, exist_ok=True)
    archived = OUT / 'source_archive/scripts/evaluate_li2023_r105.py'
    archived.parent.mkdir(parents=True, exist_ok=True)
    if archived.exists() and archived.read_bytes() != Path(__file__).read_bytes():
        raise RuntimeError('Evaluation source differs from archive')
    if not archived.exists():
        shutil.copy2(Path(__file__), archived)
    rows = []
    for seed in (33001, 33002, 33003):
        for scenario_seed in range(65000, 65050):
            for limit in (800, 600):
                rows.append(evaluate(seed, scenario_seed, 200, limit))
        for n in (1000, 3000):
            for index in range(3):
                rows.append(evaluate(seed, 81000 + 10 * n + index, n, 4 * n))
        print(f'evaluated {seed}', flush=True)
    with (RESULT / 'scenarios.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metrics = ('cost', 'unfairness', 'weighted_objective', 'failed_vehicles', 'gap_kwh',
               'physical_max_violation', 'empty_count', 'empty_gap', 'initial_deficit_count', 'li_reward')
    summary = []
    for n, limit in ((200, 600), (200, 800), (1000, 4000), (3000, 12000)):
        group = [r for r in rows if r['vehicles'] == n and r['capacity_kw'] == limit]
        for metric in metrics:
            means = [np.mean([r[metric] for r in group if r['training_seed'] == seed]) for seed in (33001, 33002, 33003)]
            summary.append(dict(method='Li2023', vehicles=n, capacity_kw=limit, metric=metric,
                                mean=float(np.mean(means)), run_sd=float(np.std(means, ddof=1))))
    il_path = ROOT / 'results/training_repeats/scenarios.csv'
    with il_path.open(newline='', encoding='utf8') as f:
        il = [r for r in csv.DictReader(f) if r['method'] == 'IL']
    for limit in (600, 800):
        group = [r for r in il if int(r['capacity_kw']) == limit]
        for metric in ('cost', 'unfairness', 'weighted_objective', 'failed_vehicles', 'gap_kwh', 'physical_max_violation'):
            seeds = sorted({r['training_seed'] for r in group})
            if len(seeds) != 3 or any(sum(r['training_seed'] == seed for r in group) != 50 for seed in seeds):
                raise ValueError('IL comparison must contain three runs and 50 scenarios per run')
            means = [np.mean([float(r[metric]) for r in group if r['training_seed'] == seed]) for seed in seeds]
            summary.append(dict(method='IL', vehicles=200, capacity_kw=limit, metric=metric,
                                mean=float(np.mean(means)), run_sd=float(np.std(means, ddof=1))))
    scale_path = ROOT / 'results/ideal10h_v1/scalability_scenarios.csv'
    with scale_path.open(newline='', encoding='utf8') as f:
        scale_il = [r for r in csv.DictReader(f) if r['method'] == 'IL']
    for n in (1000, 3000):
        group = [r for r in scale_il if int(r['vehicles']) == n]
        if {int(r['seed']) for r in group} != {81000 + 10 * n + i for i in range(3)}:
            raise ValueError('Unexpected IL scalability scenarios')
        for metric in ('cost', 'unfairness', 'weighted_objective', 'failed_vehicles'):
            def value(row: dict) -> float:
                """Read a comparable IL metric from one archived scenario.

                :param row: Archived IL scenario record.
                :return: Scalar metric.
                """
                if metric == 'weighted_objective':
                    return .7 * float(row['cost']) + .3 * float(row['unfairness'])
                return float(row['failed' if metric == 'failed_vehicles' else metric])
            summary.append(dict(method='IL', vehicles=n, capacity_kw=4*n, metric=metric,
                                mean=float(np.mean([value(r) for r in group])), run_sd=''))
    with (RESULT / 'summary.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sources = [Path(__file__), ROOT / 'src/evcharging/li2023.py', ROOT / 'src/evcharging/core.py',
               ROOT / 'scripts/run_li2023_r105.py', il_path, scale_path]
    inputs = [ROOT / 'artifacts/runs/ideal10h_v1/formal/inputs' / f'{seed}.npz' for seed in range(65000, 65050)]
    manifest = {'sources_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
                'inputs_sha256': {p.name: digest(p) for p in inputs},
                'models_sha256': {str(seed): digest(OUT / str(seed) / 'episode_6000.pt') for seed in (33001, 33002, 33003)},
                'results_sha256': {name: digest(RESULT / name) for name in ('scenarios.csv', 'summary.csv')}}
    (RESULT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
