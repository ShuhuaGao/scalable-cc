"""Evaluate one provenance-bound beta policy on the matched R7.7 cohort."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import torch

from evaluate_pressure21 import ARRAYS, CONFIG, ROOT, digest, load_case
from evcharging.models import imitation_network
from evcharging.simulation_schema import simulate

OUT = ROOT / 'artifacts/runs/pressure21_v1/evaluation/r77_beta'
BASELINE_SHA256 = '9c8b1c1e9a5d4e1766fb139789591a86d69f03a9fadb31fef81f23fdb5c58c7f'


def verify_provenance(beta: float, model: Path, provenance: Path) -> None:
    """Bind the checkpoint to its independent training and expert records.

    :param beta: Expert objective weight.
    :param model: Continuation checkpoint.
    :param provenance: Training summary.
    :return: None.
    """
    summary = json.loads(provenance.read_text(encoding='utf8'))
    model_hash = digest(model)
    if summary.get('variant') != 'pressure' or summary.get('seed') != 11002 or summary.get('continuation_sha256') != model_hash:
        raise RuntimeError('Training summary does not identify this checkpoint')
    if beta == .3:
        if model_hash != BASELINE_SHA256:
            raise RuntimeError('Beta 0.3 is not the frozen original model')
        return
    if summary.get('beta') != beta or summary.get('best_sha256') != model_hash:
        raise RuntimeError('Training summary beta or best checkpoint mismatch')
    run = ROOT / f'artifacts/runs/r77/beta_{int(beta*10):02d}'
    if model.resolve() != (run/'il_11002/il_model/best.pt').resolve() or provenance.resolve() != (run/'il_11002/il_model/summary.json').resolve():
        raise RuntimeError('Unexpected training output location')
    for name, relative in [('expert_manifest_sha256', 'experts/manifest.json'), ('expert_audit_sha256', 'experts/audit_complete.json'), ('cache_manifest_sha256', 'expert_cache/manifest.json')]:
        if digest(run/relative) != summary.get(name):
            raise RuntimeError(f'Training provenance mismatch: {name}')


def evaluate(task: tuple) -> dict:
    """Simulate and audit one case, retaining its complete physical trajectory.

    :param task: Case identity, model information and output directory.
    :return: Scenario-level metrics and provenance.
    """
    dataset, seed, n, limit, beta, model_path, model_hash, out = task
    torch.set_num_threads(1)
    cfg = json.loads(CONFIG.read_text(encoding='utf8'))
    scenario, input_path, input_hash = load_case(dataset, seed, n, cfg)
    model_path = Path(model_path)
    if digest(model_path) != model_hash:
        raise RuntimeError('Checkpoint hash mismatch')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('variant'), checkpoint.get('seed'), checkpoint.get('stage')) != ('pressure', 11002, 'continuation'):
        raise RuntimeError('Checkpoint identity mismatch')
    model = imitation_network()
    model.load_state_dict(checkpoint['model'])
    model.eval()

    def policy(x: np.ndarray, k: int, urgency: np.ndarray) -> float:
        """Return a policy action.

        :param x: Causal state.
        :param k: Vehicle index.
        :param urgency: Fleet urgency.
        :return: Normalized charging action.
        """
        del k, urgency
        with torch.no_grad():
            return float(model(torch.tensor(x, dtype=torch.float32)[None, :])[0, 0])

    result = simulate(scenario, limit, policy, state_schema='pressure21_v1')
    a, soc = result['final'], result['soc']
    t = np.arange(len(scenario.price))[:, None]
    active = (scenario.ta[None, :] <= t) & (t < scenario.td[None, :])
    remaining = np.maximum(scenario.td - t, 1)
    urgency = np.where(active, np.maximum(scenario.target - soc[:-1], 0) * scenario.capacity / (scenario.efficiency * scenario.power * scenario.dt * remaining), 0)
    share = np.divide(urgency, urgency.sum(axis=1, keepdims=True), out=np.zeros_like(urgency), where=urgency.sum(axis=1, keepdims=True) > 0)
    cost = a * scenario.power * scenario.price[:, None] * scenario.dt
    unfair = (a - a.sum(axis=1, keepdims=True) * share) ** 2
    checks = {'cost': np.abs(cost-result['cost']).max(), 'unfair': np.abs(unfair-result['unfair']).max(), 'dynamics': np.abs(soc[1:]-soc[:-1]-a*scenario.power*scenario.efficiency*scenario.dt/scenario.capacity).max(), 'station': np.maximum(a@scenario.power-limit, 0).max(), 'lower_action': np.maximum(-a, 0).max(), 'upper_action': np.maximum(a-1, 0).max(), 'disconnected': np.abs(a[~active]).max(initial=0), 'initial': np.abs(soc[0]-scenario.arrival).max(), 'lower_soc': np.maximum(.2-soc, 0).max(), 'upper_soc': np.maximum(soc-1, 0).max(), 'overcharge': np.maximum(soc-scenario.target, 0).max()}
    residual = float(max(checks.values()))
    if not np.isfinite(residual) or residual > cfg['physical_tolerance']:
        raise RuntimeError(f'Physical audit failed: {task}: {checks}')
    gap = np.maximum(scenario.target-soc[scenario.td, np.arange(n)], 0)
    failed = int((gap > cfg['soc_tolerance']).sum())
    gap_kwh = float(gap @ scenario.capacity)
    demand = float((scenario.target-scenario.arrival) @ scenario.capacity)
    c, d = float(cost.sum()), float(unfair.sum())
    path = Path(out) / f'{dataset}_{n}_{seed}_{limit}.npz'
    np.savez_compressed(path, **{key: result[key] for key in ARRAYS})
    row = {'beta': beta, 'dataset': dataset, 'seed': seed, 'vehicles': n, 'capacity_kw': limit, 'cost': c, 'unfairness': d, 'cost_per_ev': c/n, 'unfairness_per_ev': d/n, 'all_met': int(failed == 0), 'failed_vehicles': failed, 'vehicle_met_percent': 100*(1-failed/n), 'gap_kwh': gap_kwh, 'battery_demand_kwh': demand, 'gap_demand_percent': 100*gap_kwh/demand, 'physical_max_residual': residual, 'input_path': str(input_path), 'input_sha256': input_hash, 'model_sha256': model_hash, 'trajectory_path': str(path), 'trajectory_sha256': digest(path)}
    path.with_suffix('.json').write_text(json.dumps(row, indent=2), encoding='utf8')
    return row


def main() -> None:
    """Run one specified policy and retain a non-overwriting manifest.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1–4 workers')
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    beta = float(manifest['beta'])
    if beta not in (0.1, 0.3, 0.5) or manifest['training_seed'] != 11002:
        raise ValueError('Model is outside the frozen protocol')
    model = Path(manifest['checkpoint'])
    if digest(model) != manifest['checkpoint_sha256']:
        raise RuntimeError('Checkpoint hash mismatch')
    provenance = Path(manifest['training_provenance'])
    if digest(provenance) != manifest['training_provenance_sha256']:
        raise RuntimeError('Training provenance hash mismatch')
    verify_provenance(beta, model, provenance)
    cfg = json.loads(CONFIG.read_text(encoding='utf8'))
    out = OUT / f'beta_{beta:.2f}'
    out.mkdir(parents=True, exist_ok=False)
    cases = [('small', seed, 10, 40) for seed in cfg['small_seeds']]
    cases += [('scale', 81000+n*10+i, n, 4*n) for n in (100, 1000, 3000) for i in range(3)]
    cases += [('formal', seed, 200, 600) for seed in cfg['formal_seeds']]
    tasks = [(dataset, seed, n, limit, beta, str(model), manifest['checkpoint_sha256'], str(out)) for dataset, seed, n, limit in cases]
    record = {'model_manifest': manifest, 'model_manifest_sha256': digest(args.manifest), 'config_sha256': digest(CONFIG), 'evaluation_source_sha256': digest(Path(__file__)), 'workers': args.workers, 'cases': len(tasks)}
    (out/'manifest.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, tasks))
    if len(rows) != 79:
        raise RuntimeError('Incomplete cohort')
    with (out/'scenarios.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (out/'complete.json').write_text(json.dumps({'records': len(rows), 'scenarios_sha256': digest(out/'scenarios.csv')}, indent=2), encoding='utf8')
    print(f'beta={beta}: {len(rows)} scenarios complete', flush=True)


if __name__ == '__main__':
    main()
