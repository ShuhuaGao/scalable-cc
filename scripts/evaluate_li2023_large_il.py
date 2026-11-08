"""Evaluate all three frozen IL actors on the fixed large-fleet scenarios."""
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
from evcharging.core import generate
from evcharging.models import imitation_network
from evcharging.simulation import simulate
from run_li2023_r105 import OUT, RESULT


def main() -> None:
    """Compute IL large-fleet records and update the comparison summary.

    :return: None.
    """
    torch.set_num_threads(1)
    archived = OUT / 'source_archive/scripts/evaluate_li2023_large_il.py'
    if archived.exists() and archived.read_bytes() != Path(__file__).read_bytes():
        raise RuntimeError('Archived script differs')
    if not archived.exists():
        shutil.copy2(Path(__file__), archived)
    selection = json.loads((ROOT / 'artifacts/runs/training_repeats/selection.json').read_text(encoding='utf8'))
    models = [r for r in selection if r['method'] == 'IL']
    rows = []
    for item in models:
        model_path = Path(item['path'])
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('IL model hash mismatch')
        model = imitation_network()
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True)['model'])
        model.eval()
        def policy(x: np.ndarray, k: int, urgency: np.ndarray) -> float:
            """Apply the frozen IL actor to one causal state.

            :param x: Causal state.
            :param k: Pile index.
            :param urgency: Current urgency values.
            :return: Normalized action.
            """
            del k, urgency
            with torch.no_grad():
                return float(model(torch.tensor(x, dtype=torch.float32)[None, :])[0, 0])
        for n in (1000, 3000):
            for i in range(3):
                seed = 81000 + 10*n + i
                s = generate(seed, n, fixed_day=True)
                r = simulate(s, 4*n, policy, causal=True)
                cost, unfair = float(r['cost'].sum()), float(r['unfair'].sum())
                gap = np.maximum(s.target-r['soc'][s.td, np.arange(n)], 0)
                folder = OUT / 'evaluation' / f"IL_{item['training_seed']}"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / f'{n}_{seed}.npz', soc=r['soc'], final=r['final'], cost=r['cost'], unfair=r['unfair'])
                rows.append(dict(method='IL', training_seed=item['training_seed'], seed=seed,
                                 vehicles=n, capacity_kw=4*n, cost=cost, unfairness=unfair,
                                 weighted_objective=.7*cost+.3*unfair,
                                 failed_vehicles=int((gap > 1e-4).sum()), gap_kwh=float(gap@s.capacity),
                                 model_sha256=item['sha256']))
                print(f"IL {item['training_seed']} {n} {seed}", flush=True)
    path = RESULT / 'large_il_scenarios.csv'
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary_path = RESULT / 'summary.csv'
    with summary_path.open(newline='', encoding='utf8') as f:
        summary = [r for r in csv.DictReader(f) if not (r['method'] == 'IL' and int(r['vehicles']) in (1000, 3000))]
    for n in (1000, 3000):
        for metric in ('cost', 'unfairness', 'weighted_objective', 'failed_vehicles', 'gap_kwh'):
            means = [np.mean([r[metric] for r in rows if r['vehicles'] == n and r['training_seed'] == item['training_seed']]) for item in models]
            summary.append(dict(method='IL', vehicles=n, capacity_kw=4*n, metric=metric,
                                mean=float(np.mean(means)), run_sd=float(np.std(means, ddof=1))))
    with summary_path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    manifest_path = RESULT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for source in (Path(__file__), ROOT / 'src/evcharging/simulation.py', ROOT / 'src/evcharging/models.py', ROOT / 'artifacts/runs/training_repeats/selection.json'):
        manifest['sources_sha256'][source.relative_to(ROOT).as_posix()] = digest(source)
    manifest['il_model_sha256'] = {str(item['training_seed']): item['sha256'] for item in models}
    for filename in ('summary.csv', 'large_il_scenarios.csv'):
        manifest['results_sha256'][filename] = digest(RESULT / filename)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
