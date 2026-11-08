"""Summarize matched beta-sensitivity policy evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evaluate_pressure21 import ROOT, digest

SOURCE = ROOT / 'artifacts/runs/pressure21_v1/evaluation/r77_beta'
OUT = ROOT / 'results/pressure21_v1/r77'


def main() -> None:
    """Verify paired input identities and write cohort and paired summaries.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir
    all_rows = {}
    sources = {}
    for beta in (0.1, 0.3, 0.5):
        folder = SOURCE / f'beta_{beta:.2f}'
        path = folder / 'scenarios.csv'
        complete = json.loads((folder/'complete.json').read_text(encoding='utf8'))
        if complete['scenarios_sha256'] != digest(path):
            raise RuntimeError(f'Changed scenario summary: {path}')
        with path.open(encoding='utf8', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 79 or complete['records'] != 79:
            raise RuntimeError('Incomplete evaluation')
        keyed = {(r['dataset'], r['seed'], r['vehicles'], r['capacity_kw']): r for r in rows}
        if len(keyed) != 79 or any(float(r['beta']) != beta for r in rows):
            raise RuntimeError('Duplicate case or incorrect beta')
        all_rows[beta] = keyed
        sources[str(path)] = digest(path)
    ref = all_rows[.3]
    for beta, rows in all_rows.items():
        if rows.keys() != ref.keys() or any(rows[k]['input_sha256'] != ref[k]['input_sha256'] for k in ref):
            raise RuntimeError(f'Unmatched input at beta={beta}')
    summaries, pairs = [], []
    for beta, cases in all_rows.items():
        groups = defaultdict(list)
        for row in cases.values():
            groups[(row['dataset'], row['vehicles'], row['capacity_kw'])].append(row)
        for (dataset, vehicles, capacity), rows in groups.items():
            n = int(vehicles)
            summaries.append({'beta': beta, 'dataset': dataset, 'vehicles': n, 'capacity_kw': int(capacity), 'scenarios': len(rows), 'mean_cost': mean(float(r['cost']) for r in rows), 'mean_unfairness': mean(float(r['unfairness']) for r in rows), 'mean_cost_per_ev': mean(float(r['cost_per_ev']) for r in rows), 'mean_unfairness_per_ev': mean(float(r['unfairness_per_ev']) for r in rows), 'all_met_percent': 100*mean(int(r['all_met']) for r in rows), 'vehicle_met_percent': 100*(1-sum(int(r['failed_vehicles']) for r in rows)/(len(rows)*n)), 'mean_gap_kwh_per_day': mean(float(r['gap_kwh']) for r in rows), 'gap_share_of_battery_demand_percent': 100*sum(float(r['gap_kwh']) for r in rows)/sum(float(r['battery_demand_kwh']) for r in rows)})
        if beta == .3:
            continue
        for key, row in cases.items():
            base = ref[key]
            pair = {'beta': beta, 'reference_beta': .3, 'dataset': key[0], 'seed': int(key[1]), 'vehicles': int(key[2]), 'capacity_kw': int(key[3]), 'input_sha256': row['input_sha256']}
            for metric in ('cost', 'unfairness', 'cost_per_ev', 'unfairness_per_ev', 'all_met', 'vehicle_met_percent', 'gap_kwh', 'gap_demand_percent'):
                pair[metric+'_difference'] = float(row[metric])-float(base[metric])
            pairs.append(pair)
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('cohort_summary.csv', 'paired_scenarios.csv', 'manifest.json'):
        if (output/filename).exists():
            raise FileExistsError(output/filename)
    for filename, rows in [('cohort_summary.csv', summaries), ('paired_scenarios.csv', pairs)]:
        with (output/filename).open('w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (output/'manifest.json').write_text(json.dumps({'sources_sha256': sources, 'single_training_seed': 11002, 'paired_reference_beta': .3, 'summary_script_sha256': digest(Path(__file__))}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
