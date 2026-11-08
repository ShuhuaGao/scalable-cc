"""Derive comparable objectives and adjacent beta trade-offs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/pressure21_v1/r77'


def main() -> None:
    """Compute descriptive metrics without running policy evaluation.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir
    source = OUT/'results.csv'
    with source.open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    groups = {}
    objectives = []
    for row in rows:
        beta, c, d = (float(row[k]) for k in ('beta', 'mean_cost', 'mean_unfairness'))
        key = (row['dataset'], int(row['vehicles']), int(row['capacity_kw']))
        groups.setdefault(key, {})[beta] = row
        objectives.append({'dataset': key[0], 'vehicles': key[1], 'capacity_kw': key[2], 'beta': beta, 'scenarios': int(row['scenarios']), 'cost': c, 'unfairness': d, 'common_objective_beta_0_3': .7*c+.3*d, 'own_weight_objective': (1-beta)*c+beta*d})
    trades, common = [], []
    for key, cases in groups.items():
        if set(cases) != {.1, .3, .5}:
            raise RuntimeError(f'Incomplete weights: {key}')
        values = {b: .7*float(r['mean_cost'])+.3*float(r['mean_unfairness']) for b, r in cases.items()}
        best = min(values, key=values.get)
        common.append({'dataset': key[0], 'vehicles': key[1], 'capacity_kw': key[2], 'minimum_beta': best, 'minimum_common_objective': values[best], 'beta_0_3_common_objective': values[.3], 'beta_0_3_above_minimum_percent': 100*(values[.3]/values[best]-1)})
        for a, b in ((.1, .3), (.3, .5)):
            x, y = cases[a], cases[b]
            ca, cb = float(x['mean_cost']), float(y['mean_cost'])
            da, db = float(x['mean_unfairness']), float(y['mean_unfairness'])
            trades.append({'dataset': key[0], 'vehicles': key[1], 'capacity_kw': key[2], 'from_beta': a, 'to_beta': b, 'cost_increase': cb-ca, 'cost_increase_percent': 100*(cb/ca-1), 'unfairness_reduction': da-db, 'unfairness_reduction_percent': 100*(1-db/da), 'unfairness_reduction_per_added_cost': (da-db)/(cb-ca), 'vehicle_met_percentage_point_change': float(y['vehicle_met_percent'])-float(x['vehicle_met_percent']), 'day_met_percentage_point_change': float(y['all_met_percent'])-float(x['all_met_percent']), 'gap_kwh_per_day_change': float(y['mean_gap_kwh_per_day'])-float(x['mean_gap_kwh_per_day']), 'gap_demand_percentage_point_change': float(y['gap_share_of_battery_demand_percent'])-float(x['gap_share_of_battery_demand_percent'])})
    output.mkdir(parents=True, exist_ok=True)
    for name in ('derived_objectives.csv', 'derived_tradeoffs.csv', 'derived_analysis.json'):
        if (output/name).exists():
            raise FileExistsError(output/name)
    for name, data in [('derived_objectives.csv', objectives), ('derived_tradeoffs.csv', trades)]:
        with (output/name).open('w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0]); writer.writeheader(); writer.writerows(data)
    payload = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'own_weight_note': 'Different beta values define different objective functions; their values must not be ranked directly.', 'common_objective_reference_beta': .3, 'common_objective_comparisons': common, 'objectives': objectives, 'adjacent_tradeoffs': trades}
    (output/'derived_analysis.json').write_text(json.dumps(payload, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
