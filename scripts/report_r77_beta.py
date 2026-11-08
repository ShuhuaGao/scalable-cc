"""Create beta-sensitivity results from the verified cohort summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/pressure21_v1/r77'


def main() -> None:
    """Report cohort values and relative changes without modifying evaluations.

    :return: None.
    """
    with (OUT/'cohort_summary.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r['vehicles']), float(r['beta'])))
    floats = ('beta', 'mean_cost', 'mean_unfairness', 'mean_cost_per_ev', 'mean_unfairness_per_ev', 'all_met_percent', 'vehicle_met_percent', 'mean_gap_kwh_per_day', 'gap_share_of_battery_demand_percent')
    for row in rows:
        for key in floats:
            row[key] = float(row[key])
        for key in ('vehicles', 'capacity_kw', 'scenarios'):
            row[key] = int(row[key])
    if len(rows) != 15:
        raise RuntimeError('Expected 15 cohort results')
    reference = {(r['dataset'], r['vehicles'], r['capacity_kw']): r for r in rows if r['beta'] == .3}
    changes = []
    for row in rows:
        if row['beta'] == .3:
            continue
        base = reference[row['dataset'], row['vehicles'], row['capacity_kw']]
        changes.append({'dataset': row['dataset'], 'vehicles': row['vehicles'], 'capacity_kw': row['capacity_kw'], 'beta': row['beta'], 'cost_change_percent': 100*(row['mean_cost']/base['mean_cost']-1), 'unfairness_change_percent': 100*(row['mean_unfairness']/base['mean_unfairness']-1)})
    for name in ('results.csv', 'results.json'):
        if (OUT/name).exists():
            raise FileExistsError(OUT/name)
    with (OUT/'results.csv').open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (OUT/'results.json').write_text(json.dumps({'cohorts': rows, 'relative_to_beta_0.3': changes, 'note': 'Single model per beta; no training-run uncertainty.'}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
