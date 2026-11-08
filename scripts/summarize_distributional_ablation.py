"""Audit and summarize paired 21- and 27-feature evaluations."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import load_scenario

BASE = ROOT / 'artifacts/runs/pressure21_v1/evaluation'
DEFAULT_RUN_ID = 'r102_distributional_v1'
SEEDS = (11002, 11003, 11004)
METRICS = ('cost', 'unfairness', 'weighted_objective', 'all_met',
           'vehicle_met_percent', 'gap_kwh', 'daily_max_dk')


def digest(path: Path) -> str:
    """Hash a file.

    :param path: File to hash.
    :return: SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    """Read rows from a CSV file.

    :param path: Source CSV path.
    :return: String-valued rows.
    """
    with path.open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    """Write homogeneous rows to a CSV file.

    :param path: Output CSV path.
    :param rows: Nonempty rows with identical fields.
    :return: None.
    """
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def selected(row: dict) -> bool:
    """Select the four prescribed cohorts.

    :param row: Archived scenario row.
    :return: Whether the row belongs to a prescribed cohort.
    """
    n, cap = int(row['vehicles']), int(row['capacity_kw'])
    return (n == 200 and cap in (600, 800)) or (n in (1000, 3000) and cap == 4*n)


def audit(row: dict, method: str) -> dict:
    """Recompute a trajectory's daily metrics and check its archived data.

    :param row: Archived scenario metrics and source paths.
    :param method: Feature-set label.
    :return: Audited daily metrics and provenance.
    """
    n, day, cap, train = (int(row[k]) for k in ('vehicles', 'seed', 'capacity_kw', 'training_seed'))
    inp, traj = Path(row['input_path']), Path(row['trajectory_path'])
    if digest(inp) != row['input_sha256'] or digest(traj) != row['trajectory_sha256']:
        raise ValueError(f'Hash mismatch: {traj}')
    s = load_scenario(str(inp))
    with np.load(traj) as data:
        a, soc, stored_cost, stored_unfair = (data[k].copy() for k in ('final', 'soc', 'cost', 'unfair'))
    t = np.arange(len(s.price))[:, None]
    active = (s.ta[None, :] <= t) & (t < s.td[None, :])
    remaining = np.maximum(s.td - t, 1)
    urgency = np.where(active, np.maximum(s.target-soc[:-1], 0)*s.capacity /
                       (s.efficiency*s.power*s.dt*remaining), 0)
    shares = np.divide(urgency, urgency.sum(axis=1, keepdims=True), out=np.zeros_like(urgency),
                       where=urgency.sum(axis=1, keepdims=True) > 0)
    cost = a*s.power*s.price[:, None]*s.dt
    unfair = (a-a.sum(axis=1, keepdims=True)*shares)**2
    gap = np.maximum(s.target-soc[s.td, np.arange(n)], 0)
    failed = int((gap > 0.0001).sum())
    c, d = float(cost.sum()), float(unfair.sum())
    values = dict(cost=c, unfairness=d, weighted_objective=.7*c+.3*d,
                  all_met=int(failed == 0), vehicle_met_percent=100*(1-failed/n),
                  gap_kwh=float(gap@s.capacity), daily_max_dk=float(unfair.sum(axis=0).max()))
    if not np.allclose(cost, stored_cost, rtol=1e-10, atol=1e-10) or not np.allclose(unfair, stored_unfair, rtol=1e-10, atol=1e-10):
        raise ValueError(f'Array mismatch: {traj}')
    for k, v in values.items():
        if k != 'daily_max_dk' and not np.isclose(float(row[k]), v, rtol=1e-10, atol=1e-10):
            raise ValueError(f'Metric mismatch {k}: {traj}')
    return dict(method=method, dataset=row['dataset'], vehicles=n, capacity_kw=cap,
                training_seed=train, seed=day, **values, failed_vehicles=failed,
                input_sha256=row['input_sha256'], trajectory_sha256=row['trajectory_sha256'])


def load_and_audit(include_enhanced: bool, evaluation: Path) -> tuple[list[dict], dict]:
    """Load, independently audit and match the requested trajectories.

    :param include_enhanced: Whether a complete enhanced cohort is required.
    :param evaluation: Enhanced evaluation directory for the selected run.
    :return: Daily rows and rows indexed by feature set and paired key.
    """
    baseline = [r for dataset in ('formal', 'scale') for r in read(BASE/dataset/'scenarios.csv') if selected(r)]
    if len(baseline) != 318:
        raise ValueError(f'Expected 318 baseline trajectories, got {len(baseline)}')
    rows = [audit(r, '21') for r in baseline]
    new_path = evaluation/'scenarios.csv'
    if include_enhanced:
        complete = json.loads((evaluation/'complete.json').read_text(encoding='utf8'))
        if complete['records'] != 318 or complete['scenarios_sha256'] != digest(new_path):
            raise ValueError('Enhanced evaluation completion manifest does not match')
        enhanced = [r for r in read(new_path) if selected(r)]
        if len(enhanced) != 318:
            raise ValueError(f'Expected 318 enhanced trajectories, got {len(enhanced)}')
        rows.extend(audit(r, '27') for r in enhanced)
    keys = ('vehicles', 'capacity_kw', 'training_seed', 'seed')
    by_method = {method: {tuple(r[k] for k in keys): r for r in rows if r['method'] == method} for method in {r['method'] for r in rows}}
    if len(by_method['21']) != 318:
        raise ValueError('Duplicate baseline keys')
    if '27' in by_method:
        if by_method['21'].keys() != by_method['27'].keys():
            raise ValueError('Unmatched keys')
        for key in by_method['21']:
            if by_method['21'][key]['input_sha256'] != by_method['27'][key]['input_sha256']:
                raise ValueError(f'Unmatched input: {key}')
    return rows, by_method


def aggregate(by_method: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Aggregate within runs, then compare same-seed same-day observations.

    :param by_method: Audited rows keyed by feature set and paired key.
    :return: Run summaries, group summaries, daily differences and comparisons.
    """
    runs, groups, paired, comparisons = [], [], [], []
    for n, cap in ((200, 600), (200, 800), (1000, 4000), (3000, 12000)):
        days = 50 if n == 200 else 3
        expected = set(range(65000, 65050)) if n == 200 else set(range(81000+10*n, 81003+10*n))
        for method, index in by_method.items():
            for train in SEEDS:
                subset = [r for r in index.values() if (r['vehicles'], r['capacity_kw'], r['training_seed']) == (n, cap, train)]
                if len(subset) != days or {r['seed'] for r in subset} != expected:
                    raise ValueError(f'Incomplete cohort {method} {n} {cap} {train}')
                runs.append(dict(method=method, vehicles=n, capacity_kw=cap, training_seed=train, days=days,
                                 **{k: mean(r[k] for r in subset) for k in METRICS}))
            subset = [r for r in runs if (r['method'], r['vehicles'], r['capacity_kw']) == (method, n, cap)]
            groups.append(dict(method=method, vehicles=n, capacity_kw=cap, days=days,
                               **{k: mean(r[k] for r in subset) for k in METRICS},
                               **{k+'_sd': stdev(r[k] for r in subset) for k in METRICS}))
        if '27' in by_method:
            for train in SEEDS:
                for day in sorted(expected):
                    key = (n, cap, train, day)
                    old, new = by_method['21'][key], by_method['27'][key]
                    paired.append(dict(vehicles=n, capacity_kw=cap, training_seed=train, seed=day,
                                       **{k+'_delta': new[k]-old[k] for k in METRICS},
                                       all_met_delta_percentage_points=100*(new['all_met']-old['all_met'])))
            old_group = next(r for r in groups if (r['method'], r['vehicles'], r['capacity_kw']) == ('21', n, cap))
            new_group = next(r for r in groups if (r['method'], r['vehicles'], r['capacity_kw']) == ('27', n, cap))
            rng = np.random.default_rng(1022026)
            for metric in METRICS:
                per_seed = [next(r[metric] for r in runs if (r['method'], r['vehicles'], r['capacity_kw'], r['training_seed']) == ('27', n, cap, train)) -
                            next(r[metric] for r in runs if (r['method'], r['vehicles'], r['capacity_kw'], r['training_seed']) == ('21', n, cap, train)) for train in SEEDS]
                matrix = np.array([[by_method['27'][(n, cap, train, day)][metric]-by_method['21'][(n, cap, train, day)][metric]
                                    for train in SEEDS] for day in sorted(expected)])
                day_delta = matrix.mean(axis=1)
                ci = np.quantile(day_delta[rng.integers(0, days, size=(20000, days))].mean(axis=1), [.025, .975]) if n == 200 else (None, None)
                delta = new_group[metric]-old_group[metric]
                comparisons.append(dict(vehicles=n, capacity_kw=cap, metric=metric,
                                        baseline=old_group[metric], enhanced=new_group[metric], delta=delta,
                                        delta_percentage_points=100*delta if metric == 'all_met' else (delta if metric == 'vehicle_met_percent' else None),
                                        percent_delta=100*delta/old_group[metric] if abs(old_group[metric]) > 1e-10 else None,
                                        delta_11002=per_seed[0], delta_11003=per_seed[1], delta_11004=per_seed[2],
                                        bootstrap_95_low=ci[0], bootstrap_95_high=ci[1],
                                        days_increase=int((day_delta > 0).sum()), days_decrease=int((day_delta < 0).sum()),
                                        largest_abs_day=sorted(expected)[int(np.argmax(np.abs(day_delta)))],
                                        largest_abs_day_delta=day_delta[np.argmax(np.abs(day_delta))]))
    return runs, groups, paired, comparisons


def verify_archive(groups: list[dict]) -> None:
    """Check baseline aggregates against the archived published ablation table.

    :param groups: Aggregated group records.
    :return: None; raises on disagreement.
    """
    archive = json.loads((ROOT/'results/pressure21_v1/ablation_summary.json').read_text(encoding='utf8'))
    for group in (g for g in groups if g['method'] == '21'):
        source = next(g for g in archive['groups'] if g['method'] == 'Full' and g['vehicles'] == group['vehicles'] and g['capacity_kw'] == group['capacity_kw'])
        for metric in METRICS[:-1]:
            if not np.isclose(group[metric], source[metric]['mean'], rtol=1e-10, atol=1e-10):
                raise ValueError(f'Archived summary mismatch: {group} {metric}')
def output_results(rows: list[dict], runs: list[dict], groups: list[dict],
                   paired: list[dict], comparisons: list[dict], include_enhanced: bool,
                   evaluation: Path, destination: Path) -> None:
    """Stage validated outputs and publish them only after all calculations succeed.

    :param rows: Audited daily rows.
    :param runs: Within-run means.
    :param groups: Means and standard deviations across runs.
    :param paired: Paired daily differences.
    :param comparisons: Aggregate differences and conditional intervals.
    :param include_enhanced: Whether this is the complete paired analysis.
    :param evaluation: Enhanced evaluation directory for the selected run.
    :param destination: Output directory for the selected run.
    :return: None.
    """
    target = destination if include_enhanced else destination/'baseline_only'
    target.mkdir(parents=True, exist_ok=True)
    new_path = evaluation/'scenarios.csv'
    metadata = {'status': 'paired_complete' if include_enhanced else 'baseline_only',
                'baseline_trajectories': 318, 'enhanced_trajectories': 318 if include_enhanced else 0,
                'cohort': 'same training seed and day with matching input SHA-256; each 200-EV capacity has 50 days; each large fleet has 3 days',
                'daily_max_dk': 'maximum across vehicles of sum across times of squared signed action deviations; no energy normalization',
                'summary': 'first mean over days within each training seed, then mean and sample SD over the three seeds',
                'bootstrap': '200-EV only; percentile 95%, 20000 day resamples, same sampled days shared across three fixed paired training seeds; conditional on these models',
                'units': {'all_met': 'daily indicator and day fraction in [0,1]; multiply aggregate deltas by 100 for percentage points', 'vehicle_met_percent': 'percent; differences are percentage points', 'gap_kwh': 'kWh per day'},
                'baseline_archive_summary_verified': True,
                'source_hashes': {str(p): digest(p) for p in [BASE/'formal/scenarios.csv', BASE/'scale/scenarios.csv', ROOT/'results/pressure21_v1/ablation_summary.json'] + ([new_path] if include_enhanced else [])}}
    with tempfile.TemporaryDirectory(dir=target) as stage_dir:
        stage = Path(stage_dir)
        files = {'daily.csv': rows, 'run_summary.csv': runs, 'summary.csv': groups}
        if include_enhanced:
            files.update({'paired_daily.csv': paired, 'paired_summary.csv': comparisons})
        for name, data in files.items():
            write(stage/name, data)
        metadata['output_hashes'] = {name: digest(stage/name) for name in files}
        (stage/'audit.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
        names = (*files, 'audit.json')
        backups = stage/'backups'
        backups.mkdir()
        existing = {name for name in names if (target/name).exists()}
        for name in existing:
            shutil.copy2(target/name, backups/name)
        published = []
        try:
            for name in names:
                os.replace(stage/name, target/name)
                published.append(name)
        except OSError:
            for name in reversed(published):
                if name in existing:
                    os.replace(backups/name, target/name)
                else:
                    (target/name).unlink()
            raise


def main() -> None:
    """Run the paired analysis or explicitly request a separate baseline audit.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--run-id', default=DEFAULT_RUN_ID,
                        help='Run directory name under artifacts/runs and results.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.run_id):
        parser.error('--run-id must contain only letters, numbers, underscores or hyphens')
    evaluation = ROOT/'artifacts/runs'/args.run_id/'evaluation/formal'
    destination = ROOT/'results'/args.run_id
    rows, by_method = load_and_audit(not args.baseline_only, evaluation)
    runs, groups, paired, comparisons = aggregate(by_method)
    verify_archive(groups)
    output_results(rows, runs, groups, paired, comparisons, not args.baseline_only,
                   evaluation, destination)
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
