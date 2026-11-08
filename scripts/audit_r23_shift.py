"""Independently recompute distribution-shift results from saved trajectories."""
import csv
import argparse
import json
from pathlib import Path
import sys
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'src'))
from evcharging.core import load_scenario

A = R / 'artifacts/runs/pressure21_v1/r23'
O = R / 'results/pressure21_v1/r23'


def main(output_dir: Path = O) -> None:
    """Recalculate daily statistics, physical checks, and paired summaries.

    :return: None.
    """
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Output directory must be new or empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    inp = []
    max_residual = 0.0
    reuse_max = 0.0
    import hashlib
    selection = json.loads((R / 'artifacts/runs/pressure21_v1/il_selection.json').read_text(encoding='utf8'))
    expected_model = next(x for x in selection['models'] if x['seed'] == 11002)
    model = R / 'artifacts/runs/pressure21_v1/il_11002/il_model/best.pt'
    actual_model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    assert expected_model['state_schema'] == 'pressure21_v1' and actual_model_hash == expected_model['sha256']
    with (R / 'results/pressure21_v1/r26/d_daily.csv').open(encoding='utf8', newline='') as f:
        baseline = {(int(x['seed']), int(x['limit'])): x for x in csv.DictReader(f) if x['distribution'] == 'nominal' and x['method'] == 'UPA'}
    for kind in ('equipment', 'price_level', 'price_roll'):
        for seed in range(65000, 65050):
            s = load_scenario(str(A / f'input_{kind}_{seed}.npz'))
            original = load_scenario(str(R / f'artifacts/runs/ideal10h_v1/formal/inputs/{seed}.npz'))
            for field in ('ta', 'td', 'arrival', 'target', 'efficiency'):
                assert np.array_equal(getattr(s, field), getattr(original, field))
            if kind == 'equipment':
                assert np.array_equal(s.price, original.price)
                with np.load(A / f'draws_{seed}.npz') as d:
                    cr, pr = d['capacity_raw'], d['power_raw']
                assert np.array_equal(cr, np.random.default_rng(np.random.SeedSequence([seed, 2301, 1])).normal(35, 5, 200))
                assert np.array_equal(pr, np.random.default_rng(np.random.SeedSequence([seed, 2301, 2])).normal(7, 1, 200))
                assert np.array_equal(s.capacity, np.clip(cr, 20, 40)) and np.array_equal(s.power, np.clip(pr, 4, 8))
                inp.append({'seed': seed, 'capacity_mean': s.capacity.mean(), 'capacity_sd': s.capacity.std(ddof=1), 'capacity_clip_low': np.mean(cr < 20), 'capacity_clip_high': np.mean(cr > 40), 'power_mean': s.power.mean(), 'power_sd': s.power.std(ddof=1), 'power_clip_low': np.mean(pr < 4), 'power_clip_high': np.mean(pr > 8)})
            else:
                assert np.array_equal(s.capacity, original.capacity) and np.array_equal(s.power, original.power)
                expected = original.price * 1.2 if kind == 'price_level' else np.roll(original.price, 8)
                assert np.array_equal(s.price, expected)
                if kind == 'price_roll':
                    assert np.array_equal(np.sort(s.price), np.sort(original.price))
            individual = (s.target - s.arrival) * s.capacity > (s.td - s.ta) * s.power * s.efficiency * s.dt + 1e-9
            for limit in (600, 800):
                feas = json.loads((A / f'feas_{kind}_{seed}_{limit}.json').read_text(encoding='utf8'))
                for method in ('Full', 'UPA'):
                    with np.load(A / f'traj_{kind}_{seed}_{limit}_{method}.npz') as z:
                        a, soc, stored_cost, stored_d = z['final'], z['soc'], z['cost'], z['unfair']
                    active = (np.arange(56)[:, None] >= s.ta) & (np.arange(56)[:, None] < s.td)
                    remaining = np.maximum(s.td - np.arange(56)[:, None], 1)
                    u = np.where(active, np.maximum(s.target - soc[:-1], 0) * s.capacity / (s.efficiency * s.power * s.dt * remaining), 0)
                    usum = u.sum(axis=1, keepdims=True)
                    shares = np.divide(u, usum, out=np.zeros_like(u), where=usum > 0)
                    cost = a * s.power * s.price[:, None] * s.dt
                    unfair = (a - a.sum(axis=1, keepdims=True) * shares) ** 2
                    residual = max(float(np.max(np.abs(soc[1:] - soc[:-1] - a * s.power * s.efficiency * s.dt / s.capacity))), float(np.max(np.abs(a[~active]), initial=0)), float(np.max(np.maximum(a @ s.power - limit, 0))), float(np.max(np.maximum(-a, 0))), float(np.max(np.maximum(a - 1, 0))), float(np.max(np.maximum(soc - s.target, 0))), float(np.max(np.abs(soc[0] - s.arrival))), float(np.max(np.abs(cost - stored_cost))), float(np.max(np.abs(unfair - stored_d))))
                    max_residual = max(max_residual, residual)
                    if kind != 'equipment' and method == 'UPA':
                        with np.load(R / f'artifacts/runs/pressure21_v1/r26_diagnostics/d_nominal_{seed}_{limit}_upa.npz') as z:
                            reuse_max = max(reuse_max, float(np.max(np.abs(a-z['final']))), float(np.max(np.abs(soc-z['soc']))))
                            original_cost = float((z['final'] * original.power * original.price[:, None] * original.dt).sum())
                            ref = baseline[seed, limit]
                            assert abs(original_cost - float(ref['cost'])) < 1e-8
                            assert abs(float(z['unfair'].sum()) - float(ref['D'])) < 1e-8
                    gap = np.maximum(s.target - soc[s.td, np.arange(200)], 0) * s.capacity
                    demand = (s.target-s.arrival) @ s.capacity
                    rows.append({'kind': kind, 'seed': seed, 'limit': limit, 'method': method, 'feasibility': feas['status'], 'individual_unreachable': int(individual.sum()), 'cost': cost.sum(), 'D': unfair.sum(), 'met_ev_percent': 100*np.mean(gap / s.capacity <= 1e-4), 'all_met_percent': 100*int(np.all(gap / s.capacity <= 1e-4)), 'gap_kwh': gap.sum(), 'gap_fraction': gap.sum()/demand, 'purchased_kwh': (a*s.power*s.dt).sum(), 'battery_demand_kwh': demand, 'grid_demand_kwh': demand/s.efficiency, 'residual': residual})
    assert len(rows) == 600 and max_residual < 1e-8 and reuse_max == 0
    def write(name: str, data: list[dict]) -> None:
        with (output_dir / name).open('x', encoding='utf8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    write('daily.csv', rows); write('equipment_inputs.csv', inp)
    metrics = ('cost', 'D', 'met_ev_percent', 'all_met_percent', 'gap_kwh', 'gap_fraction', 'purchased_kwh', 'battery_demand_kwh', 'grid_demand_kwh')
    summary = []
    for kind in ('equipment', 'price_level', 'price_roll'):
        for limit in (600, 800):
            for subset in ('all', 'feasible'):
                group = [r for r in rows if r['kind'] == kind and r['limit'] == limit and (subset == 'all' or r['feasibility'] == 'feasible')]
                full = sorted((r for r in group if r['method']=='Full'), key=lambda r:r['seed'])
                upa = sorted((r for r in group if r['method']=='UPA'), key=lambda r:r['seed'])
                assert [r['seed'] for r in full] == [r['seed'] for r in upa]
                for metric in metrics:
                    x = np.array([r[metric] for r in full]); y = np.array([r[metric] for r in upa]); diff = x-y
                    if not len(x): continue
                    summary.append({'kind':kind,'limit':limit,'subset':subset,'n':len(x),'metric':metric,'full_mean':x.mean(),'full_sd':x.std(ddof=1) if len(x)>1 else '', 'upa_mean':y.mean(),'upa_sd':y.std(ddof=1) if len(x)>1 else '', 'difference_mean':diff.mean(),'difference_sd':diff.std(ddof=1) if len(x)>1 else '', 'reduction_percent_ratio_of_means':100*(y.mean()-x.mean())/y.mean() if y.mean()!=0 else ''})
    write('summary.csv', summary)
    with (output_dir / 'audit.json').open('x', encoding='utf8') as f:
        f.write(json.dumps({'n':len(rows),'max_physical_or_saved_metric_residual':max_residual,'upa_reuse_max_difference':reuse_max,'raw_draw_streams_verified':100,'model_sha256':actual_model_hash,'status':'passed'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=O, help='New or empty directory for recomputed reports')
    main(parser.parse_args().output_dir)
