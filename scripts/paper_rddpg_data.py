"""Evaluate the frozen RDDPG actor for main-paper plots and tabulation."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import generate, load_scenario
from evcharging.li2023 import actor, simulate
from run_li2023_r105 import numpy_policy

BASE = ROOT / 'artifacts/runs/ideal10h_v1'
RUN = ROOT / 'artifacts/runs/li2023_r105'
OUTPUT = RUN / 'paper_33001'
RESULT = ROOT / 'results/li2023_r105'


def digest(path: Path) -> str:
    """Hash an input or output file.

    :param path: File to hash.
    :return: SHA-256 hexadecimal hash.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common_metrics(s, p: np.ndarray, soc: np.ndarray) -> dict:
    """Compute the manuscript's common metrics from a power trajectory.

    :param s: Fixed-day input scenario.
    :param p: Per-vehicle power trajectory in kW.
    :param soc: SoC trajectory.
    :return: Cost and unfairness arrays and departure gaps.
    """
    t = np.arange(len(s.price))[:, None]
    active = (s.ta <= t) & (t < s.td)
    urgency = np.where(active, np.maximum(s.target-soc[:-1], 0)*s.capacity /
                       (s.efficiency*s.power*s.dt*np.maximum(s.td-t, 1)), 0.)
    denom = urgency.sum(axis=1, keepdims=True)
    share = np.divide(urgency, denom, out=np.zeros_like(urgency), where=denom > 0)
    final = p/s.power
    return dict(final=final, soc=soc, cost=p*s.price[:, None]*s.dt,
                unfair=(final-final.sum(axis=1, keepdims=True)*share)**2,
                gap=np.maximum(s.target-soc[s.td, np.arange(len(s.ta))], 0.))


def main() -> None:
    """Evaluate new scenarios, compute table rows and archive provenance.

    :return: None.
    """
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ckpt = RUN / '33001/episode_6000.pt'
    manifest = json.loads((RESULT / 'manifest.json').read_text(encoding='utf8'))
    if digest(ckpt) != manifest['models_sha256']['33001']:
        raise RuntimeError('RDDPG actor differs from frozen model')
    model = actor(); model.load_state_dict(torch.load(ckpt, weights_only=True)['model'])
    policy = numpy_policy(model)
    hashes = {ckpt.relative_to(ROOT).as_posix(): digest(ckpt)}
    newrows = []
    for n, seeds in ((10, range(76000, 76020)), (100, range(82000, 82003)), (500, range(86000, 86003))):
        for seed in seeds:
            s = generate(seed, n, fixed_day=True)
            src = BASE / ('small_experts' if n == 10 else 'scalability') / (f'seed_{seed}.npz' if n == 10 else f'input_{n}_{seed}.npz')
            with np.load(src) as d:
                keys = ('ta','td','arrival','target','capacity','power','price') if n == 10 else vars(s).keys()
                if any(not np.array_equal(d[k], np.asarray(getattr(s,k))) for k in keys):
                    raise RuntimeError(f'Input mismatch: {src}')
            hashes[src.relative_to(ROOT).as_posix()] = digest(src)
            r = simulate(s, 4*n, policy)
            v = common_metrics(s, r['power'], r['soc'])
            path = OUTPUT / f'{n}_{seed}.npz'
            np.savez_compressed(path, **v, power=r['power'], boundary=r['boundary'],
                                empty_count=r['empty_count'], empty_gap=r['empty_gap'])
            newrows.append(dict(vehicles=n, seed=seed, method='RDDPG', cost=float(v['cost'].sum()),
                                unfairness=float(v['unfair'].sum()), failed=int((v['gap'] > 1e-4).sum())))
    with (OUTPUT / 'scenarios.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(newrows[0])); w.writeheader(); w.writerows(newrows)
    table = []
    for limit in (800,600):
        cost = []; unfair = []; energy = []; gaps = []; gaps_kwh = []; failures = []
        for seed in range(65000,65050):
            s = load_scenario(str(BASE / 'formal/inputs' / f'{seed}.npz'))
            path = RUN / 'evaluation/33001' / f'200_{seed}_{limit}.npz'
            with np.load(path) as d:
                p, soc = d['power'], d['soc']
                v = common_metrics(s,p,soc)
                cost.append(v['cost'].sum()); unfair.append(v['unfair'].sum()); energy.append(p.sum()*s.dt)
                gap=v['gap']; mask=gap>1e-4; failures.append(mask.sum())
                gaps.extend(gap[mask]*100); gaps_kwh.extend(gap[mask]*s.capacity[mask])
            hashes[path.relative_to(ROOT).as_posix()]=digest(path)
        table.append(dict(capacity_kw=limit, method='RDDPG', scenarios=50,
                          all_met_percent=100*np.mean(np.array(failures)==0),
                          vehicle_met_percent=100*(1-sum(failures)/10000),
                          mean_failed_gap_pp=float(np.mean(gaps)) if gaps else 0.,
                          mean_failed_gap_kwh=float(np.mean(gaps_kwh)) if gaps_kwh else 0.,
                          mean_cost=float(np.mean(cost)), cost_per_purchased_kwh=float(sum(cost)/sum(energy)),
                          mean_unfairness=float(np.mean(unfair))))
    with (RESULT / 'main_paper_summary.csv').open('w', newline='', encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    (RESULT / 'main_paper_summary.json').write_text(json.dumps(table, indent=2),encoding='utf8')
    for p in (Path(__file__), ROOT/'src/evcharging/core.py', ROOT/'src/evcharging/li2023.py', BASE/'small_evaluation/scenarios.csv', BASE/'scalability/scenarios.csv'):
        hashes[p.relative_to(ROOT).as_posix()]=digest(p)
    (OUTPUT/'manifest.json').write_text(json.dumps({'sources':hashes,'model_seed':33001,'outputs':{p.name:digest(p) for p in OUTPUT.glob('*.npz')}},indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
