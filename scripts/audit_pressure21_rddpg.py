"""Recompute Li-criterion comparisons with the new Pressure training runs."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean,stdev

import numpy as np

from evaluate_pressure21 import ROOT,OUT,digest
from evcharging.core import load_scenario
from audit_li2023_r105 import metrics
from paper_rddpg_data import common_metrics


def read(path: Path) -> list[dict]:
    """Read a CSV file.

    :param path: CSV file.
    :return: Rows.
    """
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def main() -> None:
    """Verify all three frozen RDDPG runs and pair them with new Pressure runs.

    :return: None.
    """
    result=ROOT/'results/pressure21_v1';li=ROOT/'results/li2023_r105';manifest=json.loads((li/'manifest.json').read_text())
    for name in ('scenarios.csv','audit_scenarios.csv','summary.csv','audit_summary.csv'):
        if digest(li/name)!=manifest['results_sha256'][name]:raise RuntimeError(f'RDDPG results hash mismatch: {name}')
    for seed in (33001,33002,33003):
        model=ROOT/f'artifacts/runs/li2023_r105/{seed}/episode_6000.pt'
        if digest(model)!=manifest['models_sha256'][str(seed)]:raise RuntimeError('RDDPG model changed')
    original=read(li/'scenarios.csv');old_audit=read(li/'audit_scenarios.csv'); old_map={(int(r['training_seed']),int(r['vehicles']),int(r['scenario_seed']),int(r['capacity_kw'])):r for r in old_audit if r['method']=='Li2023'}
    pressure_formal=read(OUT/'formal/scenarios.csv');pressure_scale=read(OUT/'scale/scenarios.csv');rows=[]
    for r in original:
        model=int(r['training_seed']);n=int(r['vehicles']);seed=int(r['seed']);cap=int(r['capacity_kw'])
        input_path=ROOT/f'artifacts/runs/ideal10h_v1/formal/inputs/{seed}.npz' if n==200 else ROOT/f'artifacts/runs/ideal10h_v1/scalability/input_{n}_{seed}.npz'
        expected=manifest['inputs_sha256'][input_path.name] if n==200 else manifest['verified_scale_input_sha256'][input_path.relative_to(ROOT).as_posix()]
        if digest(input_path)!=expected:raise RuntimeError('RDDPG scenario identity mismatch')
        s=load_scenario(str(input_path));path=ROOT/f'artifacts/runs/li2023_r105/evaluation/{model}/{n}_{seed}_{cap}.npz'
        with np.load(path) as a:
            power=a['power'];soc=a['soc'];v=common_metrics(s,power,soc);cost,loss=metrics(s,soc,power)
        gap=v['gap'];values={'cost':float(v['cost'].sum()),'unfairness':float(v['unfair'].sum()),'gap_kwh':float(gap@s.capacity),'failed_vehicles':int((gap>1e-4).sum())}
        for key,value in values.items():
            if not np.isclose(float(r[key]),value,rtol=1e-10,atol=1e-9):raise RuntimeError(f'RDDPG reconstructed metric differs: {path} {key}')
        old=old_map[model,n,seed,cap]
        if not np.isclose(float(old['li_cost_fair_loss']),loss,rtol=1e-10,atol=1e-9):raise RuntimeError(f'RDDPG Li criterion differs: {path}')
        rows.append({'method':'RDDPG','training_seed':model,'vehicles':n,'seed':seed,'capacity_kw':cap,'cost':cost,'li_cost_fair_loss':loss,'trajectory_sha256':digest(path),'input_sha256':digest(input_path),'model_sha256':manifest['models_sha256'][str(model)]})
    if len(rows)!=318:raise RuntimeError('RDDPG three-run evaluation incomplete')
    for n,cap in ((200,600),(200,800),(1000,4000),(3000,12000)):
        source=pressure_formal if n==200 else pressure_scale
        for r in source:
            if int(r['vehicles'])!=n or int(r['capacity_kw'])!=cap:continue
            input_path=Path(r['input_path']);path=Path(r['trajectory_path'])
            if digest(input_path)!=r['input_sha256'] or digest(path)!=r['trajectory_sha256']:raise RuntimeError('Pressure trajectory or input changed')
            s=load_scenario(str(input_path))
            with np.load(path) as data:cost,loss=metrics(s,data['soc'],data['final']*s.power)
            rows.append({'method':'IL','training_seed':int(r['training_seed']),'vehicles':n,'seed':int(r['seed']),'capacity_kw':cap,'cost':cost,'li_cost_fair_loss':loss,'trajectory_sha256':digest(path),'input_sha256':digest(input_path),'model_sha256':r['model_sha256']})
    report=[]
    for n,cap in ((200,600),(200,800),(1000,4000),(3000,12000)):
        for method in ('IL','RDDPG'):
            group=[r for r in rows if r['vehicles']==n and r['capacity_kw']==cap and r['method']==method]
            seeds=sorted({r['training_seed'] for r in group});run_means=[mean(r['li_cost_fair_loss'] for r in group if r['training_seed']==seed) for seed in seeds]
            expected=50 if n==200 else 3
            if len(seeds)!=3 or any(sum(r['training_seed']==seed for r in group)!=expected for seed in seeds):raise RuntimeError('Incomplete paired cohort')
            report.append({'vehicles':n,'capacity_kw':cap,'method':method,'training_seeds':seeds,'days_per_run':expected,'run_means':run_means,'mean':mean(run_means),'sample_sd':stdev(run_means)})
    result.mkdir(parents=True,exist_ok=True)
    with (result/'li_criterion_scenarios.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (result/'li_criterion_summary.json').write_text(json.dumps({'rows':report,'trajectory_count':len(rows),'rddpg_source_manifest_sha256':digest(li/'manifest.json'),'scenario_output_sha256':digest(result/'li_criterion_scenarios.csv'),'definition':'Li criterion .6 cost + .4 absolute slotwise cost deviation against Li urgency share; average within run, then three-run sample SD'},indent=2),encoding='utf8')
    print('Verified 318 RDDPG and 318 new Pressure Li-criterion trajectories')


if __name__=='__main__':main()
