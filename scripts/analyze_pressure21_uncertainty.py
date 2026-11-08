"""Summarize Pressure21 scenario uncertainty and training-run variation."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean,stdev

import numpy as np

from evaluate_pressure21 import CONFIG,ROOT,OUT,digest
from analyze_evaluation_uncertainty import analyze
from pressure21_current_rl import PPO_RUN, current_rl, local_path, selection


def read(path: Path) -> list[dict]:
    """Read scenario rows.

    :param path: CSV path.
    :return: Row dictionaries.
    """
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def main() -> None:
    """Create new 900-record archive and paired scenario-bootstrap summaries.

    :return: None.
    """
    cfg=json.loads(CONFIG.read_text());dest=ROOT/'results/pressure21_v1';primary=read(OUT/'formal/scenarios.csv');rl=current_rl('formal');source=primary+rl
    if len(source)!=900:raise RuntimeError('Expected 900 training-repeat records')
    historical=read(ROOT/'artifacts/runs/ideal10h_v1/formal/scenarios.csv');feasible={(int(r['seed']),int(r['capacity_kw'])):r['feasibility'] for r in historical if r['method']=='UC'}
    counts={}
    for row in source:
        key=(row['method'],int(row['capacity_kw']),int(row['training_seed']));counts[key]=counts.get(key,0)+1
        row['feasibility']=feasible[int(row['seed']),int(row['capacity_kw'])]
        if row['feasibility']!='feasible':raise RuntimeError('Non-feasible input in formal cohort')
        if digest(local_path(row['trajectory_path']))!=row['trajectory_sha256'] or digest(local_path(row['input_path']))!=row['input_sha256']:raise RuntimeError('Source trajectory/input changed')
    if len(counts)!=18 or set(counts.values())!={50}:raise RuntimeError('Training-run identity/count mismatch')
    with (dest/'training_repeats.csv').open('w',newline='',encoding='utf8') as f:
        fields=list(dict.fromkeys(k for row in source for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(source)
    main_seeds={'IL':11002,'DDQN':21002,'PPO':int(selection()['ppo_main']['seed'])}; main=[r for r in source if int(r['training_seed'])==main_seeds[r['method']]];refs=read(dest/'formal_references.csv');main+=refs
    metrics={'cost':'cost','unfairness':'unfairness','weighted_objective':'weighted_objective','scenario_failure_rate':'failed_vehicles','vehicle_failure_rate':'failed_vehicles','total_shortfall_kwh':'gap_kwh','cost_per_purchased_kwh':'cost'}
    for row in main:
        row['seed']=str(row['seed']);row['capacity_kw']=str(row['capacity_kw'])
        row['final_energy_kwh']=row['purchased_kwh']
    bootstrap=analyze(main,('capacity_kw',),metrics,('UC','DDQN','PPO','UPA','RDDPG'),10000,20261101)
    old_small=read(ROOT/'artifacts/runs/ideal10h_v1/small_evaluation/scenarios.csv')
    ours_small=read(OUT/'small/scenarios.csv');rl_small=current_rl('small');li_small=read(ROOT/'artifacts/runs/li2023_r105/paper_33001/scenarios.csv')
    small_rows=[]
    for method in ('IL','IDEAL','UC','DDQN','PPO','RDDPG'):
        items=ours_small if method=='IL' else rl_small if method in ('DDQN','PPO') else li_small if method=='RDDPG' else old_small
        selected=[r for r in items if r['method']==method and int(r['seed']) in cfg['small_seeds'] and (method!='RDDPG' or int(r['vehicles'])==10)]
        if len(selected)!=20:raise RuntimeError(f'Unmatched small-scenario cohort: {method}')
        for source_row in selected:
            row=dict(source_row);row['seed']=str(row['seed']);row['weighted_objective']=.7*float(row['cost'])+.3*float(row['unfairness']);small_rows.append(row)
    small_bootstrap=analyze(small_rows,(),{'cost':'cost','unfairness':'unfairness','weighted_objective':'weighted_objective'},('IDEAL','UC','DDQN','PPO','RDDPG'),10000,20261102)
    runs=[]
    for method in ('IL','DDQN','PPO'):
        for cap in cfg['formal_capacities_kw']:
            seeds=sorted({int(r['training_seed']) for r in source if r['method']==method});per=[]
            for train in seeds:
                group=[r for r in source if r['method']==method and int(r['capacity_kw'])==cap and int(r['training_seed'])==train]
                item={'training_seed':train}
                for key in ('cost','unfairness','weighted_objective','all_met','vehicle_met_percent','gap_kwh'):item[key]=mean(float(r[key]) for r in group)
                item['cost_per_purchased_kwh']=sum(float(r['cost']) for r in group)/sum(float(r['purchased_kwh']) for r in group);per.append(item)
            runs.append({'method':method,'capacity_kw':cap,'run_means':per,'statistics':{key:{'mean':mean(x[key] for x in per),'sample_sd':stdev(x[key] for x in per)} for key in per[0] if key!='training_seed'}})
    result={'bootstrap':bootstrap,'small_bootstrap':small_bootstrap,'training_repeats':runs,'bootstrap_samples':10000,'bootstrap_seed':20261101,'small_bootstrap_seed':20261102,'unit':'paired independent scenarios within capacity','sources_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in (OUT/'formal/scenarios.csv',OUT/'rl/formal/scenarios.csv',dest/'formal_references.csv',OUT/'small/scenarios.csv',OUT/'rl/small/scenarios.csv',PPO_RUN/'evaluation/formal/scenarios.csv',PPO_RUN/'evaluation/small/scenarios.csv',PPO_RUN/'selection.json',ROOT/'artifacts/runs/ideal10h_v1/small_evaluation/scenarios.csv',ROOT/'artifacts/runs/li2023_r105/paper_33001/scenarios.csv',ROOT/'scripts/analyze_evaluation_uncertainty.py')},'repeat_csv_sha256':digest(dest/'training_repeats.csv')}
    (dest/'uncertainty_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('Saved 900 training-repeat records and paired bootstrap summaries')


if __name__=='__main__':main()
