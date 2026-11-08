"""Compute manuscript figure statements from matched Pressure21 trajectories."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

from evaluate_pressure21 import ROOT,OUT,CONFIG,digest
from pressure21_current_rl import PPO_RUN, current_rl, local_path
from evcharging.core import generate


def read(path: Path) -> list[dict]:
    """Read scenario metrics.

    :param path: CSV source.
    :return: Row dictionaries.
    """
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def main() -> None:
    """Summarize per-EV and per-day comparisons underlying Figures 9–15.

    :return: None.
    """
    cfg=json.loads(CONFIG.read_text());base=ROOT/'artifacts/runs/ideal10h_v1';li=ROOT/'artifacts/runs/li2023_r105/paper_33001'; small=read(OUT/'small/scenarios.csv');rl=current_rl('small');old=read(base/'small_evaluation/scenarios.csv');rddpg=read(li/'scenarios.csv');s=generate(76000,10,fixed_day=True)
    paths={'IL':OUT/'small/11002_10_76000_40.npz','DDQN':OUT/'rl/small/DDQN_21002_10_76000_40.npz','PPO':local_path(next(r['trajectory_path'] for r in rl if r['method']=='PPO' and int(r['seed'])==76000)),'UC':base/'small_evaluation/76000_UC.npz','IDEAL':base/'small_evaluation/76000_IDEAL.npz','RDDPG':li/'10_76000.npz'}
    values={}
    for name,path in paths.items():
        with np.load(path) as r:
            energy=(r['soc'][s.td,np.arange(10)]-s.arrival)*s.capacity/s.efficiency
            unit_cost=np.divide(r['cost'].sum(axis=0),energy,out=np.zeros(10),where=energy>0);unit_unfair=np.divide(r['unfair'].sum(axis=0),energy,out=np.zeros(10),where=energy>0)
            c=float(r['cost'].sum());d=float(r['unfair'].sum())
            values[name]={'cost':c,'unfairness':d,'weighted_objective':.7*c+.3*d,'unit_cost_by_ev':unit_cost.tolist(),'unit_unfair_by_ev':unit_unfair.tolist(),'trajectory_sha256':digest(path)}
    counts={name:{'lower_unit_cost':sum(a<b for a,b in zip(values['IL']['unit_cost_by_ev'],values[name]['unit_cost_by_ev'])),'lower_unit_unfair':sum(a<b for a,b in zip(values['IL']['unit_unfair_by_ev'],values[name]['unit_unfair_by_ev']))} for name in ('UC','DDQN','PPO','RDDPG')}
    daily=[]
    for seed in cfg['small_seeds']:
        baseline=next(r for r in old if int(r['seed'])==seed and r['method']=='UC')
        for name in ('IDEAL','IL','UC','DDQN','PPO','RDDPG'):
            source=small if name=='IL' else rl if name in ('DDQN','PPO') else rddpg if name=='RDDPG' else old
            row=next(r for r in source if int(r['seed'])==seed and r['method']==name and (name!='RDDPG' or int(r['vehicles'])==10))
            failed=int(row['failed_vehicles'] if name in ('IL','DDQN','PPO') else row['failed'])
            daily.append({'seed':seed,'method':name,'cost_reduction_vs_UC_percent':100*(1-float(row['cost'])/float(baseline['cost'])),'unfairness_reduction_vs_UC_percent':100*(1-float(row['unfairness'])/float(baseline['unfairness'])),'failed':failed})
    per_day=[]
    for name in ('IDEAL','IL','UC','DDQN','PPO','RDDPG'):
        selected=[r for r in daily if r['method']==name]
        per_day.append({'method':name,'mean_daily_cost_reduction_vs_UC_percent':mean(r['cost_reduction_vs_UC_percent'] for r in selected),'mean_daily_unfairness_reduction_vs_UC_percent':mean(r['unfairness_reduction_vs_UC_percent'] for r in selected),'failed_vehicles':sum(r['failed'] for r in selected),'vehicle_met_percent':100*(1-sum(r['failed'] for r in selected)/200),'all_met_scenarios':sum(r['failed']==0 for r in selected)})
    scale=read(OUT/'scale/scenarios.csv');uc=read(base/'scalability/scenarios.csv');li_scale=read(ROOT/'results/li2023_r105/scenarios.csv');groups=[]
    for n in (1000,3000):
        for method in ('IL','UC','RDDPG'):
            seeds=(11002,11003,11004) if method=='IL' else (33001,33002,33003) if method=='RDDPG' else (None,)
            for train in seeds:
                if method=='IL':records=[r for r in scale if int(r['vehicles'])==n and int(r['training_seed'])==train]
                elif method=='UC':records=[r for r in uc if int(r['vehicles'])==n and r['method']=='UC']
                else:records=[r for r in li_scale if int(r['vehicles'])==n and int(r['training_seed'])==train]
                if len(records)!=3:raise RuntimeError('Incomplete scale comparison')
                groups.append({'vehicles':n,'method':method,'training_seed':train,'cost_per_ev':mean(float(r['cost'])/n for r in records),'unfairness_per_ev':mean(float(r['unfairness'])/n for r in records),'failed_vehicles':sum(int(r.get('failed_vehicles',r.get('failed',0))) for r in records)})
    sources=[OUT/'small/scenarios.csv',OUT/'rl/small/scenarios.csv',PPO_RUN/'evaluation/small/scenarios.csv',PPO_RUN/'selection.json',OUT/'scale/scenarios.csv',base/'small_evaluation/scenarios.csv',base/'scalability/scenarios.csv',li/'scenarios.csv',ROOT/'results/li2023_r105/scenarios.csv']
    out={'example_76000':values,'example_unit_metric_lower_counts':counts,'small_20_per_day_reduction':per_day,'small_daily':daily,'scale_runs':groups,'sources_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sources}}
    result=ROOT/'results/pressure21_v1';(result/'figure_editorial_values.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('Figure-derived editorial values written')


if __name__=='__main__':main()
