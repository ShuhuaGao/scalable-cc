"""Compute manuscript and response numbers from audited Pressure21 trajectories."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean,stdev

import numpy as np

from evaluate_pressure21 import ROOT,OUT,CONFIG,digest,load_case
from pressure21_current_rl import PPO_RUN, current_rl, local_path, selection
from evcharging.core import load_scenario


def read(path: Path) -> list[dict]:
    """Read CSV rows.

    :param path: CSV path.
    :return: Row dictionaries.
    """
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def main() -> None:
    """Report full-table, SPP, small-fleet and paired-bootstrap evidence.

    :return: None.
    """
    cfg=json.loads(CONFIG.read_text());result=ROOT/'results/pressure21_v1';primary=read(OUT/'formal/scenarios.csv');refs=read(result/'formal_references.csv');rl=current_rl('formal');small=read(OUT/'small/scenarios.csv');small_rl=current_rl('small')
    for folder in (OUT/'formal',OUT/'small',OUT/'rl/formal',OUT/'rl/small'):
        done=json.loads((folder/'complete.json').read_text());h=done.get('sha256',done.get('scenarios_sha256'))
        if digest(folder/'scenarios.csv')!=h:raise RuntimeError('Incomplete evaluation data')
    selected={'IL':11002,'DDQN':21002,'PPO':int(selection()['ppo_main']['seed'])}
    by_case={}
    for method,train in selected.items():
        source=primary if method=='IL' else rl
        for row in source:
            if row['method']==method and int(row['training_seed'])==train:by_case[method,int(row['seed']),int(row['capacity_kw'])]=row
    for row in refs:by_case[row['method'],int(row['seed']),int(row['capacity_kw'])]=row
    main_methods=('IL','UC','DDQN','PPO','UPA','RDDPG');rows=[];vehicles=[];spp=[];demand={600:0.,800:0.}
    for cap in cfg['formal_capacities_kw']:
        for seed in cfg['formal_seeds']:
            s=load_scenario(str(ROOT/cfg['formal_input_dir']/f'{seed}.npz'));demand[cap]+=float(((s.target-s.arrival)*s.capacity).sum())
            for method in main_methods:
                row=by_case[method,seed,cap];path=local_path(row['trajectory_path'])
                if digest(path)!=row['trajectory_sha256']:raise RuntimeError('Trajectory hash mismatch')
                with np.load(path) as data:
                    soc=data['soc'].copy();gap=np.maximum(s.target-soc[s.td,np.arange(200)],0);failed=gap>cfg['soc_tolerance'];energy_gap=gap*s.capacity
                    for k in np.flatnonzero(failed):vehicles.append({'method':method,'seed':seed,'capacity_kw':cap,'vehicle':int(k),'gap_pp':float(gap[k]*100),'gap_kwh':float(energy_gap[k])})
                    if method in selected or method in ('UC','UPA'):
                        arrays={key:data[key].copy() for key in ('raw','clip','scale','final')}
                        energies={key:float((value*s.power*s.dt).sum()) for key,value in arrays.items()}
                        scaling=np.any(np.abs(arrays['clip']-arrays['scale'])>1e-10,axis=1);active=(s.ta[None,:]<=np.arange(len(s.price))[:,None])&(np.arange(len(s.price))[:,None]<s.td[None,:]);needed=np.any(active & (soc[:-1] < s.target[None,:]-1e-12),axis=1)
                        before=np.maximum(s.target-soc[np.maximum(s.td-1,0),np.arange(200)],0)
                        spp.append({'method':method,'seed':seed,'capacity_kw':cap,**{key+'_kwh':value for key,value in energies.items()},'scaling_slots':int(scaling.sum()),'active_need_slots':int(needed.sum()),'added_kwh':energies['final']-energies['scale'],'last_slot_battery_gap_kwh':float(before@s.capacity)})
                rows.append({'method':method,'seed':seed,'capacity_kw':cap,**{k:float(row[k]) for k in ('cost','unfairness','weighted_objective','all_met','failed_vehicles','vehicle_met_percent','gap_kwh','purchased_kwh')}})
    table=[];comparisons=[]
    for cap in cfg['formal_capacities_kw']:
        for method in main_methods:
            selected_rows=[r for r in rows if r['method']==method and r['capacity_kw']==cap];miss=[v for v in vehicles if v['method']==method and v['capacity_kw']==cap]
            record={'method':method,'capacity_kw':cap,'scenarios':50,'all_met_percent':100*mean(r['all_met'] for r in selected_rows),'vehicle_met_percent':mean(r['vehicle_met_percent'] for r in selected_rows),'failed_vehicles':len(miss),'mean_failed_gap_pp':mean(x['gap_pp'] for x in miss) if miss else 0.,'mean_failed_gap_kwh':mean(x['gap_kwh'] for x in miss) if miss else 0.,'mean_total_positive_gap_kwh_per_day':mean(r['gap_kwh'] for r in selected_rows),'positive_gap_share_of_battery_demand_percent':100*sum(r['gap_kwh'] for r in selected_rows)/demand[cap],'mean_cost':mean(r['cost'] for r in selected_rows),'mean_unfairness':mean(r['unfairness'] for r in selected_rows),'mean_weighted_objective':mean(r['weighted_objective'] for r in selected_rows),'cost_per_purchased_kwh':sum(r['cost'] for r in selected_rows)/sum(r['purchased_kwh'] for r in selected_rows)}
            table.append(record)
        ours=next(r for r in table if r['capacity_kw']==cap and r['method']=='IL')
        for name in ('UC','DDQN','PPO','UPA','RDDPG'):
            other=next(r for r in table if r['capacity_kw']==cap and r['method']==name)
            comparisons.append({'capacity_kw':cap,'comparator':name,**{k+'_reduction_percent':100*(1-ours[k]/other[k]) for k in ('mean_cost','mean_unfairness','mean_weighted_objective','cost_per_purchased_kwh')}})
    spp_summary=[]
    for cap in cfg['formal_capacities_kw']:
        for method in ('IL','UC','DDQN','PPO','UPA'):
            group=[x for x in spp if x['capacity_kw']==cap and x['method']==method]
            item={'method':method,'capacity_kw':cap,**{key+'_mean_per_day':mean(x[key] for x in group) for key in ('raw_kwh','clip_kwh','scale_kwh','final_kwh','added_kwh','scaling_slots','active_need_slots','last_slot_battery_gap_kwh')}}
            item['added_share_of_final_percent']=100*sum(x['added_kwh'] for x in group)/sum(x['final_kwh'] for x in group)
            item['scaling_share_all_slots_percent']=100*sum(x['scaling_slots'] for x in group)/(len(group)*56)
            item['scaling_share_active_need_slots_percent']=100*sum(x['scaling_slots'] for x in group)/sum(x['active_need_slots'] for x in group)
            spp_summary.append(item)
    teacher=ROOT/'artifacts/runs/ideal10h_v1/small_evaluation/scenarios.csv';old=read(teacher);rddpg=read(ROOT/'artifacts/runs/li2023_r105/paper_33001/scenarios.csv');small_report=[]
    for method in ('IDEAL','IL','UC','DDQN','PPO','RDDPG'):
        src=small if method=='IL' else small_rl if method in ('DDQN','PPO') else rddpg if method=='RDDPG' else old
        group=[r for r in src if r['method']==method and int(r['seed']) in cfg['small_seeds'] and (method!='RDDPG' or int(r['vehicles'])==10)]
        if len(group)!=20:raise RuntimeError(f'Wrong small cohort for {method}')
        cost=mean(float(r['cost']) for r in group); unfair=mean(float(r['unfairness']) for r in group)
        small_report.append({'method':method,'cost':cost,'unfairness':unfair,'weighted_objective':.7*cost+.3*unfair})
    ideal=small_report[0]
    for row in small_report:
        for k in ('cost','unfairness','weighted_objective'):row[k+'_ratio_to_IDEAL']=row[k]/ideal[k]
    bootstrap=[];rng=np.random.default_rng(20261101)
    for cap in cfg['formal_capacities_kw']:
        days=sorted(cfg['formal_seeds']);indices=rng.integers(0,len(days),size=(10000,len(days)))
        for comparator in ('UC','DDQN','PPO','UPA','RDDPG'):
            left=[next(r for r in rows if r['method']=='IL' and r['capacity_kw']==cap and r['seed']==day) for day in days]
            right=[next(r for r in rows if r['method']==comparator and r['capacity_kw']==cap and r['seed']==day) for day in days]
            for key in ('cost','unfairness','weighted_objective','gap_kwh','vehicle_met_percent'):
                a=np.array([r[key] for r in left]);b=np.array([r[key] for r in right]);difference=a-b;draws=difference[indices].mean(axis=1)
                bootstrap.append({'capacity_kw':cap,'comparator':comparator,'metric':key,'ours_minus_comparator':float(difference.mean()),'ci95_low':float(np.quantile(draws,.025)),'ci95_high':float(np.quantile(draws,.975))})
    repeats=[]
    for method in ('IL','DDQN','PPO'):
        src=primary if method=='IL' else rl;training=sorted({int(r['training_seed']) for r in src if r['method']==method})
        for cap in cfg['formal_capacities_kw']:
            means=[]
            for train in training:
                group=[r for r in src if r['method']==method and int(r['training_seed'])==train and int(r['capacity_kw'])==cap]
                if len(group)!=50:raise RuntimeError('Incomplete training-repeat cohort')
                metrics={k:mean(float(r[k]) for r in group) for k in ('cost','unfairness','weighted_objective','all_met','vehicle_met_percent','gap_kwh')}
                metrics['cost_per_purchased_kwh']=sum(float(r['cost']) for r in group)/sum(float(r['purchased_kwh']) for r in group)
                means.append({'training_seed':train,**metrics})
            repeats.append({'method':method,'capacity_kw':cap,'run_means':means,'across_runs':{k:{'mean':mean(r[k] for r in means),'sample_sd':stdev(r[k] for r in means)} for k in means[0] if k!='training_seed'}})
    output={'formal_table':table,'relative_comparisons':comparisons,'spp':spp_summary,'small_20':small_report,'bootstrap':bootstrap,'training_repeats':repeats,'battery_demand_kwh_total':demand,'bootstrap_protocol':'10000 paired day resamples per capacity, seed 20261101; intervals are descriptive, no multiple-comparison adjustment','sources_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in [OUT/'formal/scenarios.csv',OUT/'rl/formal/scenarios.csv',OUT/'small/scenarios.csv',OUT/'rl/small/scenarios.csv',PPO_RUN/'evaluation/formal/scenarios.csv',PPO_RUN/'evaluation/small/scenarios.csv',PPO_RUN/'selection.json',result/'formal_references.csv',teacher,ROOT/'artifacts/runs/li2023_r105/paper_33001/scenarios.csv']}}
    (result/'editorial_values.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    with (result/'editorial_vehicles.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=vehicles[0]);w.writeheader();w.writerows(vehicles)
    with (result/'editorial_spp_scenarios.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=spp[0]);w.writeheader();w.writerows(spp)
    print('Editorial values generated from matched trajectories')


if __name__=='__main__':main()
