"""Evaluate frozen Pressure state-group ablations on the original formal cohorts."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evcharging.models import imitation_network
from evcharging.simulation_schema import simulate
from evcharging.state_ablation import mask_state
from evaluate_pressure21 import CONFIG,OUT,ARRAYS,digest,load_case

VARIANTS={'LocalRefined':'local','WithoutA':'without_a','WithoutB':'without_b','WithoutC':'without_c'}


def model_info(name: str,seed: int) -> tuple[Path,dict]:
    """Identify the promoted model and its pre-test selection record.

    :param name: Variant name.
    :param seed: Independent training seed.
    :return: Model path and frozen selection record.
    """
    variant=VARIANTS[name]
    selection=json.loads((ROOT/'artifacts/runs/pressure21_v1/ablation_selection.json').read_text())
    if selection['state_schema']!='pressure21_v1' or selection['formal_test_used_for_selection']:raise RuntimeError('Invalid selection protocol')
    record=next(r for r in selection['models'] if r['variant']==variant and r['seed']==seed)
    return Path(record['path']),record


def run_one(task: tuple[str,str,int,int,int,int,str]) -> dict:
    """Evaluate and physically audit one model on one frozen paper scenario.

    :param task: Dataset, model variant, training seed, day, fleet, limit and output.
    :return: Scenario-level metrics and provenance.
    """
    dataset,name,train,seed,n,limit,out=task
    torch.set_num_threads(1);cfg=json.loads(CONFIG.read_text());s,input_path,input_hash=load_case(dataset,seed,n,cfg)
    path,summary=model_info(name,train);variant=VARIANTS[name]
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if (ck.get('variant'),ck.get('seed'),ck.get('stage'))!=(variant,train,'continuation') or digest(path)!=summary['sha256']: raise RuntimeError('Frozen model mismatch')
    model=imitation_network();model.load_state_dict(ck['model']);model.eval()
    def policy(x: np.ndarray,k: int,urgency: np.ndarray) -> float:
        """Apply the model-specific training mask to the causal Pressure state.

        :param x: Full Pressure features.
        :param k: Vehicle index.
        :param urgency: Fleet urgency.
        :return: Normalized action.
        """
        del k,urgency
        x=mask_state(x,variant)
        with torch.no_grad():return float(model(torch.tensor(x,dtype=torch.float32)[None,:])[0,0])
    r=simulate(s,limit,policy,state_schema='pressure21_v1'); a,soc=r['final'],r['soc'];t=np.arange(len(s.price))[:,None];active=(s.ta[None,:]<=t)&(t<s.td[None,:]);rem=np.maximum(s.td-t,1)
    urg=np.where(active,np.maximum(s.target-soc[:-1],0)*s.capacity/(s.efficiency*s.power*s.dt*rem),0);shares=np.divide(urg,urg.sum(axis=1,keepdims=True),out=np.zeros_like(urg),where=urg.sum(axis=1,keepdims=True)>0)
    cost=a*s.power*s.price[:,None]*s.dt;unfair=(a-a.sum(axis=1,keepdims=True)*shares)**2
    checks=[np.abs(cost-r['cost']).max(),np.abs(unfair-r['unfair']).max(),np.abs(soc[1:]-soc[:-1]-a*s.power*s.efficiency*s.dt/s.capacity).max(),np.maximum(a@s.power-limit,0).max(),np.maximum(-a,0).max(),np.maximum(a-1,0).max(),np.abs(a[~active]).max(initial=0),np.abs(soc[0]-s.arrival).max(),np.maximum(.2-soc,0).max(),np.maximum(soc-1,0).max(),np.maximum(soc-s.target,0).max()]
    residual=float(max(checks))
    if not np.isfinite(residual) or residual>cfg['physical_tolerance']: raise RuntimeError(f'Physical audit failed: {task}')
    gap=np.maximum(s.target-soc[s.td,np.arange(n)],0);failed=int((gap>cfg['soc_tolerance']).sum());c=float(cost.sum());d=float(unfair.sum());energy=float((a*s.power*s.dt).sum())
    destination=Path(out)/f'{name}_{train}_{n}_{seed}_{limit}.npz';np.savez_compressed(destination,**{key:r[key] for key in ARRAYS})
    row={'dataset':dataset,'method':name,'training_seed':train,'seed':seed,'vehicles':n,'capacity_kw':limit,'cost':c,'unfairness':d,'weighted_objective':.7*c+.3*d,'all_met':int(failed==0),'failed_vehicles':failed,'vehicle_met_percent':100*(1-failed/n),'gap_kwh':float(gap@s.capacity),'purchased_kwh':energy,'total_spp_l1':float(np.abs(r['raw']-a).sum()),'physical_max_residual':residual,'input_path':str(input_path),'input_sha256':input_hash,'model_sha256':digest(path),'trajectory_path':str(destination),'trajectory_sha256':digest(destination)}
    destination.with_suffix('.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    return row


def main() -> None:
    """Run formal or scalability ablations with fixed preselected models.

    :return: None.
    """
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['formal','scale'],required=True);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Workers must be 1-4')
    cfg=json.loads(CONFIG.read_text());out=OUT/'ablation'/args.dataset
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    sources=[CONFIG,Path(__file__),ROOT/'scripts/evaluate_pressure21.py',ROOT/'artifacts/runs/pressure21_v1/ablation_selection.json',ROOT/'src/evcharging/state_schema.py',ROOT/'src/evcharging/state_ablation.py',ROOT/'src/evcharging/core.py',ROOT/'src/evcharging/simulation_schema.py',ROOT/'src/evcharging/models.py']
    models={}
    for name in VARIANTS:
        for seed in cfg['training_seeds']:
            model,meta=model_info(name,seed);ck=torch.load(model,map_location='cpu',weights_only=True)
            if digest(model)!=meta['sha256'] or (ck.get('variant'),ck.get('seed'),ck.get('stage'))!=(VARIANTS[name],seed,'continuation'):raise RuntimeError(f'Model mismatch: {name} {seed}')
            models[f'{name}_{seed}']={'path':str(model),'sha256':digest(model)}
    (out/'manifest.json').write_text(json.dumps({'models':models,'sources_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sources},'dataset':args.dataset,'workers':args.workers,'threads_per_process':1},indent=2),encoding='utf8')
    if args.dataset=='formal':tasks=[(args.dataset,name,train,seed,200,cap,str(out)) for name in VARIANTS for train in cfg['training_seeds'] for seed in cfg['formal_seeds'] for cap in cfg['formal_capacities_kw']]
    else:tasks=[(args.dataset,name,train,81000+10*n+i,n,4*n,str(out)) for name in VARIANTS for train in cfg['training_seeds'] for n in cfg['scale_additional_training_sizes'] for i in range(3)]
    rows=[]
    if args.workers==1:
        for i,task in enumerate(tasks,1):
            rows.append(run_one(task))
            if i%10==0: print(args.dataset,'ablation',i,'/',len(tasks),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,row in enumerate(pool.map(run_one,tasks),1):
                rows.append(row)
                if i%10==0:print(args.dataset,'ablation',i,'/',len(tasks),flush=True)
    with (out/'scenarios.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'complete.json').write_text(json.dumps({'records':len(rows),'sha256':digest(out/'scenarios.csv')},indent=2),encoding='utf8')
    print(args.dataset,'ablation complete',len(rows),flush=True)


if __name__=='__main__':main()
