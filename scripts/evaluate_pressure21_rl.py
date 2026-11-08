"""Evaluate historical RL archives, or a specified frozen selection."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evcharging.models import network
from evcharging.simulation_schema import simulate
from evaluate_pressure21 import CONFIG,OUT,ARRAYS,digest,load_case
from pressure21_current_rl import local_path


def run_one(task: tuple[str,str,int,int,int,int,str], *, pressure_reference: bool = False) -> dict:
    """Evaluate one selected RL checkpoint and audit one daily trajectory.

    :param task: Dataset, method, training seed, day seed, fleet, limit, output.
    :param pressure_reference: Reproduce the repeated-aggregation scheduler used by some archived runs.
    :return: Scenario metrics and provenance.
    """
    dataset,method,train,seed,n,limit,out,*extra=task;torch.set_num_threads(1);cfg=json.loads(CONFIG.read_text());s,input_path,input_hash=load_case(dataset,seed,n,cfg)
    selection_path=Path(extra[0]) if extra else ROOT/'artifacts/runs/pressure21_v1/rl_selection.json'; selection=json.loads(selection_path.read_text());key='ddqn' if method=='DDQN' else 'ppo_per_run'
    choice=next(x for x in selection[key] if x['seed']==train);path=local_path(choice['path'])
    if choice['state_schema']!='pressure21_v1' or digest(path)!=choice['sha256'] or selection['formal_test_used_for_selection']:raise RuntimeError('RL selection is not frozen before formal testing')
    ck=torch.load(path,map_location='cpu',weights_only=(method!='DDQN'))
    if ck['config'].get('state_schema')!='pressure21_v1':raise RuntimeError('Checkpoint schema mismatch')
    model=network(6 if method=='DDQN' else 2);model.load_state_dict(ck['model']);model.eval()
    def policy(x: np.ndarray,k: int,urgency: np.ndarray) -> float:
        """Return a deterministic DDQN or PPO action.

        :param x: Current Pressure state.
        :param k: Vehicle index.
        :param urgency: Fleet urgency.
        :return: Normalized proposed action.
        """
        del k,urgency
        with torch.no_grad():
            z=model(torch.tensor(x,dtype=torch.float32)[None,:])
            if method=='DDQN':return float(z.argmax())/5
            ab=torch.nn.functional.softplus(z)+1;return float(ab[0,0]/ab.sum())
    r=simulate(s,limit,policy,state_schema='pressure21_v1',pressure_reference=pressure_reference);a,soc=r['final'],r['soc'];t=np.arange(len(s.price))[:,None];active=(s.ta[None,:]<=t)&(t<s.td[None,:]);rem=np.maximum(s.td-t,1)
    urg=np.where(active,np.maximum(s.target-soc[:-1],0)*s.capacity/(s.efficiency*s.power*s.dt*rem),0);shares=np.divide(urg,urg.sum(axis=1,keepdims=True),out=np.zeros_like(urg),where=urg.sum(axis=1,keepdims=True)>0)
    cost=a*s.power*s.price[:,None]*s.dt;unfair=(a-a.sum(axis=1,keepdims=True)*shares)**2
    checks=[np.abs(cost-r['cost']).max(),np.abs(unfair-r['unfair']).max(),np.abs(soc[1:]-soc[:-1]-a*s.power*s.efficiency*s.dt/s.capacity).max(),np.maximum(a@s.power-limit,0).max(),np.maximum(-a,0).max(),np.maximum(a-1,0).max(),np.abs(a[~active]).max(initial=0),np.abs(soc[0]-s.arrival).max(),np.maximum(.2-soc,0).max(),np.maximum(soc-1,0).max(),np.maximum(soc-s.target,0).max()]
    residual=float(max(checks))
    if not np.isfinite(residual) or residual>cfg['physical_tolerance']:raise RuntimeError(f'Physical audit failed: {task}')
    gap=np.maximum(s.target-soc[s.td,np.arange(n)],0);failed=int((gap>cfg['soc_tolerance']).sum());c=float(cost.sum());d=float(unfair.sum());energy=float((a*s.power*s.dt).sum())
    dest=Path(out)/f'{method}_{train}_{n}_{seed}_{limit}.npz';np.savez_compressed(dest,**{key:r[key] for key in ARRAYS})
    row={'dataset':dataset,'method':method,'run_id':f'{selection_path.parent.name}/{method.lower()}_{train}','training_seed':train,'seed':seed,'vehicles':n,'capacity_kw':limit,'cost':c,'unfairness':d,'weighted_objective':.7*c+.3*d,'all_met':int(failed==0),'failed_vehicles':failed,'vehicle_met_percent':100*(1-failed/n),'gap_kwh':float(gap@s.capacity),'purchased_kwh':energy,'total_spp_l1':float(np.abs(r['raw']-a).sum()),'physical_max_residual':residual,'input_path':str(input_path),'input_sha256':input_hash,'model_sha256':digest(path),'trajectory_path':str(dest),'trajectory_sha256':digest(dest)}
    dest.with_suffix('.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    return row


def main() -> None:
    """Evaluate the three runs per RL method and the main small-scale pair.

    :return: None.
    """
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['development','formal','small'],required=True);parser.add_argument('--workers',type=int,default=3);parser.add_argument('--selection',type=Path,default=ROOT/'artifacts/runs/public_rl/rl_selection.json');parser.add_argument('--output',type=Path,default=ROOT/'outputs/public_rl');args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Workers must be 1-4')
    cfg=json.loads(CONFIG.read_text());selection_path=args.selection.resolve();selection=json.loads(selection_path.read_text())
    if selection['state_schema']!='pressure21_v1' or selection['formal_test_used_for_selection']:raise RuntimeError('Invalid model selection protocol')
    choices=[('DDQN',r['seed']) for r in selection['ddqn']]+[('PPO',r['seed']) for r in selection['ppo_per_run']]
    if len(choices)!=6 or len(set(choices))!=6:raise RuntimeError('Expected three independent runs per method')
    main_ppo=selection['ppo_main']; main_ddqn=selection['main_ddqn_seed']
    selected=next(r for r in selection['ppo_per_run'] if r['seed']==main_ppo['seed'])
    if selected['sha256']!=main_ppo['sha256']:raise RuntimeError('Main PPO checkpoint requires a separate evaluation')
    out=args.output/args.dataset
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    sources=[CONFIG,Path(__file__),ROOT/'scripts/evaluate_pressure21.py',selection_path,ROOT/'src/evcharging/core.py',ROOT/'src/evcharging/state_schema.py',ROOT/'src/evcharging/simulation_schema.py',ROOT/'src/evcharging/models.py']
    (out/'manifest.json').write_text(json.dumps({'sources_sha256':{p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name:digest(p) for p in sources},'choices':choices,'main_ddqn_seed':main_ddqn,'main_ppo_seed':main_ppo['seed'],'threads_per_process':1,'workers':args.workers},indent=2),encoding='utf8')
    if args.dataset=='development':tasks=[('development',method,train,seed,200,cap,str(out)) for method,train in [('DDQN',main_ddqn),('PPO',main_ppo['seed'])] for seed in cfg['development_seeds'] for cap in cfg['formal_capacities_kw']]
    elif args.dataset=='formal':tasks=[('formal',method,train,seed,200,cap,str(out)) for method,train in choices for seed in cfg['formal_seeds'] for cap in cfg['formal_capacities_kw']]
    else:tasks=[('small',method,train,seed,10,40,str(out)) for method,train in [('DDQN',main_ddqn),('PPO',main_ppo['seed'])] for seed in cfg['small_seeds']]
    tasks=[(*task, str(selection_path)) for task in tasks]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,row in enumerate(pool.map(run_one,tasks),1):
            rows.append(row)
            if i%10==0:print(args.dataset,'RL',i,'/',len(tasks),flush=True)
    with (out/'scenarios.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'complete.json').write_text(json.dumps({'records':len(rows),'sha256':digest(out/'scenarios.csv')},indent=2),encoding='utf8');print(args.dataset,'RL complete',len(rows),flush=True)


if __name__=='__main__':main()
