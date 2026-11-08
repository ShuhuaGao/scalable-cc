"""Run the original order-sensitivity design with the selected Pressure policy."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'artifacts/runs/pressure21_v1/order_sensitivity'
sys.path.insert(0,str(ROOT/'src'))
from evcharging.core import Scenario, load_scenario
from evcharging.models import imitation_network
import evcharging.simulation_schema as simulation
from evcharging.state_schema import build_state

CONFIG=ROOT/'configs/order_pressure21.json'
FIELDS=('ta','td','arrival','target','capacity','power')
TRAJECTORY=('soc','raw','clip','scale','final','cost','unfair')


def sha(path: Path) -> str:
    """Return the SHA-256 of a file.

    :param path: Input file.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def permute(s: Scenario, order: np.ndarray) -> Scenario:
    """Map new processing indices to original EVs without changing price.

    :param s: Original scenario.
    :param order: Complete new-index-to-original-index permutation.
    :return: Scenario with all EV fields permuted together.
    """
    if not np.array_equal(np.sort(order),np.arange(len(s.ta))): raise ValueError('Order must be a complete permutation')
    return replace(s, **{name:getattr(s,name)[order].copy() for name in FIELDS})


def pressure_state(s: Scenario, t: int, k: int, soc: np.ndarray, proposed: np.ndarray, limit: float) -> np.ndarray:
    """Provide the frozen Pressure representation to the original simulator.

    :param s: Current permuted scenario.
    :param t: Operating-day slot.
    :param k: Vehicle's current processing index.
    :param soc: Current state of charge.
    :param proposed: Raw proposals already generated in this slot.
    :param limit: Station power capacity.
    :return: The selected 21-dimensional causal state.
    """
    return build_state(s,t,k,soc,proposed,limit,'pressure21_v1')


def mapped_simulation(s: Scenario, order: np.ndarray, model: torch.nn.Module, limit: int) -> dict:
    """Run original simulation/SPP in an order and map vehicle arrays back.

    :param s: Original scenario.
    :param order: New processing index to original vehicle index.
    :param model: Frozen Pressure network.
    :param limit: Station capacity.
    :return: Original-vehicle-order trajectories and decision-order metadata.
    """
    p=permute(s,order)
    def policy(x: np.ndarray,k: int,urgency: np.ndarray) -> float:
        """Evaluate a deterministic action from the frozen network.

        :param x: Pressure features.
        :param k: Vehicle's permuted index.
        :param urgency: Current fleet urgency.
        :return: Normalized proposed action.
        """
        del k,urgency
        with torch.no_grad(): return float(model(torch.tensor(x,dtype=torch.float32)[None,:])[0,0])
    r=simulation.simulate(p,limit,policy,state_schema='pressure21_v1')
    inv=np.argsort(order)
    mapped={key:r[key][:,inv] for key in TRAJECTORY}
    records=r['records'].copy(); records[:,1]=order[records[:,1]]
    mapped.update(states=r['states'],records=records,order=order.copy())
    return mapped


def audit(s: Scenario, trajectory: dict, cfg: dict) -> dict:
    """Reconstruct physical constraints, objective, service and energy gap.

    :param s: Original scenario.
    :param trajectory: Mapped trajectory arrays.
    :param cfg: Frozen experimental configuration.
    :return: Scenario-level metrics and physical residuals.
    """
    active=(np.arange(len(s.price))[:,None]>=s.ta)&(np.arange(len(s.price))[:,None]<s.td)
    action,soc=trajectory['final'],trajectory['soc']
    remain=np.maximum(s.td-np.arange(len(s.price))[:,None],1)
    urgency=np.where(active,np.maximum(s.target-soc[:-1],0)*s.capacity/(s.efficiency*s.power*s.dt*remain),0)
    totals=urgency.sum(axis=1,keepdims=True); shares=np.divide(urgency,totals,out=np.zeros_like(urgency),where=totals>0)
    cost=action*s.power*s.price[:,None]*s.dt; unfair=(action-action.sum(axis=1,keepdims=True)*shares)**2
    residuals={'dynamics':np.abs(soc[1:]-soc[:-1]-action*s.power*s.efficiency*s.dt/s.capacity).max(),'station':np.maximum(action@s.power-cfg['capacity_kw'],0).max(),'action_lower':np.maximum(-action,0).max(),'action_upper':np.maximum(action-1,0).max(),'disconnected':np.abs(action[~active]).max(initial=0),'soc_lower':np.maximum(.2-soc,0).max(),'soc_upper':np.maximum(soc-1,0).max(),'overcharge':np.maximum(soc-s.target,0).max(),'initial_soc':np.abs(soc[0]-s.arrival).max(),'cost':np.abs(trajectory['cost']-cost).max(),'unfairness':np.abs(trajectory['unfair']-unfair).max()}
    residual=float(max(residuals.values()))
    if not np.isfinite(residual) or residual>cfg['physical_tolerance']: raise RuntimeError(f'Physical audit failed: {residuals}')
    gap=np.maximum(s.target-soc[s.td,np.arange(len(s.ta))],0); failed=int((gap>cfg['soc_tolerance']).sum()); c,d=float(cost.sum()),float(unfair.sum())
    return {'cost':c,'unfairness':d,'objective':.7*c+.3*d,'all_met':int(failed==0),'vehicle_met_percent':100*(1-failed/len(s.ta)),'failed_vehicles':failed,'gap_kwh':float(gap@s.capacity),'max_residual':residual,**{f'residual_{key}':float(value) for key,value in residuals.items()}}


def model_for(cfg: dict) -> torch.nn.Module:
    """Load the prespecified selected Pressure checkpoint.

    :param cfg: Frozen experimental configuration.
    :return: Evaluation-mode model.
    """
    torch.set_num_threads(1); model=imitation_network(); checkpoint=torch.load(ROOT/cfg['model'],map_location='cpu',weights_only=True)
    if checkpoint['variant']!='pressure' or checkpoint['seed']!=11002: raise ValueError('Unexpected checkpoint')
    model.load_state_dict(checkpoint['model']); model.eval(); return model


def preflight() -> None:
    """Freeze configuration, inputs, code, model, orders and identity references.

    :return: None.
    """
    cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig')); old=ROOT/cfg['orders']; freeze=json.loads((ROOT/cfg['model_freeze']).read_text()); selected=next(x for x in freeze['models'] if x['seed']==11002)
    if sha(ROOT/cfg['model'])!=selected['sha256'] or selected['state_schema']!='pressure21_v1': raise RuntimeError('Pressure model or API changed')
    rng=np.random.default_rng(cfg['order_seed'])
    with np.load(old) as archived:
        if set(archived.files)!={str(seed) for seed in cfg['scenario_seeds']}: raise RuntimeError('Original order set differs')
        for seed in cfg['scenario_seeds']:
            expected=np.stack([rng.permutation(cfg['vehicles']) for _ in range(cfg['orders_per_scenario'])])
            if not np.array_equal(expected,archived[str(seed)]): raise RuntimeError(f'Original RNG order mismatch: {seed}')
    paths=[CONFIG,Path(__file__),ROOT/cfg['orders'],ROOT/cfg['model'],ROOT/cfg['model_freeze'],ROOT/cfg['features'],ROOT/'src/evcharging/core.py',ROOT/'src/evcharging/simulation_schema.py',ROOT/'src/evcharging/models.py']
    paths += [ROOT/cfg['input_dir']/f'{seed}.npz' for seed in cfg['scenario_seeds']]
    paths += [ROOT/cfg['identity_reference_template'].format(seed=seed) for seed in cfg['scenario_seeds']]
    record={'config':cfg,'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'threads':1,'order_mapping':'new processing index to original vehicle index, fixed all day','identity_reference_fields':list(TRAJECTORY),'identity_reference_does_not_include':['states','records'],'order_reproduced_from_seed':True}
    dest=HERE/'manifest.json'
    if dest.exists(): raise FileExistsError('Preflight manifest already exists')
    HERE.mkdir(parents=True,exist_ok=True)
    (HERE/'execution').mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),HERE/'execution/order_pressure21.py'); shutil.copyfile(CONFIG,HERE/'execution/order_pressure21.json')
    dest.write_text(json.dumps(record,indent=2)); print('Preflight frozen; orders reproduce exactly.',flush=True)


def verify_manifest() -> dict:
    """Verify all frozen inputs and sources before executing any scenario.

    :return: Experimental configuration.
    """
    record=json.loads((HERE/'manifest.json').read_text());
    for name,value in record['sha256'].items():
        if sha(ROOT/name)!=value: raise RuntimeError(f'Frozen source or input changed: {name}')
    return record['config']


def identities() -> None:
    """Produce the five actual identity runs and check seven archived arrays.

    :return: None.
    """
    cfg=verify_manifest(); model=model_for(cfg); dest=HERE/'trajectories'; dest.mkdir(exist_ok=True); checks=[]
    for seed in cfg['scenario_seeds']:
        s=load_scenario(str(ROOT/cfg['input_dir']/f'{seed}.npz')); identity=np.arange(len(s.ta)); result=mapped_simulation(s,identity,model,cfg['capacity_kw'])
        reference=ROOT/cfg['identity_reference_template'].format(seed=seed)
        with np.load(reference) as old:
            for key in TRAJECTORY:
                if not np.array_equal(result[key],old[key]): raise RuntimeError(f'Identity mismatch: {seed} {key}')
        metrics=audit(s,result,cfg); path=dest/f'{seed}_-01.npz'
        if path.exists(): raise FileExistsError(path)
        np.savez_compressed(path,**result); checks.append({'seed':seed,'equal_fields':list(TRAJECTORY),'reference_sha256':sha(reference),'trajectory_sha256':sha(path),'metrics':metrics})
        print('identity',seed,'matched seven arrays',flush=True)
    (HERE/'identity_checks.json').write_text(json.dumps(checks,indent=2))


def evaluate(seed: int) -> int:
    """Evaluate one day under the 100 fixed original permutations.

    :param seed: Day seed.
    :return: Completed day seed.
    """
    cfg=verify_manifest(); model=model_for(cfg); s=load_scenario(str(ROOT/cfg['input_dir']/f'{seed}.npz'))
    if len(s.ta)!=cfg['vehicles'] or s.beta!=cfg['beta']: raise ValueError('Input configuration mismatch')
    checks=json.loads((HERE/'identity_checks.json').read_text()); identity=next(x for x in checks if x['seed']==seed); dest=HERE/'trajectories'; ipath=dest/f'{seed}_-01.npz'
    if sha(ipath)!=identity['trajectory_sha256']: raise RuntimeError('Identity trajectory changed')
    rows=[{'seed':seed,'order_index':-1,**identity['metrics'],'trajectory_sha256':sha(ipath)}]
    with np.load(ROOT/cfg['orders']) as old: orders=old[str(seed)].copy()
    for i,order in enumerate(orders):
        path=dest/f'{seed}_{i:03}.npz'
        if path.exists(): raise FileExistsError(path)
        result=mapped_simulation(s,order,model,cfg['capacity_kw']); metrics=audit(s,result,cfg); np.savez_compressed(path,**result)
        rows.append({'seed':seed,'order_index':i,**metrics,'trajectory_sha256':sha(path)})
        with (HERE/f'{seed}.csv').open('w',newline='',encoding='utf8') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        if (i+1)%10==0: print(seed,i+1,'/ 100',flush=True)
    return seed


def summarize(cfg: dict) -> None:
    """Summarize random orders within days and then across the five days.

    :param cfg: Frozen experimental configuration.
    :return: None.
    """
    rows=[]
    for seed in cfg['scenario_seeds']:
        with (HERE/f'{seed}.csv').open(newline='',encoding='utf8') as stream: data=list(csv.DictReader(stream))
        if len(data)!=101 or [int(r['order_index']) for r in data]!=list(range(-1,100)): raise RuntimeError('Incomplete or misordered data')
        base,random=data[0],data[1:]; row={'seed':seed}
        for key in ('objective','cost','unfairness','all_met','vehicle_met_percent','gap_kwh'):
            a=np.asarray([float(x[key]) for x in random]); b=float(base[key]); row.update({f'{key}_original':b,f'{key}_mean':float(a.mean()),f'{key}_sd':float(a.std(ddof=1)),f'{key}_min':float(a.min()),f'{key}_max':float(a.max())})
            if key in ('objective','cost','unfairness'):
                for label,value in (('mean',a.mean()),('min',a.min()),('max',a.max())): row[f'{key}_{label}_relative_percent']=100*(float(value)/b-1)
                row[f'{key}_cv_percent']=100*float(a.std(ddof=1))/float(a.mean())
                row[f'{key}_max_abs_relative_percent']=float(np.max(np.abs(100*(a/b-1))))
        row['max_residual']=max(float(x['max_residual']) for x in data); rows.append(row)
    with (HERE/'scenario_summary.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    across={key:{'mean':float(np.mean([row[key] for row in rows])),'sample_sd':float(np.std([row[key] for row in rows],ddof=1))} for key in rows[0] if key!='seed'}
    (HERE/'summary.json').write_text(json.dumps({'scenarios':rows,'across_scenarios':across,'orders_sha256':sha(ROOT/cfg['orders']),'manifest_sha256':sha(HERE/'manifest.json')},indent=2))


def main() -> None:
    """Run the frozen order experiment with up to three worker processes.

    :return: None.
    """
    cfg=verify_manifest()
    if not (HERE/'identity_checks.json').exists(): raise RuntimeError('Run identity checks first')
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for seed in pool.map(evaluate,cfg['scenario_seeds']): print('finished',seed,flush=True)
    summarize(cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--preflight',action='store_true'); parser.add_argument('--identities',action='store_true'); args=parser.parse_args()
    if args.preflight: preflight()
    elif args.identities: identities()
    else: main()
