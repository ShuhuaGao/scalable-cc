"""Audit existing state-independent formal references and summarize Pressure."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean,stdev

import numpy as np

from evaluate_pressure21 import ROOT,OUT,CONFIG,digest
from evcharging.core import load_scenario
from paper_rddpg_data import common_metrics


def read_csv(path: Path) -> list[dict]:
    """Read a UTF-8 CSV file.

    :param path: CSV path.
    :return: Rows.
    """
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def main() -> None:
    """Verify references and report matched scenario and training-run summaries.

    :return: None.
    """
    cfg=json.loads(CONFIG.read_text());out=ROOT/'results/pressure21_v1';out.mkdir(parents=True,exist_ok=True)
    primary=read_csv(OUT/'formal/scenarios.csv'); reference=ROOT/'artifacts/runs/ideal10h_v1/formal';origin=json.loads((reference/'manifest.json').read_text()); audit=read_csv(ROOT/'results/ideal10h_v1/physical_audit.csv'); audit_map={(int(r['seed']),int(r['capacity_kw']),r['method']):r for r in audit}
    if len(primary)!=300:raise RuntimeError('Pressure formal evaluation incomplete')
    li_manifest=ROOT/'artifacts/runs/li2023_r105/paper_33001/manifest.json';li=json.loads(li_manifest.read_text());rows=[]
    for seed in cfg['formal_seeds']:
        input_path=reference/'inputs'/f'{seed}.npz'
        if digest(input_path)!=origin['input_sha256'][str(seed)]:raise RuntimeError(f'Historical formal input changed: {seed}')
        s=load_scenario(str(input_path))
        for limit in cfg['formal_capacities_kw']:
            for method in ('UC','UPA','RDDPG'):
                if method=='RDDPG':
                    path=ROOT/f'artifacts/runs/li2023_r105/evaluation/33001/200_{seed}_{limit}.npz';key=path.relative_to(ROOT).as_posix();expected=li['sources'].get(key)
                    if expected is None or digest(path)!=expected:raise RuntimeError(f'RDDPG trajectory changed: {path}')
                    with np.load(path) as d:
                        result=common_metrics(s,d['power'],d['soc']);a,soc=result['final'],result['soc']
                else:
                    path=reference/f'{seed}_{limit}_{method}.npz'
                    with np.load(path) as d:a=d['final'].copy();soc=d['soc'].copy()
                t=np.arange(len(s.price))[:,None];active=(s.ta[None,:]<=t)&(t<s.td[None,:]);rem=np.maximum(s.td-t,1)
                urg=np.where(active,np.maximum(s.target-soc[:-1],0)*s.capacity/(s.efficiency*s.power*s.dt*rem),0);shares=np.divide(urg,urg.sum(axis=1,keepdims=True),out=np.zeros_like(urg),where=urg.sum(axis=1,keepdims=True)>0)
                costs=a*s.power*s.price[:,None]*s.dt;unfair=(a-a.sum(axis=1,keepdims=True)*shares)**2
                gap=np.maximum(s.target-soc[s.td,np.arange(len(s.ta))],0);failed=int((gap>cfg['soc_tolerance']).sum());energy=float((a*s.power*s.dt).sum()); c=float(costs.sum());d=float(unfair.sum())
                residual=float(max(np.abs(soc[1:]-soc[:-1]-a*s.power*s.efficiency*s.dt/s.capacity).max(),np.maximum(a@s.power-limit,0).max(),np.maximum(-a,0).max(),np.maximum(a-1,0).max(),np.abs(a[~active]).max(initial=0),np.abs(soc[0]-s.arrival).max(),np.maximum(.2-soc,0).max(),np.maximum(soc-1,0).max(),np.maximum(soc-s.target,0).max()))
                if residual>1e-8:raise RuntimeError(f'Physical failure {path}: {residual}')
                if method!='RDDPG':
                    original=audit_map[seed,limit,method]
                    for k,v in [('cost',c),('unfairness',d),('gap_kwh',float(gap@s.capacity)),('failed_vehicles',failed)]:
                        if not np.isclose(float(original[k]),v,rtol=1e-10,atol=1e-10):raise RuntimeError(f'Reference audit changed {path} {k}')
                rows.append({'dataset':'formal','method':method,'seed':seed,'vehicles':200,'capacity_kw':limit,'cost':c,'unfairness':d,'weighted_objective':.7*c+.3*d,'all_met':int(failed==0),'failed_vehicles':failed,'vehicle_met_percent':100*(1-failed/200),'gap_kwh':float(gap@s.capacity),'purchased_kwh':energy,'physical_max_residual':residual,'input_sha256':digest(input_path),'trajectory_path':str(path),'trajectory_sha256':digest(path)})
    with (out/'formal_references.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    report=[]
    for cap in cfg['formal_capacities_kw']:
        for method in ('IL','UC','UPA','RDDPG'):
            seeds=cfg['training_seeds'] if method=='IL' else [None]
            for train in seeds:
                selected=[r for r in (primary if method=='IL' else rows) if int(r['capacity_kw'])==cap and r['method']==method and (train is None or int(r['training_seed'])==train)]
                if len(selected)!=50:raise RuntimeError('Unpaired formal cohort')
                m={'method':method,'capacity_kw':cap,'training_seed':train,'scenarios':len(selected)}
                for k in ('cost','unfairness','weighted_objective','all_met','vehicle_met_percent','gap_kwh','purchased_kwh'):
                    m[k]=mean(float(r[k]) for r in selected)
                m['cost_per_purchased_kwh']=sum(float(r['cost']) for r in selected)/sum(float(r['purchased_kwh']) for r in selected)
                m['failed_vehicles']=sum(int(r['failed_vehicles']) for r in selected)
                report.append(m)
    repeat=[]
    for cap in cfg['formal_capacities_kw']:
        subset=[r for r in report if r['method']=='IL' and r['capacity_kw']==cap]
        repeat.append({'capacity_kw':cap,'runs':len(subset),'statistics':{k:{'mean':mean(r[k] for r in subset),'sample_sd':stdev(r[k] for r in subset)} for k in ('cost','unfairness','weighted_objective','all_met','vehicle_met_percent','gap_kwh','purchased_kwh','cost_per_purchased_kwh')}})
    result={'formal_run_summaries':report,'pressure_across_training_runs':repeat,'formal_references_sha256':digest(out/'formal_references.csv'),'pressure_scenarios_sha256':digest(OUT/'formal/scenarios.csv'),'reference_audit_sha256':digest(ROOT/'results/ideal10h_v1/physical_audit.csv'),'rddpg_manifest_sha256':digest(li_manifest),'note':'DDQN/PPO pending retraining; baseline references are not independent training repetitions.'}
    (out/'formal_summary_partial.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('Audited 300 state-independent formal references; saved partial summary')


if __name__=='__main__':main()
