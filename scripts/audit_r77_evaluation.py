"""Independently recompute saved R77 evaluation metrics without simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/runs/pressure21_v1/evaluation/r77_beta'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betas', nargs='+', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/pressure21_evaluation.json').read_text())
    rows_by_beta = {}; max_errors = {}; physical_max = 0.; failures = []
    for beta in args.betas:
        folder = BASE/f'beta_{beta:.2f}'
        rows = list(csv.DictReader((folder/'scenarios.csv').open(encoding='utf8', newline='')))
        if len(rows) != 79:
            raise ValueError(f'Incomplete cohort for {beta}')
        identities = {}
        for row in rows:
            n = int(row['vehicles']); limit = float(row['capacity_kw'])
            identity = (row['dataset'], int(row['seed']), n, limit)
            if identity in identities or float(row['beta']) != beta:
                raise ValueError(f'Duplicate or wrong beta: {identity}')
            inp = Path(row['input_path']); traj = Path(row['trajectory_path'])
            if sha(inp) != row['input_sha256'] or sha(traj) != row['trajectory_sha256']:
                raise ValueError(f'Hash mismatch: {identity}')
            identities[identity] = row['input_sha256']
            with np.load(inp) as z:
                v = {k:z[k].copy() for k in ('ta','td','arrival','target','capacity','power','price','efficiency','dt')}
            with np.load(traj) as z:
                a = z['final'].copy(); soc = z['soc'].copy(); stored_c = z['cost'].copy(); stored_d = z['unfair'].copy()
            steps = len(v['price']); t = np.arange(steps)[:,None]
            if a.shape != (steps,n) or soc.shape != (steps+1,n):
                raise ValueError(f'Wrong trajectory dimensions: {identity}')
            active = (v['ta'] <= t) & (t < v['td'])
            g = np.where(active, np.maximum(v['target']-soc[:-1],0)*v['capacity']/(v['efficiency']*v['power']*v['dt']*np.maximum(v['td']-t,1)),0)
            den = g.sum(axis=1,keepdims=True)
            frac = np.divide(g,den,out=np.zeros_like(g),where=den>0)
            c = a*v['power']*v['price'][:,None]*v['dt']
            d = (a-frac*a.sum(axis=1,keepdims=True))**2
            checks = [np.abs(c-stored_c).max(),np.abs(d-stored_d).max(),np.abs(soc[1:]-soc[:-1]-a*v['power']*v['efficiency']*v['dt']/v['capacity']).max(),np.maximum(a@v['power']-limit,0).max(),np.maximum(-a,0).max(),np.maximum(a-1,0).max(),np.abs(a[~active]).max(initial=0),np.abs(soc[0]-v['arrival']).max(),np.maximum(.2-soc,0).max(),np.maximum(soc-1,0).max(),np.maximum(soc-v['target'],0).max()]
            residual=float(max(checks)); physical_max=max(physical_max,residual)
            gap=np.maximum(v['target']-soc[v['td'],np.arange(n)],0); failed=int((gap>config['soc_tolerance']).sum()); energy=float(gap@v['capacity']); demand=float((v['target']-v['arrival'])@v['capacity']); C=float(c.sum()); D=float(d.sum())
            expected={'cost':C,'unfairness':D,'cost_per_ev':C/n,'unfairness_per_ev':D/n,'all_met':int(failed==0),'failed_vehicles':failed,'vehicle_met_percent':100*(1-failed/n),'gap_kwh':energy,'battery_demand_kwh':demand,'gap_demand_percent':100*energy/demand,'physical_max_residual':residual}
            for key,value in expected.items():
                error=abs(value-float(row[key])); max_errors[key]=max(max_errors.get(key,0.),error)
                if not np.isclose(value,float(row[key]),rtol=1e-10,atol=1e-10): failures.append([beta,identity,key,error])
            if not np.isfinite(residual) or residual>config['physical_tolerance']: failures.append([beta,identity,'physical',residual])
        rows_by_beta[str(beta)] = identities
    first=next(iter(rows_by_beta.values()))
    matched=all(x==first for x in rows_by_beta.values())
    report={'betas':args.betas,'counts':{k:len(v) for k,v in rows_by_beta.items()},'same_case_identities_and_input_hashes':matched,'csv_max_absolute_errors':max_errors,'physical_max_residual':physical_max,'failures':failures,'source_sha256':sha(Path(__file__)),'csv_sha256':{str(b):sha(BASE/f'beta_{b:.2f}'/'scenarios.csv') for b in args.betas}}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
    if failures or not matched: raise RuntimeError('Independent audit failed')


if __name__ == '__main__': main()
