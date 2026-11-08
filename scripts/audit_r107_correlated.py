"""Independently audit correlated-arrival scenarios, trajectories and metrics."""
import csv,sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from evcharging.core import load_scenario
A=R/'artifacts/runs/pressure21_v1/r107_correlated';O=R/'results/pressure21_v1/r107'
def sha(p: Path) -> str:
    """Hash a file.

    :param p: File path.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main() -> None:
    """Recalculate every output, physical residual and input diagnostic.

    :return: None.
    """
    rows=list(csv.DictReader((O/'daily.csv').open(encoding='utf8',newline=''))); assert len(rows)==206
    diagnostics=[]; maxres=0.; arrivals=[];zs=[]
    for n,seed in [(200,i) for i in range(65000,65050)]+[(3000,i) for i in range(111000,111003)]:
        p=A/f'input_{n}_{seed}.npz';source=R/(f'artifacts/runs/ideal10h_v1/formal/inputs/{seed}.npz' if n==200 else f'artifacts/runs/ideal10h_v1/scalability/input_{n}_{seed}.npz')
        s=load_scenario(str(p));old=load_scenario(str(source))
        for k in ('arrival','target','capacity','power','price','efficiency','dt','beta'): assert np.array_equal(np.asarray(getattr(s,k)),np.asarray(getattr(old,k)))
        with np.load(p) as z:
            required=np.ceil(s.capacity*(s.target-s.arrival)/(s.efficiency*s.power*s.dt)).astype(int)
            assert np.array_equal(required,z['required']) and np.all(s.td-s.ta>=required)
            common=float(z['common']); latent=z['z'].copy()
            assert np.allclose(latent,(common+z['eps'])/np.sqrt(2),rtol=0,atol=1e-14)
        counts=np.bincount(s.ta,minlength=17); diagnostics.append(dict(n=n,seed=seed,common=common,arrival_mean=float(s.ta.mean()),arrival_sd=float(s.ta.std(ddof=1)),max_slot_fraction=float(counts.max()/n),latent_mean=float(latent.mean()),latent_sd=float(latent.std(ddof=1))))
        if n==200: arrivals.append(s.ta);zs.append(latent)
        for row in (x for x in rows if int(x['n'])==n and int(x['seed'])==seed):
            limit=int(row['limit']);method=row['method'];q=A/f'traj_{n}_{seed}_{limit}_{method}.npz'
            assert sha(p)==row['input_sha256'] and sha(source)==row['source_sha256'] and sha(q)==row['trajectory_sha256']
            with np.load(q) as z: a=z['final'];soc=z['soc'];stored_c=z['cost'];stored_d=z['unfair']
            t=np.arange(56)[:,None];active=(t>=s.ta)&(t<s.td);rem=np.maximum(s.td-t,1)
            u=np.where(active,np.maximum(s.target-soc[:-1],0)*s.capacity/(s.efficiency*s.power*s.dt*rem),0)
            share=np.divide(u,u.sum(axis=1,keepdims=True),out=np.zeros_like(u),where=u.sum(axis=1,keepdims=True)>0)
            c=a*s.power*s.price[:,None]*s.dt;d=(a-a.sum(axis=1,keepdims=True)*share)**2
            checks=[np.abs(soc[0]-s.arrival).max(),np.abs(soc[1:]-soc[:-1]-a*s.power*s.efficiency*s.dt/s.capacity).max(),np.maximum(a@s.power-limit,0).max(),np.maximum(-a,0).max(),np.maximum(a-1,0).max(),np.abs(a[~active]).max(initial=0),np.maximum(.2-soc,0).max(),np.maximum(soc-1,0).max(),np.maximum(soc-s.target,0).max(),np.abs(c-stored_c).max(),np.abs(d-stored_d).max()]
            res=float(max(checks));maxres=max(maxres,res);assert res<1e-8
            gap=np.maximum(s.target-soc[s.td,np.arange(n)],0); failed=int((gap>1e-4).sum())
            for key,value in [('cost',c.sum()),('D',d.sum()),('gap_kwh',gap@s.capacity),('vehicle_met_percent',100*(1-failed/n)),('failed',failed),('all_met',int(failed==0))]: assert np.isclose(float(row[key]),value,rtol=1e-10,atol=1e-8),(seed,method,key)
    with (O/'input_diagnostics.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=diagnostics[0]);w.writeheader();w.writerows(diagnostics)
    arr=np.asarray(arrivals); zz=np.asarray(zs); ii=np.triu_indices(200,1)
    correlation=lambda x: float(np.corrcoef(x,rowvar=False)[ii].mean())
    summary=[]
    for n,limit in ((200,600),(200,800),(3000,12000)):
        for subset in ('all','feasible','infeasible','unknown'):
            group=[r for r in rows if int(r['n'])==n and int(r['limit'])==limit and (subset=='all' or r['feasibility']==subset)]
            for method in ('Ours','UPA'):
                data=[r for r in group if r['method']==method]
                if not data:continue
                for metric in ('cost','D','vehicle_met_percent','all_met','gap_kwh'):
                    v=np.array([float(r[metric])*(100 if metric=='all_met' else 1) for r in data]);paired={r['seed']:float(r[metric]) for r in group if r['method']=='UPA'}
                    diff=np.array([(float(r[metric])-paired[r['seed']])*(100 if metric=='all_met' else 1) for r in data])
                    summary.append(dict(n=n,limit=limit,subset=subset,method=method,metric=metric,count=len(v),mean=v.mean(),sd=v.std(ddof=1) if len(v)>1 else '',paired_ours_minus_upa_mean=diff.mean() if method=='Ours' else '',paired_ours_minus_upa_sd=diff.std(ddof=1) if method=='Ours' and len(v)>1 else ''))
    with (O/'summary.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    report=dict(rows=len(rows),max_physical_or_metric_residual=maxres,mean_pairwise_arrival_correlation_200=correlation(arr),mean_pairwise_latent_correlation_200=correlation(zz),input_diagnostics_sha256=sha(O/'input_diagnostics.csv'),summary_sha256=sha(O/'summary.csv'),audit_source_sha256=sha(Path(__file__)))
    (O/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
