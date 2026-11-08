"""Run the prespecified correlated-arrival experiment in a new archive."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import csv, json, hashlib, sys, shutil, concurrent.futures
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from scipy.special import ndtr
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from evcharging.core import Scenario, load_scenario, fixed_feasibility
from evcharging.simulation_schema import simulate as linear
from run_r26_diagnostics import model, policy, simulate as baseline

A=R/'artifacts/runs/pressure21_v1/r107_correlated'
O=R/'results/pressure21_v1/r107'
MODEL=R/'artifacts/runs/pressure21_v1/il_11002/il_model/best.pt'
FIELDS=('arrival','target','capacity','power','price','efficiency','dt','beta')

def digest(p: Path) -> str:
    """Return a file SHA-256.

    :param p: File path.
    :return: Digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()

def pmf(mu: float, sd: float, low: int, high: int) -> np.ndarray:
    """Compute round-then-clip normal probabilities.

    :param mu: Normal mean.
    :param sd: Normal standard deviation.
    :param low: Minimum integer.
    :param high: Maximum integer.
    :return: Probability mass on all integers from low to high.
    """
    cuts=(np.arange(low,high)+0.5-mu)/sd
    cdf=ndtr(cuts)
    return np.diff(np.r_[0.,cdf,1.])

def transform(s: Scenario, seed: int) -> tuple[Scenario, dict[str, np.ndarray | float]]:
    """Draw conditional feasible times using one common Gaussian factor.

    :param s: Original fixed-day scenario.
    :param seed: Original scenario seed.
    :return: Modified scenario and random latent arrays.
    """
    n=len(s.ta); base=seed+120000
    common=np.random.default_rng(np.random.SeedSequence([base,1])).normal()
    eps=np.random.default_rng(np.random.SeedSequence([base,2])).normal(size=n)
    ud=np.random.default_rng(np.random.SeedSequence([base,3])).random(n)
    z=(common+eps)/np.sqrt(2); ua=ndtr(z)
    pa=pmf(8,4,0,16); pd=pmf(40,4,32,56)
    aa=np.arange(17); dd=np.arange(32,57)
    required=np.ceil(s.capacity*(s.target-s.arrival)/(s.efficiency*s.power*s.dt)).astype(int)
    ta=np.empty(n,dtype=int);td=np.empty(n,dtype=int)
    for need in np.unique(required):
        idx=np.flatnonzero(required==need)
        w=pa[:,None]*pd[None,:]*(dd[None,:]-aa[:,None]>=need)
        marg=w.sum(axis=1); total=marg.sum()
        if total<=0: raise ValueError(f'No feasible parking window for required={need}')
        c=np.cumsum(marg/total);c[-1]=1
        ta[idx]=np.searchsorted(c,ua[idx],side='right')
        for a in np.unique(ta[idx]):
            group=idx[ta[idx]==a]; weights=w[a]/marg[a];cdf=np.cumsum(weights);cdf[-1]=1
            td[group]=dd[np.searchsorted(cdf,ud[group],side='right')]
    if np.any(td-ta<required): raise RuntimeError('Individual reachability failed')
    return replace(s,ta=ta,td=td),dict(common=common,eps=eps,z=z,arrival_uniform=ua,departure_uniform=ud,required=required)

def case(task: tuple[int, int]) -> list[dict[str, object]]:
    """Evaluate one day and its capacity settings.

    :param task: Number of vehicles and source seed.
    :return: Per-method records.
    """
    n,seed=task
    source=(R/f'artifacts/runs/ideal10h_v1/formal/inputs/{seed}.npz' if n==200 else R/f'artifacts/runs/ideal10h_v1/scalability/input_{n}_{seed}.npz')
    old=load_scenario(str(source));s,latent=transform(old,seed)
    for k in FIELDS:
        if not np.array_equal(np.asarray(getattr(old,k)),np.asarray(getattr(s,k))): raise RuntimeError(k)
    name=f'{n}_{seed}';inp=A/f'input_{name}.npz';np.savez_compressed(inp,**vars(s),**latent)
    m=model();out=[]
    for limit in ((600,800) if n==200 else (12000,)):
        status,res=fixed_feasibility(s,limit)
        for method in ('Ours','UPA'):
            r=linear(s,limit,policy(m),state_schema='pressure21_v1') if method=='Ours' else baseline(s,limit,None,'upa')
            path=A/f'traj_{name}_{limit}_{method}.npz'
            np.savez_compressed(path,**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
            soc=r['soc'];gap=np.maximum(s.target-soc[s.td,np.arange(n)],0);failed=int((gap>1e-4).sum())
            out.append(dict(n=n,seed=seed,limit=limit,method=method,feasibility=status,lp_residual=res,cost=float(r['cost'].sum()),D=float(r['unfair'].sum()),vehicle_met_percent=100*(1-failed/n),all_met=int(failed==0),failed=failed,gap_kwh=float(gap@s.capacity),source_sha256=digest(source),input_sha256=digest(inp),trajectory_sha256=digest(path),model_sha256=digest(MODEL)))
    return out

def main() -> None:
    """Execute the complete predefined cohort and retain provenance.

    :return: None.
    """
    if A.exists() or O.exists(): raise FileExistsError('Output directory already exists')
    A.mkdir(parents=True);O.mkdir(parents=True)
    sourcefiles=[Path(__file__),R/'scripts/run_r26_diagnostics.py',R/'src/evcharging/core.py',R/'src/evcharging/simulation_schema.py',R/'src/evcharging/state_schema.py',R/'src/evcharging/models.py']
    snap=A/'source';snap.mkdir()
    for p in sourcefiles: shutil.copy2(p,snap/p.name)
    config=dict(rho_latent=.5,streams={'common':1,'individual':2,'departure':3},seed='source seed + 120000',source_sha256={p.relative_to(R).as_posix():digest(p) for p in sourcefiles},model_sha256=digest(MODEL),cases='200:65000-65049 at 600,800 kW; 3000:111000-111002 at 12000 kW',method='conditional rounded clipped N(8,4) arrival, N(40,4) departure; no redraw')
    (A/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    tasks=[(200,i) for i in range(65000,65050)]+[(3000,i) for i in range(111000,111003)]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for result in pool.map(case,tasks):
            rows.extend(result);print(f'completed {len(rows)} / 206 method-capacity-days',flush=True)
    with (O/'daily.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (A/'complete.json').write_text(json.dumps({'rows':len(rows),'daily_sha256':digest(O/'daily.csv')},indent=2),encoding='utf8')
if __name__=='__main__': main()
