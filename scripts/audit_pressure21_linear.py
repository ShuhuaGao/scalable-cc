"""Audit compact Pressure scheduler against the historical reference."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import load_scenario
from evcharging.models import imitation_network
from evcharging.simulation_schema import simulate


def digest(p: Path) -> str:
    """Hash an input file.

    :param p: Path to the input file.
    :return: SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> None:
    """Run the bounded audit into a new output directory.

    :return: None.
    """
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/pressure21_linear_audit_final_20261106', help='New output directory; relative paths are relative to code/')
    out=parser.parse_args().output
    if not out.is_absolute(): out=ROOT/out
    out.mkdir(parents=True, exist_ok=False)
    model_path=ROOT/'artifacts/runs/pressure21_v1/il_11002/il_model/best.pt'
    model=imitation_network(); model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True)['model']); model.eval()
    def policy(x: np.ndarray, k: int, u: np.ndarray) -> float:
        """Evaluate the frozen policy on a state.

        :param x: Current policy state.
        :param k: Original vehicle index.
        :param u: Full-fleet urgency vector.
        :return: Proposed normalized action.
        """
        with torch.no_grad(): return float(model(torch.tensor(x,dtype=torch.float32)[None,:])[0,0])
    paths=[ROOT/'src/evcharging/simulation_schema.py', ROOT/'src/evcharging/state_schema.py', ROOT/'src/evcharging/core.py', ROOT/'src/evcharging/models.py', Path(__file__)]
    manifest={'model':str(model_path),'model_sha256':digest(model_path),'sources_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},'threads':1,'tolerance':1e-6,'cases':[]}
    (out/'audit.json').write_text(json.dumps(manifest,indent=2))
    cases=[('development',64000,200,600),('development',64000,200,800)]+[('scale',111000+i,3000,12000) for i in range(3)]
    for kind, seed, n, limit in cases:
        path=ROOT/(f'artifacts/runs/ideal10h_v1/development/inputs/{seed}.npz' if kind=='development' else f'artifacts/runs/ideal10h_v1/scalability/input_{n}_{seed}.npz')
        s=load_scenario(str(path)); a=simulate(s,limit,policy,state_schema='pressure21_v1'); b=simulate(s,limit,policy,state_schema='pressure21_v1',pressure_reference=True)
        errors={key:float(np.max(np.abs(a[key]-b[key]),initial=0)) for key in ('states','raw','clip','scale','final','soc','cost','unfair','records')}
        met=[]; physical=[]
        for r in (a,b):
            gap=np.maximum(s.target-r['soc'][s.td,np.arange(n)],0); met.append(int((gap>1e-4).sum()))
            active=(s.ta[None,:]<=np.arange(len(s.price))[:,None]) & (np.arange(len(s.price))[:,None]<s.td[None,:]); final=r['final']; soc=r['soc']
            physical.append({'station':float(np.maximum(final@s.power-limit,0).max(initial=0)), 'disconnected':float(np.abs(final[~active]).max(initial=0)), 'dynamics':float(np.abs(soc[1:]-soc[:-1]-final*s.power*s.efficiency*s.dt/s.capacity).max(initial=0)), 'overcharge':float(np.maximum(soc-s.target,0).max(initial=0)), 'action_lower':float(np.maximum(-final,0).max(initial=0)), 'action_upper':float(np.maximum(final-1,0).max(initial=0))})
        row={'input':str(path),'input_sha256':digest(path),'limit':limit,'max_abs_error':errors,'failed_vehicles':met,'physical':physical,'seconds_linear':a['seconds'],'seconds_reference':b['seconds'],'cost_total': [float(a['cost'].sum()), float(b['cost'].sum())], 'unfair_total':[float(a['unfair'].sum()), float(b['unfair'].sum())]}
        manifest['cases'].append(row); (out/'audit.json').write_text(json.dumps(manifest,indent=2))
        print(kind,seed,limit,errors,met,flush=True)
        if max(errors.values())>1e-6 or met[0]!=met[1] or max(max(p.values()) for p in physical)>1e-8: raise RuntimeError('Audit failed')
if __name__=='__main__': main()
