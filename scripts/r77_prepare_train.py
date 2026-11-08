"""Audit matched expert cases, build the Pressure cache, and train one policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/runs/r77'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from expert_ideal10h import generate, extended_price, input_digest, causal_state
import build_pressure21_cache as cache
import train_il_pressure21 as trainer

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--beta', type=float, choices=[0.1, 0.5], required=True)
args = parser.parse_args()
beta = args.beta
dest = BASE / f'beta_{int(beta*10):02d}'
experts = dest / 'experts'
frozen = json.loads((BASE/'frozen.json').read_text(encoding='utf8'))
full_manifest = BASE/'manifest.json'
if sha(full_manifest) != frozen['manifest_sha256']:
    raise RuntimeError('Frozen full manifest hash mismatch')
all_entries = json.loads(full_manifest.read_text(encoding='utf8'))
original_manifest = ROOT/'artifacts/runs/ideal10h_v1/experts/manifest.json'
if sha(original_manifest) != frozen['input_manifest_sha256']:
    raise RuntimeError('Original manifest hash mismatch')
original_entries = json.loads(original_manifest.read_text(encoding='utf8'))
expected = [dict(e, beta=b) for e in original_entries for b in (0.1, 0.5)]
if all_entries != expected or len(all_entries) != 900:
    raise RuntimeError('Full manifest differs from frozen matched cases')
entries = [dict((k,v) for k,v in e.items() if k != 'beta') for e in all_entries if e['beta'] == beta]
keys = [(e['split'],e['seed'],e['vehicles'],e['rate']) for e in entries]
if len(set(keys)) != 450 or sum(e['split']=='train' for e in entries)!=360 or sum(e['split']=='validation' for e in entries)!=90:
    raise RuntimeError('Duplicate case or incorrect split counts')
frozen_sources = {key.replace('\\', '/'): value for key, value in frozen['source_sha256'].items()}
expected_sources = {name: frozen_sources['code/'+name] for name in ('scripts/expert_ideal10h.py','src/evcharging/core.py','PYSOLVER/nlp2a.py','PYSOLVER/ev.py')}
expected_sources['r77/worker.py'] = frozen_sources['code/artifacts/runs/r77/worker.py']
for name, digest in expected_sources.items():
    p = BASE/'worker.py' if name == 'r77/worker.py' else ROOT/name
    if sha(p) != digest:
        raise RuntimeError(f'Expert source drift: {name}')
manifest = experts / 'manifest.json'
manifest.write_text(json.dumps(entries, indent=2), encoding='utf8')
accepted = {}; errors = []
for entry in entries:
    p = experts / entry['file']; report = p.with_suffix('.json')
    try:
        r = json.loads(report.read_text(encoding='utf8'))
        assert r['source_sha256'] == expected_sources, 'expert source hash mismatch'
        n=entry['vehicles']; limit=n*entry['rate']; s=generate(entry['seed'],n,fixed_day=True); s.beta=beta
        assert r['complete'] and r['seed']==entry['seed'] and r['n']==n and r['limit']==limit
        assert r['starts']==4 and r['seconds_per_window']==3*n and r['input_sha256']==input_digest(s,limit)
        assert all(w['checked_residual']<=1e-4 for w in r['windows'])
        with np.load(p) as z:
            for key in ('ta','td','arrival','target','capacity','power','price'):
                assert np.array_equal(z[key],getattr(s,key)),key
            assert float(z['beta'])==beta and float(z['limit'])==limit
            assert np.array_equal(z['price_extended'],extended_price(entry['seed'],s))
            soc,a,x,y,records=(z[k].copy() for k in ('soc','actions','x','y','records'))
        active=(s.ta[None,:]<=np.arange(56)[:,None]) & (np.arange(56)[:,None]<s.td[None,:])
        assert np.array_equal(records,np.argwhere(active))
        assert all(np.isfinite(v).all() for v in (soc,a,x,y))
        checks=[np.max(np.abs(soc[0]-s.arrival)),np.max(np.abs(soc[1:]-soc[:-1]-a*s.power*s.efficiency*s.dt/s.capacity)),np.max(np.maximum(-a,0)),np.max(np.maximum(a-1,0)),np.max(np.abs(a[~active]),initial=0),np.max(np.maximum(a@s.power-limit,0)),np.max(np.maximum(soc-s.target,0)),np.max(np.abs(soc[s.td,np.arange(n)]-s.target)),np.max(np.abs(y-a[records[:,0],records[:,1]]))]
        assert max(checks)<=1e-4, max(checks)
        for j,(t,k) in enumerate(records):
            assert np.allclose(x[j],causal_state(s,int(t),int(k),soc[t],a[t],limit),rtol=0,atol=1e-10)
        accepted[entry['file']]={'npz_sha256':sha(p),'report_sha256':sha(report),'max_residual':float(max(checks))}
    except Exception as exc:
        errors.append([entry['file'],repr(exc)])
audit={'schema':'ideal10h-expert-audit-v1','manifest_sha256':sha(manifest),'full_manifest_sha256':sha(full_manifest),'audit_source_sha256':sha(Path(__file__)),'expected_expert_sources':expected_sources,'entries':accepted,'errors':errors,'beta':beta}
(experts/('audit_status.json' if errors else 'audit_complete.json')).write_text(json.dumps(audit,indent=2),encoding='utf8')
if errors: raise RuntimeError(f'{len(errors)} audit failures: {errors[:5]}')
training_sources=json.loads((BASE/'training_source_manifest.json').read_text(encoding='utf8'))
for source, rec in training_sources.items():
    if sha(ROOT/source)!=rec['sha256'] or sha(ROOT/rec['archive'])!=rec['sha256']:
        raise RuntimeError(f'Training source drift: {source}')
cache.HERE=dest; cache.EXPERT=experts; cache.main()
trainer.HERE=dest
trainer.run('pressure',11002,f'r77/beta_{int(beta*10):02d}')
summary=dest/'il_11002/il_model/summary.json'
data=json.loads(summary.read_text(encoding='utf8')); data.update(beta=beta,expert_manifest_path=str(manifest.resolve()),expert_manifest_sha256=sha(manifest),expert_audit_path=str((experts/'audit_complete.json').resolve()),expert_audit_sha256=sha(experts/'audit_complete.json'),cache_manifest_path=str((dest/'expert_cache/manifest.json').resolve()),cache_manifest_sha256=sha(dest/'expert_cache/manifest.json'),best_path=str((dest/'il_11002/il_model/best.pt').resolve()),best_sha256=sha(dest/'il_11002/il_model/best.pt'),pipeline_sha256=sha(Path(__file__)))
summary.write_text(json.dumps(data,indent=2),encoding='utf8')
