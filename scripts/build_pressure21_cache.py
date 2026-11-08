"""Build audited teacher-forced feature caches from the original expert data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'artifacts/runs/pressure21_v1'
sys.path.insert(0, str(ROOT / "src"))
from evcharging.core import causal_state, load_scenario
from evcharging.state_schema import build_state

EXPERT = ROOT / "artifacts/runs/ideal10h_v1/experts"


def digest(path: Path) -> str:
    """Return the SHA-256 of a file.

    :param path: File to hash.
    :return: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Reconstruct all active decisions and archive cache provenance.

    :return: None.
    """
    output = HERE / "expert_cache"
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = EXPERT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((EXPERT / "audit_complete.json").read_text())
    if audit["schema"] != "ideal10h-expert-audit-v1" or audit["manifest_sha256"] != digest(manifest_path) or set(audit['entries']) != {e['file'] for e in manifest}:
        raise RuntimeError("Expert audit does not match manifest")
    result = {}
    for split in ("train", "validation"):
        pressure, yall, positions, groups, group_powers, group_limits, sources, slots = [], [], [], [], [], [], [], []
        offset = 0
        split_entries = [entry for entry in manifest if entry['split'] == split]
        for number, entry in enumerate(split_entries, 1):
            path = EXPERT / entry['file']; rec = audit['entries'][entry['file']]
            if digest(path) != rec['npz_sha256'] or digest(path.with_suffix('.json')) != rec['report_sha256']:
                raise RuntimeError(f"Expert audit hash mismatch: {path}")
            s = load_scenario(str(path))
            with np.load(path) as z:
                soc, actions, original_x, labels, records, limit = (z[k].copy() for k in ('soc','actions','x','y','records','limit'))
            if not np.allclose(labels, actions[records[:,0], records[:,1]], rtol=0, atol=1e-10):
                raise ValueError(f"Expert label mismatch: {path}")
            for j, (t, k) in enumerate(records):
                original = causal_state(s, int(t), int(k), soc[t], actions[t], float(limit))
                if not np.allclose(original, original_x[j], rtol=0, atol=1e-10):
                    raise ValueError(f"Original state reconstruction mismatch: {path} row {j}")
                pressure.append(build_state(s, int(t), int(k), soc[t], actions[t], float(limit), 'pressure21_v1'))
            for t in np.unique(records[:,0]):
                local = np.flatnonzero(records[:,0] == t)
                ids = records[local,1]
                if len(ids)>1 and not np.all(np.diff(ids)>0): raise ValueError("Invalid sequential order")
                groups.append(local+offset); group_powers.append(s.power[ids]); group_limits.append(float(limit)); sources.append(entry['file']); slots.append(int(t))
            positions.extend(records); yall.extend(labels); offset += len(labels)
            if number % 50 == 0: print(split, number, '/', len(split_entries), 'samples', offset, flush=True)
        width=max(len(g) for g in groups)
        index=np.full((len(groups),width),-1,dtype=np.int64); powers=np.zeros((len(groups),width),dtype=np.float64)
        for j,(group,power) in enumerate(zip(groups,group_powers)):
            index[j,:len(group)]=group; powers[j,:len(group)]=power
        path=output/f'{split}.npz'
        np.savez_compressed(path, x_pressure=np.asarray(pressure,dtype=np.float32), y=np.asarray(yall,dtype=np.float32), records=np.asarray(positions,dtype=np.int64), group_index=index, group_power=powers, group_limit=np.asarray(group_limits,dtype=np.float64), group_source=np.asarray(sources), group_slot=np.asarray(slots,dtype=np.int64))
        result[split]={'expert_trajectories':len(split_entries),'samples':offset,'groups':len(groups),'sha256':digest(path)}
    result['sources_sha256']={path.relative_to(ROOT).as_posix():digest(path) for path in (Path(__file__),ROOT/'src/evcharging/state_schema.py',ROOT/'configs/pressure21_il.json',ROOT/'src/evcharging/core.py',manifest_path,EXPERT/'audit_complete.json')}
    (output/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
