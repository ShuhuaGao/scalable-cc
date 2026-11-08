"""Freeze freshly trained Pressure-state ablation checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/runs/pressure21_v1'


def sha(path: Path) -> str:
    """Return a file's SHA-256 digest.

    :param path: Input file.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify and publish the twelve selected ablation checkpoints.

    :return: None.
    """
    records=[]
    for variant in ('local','without_a','without_b','without_c'):
        for seed in (11002,11003,11004):
            folder=OUT/'models'/f'{variant}_{seed}'; source=folder/'continuation_best.pt'; summary=json.loads((folder/'summary.json').read_text()); curve=json.loads((folder/'continuation_curve.json').read_text()); ck=torch.load(source,map_location='cpu',weights_only=True); epoch=int(np.argmin([x['autoregressive_validation_mse'] for x in curve]))
            if len(curve)!=21 or epoch!=ck['epoch'] or ck['seed']!=seed or ck['variant']!=variant or ck['state_schema']!='pressure21_v1' or sha(source)!=summary['continuation_sha256']: raise RuntimeError(f'Invalid ablation selection: {variant} {seed}')
            target=OUT/'ablation_models'/f'{variant}_{seed}'/'best.pt'; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): raise FileExistsError(target)
            shutil.copyfile(source,target); records.append({'variant':variant,'seed':seed,'path':str(target.resolve()),'sha256':sha(target),'state_schema':'pressure21_v1','selected_epoch':epoch,'validation_mse':ck['validation_mse']})
    dest=OUT/'ablation_selection.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps({'state_schema':'pressure21_v1','selection':'minimum expert autoregressive validation MSE including epoch 0','formal_test_used_for_selection':False,'models':records},indent=2)); print('Frozen twelve state-ablation policies.')


if __name__=='__main__': main()
