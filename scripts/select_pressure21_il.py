"""Freeze freshly trained Pressure policies on expert-validation selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    """Verify the per-run expert-validation minima and freeze their models.

    :return: None.
    """
    result=[]
    for seed in (11002,11003,11004):
        folder=OUT/f'il_{seed}/il_model'; path=folder/'best.pt'; rec=json.loads((folder/'summary.json').read_text()); curve=json.loads((folder/'continuation_curve.json').read_text()); ck=torch.load(path,map_location='cpu',weights_only=True)
        selected=int(np.argmin([r['autoregressive_validation_mse'] for r in curve]))
        if len(curve)!=21 or selected!=ck['epoch'] or ck['seed']!=seed or ck['state_schema']!='pressure21_v1' or sha(path)!=rec['continuation_sha256']: raise RuntimeError(f'IL selection mismatch: {seed}')
        result.append({'seed':seed,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'state_schema':'pressure21_v1','selection':'minimum expert autoregressive validation MSE including epoch 0','epoch':selected})
    dest=OUT/'il_selection.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps({'state_schema':'pressure21_v1','models':result},indent=2)); print('Frozen three independently trained IL policies.')


if __name__=='__main__': main()
