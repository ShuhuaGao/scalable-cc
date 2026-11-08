"""Select checkpoints from the current-rate three-run RL training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evcharging.models import network
from train_ppo_pressure21 import validate

OUT=ROOT/'artifacts/runs/public_rl'


def sha(path: Path) -> str:
    """Return a file's SHA-256.

    :param path: File to hash.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking(item: dict) -> tuple:
    """Rank PPO checkpoints by the prespecified common-validation rule.

    :param item: Validation metrics.
    :return: Sort key.
    """
    return (item['failed'], 0.0 if item['failed']==0 else item['gap_kwh'], item['objective'])


def main() -> None:
    """Select three per-run and one main-comparison PPO model, and three DDQN models.

    :return: None.
    """
    if not (OUT/'rl_training_complete.json').exists(): raise RuntimeError('All RL training must finish before selection')
    cfg=json.loads((ROOT/'configs/pressure21_v1.json').read_text()); torch.set_num_threads(1)
    ddqn=[]
    for seed in (21002,21003,21004):
        folder=OUT/f'ddqn_{seed}/ddqn_model'; result=json.loads((folder/'selected.json').read_text()); path=folder/f"episode_{result['episode']}.pt"; check=torch.load(path,map_location='cpu',weights_only=False)
        if check['config']['state_schema']!='pressure21_v1' or check['config']['seed']!=seed: raise ValueError('DDQN checkpoint schema mismatch')
        ddqn.append({'seed':seed,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'selected':result,'state_schema':'pressure21_v1'})
    common=dict(cfg['ppo'],validation_seed_start=80000,validation_count=20); candidates=[]
    for seed in (24003,24004,24005):
        paths=sorted((OUT/f'ppo_{seed}/ppo_model').glob('episode_*.pt'))
        if len(paths)!=10: raise RuntimeError(f'Expected ten PPO checkpoints for {seed}')
        for path in paths:
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if checkpoint['config']['state_schema']!='pressure21_v1' or checkpoint['config']['seed']!=seed: raise ValueError('PPO checkpoint schema mismatch')
            actor=network(2); actor.load_state_dict(checkpoint['model']); actor.eval(); metrics=validate(actor,common,cfg['scenario']['beta'],cfg['evaluation']['soc_tolerance']); metrics.update(seed=seed,episode=checkpoint['episode'],path=path.relative_to(ROOT).as_posix(),sha256=sha(path),state_schema='pressure21_v1'); candidates.append(metrics); print('validated',seed,checkpoint['episode'],flush=True)
    per_run=[min((r for r in candidates if r['seed']==seed),key=ranking) for seed in (24003,24004,24005)]
    main_choice=min((r for r in candidates if r['seed'] in (24003,24004)),key=ranking)
    result={'state_schema':'pressure21_v1','common_validation':{'seed_start':80000,'count':20,'rule':'fewest target failures; if nonzero, total energy gap; then objective'},'ddqn':ddqn,'ppo_per_run':per_run,'ppo_main':main_choice,'ppo_candidates':candidates,'main_ddqn_seed':21002,'formal_test_used_for_selection':False}
    dest=OUT/'rl_selection.json'
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2)); print('Selected main PPO:',main_choice['seed'],main_choice['episode'],flush=True)


if __name__=='__main__': main()
