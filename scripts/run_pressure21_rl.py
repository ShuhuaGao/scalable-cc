"""Train three DDQN and three current-rate PPO runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/runs/public_rl'


def sha(path: Path) -> str:
    """Hash a source or configuration file.

    :param path: File to hash.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Launch the six runs with the selected worker limit.

    :return: None.
    """
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
    import argparse
    parser = argparse.ArgumentParser(description="Train three DDQN and three PPO runs with the current configuration.")
    parser.add_argument('--workers', type=int, default=1, choices=(1, 2, 3))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config_dir = ROOT/'configs/rl_repeats'
    logs=OUT/'logs'; logs.mkdir(parents=True,exist_ok=True)
    jobs=[('ddqn',21002),('ddqn',21003),('ppo',24003),('ddqn',21004),('ppo',24004),('ppo',24005)]
    active=[]
    while jobs or active:
        while jobs and len(active)<args.workers:
            method,seed=jobs.pop(0); run=f'public_rl/{method}_{seed}'; target=OUT/f'{method}_{seed}'/f'{method}_model'
            if target.exists(): raise FileExistsError(f'Existing run retained: {target}')
            out=(logs/f'{method}_{seed}.log').open('w',encoding='utf8'); err=(logs/f'{method}_{seed}.err').open('w',encoding='utf8'); config=config_dir/f'{method}_{seed}.json'
            cmd=[sys.executable,str(ROOT/'scripts'/f'train_{method}_pressure21.py'),'--config',str(config),'--run-id',run]
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err); active.append((method,seed,proc,out,err,time.time())); print('started',method,seed,'pid',proc.pid,flush=True)
        for item in active[:]:
            method,seed,proc,out,err,start=item; code=proc.poll()
            if code is not None:
                out.close();err.close();active.remove(item); print('finished',method,seed,'exit',code,'elapsed_seconds',round(time.time()-start),flush=True)
                if code: raise RuntimeError(f'Training failed {method} {seed} exit {code}')
        time.sleep(2)
    (OUT/'rl_training_complete.json').write_text(json.dumps({'completed':6,'schema':'pressure21_v1'},indent=2))


if __name__=='__main__': main()
