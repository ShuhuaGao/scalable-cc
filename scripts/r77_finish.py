"""Wait for expert workers and run the two independent audits and trainings."""
import subprocess
import sys
import time
from pathlib import Path
import json

import psutil

base = Path(__file__).resolve().parents[1] / 'artifacts/runs/r77'
pids = [int(x) for x in sys.argv[1:]]
workers = []
for pid in pids:
    try:
        process = psutil.Process(pid)
        workers.append((pid, process.create_time()))
    except psutil.NoSuchProcess:
        pass
manifest = json.loads((base/'manifest.json').read_text(encoding='utf8'))
pending = {'0.1': 'beta_01', '0.5': 'beta_05'}
while pending:
    alive = False
    for pid, created in workers:
        try:
            alive |= psutil.Process(pid).create_time() == created
        except psutil.NoSuchProcess:
            pass
    for beta, directory in list(pending.items()):
        expected = [entry for entry in manifest if entry['beta'] == float(beta)]
        count = sum((base/directory/'experts'/Path(entry['file']).with_suffix('.json')).exists() for entry in expected)
        if count == 450 or not alive:
            print('Starting audit and training for beta',beta,'reports',count,flush=True)
            result=subprocess.run([sys.executable,str(base/'prepare_train.py'),'--beta',beta],check=False)
            print('Finished beta',beta,'exit',result.returncode,flush=True)
            del pending[beta]
    if pending:
        time.sleep(20)
