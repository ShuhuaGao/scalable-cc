"""Run frozen matched beta expert cases in independent shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import expert_ideal10h as expert

parser = argparse.ArgumentParser()
parser.add_argument('--shard', type=int, required=True)
args = parser.parse_args()
base = Path(__file__).resolve().parents[1] / 'artifacts/runs/r77'
manifest = json.loads((base / 'manifest.json').read_text(encoding='utf8'))
original_generate = expert.generate
expert.SOURCE_SHA256['r77/worker.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for i, entry in enumerate(manifest):
    if i % 4 != args.shard:
        continue
    beta = entry['beta']
    def matched_generate(*a: Any, **kw: Any) -> expert.Scenario:
        """Generate a matched scenario with the requested objective weight.

        :param a: Positional generator arguments.
        :param kw: Keyword generator arguments.
        :return: Scenario carrying the selected weight.
        """
        scenario = original_generate(*a, **kw)
        scenario.beta = beta
        return scenario
    expert.generate = matched_generate
    stem = base / f'beta_{int(beta*10):02d}' / 'experts' / Path(entry['file']).stem
    if stem.with_suffix('.json').exists() or stem.with_suffix('.npz').exists():
        print('retained existing:', stem, flush=True)
        continue
    print('START', i, entry, 'pid', os.getpid(), 'time', time.time(), flush=True)
    try:
        expert.run(entry['seed'], entry['vehicles'], entry['vehicles'] * entry['rate'], stem, 4, 3 * entry['vehicles'])
    except Exception as exc:
        print('FAILED', i, repr(exc), flush=True)
    print('END', i, 'time', time.time(), flush=True)
