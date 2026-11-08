"""Measure frozen Ours, UPA and RDDPG on nine archived scenarios."""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import Scenario, load_scenario, spp, urgency_allocation
from evcharging.li2023 import actor, simulate as li_simulate
from evcharging.models import imitation_network
from evcharging.simulation import simulate as upa_simulate
from evcharging.simulation_schema import simulate as ours_simulate
from run_li2023_r105 import numpy_policy

INPUT = ROOT / 'artifacts/runs/pressure21_v1/evaluation/timing/scenarios.csv'
OURS = ROOT / 'artifacts/runs/pressure21_v1/il_11002/il_model/best.pt'
RDDPG = ROOT / 'artifacts/runs/li2023_r105/33001/episode_6000.pt'

def digest(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    :param path: File to hash.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()

def cases() -> list[dict]:
    """Read and select the nine archived timing inputs.

    :return: Scenario manifest rows.
    """
    with INPUT.open(newline='', encoding='utf8') as f:
        rows = [r for r in csv.DictReader(f) if int(r['vehicles']) in (100, 1000, 3000)]
    if len(rows) != 9 or any(sum(int(r['vehicles']) == n for r in rows) != 3 for n in (100, 1000, 3000)):
        raise ValueError('Expected three archived inputs per size')
    return rows

def direct_upa(s: Scenario, limit: float, profile: dict | None = None) -> dict:
    """Run the existing UPA rule without constructing unused neural states.

    :param s: Archived scenario.
    :param limit: Station power limit.
    :param profile: Optional dictionary receiving per-slot times.
    :return: Actions, physical trajectories and evaluation arrays.
    """
    n, h = len(s.ta), len(s.price)
    soc = np.zeros((h + 1, n))
    soc[0] = s.arrival
    raw = np.zeros((h, n)); clip = np.zeros_like(raw); scale = np.zeros_like(raw)
    final = np.zeros_like(raw); urgency = np.zeros_like(raw)
    if profile is not None:
        profile['slot_seconds'] = []
    for t in range(h):
        if profile is not None:
            start = time.perf_counter()
        active = (s.ta <= t) & (t < s.td)
        remaining = np.maximum(s.td - t, 1)
        urgency[t] = np.where(active, np.maximum(s.target-soc[t], 0)*s.capacity /
                              (s.efficiency*s.power*s.dt*remaining), 0.0)
        raw[t] = urgency_allocation(urgency[t], s.power, s.td-t, limit)
        clip[t], scale[t], final[t] = spp(raw[t], soc[t], s.target, s.capacity, s.power,
                                          active, s.td-t, limit, s.efficiency, s.dt)
        soc[t+1] = soc[t] + final[t]*s.power*s.efficiency*s.dt/s.capacity
        if profile is not None:
            profile['slot_seconds'].append(time.perf_counter()-start)
    total = urgency.sum(axis=1, keepdims=True)
    shares = np.divide(urgency, total, out=np.zeros_like(urgency), where=total > 0)
    unfair = (final-final.sum(axis=1, keepdims=True)*shares)**2
    return {'soc': soc, 'raw': raw, 'clip': clip, 'scale': scale, 'final': final,
            'cost': final*s.power[None, :]*s.price[:, None]*s.dt, 'unfair': unfair}

def worker(row: dict, method: str, output: Path) -> None:
    """Measure and independently verify one case.

    :param row: Archived input metadata.
    :param method: One of Ours, UPA and RDDPG.
    :param output: Existing archive directory.
    :return: None.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = Path(row['input_path'])
    if digest(path) != row['input_sha256'] or digest(OURS) != row['model_sha256']:
        raise ValueError('Archived input or Ours model hash mismatch')
    li_manifest = json.loads((ROOT / 'results/li2023_r105/manifest.json').read_text(encoding='utf8'))
    if digest(RDDPG) != li_manifest['models_sha256']['33001']:
        raise ValueError('RDDPG checkpoint hash mismatch')
    s = load_scenario(str(path))
    n, limit = len(s.ta), float(row['capacity_kw'])
    if n != int(row['vehicles']) or limit != 4*n or len(s.price) != 56:
        raise ValueError('Unexpected scenario shape or limit')
    if method == 'Ours':
        model = imitation_network()
        model.load_state_dict(torch.load(OURS, map_location='cpu', weights_only=True)['model'])
        model.eval()
        def policy(x: np.ndarray, k: int, u: np.ndarray) -> float:
            """Evaluate the frozen shared policy.

            :param x: State vector.
            :param k: Vehicle index.
            :param u: Urgency vector.
            :return: Proposed action.
            """
            with torch.no_grad():
                return float(model(torch.tensor(x, dtype=torch.float32)[None, :])[0, 0])
        for _ in range(10):
            policy(np.zeros(21), 0, np.zeros(1))
        def run(profile: dict | None) -> dict:
            """Run the learned policy.

            :param profile: Optional profiling results.
            :return: Simulation output.
            """
            return ours_simulate(s, limit, policy, state_schema='pressure21_v1', profile=profile)
        model_hash = digest(OURS)
    elif method == 'UPA':
        cache = {'t': -1, 'actions': None}
        def policy(x: np.ndarray, k: int, urgency: np.ndarray) -> float:
            """Return the archived urgency-proportional allocation.

            :param x: Causal state vector.
            :param k: Vehicle index.
            :param urgency: Fleet urgency.
            :return: Proposed normalized action.
            """
            t = int(round((1 - x[1]) * len(s.price) / 2))
            if t != cache['t']:
                cache['t'] = t
                cache['actions'] = urgency_allocation(urgency, s.power, s.td - t, limit)
            return float(cache['actions'][k])
        def reference() -> dict:
            """Run the original UPA controller.

            :return: Simulation output.
            """
            cache['t'] = -1
            return upa_simulate(s, limit, policy, causal=True)
        def run(profile: dict | None) -> dict:
            """Run the direct UPA controller.

            :param profile: Optional profiling results.
            :return: Simulation output.
            """
            return direct_upa(s, limit, profile)
        model_hash = None
    elif method == 'RDDPG':
        model = actor()
        checkpoint = torch.load(RDDPG, map_location='cpu', weights_only=True)
        if checkpoint['seed'] != 33001 or checkpoint['episode'] != 6000:
            raise ValueError('Unexpected checkpoint')
        model.load_state_dict(checkpoint['model'])
        policy = numpy_policy(model)
        for _ in range(10):
            policy(np.zeros(9, dtype=np.float32))
        def run(profile: dict | None) -> dict:
            """Run the RDDPG controller.

            :param profile: Optional profiling results.
            :return: Simulation output.
            """
            return li_simulate(s, limit, policy, profile=profile)
        model_hash = digest(RDDPG)
    else:
        raise ValueError(method)
    profile = {'memory_callback': lambda: None} if method == 'Ours' else {}
    timed = run(profile)
    standard = run(None)
    errors = {}
    for key, value in timed.items():
        if isinstance(value, np.ndarray):
            errors[key] = float(np.max(np.abs(value.astype(float) - standard[key].astype(float)), initial=0))
        elif isinstance(value, (float, int)) and key != 'seconds':
            errors[key] = abs(float(value) - float(standard[key]))
    if max(errors.values(), default=0) != 0 or len(profile['slot_seconds']) != 56:
        raise RuntimeError('Profiled and unprofiled runs differ')
    reference_errors = {}
    if method == 'UPA':
        prior = reference()
        reference_errors = {k: float(np.max(np.abs(timed[k]-prior[k]), initial=0))
                            for k in ('soc', 'raw', 'clip', 'scale', 'final', 'cost', 'unfair')}
        if max(reference_errors.values()) != 0:
            raise RuntimeError('Direct UPA differs from the original controller')
    soc = timed['soc']
    power = timed['power'] if method == 'RDDPG' else timed['final'] * s.power
    active = (s.ta[None, :] <= np.arange(56)[:, None]) & (np.arange(56)[:, None] < s.td[None, :])
    urgency = np.where(active, np.maximum(s.target - soc[:-1], 0) * s.capacity / (s.efficiency * s.power * s.dt * np.maximum(s.td - np.arange(56)[:, None], 1)), 0)
    denom = urgency.sum(axis=1, keepdims=True)
    share = np.divide(urgency, denom, out=np.zeros_like(urgency), where=denom > 0)
    actions = power / s.power
    gap = np.maximum(s.target - soc[s.td, np.arange(n)], 0)
    physical = {'station': float(np.maximum(power.sum(axis=1)-limit, 0).max(initial=0)),
                'disconnected': float(np.abs(power[~active]).max(initial=0)),
                'power_lower': float(np.maximum(-power, 0).max(initial=0)),
                'power_upper': float(np.maximum(power-s.power, 0).max(initial=0)),
                'soc_upper': float(np.maximum(soc-s.target, 0).max(initial=0)),
                'dynamics': float(np.abs(soc[1:]-soc[:-1]-power*s.efficiency*s.dt/s.capacity).max(initial=0))}
    if max(physical.values()) > 1e-8:
        raise RuntimeError('Physical residual exceeds tolerance')
    stem = f'{n}_{row["seed"]}_{method}'
    arrays = {k: v for k, v in timed.items() if isinstance(v, np.ndarray)}
    arrays['slot_seconds'] = np.asarray(profile['slot_seconds'])
    np.savez_compressed(output / (stem + '.npz'), **arrays)
    record = {'vehicles': n, 'seed': int(row['seed']), 'capacity_kw': limit, 'method': method,
              'input_sha256': digest(path), 'model_sha256': model_hash,
              'trajectory_sha256': digest(output / (stem + '.npz')),
              'seconds_mean': float(np.mean(profile['slot_seconds'])), 'seconds_max': float(np.max(profile['slot_seconds'])),
              'cost': float((power*s.price[:, None]*s.dt).sum()),
              'D': float(((actions-actions.sum(axis=1, keepdims=True)*share)**2).sum()),
              'failed': int((gap > 1e-4).sum()), 'gap_kwh': float(gap @ s.capacity),
              'physical': physical, 'profile_errors': errors, 'upa_reference_errors': reference_errors}
    (output / (stem + '.json')).write_text(json.dumps(record, indent=2), encoding='utf8')

def main() -> None:
    """Run all cases serially in fresh processes and write summaries.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--method', choices=('Ours', 'UPA', 'RDDPG'))
    args = parser.parse_args()
    rows = cases()
    out = args.output.resolve()
    if args.worker is not None:
        worker(rows[args.worker], args.method, out)
        return
    out.mkdir(parents=True, exist_ok=False)
    args.summary.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/'src/evcharging/core.py', ROOT/'src/evcharging/li2023.py', ROOT/'src/evcharging/simulation.py', ROOT/'src/evcharging/simulation_schema.py', ROOT/'src/evcharging/state_schema.py', ROOT/'src/evcharging/models.py', ROOT/'scripts/run_li2023_r105.py', ROOT/'pyproject.toml', ROOT/'uv.lock', INPUT]
    for p in sources:
        target = out/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    hardware = subprocess.check_output(['powershell', '-NoProfile', '-Command', "$c=Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors; $s=Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory; @{cpu=$c;system=$s} | ConvertTo-Json -Depth 3 -Compress"], text=True)
    order = [(i, ['Ours', 'UPA', 'RDDPG'][(j+i)%3]) for i in range(9) for j in range(3)]
    manifest = {'hardware': json.loads(hardware), 'platform': platform.platform(), 'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'threads': 1,
                'sources_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}, 'models_sha256': {'Ours': digest(OURS), 'RDDPG': digest(RDDPG)}, 'execution_order': order, 'cases': rows}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    records = []
    for i, method in order:
        print(f'{i+1}/9 {method}', flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(out), '--summary', str(args.summary), '--worker', str(i), '--method', method], cwd=ROOT, check=True)
        records.append(json.loads((out/f'{rows[i]["vehicles"]}_{rows[i]["seed"]}_{method}.json').read_text(encoding='utf8')))
    summary = []
    for n in (100, 1000, 3000):
        for method in ('Ours', 'UPA', 'RDDPG'):
            group = [x for x in records if x['vehicles'] == n and x['method'] == method]
            means = [x['seconds_mean'] for x in group]
            summary.append({'vehicles': n, 'method': method, 'cases': len(group), 'mean_seconds': float(np.mean(means)), 'min_case_mean': min(means), 'max_case_mean': max(means), 'max_observed_slot': max(x['seconds_max'] for x in group)})
    (out/'results.json').write_text(json.dumps({'records': records, 'summary': summary}, indent=2), encoding='utf8')
    (args.summary/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    with (args.summary/'summary.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (out/'complete.json').write_text(json.dumps({'records': len(records), 'manifest_sha256': digest(out/'manifest.json'), 'results_sha256': digest(out/'results.json')}, indent=2), encoding='utf8')

if __name__ == '__main__':
    main()
