"""Measure paired Pressure21 slot time and process peak working set."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import ctypes
from ctypes import wintypes
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import load_scenario
from evcharging.models import imitation_network
from evcharging.simulation_schema import simulate


class MemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS structure."""
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def memory() -> dict[str, int]:
    """Read the current process working set and lifetime peak in bytes.

    :return: Current and peak working-set sizes.
    """
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_bytes': counters.WorkingSetSize, 'peak_working_set_bytes': counters.PeakWorkingSetSize}


def digest(path: Path) -> str:
    """Hash one file.

    :param path: File to hash.
    :return: SHA-256 hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(row: dict, method: str, out: Path) -> None:
    """Run and validate one fresh-process measurement.

    :param row: Frozen historical input identity.
    :param method: Linear or reference scheduler.
    :param out: Output directory.
    :return: None.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = Path(row['input_path'])
    model_path = ROOT / 'artifacts/runs/pressure21_v1/il_11002/il_model/best.pt'
    if digest(path) != row['input_sha256'] or digest(model_path) != row['model_sha256']:
        raise RuntimeError('Frozen input or model hash mismatch')
    scenario = load_scenario(str(path))
    if len(scenario.price) != 56 or float(row['capacity_kw']) != 4 * len(scenario.ta):
        raise RuntimeError('Unexpected horizon or capacity')
    model = imitation_network()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True)['model'])
    model.eval()
    baseline = memory()
    def policy(x: np.ndarray, k: int, u: np.ndarray) -> float:
        """Evaluate the fixed network.

        :param x: State.
        :param k: Vehicle identifier.
        :param u: Urgency vector.
        :return: Normalized action.
        """
        with torch.no_grad():
            return float(model(torch.tensor(x, dtype=torch.float32)[None, :])[0, 0])
    for _ in range(10):
        policy(np.zeros(21), 0, np.zeros(1))
    warmup = memory()
    profile = {'memory_callback': memory}
    limit = float(row['capacity_kw'])
    timed = simulate(scenario, limit, policy, state_schema='pressure21_v1', pressure_reference=method == 'reference', profile=profile)
    standard = simulate(scenario, limit, policy, state_schema='pressure21_v1', pressure_reference=method == 'reference')
    keys = ('soc', 'raw', 'clip', 'scale', 'final', 'states', 'records', 'cost', 'unfair')
    errors = {key: float(np.max(np.abs(timed[key] - standard[key]), initial=0)) for key in keys}
    if max(errors.values()) != 0:
        raise RuntimeError('Timed and standard trajectories differ')
    n = len(scenario.ta)
    gap = np.maximum(scenario.target - timed['soc'][scenario.td, np.arange(n)], 0)
    active = (scenario.ta[None, :] <= np.arange(len(scenario.price))[:, None]) & (np.arange(len(scenario.price))[:, None] < scenario.td[None, :])
    final, soc = timed['final'], timed['soc']
    physical = {'station': float(np.maximum(final @ scenario.power - limit, 0).max(initial=0)), 'disconnected': float(np.abs(final[~active]).max(initial=0)), 'dynamics': float(np.abs(soc[1:] - soc[:-1] - final * scenario.power * scenario.efficiency * scenario.dt / scenario.capacity).max(initial=0)), 'overcharge': float(np.maximum(soc - scenario.target, 0).max(initial=0)), 'action_lower': float(np.maximum(-final, 0).max(initial=0)), 'action_upper': float(np.maximum(final - 1, 0).max(initial=0))}
    if max(physical.values()) > 1e-8:
        raise RuntimeError('Physical constraints failed')
    name = f"{row['vehicles']}_{row['seed']}_{method}"
    trajectory = out / (name + '.npz')
    np.savez_compressed(trajectory, **{k: timed[k] for k in keys}, slot_seconds=np.array(profile['slot_seconds']))
    record = {'vehicles': n, 'seed': int(row['seed']), 'capacity_kw': limit, 'method': method, 'input_path': str(path), 'input_sha256': digest(path), 'model_sha256': digest(model_path), 'trajectory': trajectory.name, 'trajectory_sha256': digest(trajectory), 'seconds_mean': float(np.mean(profile['slot_seconds'])), 'seconds_max': float(np.max(profile['slot_seconds'])), 'memory_load': baseline, 'memory_warmup': warmup, 'memory_preloop': profile['preloop_memory'], 'memory_postloop': profile['postloop_memory'], 'timed_standard_error': errors, 'cost': float(timed['cost'].sum()), 'unfairness': float(timed['unfair'].sum()), 'failed_vehicles': int((gap > 1e-4).sum()), 'service_percent': float(100 * np.mean(gap <= 1e-4)), 'physical': physical}
    (out / (name + '.json')).write_text(json.dumps(record, indent=2), encoding='utf8')


def main() -> None:
    """Run paired cases serially in a newly created archive.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--method', choices=('linear', 'reference'))
    args = parser.parse_args()
    out = args.output.resolve()
    source_csv = ROOT / 'artifacts/runs/pressure21_v1/evaluation/timing/scenarios.csv'
    with source_csv.open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    if args.worker is not None:
        worker(rows[args.worker], args.method, out)
        return
    if len(rows) != 21:
        raise RuntimeError('Expected 21 frozen cases')
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *sorted((ROOT/'src/evcharging').glob('*.py')), ROOT/'pyproject.toml', ROOT/'uv.lock', source_csv]
    snapshot = out/'source_snapshot'
    for p in sources:
        relative = p.relative_to(ROOT)
        dest = snapshot/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    hardware = subprocess.check_output(['powershell', '-NoProfile', '-Command', "$c = Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors; $s = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory; @{cpu=$c;system=$s} | ConvertTo-Json -Depth 3 -Compress"], text=True).strip()
    manifest = {'hardware': json.loads(hardware), 'python_executable': sys.executable, 'protocol': 'Paired serial fresh processes; alternating method order by case; 10 fixed zero-state forward warmups; all 56 slots retained.', 'timing_csv_sha256': digest(source_csv), 'sources_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}, 'platform': platform.platform(), 'processor': platform.processor(), 'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'threads': 1, 'execution_order': []}
    for i, row in enumerate(rows):
        for method in (('linear', 'reference') if i % 2 == 0 else ('reference', 'linear')):
            manifest['execution_order'].append([i, method])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    records = []
    for i, method in manifest['execution_order']:
        print(f"Case {i+1}/21 {method}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(out), '--worker', str(i), '--method', method], check=True, cwd=ROOT, env=os.environ.copy())
        name = f"{rows[i]['vehicles']}_{rows[i]['seed']}_{method}.json"
        records.append(json.loads((out/name).read_text(encoding='utf8')))
    pair_errors = {}
    for row in rows:
        stem = f"{row['vehicles']}_{row['seed']}"
        with np.load(out/(stem+'_linear.npz')) as a, np.load(out/(stem+'_reference.npz')) as b:
            errors = {k: float(np.max(np.abs(a[k]-b[k]), initial=0)) for k in ('soc', 'raw', 'clip', 'scale', 'final', 'states', 'records', 'cost', 'unfair')}
        if max(errors.values()) > 1e-6:
            raise RuntimeError('Paired trajectories differ beyond tolerance')
        pair_errors[stem] = errors
    (out/'results.json').write_text(json.dumps({'records': records, 'pair_errors': pair_errors}, indent=2), encoding='utf8')
    (out/'complete.json').write_text(json.dumps({'results_sha256': digest(out/'results.json'), 'manifest_sha256': digest(out/'manifest.json'), 'records': len(records)}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
