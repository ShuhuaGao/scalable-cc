"""Freeze and execute the prespecified R2.3 distribution shifts."""
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import run_r26_diagnostics as old
from evcharging.core import fixed_feasibility, load_scenario

R = old.R
A = R / 'artifacts/runs/pressure21_v1/r23'
O = R / 'results/pressure21_v1/r23'
KINDS = ('equipment', 'price_level', 'price_roll')


def ensure_no_cases(directory: Path, results_directory: Path | None = None) -> None:
    """Reject execution if any prescribed case output already exists.

    :param directory: Frozen archive directory.
    :return: None.
    """
    for pattern in ('draws_*.npz', 'input_*.npz', 'feas_*.json', 'traj_*.npz', 'manifest.json', 'run_started.lock'):
        if next(directory.glob(pattern), None) is not None:
            raise FileExistsError(f'Existing case or completion manifest in {directory}: {pattern}')
    if results_directory is not None and results_directory.exists() and any(results_directory.iterdir()):
        raise FileExistsError(f'Existing results in {results_directory}')


def shifted(original, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Apply the three independently specified shifts to one original day.

    :param original: Original scenario.
    :param seed: Original scenario seed.
    :return: Scenarios and raw capacity and power draws.
    """
    cap_raw = np.random.default_rng(np.random.SeedSequence([seed, 2301, 1])).normal(35, 5, 200)
    power_raw = np.random.default_rng(np.random.SeedSequence([seed, 2301, 2])).normal(7, 1, 200)
    return ({'equipment': replace(original, capacity=np.clip(cap_raw, 20, 40), power=np.clip(power_raw, 4, 8)),
             'price_level': replace(original, price=original.price * 1.2),
             'price_roll': replace(original, price=np.roll(original.price, 8))}, cap_raw, power_raw)


def reprice_upa(s, trajectory: dict) -> dict:
    """Copy a price-independent UPA trajectory and update its costs.

    :param s: Scenario with modified prices and unchanged physical inputs.
    :param trajectory: Nominal UPA trajectory.
    :return: Copied trajectory with recomputed cost and unfairness.
    """
    result = {key: value.copy() for key, value in trajectory.items()}
    result['cost'], result['unfair'] = old.totals(s, result['soc'], result['final'])
    return result


def sha(p: Path) -> str:
    """Return SHA-256 of a file.

    :param p: Input file.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def requires_archived_identity(path: str) -> bool:
    """Identify original inputs and baseline trajectories on any platform.

    :param path: Relative source path using either path separator.
    :return: Whether an existing R2.6 archive identity is mandatory.
    """
    normalized = path.replace('\\', '/')
    return ('formal/inputs/' in normalized or 'd_nominal_' in normalized
            or 'evaluation/formal/' in normalized)


def sources() -> list[Path]:
    """List all fixed sources needed by the experiment.

    :return: Paths relative to the code directory.
    """
    paths = [Path('scripts/run_r23_shift.py'), Path('docs/r23_shift_protocol.md'),
             Path('results/pressure21_v1/r26/d_daily.csv'), Path('artifacts/runs/pressure21_v1/r26_diagnostics/manifest.json')]
    paths += [Path(x) for x in ('scripts/run_r26_diagnostics.py', 'src/evcharging/core.py', 'src/evcharging/models.py', 'src/evcharging/state_schema.py', 'src/evcharging/simulation_schema.py', 'artifacts/runs/pressure21_v1/il_selection.json', 'artifacts/runs/pressure21_v1/il_11002/il_model/best.pt')]
    for seed in range(65000, 65050):
        paths.append(Path(f'artifacts/runs/ideal10h_v1/formal/inputs/{seed}.npz'))
        for limit in (600, 800):
            paths.extend([Path(f'artifacts/runs/pressure21_v1/r26_diagnostics/d_nominal_{seed}_{limit}_upa.npz'), Path(f'artifacts/runs/pressure21_v1/evaluation/formal/11002_200_{seed}_{limit}.npz')])
    return paths


def freeze() -> None:
    """Verify archived source identities and save a pre-execution manifest.

    :return: None.
    """
    A.mkdir(parents=True, exist_ok=False)
    recorded = json.loads((R / 'artifacts/runs/pressure21_v1/r26_diagnostics/manifest.json').read_text(encoding='utf8'))['sha256']
    archive = {key.replace('\\', '/'): value for key, value in recorded.items()}
    identities = {str(p): sha(R / p) for p in sources()}
    for p, digest in identities.items():
        key = p.replace('\\', '/')
        if requires_archived_identity(key):
            if key not in archive or archive[key] != digest:
                raise ValueError(f'Archived input or trajectory identity mismatch: {p}')
    old.model()
    (A / 'frozen_protocol.md').write_bytes((R / 'docs/r23_shift_protocol.md').read_bytes())
    (A / 'pre_run_manifest.json').write_text(json.dumps({'sha256': identities}, indent=2), encoding='utf8')


def task(seed: int) -> int:
    """Run all prescribed cases for one original day.

    :param seed: Original scenario seed.
    :return: Completed seed.
    """
    m = old.model()
    original = load_scenario(str(old.INPUT / f'{seed}.npz'))
    scenarios, cap_raw, power_raw = shifted(original, seed)
    np.savez_compressed(A / f'draws_{seed}.npz', capacity_raw=cap_raw, power_raw=power_raw)
    with (R / 'results/pressure21_v1/r26/d_daily.csv').open(encoding='utf8', newline='') as f:
        baseline = {(int(x['seed']), int(x['limit'])): x for x in csv.DictReader(f) if x['distribution'] == 'nominal' and x['method'] == 'UPA'}
    for kind, s in scenarios.items():
        old.save_scenario(A / f'input_{kind}_{seed}.npz', s)
        for limit in (600, 800):
            if kind == 'equipment':
                status, residual = fixed_feasibility(s, limit)
            else:
                ref = baseline[seed, limit]
                status, residual = ref['feasibility'], ref['feasibility_residual']
            (A / f'feas_{kind}_{seed}_{limit}.json').write_text(json.dumps({'status': status, 'residual': residual}), encoding='utf8')
            for method in ('Full', 'UPA'):
                if kind != 'equipment' and method == 'UPA':
                    with np.load(old.A / f'd_nominal_{seed}_{limit}_upa.npz') as z:
                        traj = {k: z[k] for k in z.files}
                    traj = reprice_upa(s, traj)
                else:
                    traj = old.simulate(s, limit, m if method == 'Full' else None, method.lower())
                np.savez_compressed(A / f'traj_{kind}_{seed}_{limit}_{method}.npz', **traj)
    return seed


def main() -> None:
    """Execute all cases after checking frozen source hashes.

    :return: None.
    """
    ensure_no_cases(A, O)
    manifest = json.loads((A / 'pre_run_manifest.json').read_text(encoding='utf8'))['sha256']
    for p, digest in manifest.items():
        if sha(R / p) != digest:
            raise ValueError(f'Source changed after freeze: {p}')
    with (A / 'run_started.lock').open('x', encoding='utf8') as marker:
        marker.write('Execution started; retain this marker even if interrupted.\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        for seed in pool.map(task, range(65000, 65050)):
            print(f'Completed {seed}', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['freeze']:
        freeze()
    elif sys.argv[1:] == ['run']:
        main()
    else:
        raise SystemExit('Use freeze or run')
