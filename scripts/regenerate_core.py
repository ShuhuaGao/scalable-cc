"""Regenerate deterministic evaluation inputs and six-method formal trajectories."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from compare_day import ROOT, records
from evcharging.core import causal_state, generate, load_scenario


def digest(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    :param path: File to hash.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs() -> None:
    """Write missing deterministic scenario files without replacing existing files.

    :return: None.
    """
    base = ROOT / 'artifacts/runs/ideal10h_v1'
    cases = [
        (base / kind / 'inputs' / f'{seed}.npz', seed, 200)
        for kind, seeds in [('development', range(64000, 64020)),
                            ('formal', range(65000, 65050))]
        for seed in seeds
    ]
    cases += [
        (base / 'scalability' / f'input_{n}_{81000 + 10*n + i}.npz',
         81000 + 10*n + i, n)
        for n in (100, 500, 1000, 1500, 2000, 2500, 3000) for i in range(3)
    ]
    for path, seed, n in cases:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix='.scenario-') as folder:
            candidate = Path(folder) / path.name
            np.savez_compressed(candidate, **vars(generate(seed, n, fixed_day=True)))
            try:
                os.link(candidate, path)
            except FileExistsError:
                if digest(path) != digest(candidate):
                    raise ValueError(f'Existing scenario differs: {path}')
    print(f'Prepared {len(cases)} scenario inputs')


def present(path: Path, expected: str) -> bool:
    """Check whether an existing destination has the expected digest.

    :param path: Destination file.
    :param expected: Expected SHA-256 digest.
    :return: Whether the correct file already exists.
    :raises ValueError: If the existing file has a different digest.
    """
    if not path.exists():
        return False
    if digest(path) != expected:
        raise ValueError(f'Existing trajectory digest mismatch: {path}')
    return True


def install(candidate: Path, destination: Path, expected: str) -> None:
    """Verify and atomically install a candidate without replacing a file.

    :param candidate: Completed file on the destination filesystem.
    :param destination: Intended output path.
    :param expected: Historical SHA-256 digest.
    :return: None.
    :raises ValueError: If the candidate or a concurrent destination differs.
    """
    if digest(candidate) != expected:
        raise ValueError(f'Regenerated trajectory digest mismatch: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(candidate, destination)
    except FileExistsError:
        present(destination, expected)


def generate_learned(method: str, dataset: str, training: int, seed: int, n: int,
                     limit: int, folder: Path, selection: Path, expected: str) -> Path:
    """Regenerate one learned trajectory with either historical scheduler.

    :param method: IL, DDQN or PPO.
    :param dataset: Evaluation dataset.
    :param training: Training seed.
    :param seed: Scenario seed.
    :param n: Fleet size.
    :param limit: Station limit.
    :param folder: Temporary output directory.
    :param selection: Published RL selection file.
    :param expected: Expected trajectory digest.
    :return: Path of the verified temporary trajectory.
    :raises ValueError: If neither scheduler reproduces the archived digest.
    """
    from evaluate_pressure21 import evaluate_one
    from evaluate_pressure21_rl import run_one

    for reference in (True, False):
        if method == 'IL':
            row = evaluate_one((dataset, training, seed, n, limit, str(folder)),
                               pressure_reference=reference)
            candidate = folder / f'{training}_{n}_{seed}_{limit}.npz'
        else:
            row = run_one((dataset, method, training, seed, n, limit, str(folder), str(selection)),
                          pressure_reference=reference)
            candidate = folder / f'{method}_{training}_{n}_{seed}_{limit}.npz'
        if row['trajectory_sha256'] == expected:
            return candidate
    raise ValueError(f'Both schedulers differ from archived {method} trajectory: {candidate}')


def learned(output: Path, dataset: str, seed: int | None, capacity: int | None) -> None:
    """Restore published learned trajectories selected by scientific records.

    :param output: Root of the repository-shaped output directory.
    :param dataset: Development, formal, small or scale dataset.
    :param seed: Optional scenario seed filter.
    :param capacity: Optional station-capacity filter.
    :return: None.
    """
    base = ROOT / 'artifacts/runs/pressure21_v1/evaluation'
    files = [(base / dataset / 'scenarios.csv', 'IL')]
    if dataset in ('formal', 'small'):
        files.extend([(base / 'rl' / dataset / 'scenarios.csv', 'DDQN'),
                      (ROOT / 'artifacts/runs/ppo_half_formal_20261107/evaluation' /
                       dataset / 'scenarios.csv', 'PPO')])
    selection = ROOT / 'artifacts/runs/pressure21_v1/published_rl_selection.json'
    tasks = []
    for source, method in files:
        with source.open(newline='', encoding='utf8') as stream:
            for row in csv.DictReader(stream):
                if row['method'] != method:
                    continue
                if seed is not None and int(row['seed']) != seed:
                    continue
                if capacity is not None and int(row['capacity_kw']) != capacity:
                    continue
                tasks.append(row)
    if not tasks:
        raise ValueError('No learned trajectories match the requested filters')
    output.mkdir(parents=True, exist_ok=True)
    for row in tasks:
        method = row['method']
        dest = output / row['trajectory_path']
        expected = row['trajectory_sha256']
        if present(dest, expected):
            continue
        with tempfile.TemporaryDirectory(dir=output, prefix='.learned-case-') as folder:
            candidate = generate_learned(
                method, dataset, int(row['training_seed']), int(row['seed']),
                int(row['vehicles']), int(row['capacity_kw']), Path(folder), selection, expected,
            )
            install(candidate, dest, expected)
        print(f"Restored {method} seed {row['training_seed']} day {row['seed']} "
              f"at {row['capacity_kw']} kW", flush=True)
    print(f'Verified {len(tasks)} learned trajectories', flush=True)


def formal(output: Path, seeds: Iterable[int], capacities: Iterable[int]) -> None:
    """Restore verified six-method archives in downstream-compatible locations.

    :param output: Root of the repository-shaped output directory.
    :param seeds: Formal scenario seeds.
    :param capacities: Station capacities in kW.
    :return: None.
    """
    from benchmark_r24_baselines import direct_upa
    from evcharging.li2023 import actor, simulate as simulate_rddpg
    from evcharging.simulation import simulate
    from run_li2023_r105 import numpy_policy

    selection = ROOT / 'artifacts/runs/pressure21_v1/published_rl_selection.json'
    rddpg = ROOT / 'artifacts/runs/li2023_r105/33001/episode_6000.pt'
    expected = json.loads((ROOT / 'results/li2023_r105/manifest.json').read_text())['models_sha256']['33001']
    if digest(rddpg) != expected:
        raise ValueError('RDDPG model digest mismatch')
    model = actor()
    model.load_state_dict(torch.load(rddpg, map_location='cpu', weights_only=True)['model'])
    model.eval()
    torch.set_num_threads(1)
    metadata_path = ROOT / 'artifacts/runs/ideal10h_v1/formal/historical_baseline_metadata.csv'
    with metadata_path.open(newline='', encoding='utf8') as stream:
        metadata = {(int(r['seed']), int(r['capacity']), r['method']): r
                    for r in csv.DictReader(stream)}
    manifest_path = ROOT / 'artifacts/runs/li2023_r105/paper_33001/manifest.json'
    li_hashes = json.loads(manifest_path.read_text())['sources']
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = ROOT / f'artifacts/runs/ideal10h_v1/formal/inputs/{seed}.npz'
        day = load_scenario(str(path))
        for limit in capacities:
            refs = records(seed, limit)
            if {r['input_sha256'] for r in refs.values()} != {digest(path)}:
                raise ValueError(f'Input digest mismatch: {path}')
            paths = {
                'IL': f'artifacts/runs/pressure21_v1/evaluation/formal/11002_200_{seed}_{limit}.npz',
                'DDQN': f'artifacts/runs/pressure21_v1/evaluation/rl/formal/DDQN_21002_200_{seed}_{limit}.npz',
                'PPO': f'artifacts/runs/ppo_half_formal_20261107/evaluation/formal/PPO_24003_200_{seed}_{limit}.npz',
                'UC': f'artifacts/runs/ideal10h_v1/formal/{seed}_{limit}_UC.npz',
                'UPA': f'artifacts/runs/ideal10h_v1/formal/{seed}_{limit}_UPA.npz',
                'RDDPG': f'artifacts/runs/li2023_r105/evaluation/33001/200_{seed}_{limit}.npz',
            }
            expected_hashes = {method: refs[method]['trajectory_sha256']
                               for method in ('IL', 'DDQN', 'PPO')}
            expected_hashes.update({method: metadata[seed, limit, method]['sha256']
                                    for method in ('UC', 'UPA')})
            expected_hashes['RDDPG'] = li_hashes[paths['RDDPG']]
            pending = [method for method, name in paths.items()
                       if not present(output / name, expected_hashes[method])]
            if not pending:
                print(f'Already verified {seed} at {limit} kW', flush=True)
                continue
            with tempfile.TemporaryDirectory(dir=output, prefix=f'.case-{seed}-{limit}-') as folder:
                temporary = Path(folder)
                for method in pending:
                    dest = output / paths[method]
                    candidate = temporary / dest.name
                    if method in ('IL', 'DDQN', 'PPO'):
                        training = {'IL': 11002, 'DDQN': 21002, 'PPO': 24003}[method]
                        candidate = generate_learned(method, 'formal', training, seed, 200,
                                                     limit, temporary, selection,
                                                     expected_hashes[method])
                    elif method in ('UC', 'UPA'):
                        if method == 'UC':
                            result = simulate(day, limit, lambda x, k, u: 1.0, causal=True)
                        else:
                            result = direct_upa(day, limit)
                            pairs = [(t, int(k)) for t in range(len(day.price))
                                     for k in np.flatnonzero((day.ta <= t) & (t < day.td))]
                            result['states'] = np.asarray([
                                causal_state(day, t, k, result['soc'][t], result['raw'][t], limit)
                                for t, k in pairs
                            ])
                            result['records'] = np.asarray(pairs, dtype=int)
                        np.testing.assert_allclose(
                            [result['cost'].sum(), result['unfair'].sum()],
                            [float(refs[method]['cost']), float(refs[method]['unfairness'])],
                            rtol=1e-6, atol=1e-7,
                        )
                        meta = metadata[seed, limit, method]
                        arrays = {key: result[key] for key in
                                  ('soc', 'raw', 'clip', 'scale', 'final', 'states',
                                   'records', 'cost', 'unfair')}
                        np.savez_compressed(candidate, **arrays, seconds=float(meta['seconds']),
                                            feasibility=meta['feasibility'],
                                            primal_residual=float(meta['primal_residual']),
                                            seed=seed, limit=limit, method=method)
                    else:
                        result = simulate_rddpg(day, limit, numpy_policy(model))
                        arrays = {key: result[key] for key in
                                  ('power', 'soc', 'shares', 'reward', 'boundary', 'boundary_rate')}
                        np.savez_compressed(candidate, **arrays)
                    install(candidate, dest, expected_hashes[method])
            print(f'Completed {seed} at {limit} kW', flush=True)


def main() -> None:
    """Parse the requested stage and regenerate its artifacts.

    :return: None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inputs', 'formal', 'learned'])
    parser.add_argument('--output', type=Path, default=ROOT,
                        help='Root of restored repository-shaped paths')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset', choices=('development', 'formal', 'small', 'scale'))
    parser.add_argument('--capacity', type=int)
    args = parser.parse_args()
    if args.stage == 'inputs':
        inputs()
    elif args.stage == 'learned':
        if not args.dataset:
            parser.error('--dataset is required for learned')
        learned(args.output, args.dataset, args.seed, args.capacity)
    else:
        if args.seed is not None and args.seed not in range(65000, 65050):
            parser.error('formal seed must be in 65000..65049')
        if args.capacity is not None and args.capacity not in (600, 800):
            parser.error('formal capacity must be 600 or 800')
        seeds = [args.seed] if args.seed is not None else range(65000, 65050)
        capacities = [args.capacity] if args.capacity is not None else (600, 800)
        formal(args.output, seeds, capacities)


if __name__ == '__main__':
    main()
