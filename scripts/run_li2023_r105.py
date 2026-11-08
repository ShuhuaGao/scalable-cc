"""Train and evaluate the adapted Li 2023 baseline; see its protocol."""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evcharging.core import generate, load_scenario
from evcharging.li2023 import PreviousEpisodeD, Replay, actor, critic, simulate

OUT = ROOT / 'artifacts/runs/li2023_r105'
RESULT = ROOT / 'results/li2023_r105'
CONFIG = dict(episodes=6000, seeds=[33001, 33002, 33003], train_scenario_start=900000,
              vehicles=10, capacity_per_ev=[4, 3.5, 3], warmup=5, batch=64,
              replay=60000, actor_lr=1e-4, critic_lr=1e-3, gamma=.99, tau=.01,
              sigma=2., decay=.9993, beta=.4, reward_scale=1., save_every=500)


def digest(path: Path) -> str:
    """Hash a file.

    :param path: File path.
    :return: SHA-256 hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive() -> None:
    """Freeze exact sources and experiment settings before formal training.

    :return: None.
    """
    folder = OUT / 'source_archive'
    folder.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), ROOT / 'src/evcharging/core.py', ROOT / 'src/evcharging/li2023.py']
    hashes = {}
    for path in files:
        name = path.relative_to(ROOT).as_posix().replace('\\', '/')
        dest = folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest) != digest(path):
            raise RuntimeError(f'Archived source differs: {name}')
        if not dest.exists():
            shutil.copy2(path, dest)
        hashes[name] = digest(dest)
    manifest = dict(config=CONFIG, source_sha256=hashes, python=platform.python_version(),
                    numpy=np.__version__, torch=torch.__version__)
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != manifest:
        raise RuntimeError('Existing manifest differs')
    path.write_text(json.dumps(manifest, indent=2), encoding='utf8')


def numpy_policy(model: nn.Module, rng: np.random.Generator | None = None,
                 sigma: float = 0.) -> Callable[[np.ndarray], float]:
    """Snapshot weights for equivalent CPU actor inference during one episode.

    :param model: Actor network.
    :param rng: Exploration generator or None for deterministic inference.
    :param sigma: Exploration noise standard deviation.
    :return: State-to-action function.
    """
    weights = [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy())
               for m in model if isinstance(m, nn.Linear)]
    def infer(state: np.ndarray) -> float:
        """Evaluate the saved actor on one state.

        :param state: Nine-element observation.
        :return: Action with optional exploration noise.
        """
        x = state
        for w, b in weights[:-1]:
            x = np.maximum(w @ x + b, 0)
        a = np.tanh(weights[-1][0] @ x + weights[-1][1]).item()
        return a + (rng.normal(0, sigma) if rng is not None else 0)
    return infer


def train(seed: int, episodes: int, output: Path) -> None:
    """Train one actor on the prescribed common training scenarios.

    :param seed: Network and exploration seed.
    :param episodes: Number of complete training episodes.
    :param output: Destination for checkpoints and training curve.
    :return: None.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    a, q, at, qt = actor(), critic(), actor(), critic()
    at.load_state_dict(a.state_dict()); qt.load_state_dict(q.state_dict())
    ao = torch.optim.Adam(a.parameters(), lr=CONFIG['actor_lr'])
    qo = torch.optim.Adam(q.parameters(), lr=CONFIG['critic_lr'])
    replay, history = Replay(CONFIG['replay']), PreviousEpisodeD()
    output.mkdir(parents=True, exist_ok=True)
    curve = []
    for episode in range(episodes):
        s = generate(CONFIG['train_scenario_start'] + episode, n=10, fixed_day=True)
        limit = 10 * CONFIG['capacity_per_ev'][episode % 3]
        prior = history.mean() if episode >= 5 else None
        r = simulate(s, limit, numpy_policy(a, rng, CONFIG['sigma'] * CONFIG['decay'] ** episode), d_history=prior)
        for row in r['transitions']:
            replay.add(*row)
        history.add(r['d'])
        if len(replay) >= CONFIG['batch']:
            st, ac, rw, ns, done = replay.sample(CONFIG['batch'], rng)
            st, ac, rw, ns, done = (torch.as_tensor(x, dtype=torch.float32) for x in (st, ac[:, None], rw[:, None], ns, done[:, None]))
            with torch.no_grad():
                y = rw + CONFIG['gamma'] * (1 - done) * qt(torch.cat((ns, at(ns)), dim=1))
            loss = (q(torch.cat((st, ac), dim=1)) - y).square().mean()
            qo.zero_grad(); loss.backward(); qo.step()
            actor_loss = -q(torch.cat((st, a(st)), dim=1)).mean()
            ao.zero_grad(); actor_loss.backward(); ao.step()
            with torch.no_grad():
                for target, source in [(at, a), (qt, q)]:
                    for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
                        tp.lerp_(sp, CONFIG['tau'])
        if not np.isfinite(r['soc']).all() or not all(torch.isfinite(p).all() for p in a.parameters()):
            raise RuntimeError(f'Nonfinite values at episode {episode + 1}')
        curve.append(dict(episode=episode+1, reward=float(r['reward'].sum()),
                          empty_count=r['empty_count'], empty_gap=r['empty_gap'],
                          next_empty_count=r['next_empty_count'], next_empty_gap=r['next_empty_gap'],
                          initial_deficit_count=r['initial_deficit_count'],
                          initial_deficit_soc=r['initial_deficit_soc']))
        if (episode + 1) % 500 == 0 or episode + 1 == episodes:
            torch.save(dict(model=a.state_dict(), episode=episode+1, seed=seed), output / f'episode_{episode+1}.pt')
            (output / 'curve.json').write_text(json.dumps(curve), encoding='utf8')
            print(f'{seed}: {episode+1}/{episodes}', flush=True)


def main() -> None:
    """Parse command-line arguments and run the requested stage.

    :return: None.
    """
    p = argparse.ArgumentParser(description="Train the RDDPG baseline in a new output directory.")
    p.add_argument('--seed', type=int, default=33001)
    p.add_argument('--episodes', type=int, default=CONFIG['episodes'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    train(args.seed, args.episodes, args.output)


if __name__ == '__main__':
    main()
