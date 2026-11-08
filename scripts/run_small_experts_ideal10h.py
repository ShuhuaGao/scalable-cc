"""Generate independent small-scale rolling expert scenarios."""
from pathlib import Path
from expert_ideal10h import run
from archive_ideal10h import ensure_source_archive


def main() -> None:
    """Solve the fixed independent 20-scenario small-scale suite.

    :return: None.
    """
    ensure_source_archive()
    root = Path(__file__).resolve().parents[1]
    dest = root / 'artifacts/runs/ideal10h_v1/small_experts'
    for seed in range(76000, 76020):
        output = dest / f'seed_{seed}'
        if output.with_suffix('.json').exists() or output.with_suffix('.npz').exists():
            print(f'Existing output retained: {output}', flush=True)
            continue
        try:
            run(seed, 10, 40.0, output, 4, 30.0)
        except Exception as exc:
            print(f'Failed {seed}: {exc}', flush=True)


if __name__ == '__main__':
    main()
