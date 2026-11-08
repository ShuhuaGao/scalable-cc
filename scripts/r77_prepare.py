"""Freeze matched beta expert cases in a fresh r77 output directory."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/runs/r77'


def digest(path: Path) -> str:
    """Hash one file.

    :param path: Input file.
    :return: SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Freeze inputs and copy executable sources without overwriting results.

    :return: None.
    """
    BASE.mkdir(parents=True, exist_ok=False)
    for name in ('worker', 'prepare_train', 'finish'):
        source = (ROOT / 'scripts' / f'r77_{name}.py').read_text(encoding='utf8')
        # The execution copies reside in r77, so retain their original root calculation.
        source = source.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path(__file__).resolve().parents[3]")
        source = source.replace("BASE = ROOT / 'artifacts/runs/r77'", "BASE = Path(__file__).resolve().parent")
        source = source.replace("base = Path(__file__).resolve().parents[1] / 'artifacts/runs/r77'", "base = Path(__file__).resolve().parent")
        (BASE / f'{name}.py').write_text(source, encoding='utf8')
    original = ROOT/'artifacts/runs/ideal10h_v1/experts/manifest.json'
    manifest = json.loads(original.read_text(encoding='utf8'))
    entries = [dict(entry, beta=beta) for entry in manifest for beta in (0.1,0.5)]
    target = BASE/'manifest.json'
    target.write_text(json.dumps(entries,indent=2),encoding='utf8')
    files = [original, ROOT/'scripts/expert_ideal10h.py', ROOT/'PYSOLVER/nlp2a.py', ROOT/'PYSOLVER/ev.py', ROOT/'src/evcharging/core.py', BASE/'worker.py']
    hashes = {'code/'+path.relative_to(ROOT).as_posix():digest(path) for path in files}
    frozen = {'betas':[0.1,0.3,0.5], 'seed':11002, 'shards':4, 'starts':4, 'window_seconds_per_ev':3, 'input_manifest_sha256':digest(original), 'manifest_sha256':digest(target), 'source_sha256':hashes}
    (BASE/'frozen.json').write_text(json.dumps(frozen,indent=2),encoding='utf8')
    sources = ('scripts/build_pressure21_cache.py','scripts/train_il_pressure21.py','src/evcharging/state_schema.py','src/evcharging/core.py','src/evcharging/models.py','configs/pressure21_il.json','pyproject.toml','uv.lock')
    source_manifest = {}
    for name in sources:
        src=ROOT/name; archive=BASE/'source_archive'/name
        archive.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,archive)
        source_manifest[name]={'sha256':digest(src),'archive':archive.relative_to(ROOT).as_posix().replace('\\','/')}
    execution=BASE/'prepare_train.py'; archive=BASE/'source_archive/prepare_train.py'
    shutil.copy2(execution,archive)
    source_manifest['artifacts/runs/r77/prepare_train.py']={'sha256':digest(execution),'archive':archive.relative_to(ROOT).as_posix().replace('\\','/')}
    (BASE/'training_source_manifest.json').write_text(json.dumps(source_manifest,indent=2),encoding='utf8')
    for name in ('scripts/expert_ideal10h.py','PYSOLVER/nlp2a.py','PYSOLVER/ev.py','PYSOLVER/knitro_env.py','configs/ideal10h_v1.json'):
        src=ROOT/name; archive=BASE/'source_archive'/name
        archive.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,archive)
    shutil.copy2(BASE/'worker.py',BASE/'source_archive/worker.py')
    environment={'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'packages':{name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','knitro')},'worker_threads':1,'worker_count':4,'starts':4,'ms_seed':0,'per_window_seconds':'3 * number of daily EVs'}
    (BASE/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf8')
    print(f'Frozen {len(entries)} matched expert cases in {BASE}')


if __name__ == '__main__':
    main()
