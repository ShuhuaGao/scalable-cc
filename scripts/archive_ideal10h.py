"""Freeze and verify source files used to generate ideal10h experts."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('scripts/expert_ideal10h.py', 'src/evcharging/core.py', 'PYSOLVER/nlp2a.py', 'PYSOLVER/ev.py')


def _create_or_verify(target: Path, data: bytes) -> None:
    """Create a fully written file atomically or verify its existing contents.

    :param target: Destination archive path.
    :param data: Exact expected bytes.
    :return: None.
    :raises RuntimeError: If an existing destination differs.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != data:
                raise RuntimeError(f'Archived file differs: {target}')
    finally:
        temporary.unlink()


def ensure_source_archive() -> dict[str, str]:
    """Create or verify the exact source archive before generating experts.

    Existing files are never overwritten; a difference requires a new run directory.

    :return: Source paths mapped to their SHA-256 digests.
    :raises RuntimeError: If the archive or its manifest differs from the source.
    """
    archive = ROOT / 'artifacts/runs/ideal10h_v1/source_archive'
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        data = (ROOT / name).read_bytes()
        _create_or_verify(archive / name, data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = archive / 'sha256.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf8')) != hashes:
            raise RuntimeError('Archived source hash manifest differs')
    else:
        _create_or_verify(manifest, json.dumps(hashes, indent=2).encode('utf8'))
    return hashes
