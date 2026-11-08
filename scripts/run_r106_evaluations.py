"""Prepare and resume paired sensitivity model-day evaluations."""

from __future__ import annotations
import argparse
import hashlib
import json
import msvcrt
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ("w64", "w256", "n30", "n60")
SEEDS = (11002, 11003, 11004)
SOURCES = (
    "scripts/evaluate_r106.py",
    "scripts/summarize_r106.py",
    "scripts/run_r106_evaluations.py",
    "scripts/summarize_distributional_ablation.py",
    "configs/pressure21_evaluation.json",
    "src/evcharging/core.py",
    "src/evcharging/simulation_schema.py",
    "src/evcharging/state_schema.py",
)


def digest(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    :param path: Input file.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(base: Path) -> None:
    """Create or verify the immutable source archive for a run.

    :param base: Run directory.
    :return: None.
    """
    archive = base / "evaluation_source_archive"
    if not archive.exists():
        with tempfile.TemporaryDirectory(
            dir=base, prefix="evaluation_source_stage_"
        ) as temporary:
            stage = Path(temporary) / "archive"
            stage.mkdir()
            hashes = {}
            for name in SOURCES:
                source = ROOT / name
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                hashes[name] = digest(source)
            (stage / "manifest.json").write_text(
                json.dumps(hashes, indent=2), encoding="utf8"
            )
            os.replace(stage, archive)
    hashes = json.loads((archive / "manifest.json").read_text(encoding="utf8"))
    if set(hashes) != set(SOURCES):
        raise ValueError("Evaluation archive source list changed")
    for name, expected in hashes.items():
        if digest(archive / name) != expected or digest(ROOT / name) != expected:
            raise ValueError(f"Evaluation source mismatch: {name}")


def preserve(path: Path) -> None:
    """Move an incomplete directory aside without deleting its contents.

    :param path: Existing output directory.
    :return: None.
    """
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.rename(path.with_name(path.name + "_interrupted_" + suffix))


def completed(path: Path) -> bool:
    """Verify a completed model evaluation's record count and CSV hash.

    :param path: Model evaluation directory.
    :return: Whether the completion marker is valid.
    """
    try:
        meta = json.loads((path / "complete.json").read_text(encoding="utf8"))
        return meta["records"] == 106 and meta["scenarios_sha256"] == digest(
            path / "scenarios.csv"
        )
    except OSError, KeyError, ValueError:
        return False


def main() -> None:
    """Prepare sources, resume finished work and run the remaining evaluations.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (
        Path(args.run_id).name != args.run_id
        or args.run_id in (".", "..")
        or not 1 <= args.jobs <= 4
    ):
        raise ValueError("Invalid run ID or job count")
    base = ROOT / "artifacts/runs" / args.run_id
    base.mkdir(parents=True, exist_ok=True)
    with (base / "evaluation_manager.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another evaluation manager is active") from exc
        prepare(base)
        if args.prepare_only:
            print("Source archive prepared and verified", flush=True)
            return
        pending = {(group, seed) for group in SETTINGS for seed in SEEDS}
        running = {}
        while pending or running:
            for key, (process, log, marker) in list(running.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    marker.unlink()
                    del running[key]
                    if code:
                        raise RuntimeError(f"Evaluation failed for {key}: {code}")
                    print("Completed", key, flush=True)
            for group, seed in sorted(pending):
                if len(running) >= args.jobs:
                    break
                model = base / group / f"il_{seed}" / "il_model"
                if not (model / "summary.json").exists():
                    continue
                dest = base / "evaluation" / group / str(seed)
                marker = base / f"evaluation_{group}_{seed}.pid"
                if marker.exists():
                    pid = int(marker.read_text())
                    if psutil.pid_exists(pid):
                        raise RuntimeError(
                            f"Previous evaluation process still exists: {pid}"
                        )
                    marker.unlink()
                if dest.exists():
                    if completed(dest):
                        pending.remove((group, seed))
                        print("Verified existing", group, seed, flush=True)
                        continue
                    preserve(dest)
                log_path = base / f"evaluation_{group}_{seed}.log"
                log = log_path.open("a", encoding="utf8")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "scripts/evaluate_r106.py"),
                        "--run-id",
                        args.run_id,
                        "--setting",
                        group,
                        "--seed",
                        str(seed),
                        "--workers",
                        "1",
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                marker.write_text(str(process.pid), encoding="utf8")
                running[(group, seed)] = (process, log, marker)
                pending.remove((group, seed))
                print("Started", group, seed, flush=True)
            if pending or running:
                time.sleep(10)
        result = ROOT / "results" / args.run_id
        if result.exists():
            try:
                audit = json.loads((result / "audit.json").read_text(encoding="utf8"))
                valid = audit["records"] == 1590 and all(
                    digest(result / name) == hash_value
                    for name, hash_value in audit["output_sha256"].items()
                )
            except OSError, KeyError, ValueError:
                valid = False
            if valid:
                print("Verified existing summary", flush=True)
                return
            preserve(result)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/summarize_r106.py"),
                "--run-id",
                args.run_id,
            ],
            cwd=ROOT,
            check=True,
        )
        print("All evaluations and analysis complete", flush=True)


if __name__ == "__main__":
    main()
