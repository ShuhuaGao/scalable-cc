"""Run the fixed Pressure21 single-factor learning-rate study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts/runs/pressure21_v1"


def sha(p: Path) -> str:
    """Return the SHA-256 digest of a file.

    :param p: File to hash.
    :return: Hexadecimal digest.
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p: Path, obj: dict) -> None:
    """Write a JSON record.

    :param p: Destination file.
    :param obj: Record to serialize.
    :return: None.
    """
    p.write_text(json.dumps(obj, indent=2), encoding="utf8")


def prepare(out: Path) -> None:
    """Freeze the experiment protocol and copy the verified execution sources.

    :param out: New output directory.
    :return: None.
    """
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads((ARCHIVE / "training_plan.json").read_text())
    snapshot = ARCHIVE / "source_archive"
    names = [
        "scripts/train_ddqn_pressure21.py",
        "scripts/train_ppo_pressure21.py",
        "src/evcharging/core.py",
        "src/evcharging/models.py",
        "src/evcharging/state_schema.py",
        "src/evcharging/simulation_schema.py",
    ]
    for name in names:
        expected = {k.replace("\\", "/"): v for k, v in plan["sources_sha256"].items()}[name]
        if sha(snapshot / name) != expected:
            raise RuntimeError(f"Archived source hash mismatch: {name}")
    out.mkdir(parents=True)
    for name in names:
        dst = out / "engine" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / name, dst)
    cfg = json.loads(
        (ARCHIVE / "configs/ddqn_21002.json").read_text(encoding="utf-8-sig")
    )
    if cfg != json.loads(
        (ARCHIVE / "configs/ppo_24003.json").read_text(encoding="utf-8-sig")
    ):
        raise RuntimeError("Main run configs differ")
    jobs = []
    for method, base in [("ddqn", 1e-05), ("ppo", 0.0001)]:
        for label, factor in [("half", 0.5), ("double", 2.0)]:
            c = json.loads(json.dumps(cfg))
            c[method]["learning_rate"] = base * factor
            path = out / "configs" / f"{method}_{label}.json"
            path.parent.mkdir(exist_ok=True)
            write(path, c)
            jobs.append(
                {
                    "method": method,
                    "label": label,
                    "learning_rate": base * factor,
                    "config": path.relative_to(out).as_posix(),
                    "config_sha256": sha(path),
                }
            )
    protocol = {
        "purpose": "Single-factor learning-rate comparison; one matched training seed per algorithm, not a multi-seed search.",
        "locked_before_training": True,
        "source": "Original execution-time Pressure21 archived training sources, verified against training_plan.json.",
        "source_sha256": {n: sha(snapshot / n) for n in names},
        "driver_sha256": sha(Path(__file__)),
        "jobs": jobs,
        "one_x": "Reuse all ten original checkpoints for DDQN seed 21002 and PPO seed 24003.",
        "training": "DDQN 1000 episodes, PPO 200 episodes; all other hyperparameters and scenario sequences held at original settings.",
        "selection": "All ten checkpoints per candidate; DDQN validation seeds 73000-73008 ranked by failures, summed SoC gap, objective; PPO common validation seeds 80000-80019 ranked by failures, kWh gap when failures > 0, objective. Ties resolve in factor order 0.5, 1, 2 then ascending episode.",
        "formal": "After selection is saved, evaluate differing winners and their 1x controls on the same archived 50 days at 600 and 800 kW. Formal results do not influence selection.",
        "ppo_scope": "The 1x candidate is seed 24003 alone, not the historical 24003/24004 two-run main selection.",
        "threads": 1,
    }
    write(out / "protocol.json", protocol)


def train(out: Path) -> None:
    """Run the four candidates using the locked source and settings.

    :param out: Prepared output directory.
    :return: None.
    """
    p = json.loads((out / "protocol.json").read_text())
    for name, expected in p["source_sha256"].items():
        if sha(out / "engine" / name) != expected:
            raise RuntimeError(f"Execution source changed: {name}")
    env = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    jobs = []
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    for job in p["jobs"]:
        method = job["method"]
        label = job["label"]
        dest = out / "engine/artifacts/runs" / f"{method}_{label}" / f"{method}_model"
        if dest.exists():
            raise FileExistsError(dest)
        stdout = (logs / f"{method}_{label}.log").open("w")
        stderr = (logs / f"{method}_{label}.err").open("w")
        config_path = out / "configs" / f"{method}_{label}.json"
        if sha(config_path) != job["config_sha256"]:
            raise RuntimeError(f"Configuration changed: {config_path}")
        cmd = [
            sys.executable,
            str(out / "engine/scripts" / f"train_{method}_pressure21.py"),
            "--config",
            str(config_path),
            "--run-id",
            f"{method}_{label}",
        ]
        proc = subprocess.Popen(
            cmd, cwd=out / "engine", env=env, stdout=stdout, stderr=stderr
        )
        jobs.append((job, proc, stdout, stderr, time.time()))
        print("started", method, label, flush=True)
    for job, proc, stdout, stderr, start in jobs:
        code = proc.wait()
        stdout.close()
        stderr.close()
        print(
            "finished",
            job["method"],
            job["label"],
            code,
            round(time.time() - start),
            flush=True,
        )
        if code:
            raise RuntimeError(f"Failed: {job}")
    write(out / "training_complete.json", {"jobs": len(jobs)})


def select(out: Path) -> None:
    """Rank all checkpoints on the locked independent validation scenarios.

    :param out: Completed training output directory.
    :return: None.
    """
    if not (out / "training_complete.json").exists():
        raise RuntimeError("Training incomplete")
    if (out / "selection.json").exists():
        raise FileExistsError(out / "selection.json")
    protocol = json.loads((out / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        if sha(out / "engine" / name) != expected:
            raise RuntimeError(f"Execution source changed: {name}")
    sys.path[:0] = [str(out / "engine/src"), str(out / "engine/scripts")]
    import torch
    from train_ddqn_pressure21 import evaluate
    from train_ppo_pressure21 import validate

    from evcharging.models import network

    torch.set_num_threads(1)
    cfg = json.loads(
        (ARCHIVE / "configs/ddqn_21002.json").read_text(encoding="utf-8-sig")
    )
    records = []
    for method in ("ddqn", "ppo"):
        for label in ("half", "original", "double"):
            folder = (
                ARCHIVE
                / f"{method}_{(21002 if method == 'ddqn' else 24003)}"
                / f"{method}_model"
                if label == "original"
                else out
                / "engine/artifacts/runs"
                / f"{method}_{label}"
                / f"{method}_model"
            )
            interval = 100 if method == "ddqn" else 20
            paths = [
                folder / f"episode_{ep}.pt"
                for ep in range(interval, interval * 11, interval)
            ]
            for path in paths:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                expected = (
                    cfg[method]["learning_rate"]
                    * {"half": 0.5, "original": 1.0, "double": 2.0}[label]
                )
                seed = 21002 if method == "ddqn" else 24003
                if (
                    ck["config"]["seed"] != seed
                    or ck["config"]["learning_rate"] != expected
                    or ck["config"]["state_schema"] != "pressure21_v1"
                ):
                    raise RuntimeError(f"Invalid checkpoint {path}")
                expected_config = dict(cfg[method], learning_rate=expected)
                if ck["config"] != expected_config or ck["episode"] != int(
                    path.stem.split("_")[1]
                ):
                    raise RuntimeError(
                        f"Checkpoint configuration or episode differs: {path}"
                    )
                model = network(6 if method == "ddqn" else 2)
                model.load_state_dict(ck["model"])
                model.eval()
                if method == "ddqn":
                    metrics = evaluate(
                        model,
                        ck["config"],
                        cfg["expert"]["capacity_per_ev_kw"],
                        cfg["evaluation"]["soc_tolerance"],
                    )
                else:
                    metrics = validate(
                        model,
                        dict(
                            ck["config"],
                            validation_seed_start=80000,
                            validation_count=20,
                        ),
                        cfg["scenario"]["beta"],
                        cfg["evaluation"]["soc_tolerance"],
                    )
                base = ROOT if label == "original" else out
                records.append(
                    {
                        "method": method,
                        "label": label,
                        "episode": ck["episode"],
                        "path": str(path),
                        "path_base": "root" if label == "original" else "output",
                        "relative_path": path.relative_to(base).as_posix(),
                        "sha256": sha(path),
                        "metrics": metrics,
                    }
                )
                print("validated", method, label, ck["episode"], flush=True)

    def rank(x: dict) -> tuple:
        """Apply the prespecified validation ordering.

        :param x: Candidate validation record.
        :return: Lexicographic ranking key.
        """
        m = x["metrics"]
        return (
            m["failed"],
            m["gap"] if x["method"] == "ddqn" else m["gap_kwh"] if m["failed"] else 0,
            m["objective"],
        )

    results = {
        m: min((r for r in records if r["method"] == m), key=rank)
        for m in ("ddqn", "ppo")
    }
    baseline = {
        m: min(
            (r for r in records if r["method"] == m and r["label"] == "original"),
            key=rank,
        )
        for m in ("ddqn", "ppo")
    }
    write(
        out / "selection.json",
        {
            "formal_test_used_for_selection": False,
            "winners": results,
            "baseline": baseline,
            "candidates": records,
        },
    )
    print(
        "winners",
        [(m, v["label"], v["episode"]) for m, v in results.items()],
        flush=True,
    )


def main() -> None:
    """Run the requested stage in a separate output directory.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase", choices=["prepare", "train", "select"], required=True
    )
    args = parser.parse_args()
    out = Path(args.output).resolve()
    {"prepare": prepare, "train": train, "select": select}[args.phase](out)


if __name__ == "__main__":
    main()
