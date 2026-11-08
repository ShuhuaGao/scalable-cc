"""Create and execute the independent MPC expert collection manifest."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from archive_ideal10h import ensure_source_archive


def main() -> None:
    """Create the manifest and optionally execute missing expert solves.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--config", default="configs/ideal10h_v1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ensure_source_archive()
    cfg = json.loads((root / args.config).read_text(encoding="utf8"))["expert"]
    output = root / "artifacts/runs/ideal10h_v1/experts"
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for split in ("train", "validation"):
        seed = cfg[f"{split}_seed_start"]
        for n_str, count in cfg[f"{split}_families"].items():
            for _ in range(count):
                for rate in cfg["capacity_per_ev_kw"]:
                    name = f"{split}_{seed}_{n_str}_{int(round(rate * 10))}decikw"
                    entries.append({"file": name + ".npz", "split": split, "seed": seed, "vehicles": int(n_str), "rate": rate})
                seed += 1
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf8")) != entries:
        raise RuntimeError("Existing expert manifest differs")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("Invalid shard index")
    if not path.exists():
        if args.execute:
            raise RuntimeError("Create manifest before launching workers")
        path.write_text(json.dumps(entries, indent=2), encoding="utf8")
    print(f"Manifest: {len(entries)} expert runs", flush=True)
    if not args.execute:
        return
    for index, entry in enumerate(entries):
        if index % args.shards != args.shard:
            continue
        stem = output / Path(entry["file"]).stem
        report = stem.with_suffix(".json")
        if report.exists() and json.loads(report.read_text(encoding="utf8")).get("complete"):
            continue
        if report.exists() or stem.with_suffix(".npz").exists():
            print(f"Incomplete output retained: {stem}", flush=True)
            continue
        command = [sys.executable, str(root / "scripts/expert_ideal10h.py"), "--seed", str(entry["seed"]), "--vehicles", str(entry["vehicles"]), "--capacity-per-ev", str(entry["rate"]), "--starts", str(cfg["starts"]), "--seconds", str(cfg["seconds_per_10_vehicles"] * entry["vehicles"] / 10), "--output", str(stem)]
        subprocess.run(command, cwd=root, check=False)


if __name__ == "__main__":
    main()
