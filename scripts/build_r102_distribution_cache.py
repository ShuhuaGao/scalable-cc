"""Build distributional inputs from audited fixed expert trajectories."""

import argparse
import hashlib, json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from evcharging.core import load_scenario
from evcharging.r102_distribution import schedule_distribution27

DEFAULT_RUN_ID = "r102_distributional_v1_reproduce"
base = R / "artifacts/runs/pressure21_v1/expert_cache"
expert = R / "artifacts/runs/ideal10h_v1/experts"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    """Reconstruct enhanced states from the verified teacher trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Run ID must be a single directory name")
    output = R / "artifacts/runs" / args.run_id / "expert_cache"
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((base / "manifest.json").read_text())
    results = {}
    for split in ["train", "validation"]:
        p = base / f"{split}.npz"
        assert sha(p) == manifest[split]["sha256"]
        with np.load(p) as z:
            data = {k: z[k].copy() for k in z.files}
        x = np.empty((len(data["y"]), 27), np.float32)
        old = None
        maxerr = 0.0
        for g, (source, t) in enumerate(zip(data["group_source"], data["group_slot"])):
            if old != source:
                ep = expert / str(source)
                s = load_scenario(str(ep))
                old = source
                with np.load(ep) as z:
                    soc = z["soc"].copy()
                    actions = z["actions"].copy()
            rows = data["group_index"][g]
            rows = rows[rows >= 0]
            ids = data["records"][rows, 1]
            assert np.all(data["records"][rows, 0] == t)
            eff = np.asarray(s.efficiency)
            eff = eff if eff.ndim == 0 else eff[ids]
            q = schedule_distribution27(
                ids,
                soc[t, ids],
                s.target[ids],
                s.capacity[ids],
                s.power[ids],
                s.td[ids] - t,
                eff,
                s.price[t],
                int(t),
                s.dt,
                data["group_limit"][g],
                lambda state, j, u: actions[t, ids[j]],
            )
            err = np.max(np.abs(q["states"][:, :21] - data["x_pressure"][rows]))
            maxerr = max(maxerr, err)
            if err > 2e-6:
                raise ValueError((source, t, err))
            x[rows] = q["states"]
        data["x_distribution"] = x
        del data["x_pressure"]
        out = output / f"{split}.npz"
        np.savez_compressed(out, **data)
        results[split] = {
            "sha256": sha(out),
            "samples": len(x),
            "groups": len(data["group_index"]),
            "max_original_feature_difference": maxerr,
            "source_sha256": sha(p),
        }
        print(split, results[split], flush=True)
    (output / "manifest.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
