"""Audit and summarize paired sensitivity model-day evaluations."""

import argparse, csv, json
import numpy as np
from pathlib import Path, PureWindowsPath
from statistics import mean, stdev
from summarize_distributional_ablation import (
    ROOT,
    BASE,
    SEEDS,
    read,
    selected,
    audit,
    digest,
    verify_archive,
)
from evcharging.core import load_scenario

SETTINGS = ("baseline", "w64", "w256", "n30", "n60")
METRICS = (
    "cost",
    "unfairness",
    "weighted_objective",
    "all_met",
    "vehicle_met_percent",
    "gap_kwh",
    "failed_vehicles",
    "failed_soc_gap_sum",
    "failed_energy_gap_sum",
    "failed_soc_gap_max",
    "failed_energy_gap_max",
)


def write(path: Path, rows: list[dict]) -> None:
    """Write homogeneous records to CSV.

    :param path: Destination.
    :param rows: Nonempty records.
    :return: None.
    """
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def enriched(row: dict, setting: str) -> tuple[dict, list[dict]]:
    """Audit a trajectory and calculate departure and physical residuals.

    :param row: Archived model-day record.
    :param setting: Model setting.
    :return: Audited record and individual failed vehicle records.
    """
    row = row.copy()
    for field in ("input_path", "trajectory_path"):
        parts = PureWindowsPath(row[field]).parts
        try:
            offset = next(
                i for i, part in enumerate(parts) if part.lower() == "artifacts"
            )
        except StopIteration as exc:
            raise ValueError(f"Missing artifacts path: {row[field]}") from exc
        row[field] = str(ROOT.joinpath(*parts[offset:]))
    x = audit(row, setting)
    s = load_scenario(row["input_path"])
    with np.load(row["trajectory_path"]) as d:
        a, soc = d["final"], d["soc"]
        gap = np.maximum(s.target - soc[s.td, np.arange(len(s.ta))], 0)
        fail = gap > 0.0001
        e = gap * s.capacity
        t = np.arange(len(s.price))[:, None]
        active = (s.ta[None, :] <= t) & (t < s.td[None, :])
        checks = {
            "station_kw": np.maximum(a @ s.power - int(row["capacity_kw"]), 0).max(),
            "action_lower": np.maximum(-a, 0).max(),
            "action_upper": np.maximum(a - 1, 0).max(),
            "disconnected": np.abs(a[~active]).max(initial=0),
            "soc_lower": np.maximum(0.2 - soc, 0).max(),
            "soc_upper": np.maximum(soc - 1, 0).max(),
            "overcharge": np.maximum(soc - s.target, 0).max(),
            "dynamics": np.abs(
                soc[1:] - soc[:-1] - a * s.power * s.efficiency * s.dt / s.capacity
            ).max(),
        }
    if max(checks.values()) > 1e-8:
        raise ValueError(f"Physical residual {row['trajectory_path']}: {checks}")
    x.update(
        failed_soc_gap_sum=float(gap[fail].sum()),
        failed_energy_gap_sum=float(e[fail].sum()),
        failed_soc_gap_max=float(gap[fail].max(initial=0)),
        failed_energy_gap_max=float(e[fail].max(initial=0)),
        **{f"violation_{k}": float(v) for k, v in checks.items()},
    )
    failures = [
        dict(
            setting=setting,
            vehicles=len(s.ta),
            capacity_kw=int(row["capacity_kw"]),
            training_seed=int(row["training_seed"]),
            seed=int(row["seed"]),
            vehicle=int(k),
            soc_gap=float(gap[k]),
            energy_gap_kwh=float(e[k]),
        )
        for k in np.flatnonzero(fail)
    ]
    return x, failures


def main() -> None:
    """Pair all audited model days and save summaries.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("Run ID must be a single directory name")
    OUT = ROOT / "results" / args.run_id
    EVAL = ROOT / "artifacts/runs" / args.run_id / "evaluation"
    archive = ROOT / "artifacts/runs" / args.run_id / "evaluation_source_archive"
    source_hashes = json.loads((archive / "manifest.json").read_text(encoding="utf8"))
    for relative, expected in source_hashes.items():
        if (
            digest(archive / relative) != expected
            or digest(ROOT / relative) != expected
        ):
            raise ValueError(f"Evaluation source mismatch: {relative}")
    raw = {
        "baseline": [
            r
            for dataset in ("formal", "scale")
            for r in read(BASE / dataset / "scenarios.csv")
            if selected(r)
        ]
    }
    for setting in SETTINGS[1:]:
        raw[setting] = []
        for seed in SEEDS:
            p = EVAL / setting / str(seed)
            meta = json.loads((p / "complete.json").read_text())
            run_manifest = json.loads((p / "manifest.json").read_text())
            for source, expected in run_manifest["source_sha256"].items():
                matches = [
                    v
                    for k, v in source_hashes.items()
                    if k.replace("\\", "/").endswith(PureWindowsPath(source).name)
                ]
                if len(matches) != 1 or matches[0] != expected:
                    raise ValueError(f"Evaluation execution source mismatch: {source}")
            model = (
                ROOT
                / "artifacts/runs"
                / args.run_id
                / setting
                / f"il_{seed}"
                / "il_model/best.pt"
            )
            if digest(model) != run_manifest["model_sha256"]:
                raise ValueError(f"Changed model: {model}")
            if meta["records"] != 106 or meta["scenarios_sha256"] != digest(
                p / "scenarios.csv"
            ):
                raise ValueError(f"Incomplete {p}")
            raw[setting].extend(read(p / "scenarios.csv"))
    rows = []
    failures = []
    by = {}
    for setting in SETTINGS:
        if len(raw[setting]) != 318:
            raise ValueError(f"Wrong count {setting}")
        by[setting] = {}
        for row in raw[setting]:
            x, f = enriched(row, setting)
            x["method"] = setting
            key = tuple(
                x[k] for k in ("vehicles", "capacity_kw", "training_seed", "seed")
            )
            if key in by[setting]:
                raise ValueError(f"Duplicate {key}")
            by[setting][key] = x
            rows.append(x)
            failures.extend(f)
        if setting != "baseline":
            if by[setting].keys() != by["baseline"].keys():
                raise ValueError("Unmatched keys")
            for key, x in by[setting].items():
                if x["input_sha256"] != by["baseline"][key]["input_sha256"]:
                    raise ValueError(f"Unmatched input {key}")
    runs = []
    groups = []
    paired = []
    comparisons = []
    for n, cap in ((200, 600), (200, 800), (1000, 4000), (3000, 12000)):
        days = (
            range(65000, 65050) if n == 200 else range(81000 + 10 * n, 81003 + 10 * n)
        )
        for setting in SETTINGS:
            for seed in SEEDS:
                subset = [by[setting][(n, cap, seed, day)] for day in days]
                runs.append(
                    dict(
                        method=setting,
                        vehicles=n,
                        capacity_kw=cap,
                        training_seed=seed,
                        days=len(subset),
                        **{k: mean(r[k] for r in subset) for k in METRICS},
                    )
                )
            subset = [
                r
                for r in runs
                if (r["method"], r["vehicles"], r["capacity_kw"]) == (setting, n, cap)
            ]
            g = dict(
                method=setting,
                vehicles=n,
                capacity_kw=cap,
                days=len(days),
                **{k: mean(r[k] for r in subset) for k in METRICS},
                **{k + "_sd": stdev(r[k] for r in subset) for k in METRICS},
            )
            g["failed_soc_gap_pooled_mean"] = (
                g["failed_soc_gap_sum"] / g["failed_vehicles"]
                if g["failed_vehicles"]
                else 0
            )
            g["failed_energy_gap_pooled_mean"] = (
                g["failed_energy_gap_sum"] / g["failed_vehicles"]
                if g["failed_vehicles"]
                else 0
            )
            groups.append(g)
        for setting in SETTINGS[1:]:
            for seed in SEEDS:
                for day in days:
                    k = (n, cap, seed, day)
                    b = by["baseline"][k]
                    a = by[setting][k]
                    paired.append(
                        dict(
                            method=setting,
                            vehicles=n,
                            capacity_kw=cap,
                            training_seed=seed,
                            seed=day,
                            **{m + "_delta": a[m] - b[m] for m in METRICS},
                        )
                    )
            rng = np.random.default_rng(1062026)
            indices = (
                rng.integers(0, len(days), size=(20000, len(days)))
                if n == 200
                else None
            )
            for metric in METRICS:
                mat = np.array(
                    [
                        [
                            by[setting][(n, cap, seed, day)][metric]
                            - by["baseline"][(n, cap, seed, day)][metric]
                            for seed in SEEDS
                        ]
                        for day in days
                    ]
                )
                daily = mat.mean(axis=1)
                sd = mat.mean(axis=0)
                ci = (
                    np.quantile(daily[indices].mean(axis=1), [0.025, 0.975])
                    if indices is not None
                    else [None, None]
                )
                comparisons.append(
                    dict(
                        method=setting,
                        vehicles=n,
                        capacity_kw=cap,
                        metric=metric,
                        delta=float(mat.mean()),
                        delta_11002=sd[0],
                        delta_11003=sd[1],
                        delta_11004=sd[2],
                        bootstrap_95_low=ci[0],
                        bootstrap_95_high=ci[1],
                        daily_delta_min=float(daily.min()),
                        daily_delta_max=float(daily.max()),
                        daily_deltas=json.dumps(dict(zip(days, daily.tolist()))),
                    )
                )
    verify_archive([dict(g, method="21") for g in groups if g["method"] == "baseline"])
    OUT.mkdir(parents=True, exist_ok=False)
    for name, data in [
        ("daily.csv", rows),
        ("run_summary.csv", runs),
        ("summary.csv", groups),
        ("paired_daily.csv", paired),
        ("paired_summary.csv", comparisons),
    ]:
        write(OUT / name, data)
    if failures:
        write(OUT / "failed_vehicles.csv", failures)
    meta = {
        "records": len(rows),
        "new_records": 1272,
        "baseline_records": 318,
        "failed_vehicle_records": len(failures),
        "bootstrap": "20000 paired date resamples, same dates for three fixed matched models; conditional on the models",
        "soc_tolerance": 0.0001,
        "gap_kwh": "sum of all positive departure gaps times battery capacity, including gaps below tolerance",
        "failed_vehicle_gaps": "only departure gaps strictly greater than tolerance; pooled means weight each failed vehicle equally",
        "max_physical_residual": max(
            max(v for k, v in r.items() if k.startswith("violation_")) for r in rows
        ),
        "source_sha256": {
            str(p): digest(p)
            for p in [
                Path(__file__),
                BASE / "formal/scenarios.csv",
                BASE / "scale/scenarios.csv",
            ]
        },
        "output_sha256": {p.name: digest(p) for p in OUT.glob("*.csv")},
    }
    (OUT / "audit.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
