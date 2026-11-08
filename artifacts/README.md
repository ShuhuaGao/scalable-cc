# Core artifacts and regeneration

This repository retains all 450 training and validation expert trajectories and their audit, the 20 small-fleet expert trajectories, the Pressure21 expert cache, deterministic scenario inputs, selected models, and compact evaluation records. The large per-day evaluation trajectory archives are omitted. `scripts/example_day.py` and `scripts/compare_day.py` use the retained files directly.

## Scenario inputs

```sh
uv run --locked python scripts/regenerate_core.py inputs
```

This fills missing 20 development and 50 formal inputs under `artifacts/runs/ideal10h_v1/{development,formal}/inputs/`, and 21 scalability inputs under `artifacts/runs/ideal10h_v1/scalability/`. Existing files are retained. Seeds and fleet sizes are fixed by the experiment protocol; formal input digests are checked against the retained records during replay.

## Expert trajectories

The archived 450 expert runs and 20 small-fleet runs already support cache reconstruction and studies requiring expert labels. To regenerate them from scratch in a fresh checkout, remove or relocate the corresponding existing archives first, install Knitro 16 and configure its license, then run:

```sh
uv run --locked python scripts/run_experts_ideal10h.py
uv run --locked python scripts/run_experts_ideal10h.py --execute
uv run --locked python scripts/run_small_experts_ideal10h.py
uv run --locked python scripts/audit_experts_ideal10h.py
```

The 450 expert solves are computationally expensive. Re-optimization may produce different trajectories and hashes. The expert audit and `scripts/build_pressure21_cache.py` use the original archived hashes; regenerated trajectories require a new audit and cache. The latter script writes `artifacts/runs/pressure21_v1/expert_cache/` and refuses an existing directory.

## Formal trajectories

```sh
uv run --locked python scripts/regenerate_core.py formal --seed 65000 --capacity 600
uv run --locked python scripts/regenerate_core.py formal
```

The first command restores one case; omit the filters for all 50 days at both capacities. The destination defaults to the original `artifacts/runs/` paths used by downstream scripts. Use `--output outputs/restored` to create the same directory structure elsewhere. Existing trajectories with the expected digest are skipped, so interrupted runs can be resumed. A differing file stops the run and is never overwritten. Each new trajectory is checked in a temporary case directory before atomic installation; temporary files are removed after the case. IL, DDQN and PPO outputs contain the raw, clipped, scaled and final actions; UC and UPA also contain the historical state and decision arrays; RDDPG includes power, SoC and its boundary and reward arrays. Each regenerated file is checked against its historical SHA-256 digest. The compact `historical_baseline_metadata.csv` preserves the original UC/UPA timing and solver-status values; those timings describe the original run, not regeneration. Tests on other software or hardware may fail a digest check even if the numerical outputs are close.

To restore all published learned trajectories in their original locations, run the following for each dataset:

```sh
uv run --locked python scripts/regenerate_core.py learned --dataset formal
uv run --locked python scripts/regenerate_core.py learned --dataset small
uv run --locked python scripts/regenerate_core.py learned --dataset development
uv run --locked python scripts/regenerate_core.py learned --dataset scale
```

The formal dataset includes all three IL, DDQN and selected half-rate PPO seeds; small includes the main seed of each method. Development and scale restore the IL runs listed in the respective retained records. Add `--seed 65003 --capacity 800` to select one formal case, or `--output outputs/restored` to choose a different destination root. The command uses the published selection and filters the historical RL table to DDQN, excluding the obsolete PPO entry. It tries the archived reference scheduler and the equivalent linear scheduler, and installs a trajectory only when its digest matches the published record. The selected scenario records and completion markers describe the original evaluations; they do not indicate that every omitted trajectory has already been restored in a fresh checkout.

To evaluate newly trained checkpoints instead, use the separate commands in [training and checkpoint selection](../docs/pressure21_training.md).

See [experiment protocol](../docs/experiment_protocol.md) for the solver and scenario definitions, and [training](../docs/pressure21_training.md) for learned-model reproduction and selection. Re-trained models may differ in floating-point details and hashes from the supplied selected models.
