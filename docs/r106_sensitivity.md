# Network width, training-set size, and charging intensity

> **Release inputs.** The twelve selected sensitivity checkpoints, training summaries and frozen subsets are included; retrain only to repeat fitting. The included expert cases and Pressure21 cache support the subset and training stages. Generate new evaluations before the aggregation stage; reference model trajectories must also be regenerated where the driver requires them. The recorded summary and daily records are in `results/r106_v2/`.

Restore the matched 21-input reference trajectories with `uv run --locked python scripts/regenerate_core.py learned --dataset formal` and `uv run --locked python scripts/regenerate_core.py learned --dataset scale`.

## Design

The two comparisons share the three frozen 21-input policies trained with two 128-unit GeLU hidden layers and all 120 training families. The width comparison changes both hidden layers together. The data comparison changes the number of independent training families. The expert labels, objective weight (0.3), policy inputs, activation functions, sequential controller, and safety post-processing are common to all settings.

| Hidden units per layer | Training families | Families by fleet size (10 / 20 / 30 / 40 / 50 EVs) | Expert days | Distinct training decisions | Trainable parameters |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 | 120 | 100 / 10 / 5 / 3 / 2 | 360 | 150,174 | 5,633 |
| 128 | 120 | 100 / 10 / 5 / 3 / 2 | 360 | 150,174 | 19,457 |
| 256 | 120 | 100 / 10 / 5 / 3 / 2 | 360 | 150,174 | 71,681 |
| 128 | 30 | 24 / 3 / 1 / 1 / 1 | 90 | 40,032 | 19,457 |
| 128 | 60 | 50 / 5 / 2 / 2 / 1 | 180 | 75,501 | 19,457 |

A family comprises a daily fleet and price realization evaluated at 4, 3.5, and 3 kW per EV. Within each fleet-size stratum, sorted families are shuffled once with NumPy seed `20261107 + fleet_size`; the 30- and 60-family sets use nested prefixes. The same selected families are used for model seeds 11002, 11003, and 11004. The validation split always contains 30 other families, 90 trajectories, and 49,308 decisions. The comparisons describe training variability on this fixed subset selection. Repeated independent subset draws would address a different source of variation.

One reference epoch presents 150,174 decisions in 294 Adam updates (293 batches of 512 and one of 158). Each smaller set is shuffled and cycled to fill the same budget. Teacher training uses a learning rate of 0.001, up to 300 reference epochs, and patience of 30 validation checks. Continuation starts at the best teacher checkpoint with fresh Adam at 0.0001. At each of its 20 reference epochs, the current model reconstructs the preceding requests at the saved expert SoCs. The minimum autoregressive validation MSE over epochs 0–20 selects the deployed checkpoint. The selected checkpoint and actual stopping epoch are recorded separately for every training seed.

## Evaluation

Each frozen model sees the same 50 test days with 200 EVs at each of 600 and 800 kW, and three days each at 1,000 and 3,000 EVs at 4 kW per EV. Each setting contributes 318 model-day records. Four new settings contribute 1,272 records alongside the 318 archived reference records. Match the setting and reference on fleet size, capacity, training seed, test day, and input hash.

Daily measures include cost, unfairness, the weighted objective (0.7 times cost plus 0.3 times unfairness), vehicle and whole-day departure attainment, and total positive battery-energy shortfall. Departure attainment uses the original SoC tolerance of 0.0001. Mean results are calculated within each training seed over common test days, then across the three seeds. Between-seed variation is the sample standard deviation of three run means. For each 200-EV capacity, paired percentile intervals resample the 50 days 20,000 times, keeping the three trained policies per setting fixed and preserving the same dates across settings and seeds. Large-fleet results use the individual paired days and their range.

## Reproduction

Run the following in Windows PowerShell from the implementation root (the repository root), with the locked environment and the original assets listed in [the Pressure21 training protocol](pressure21_training.md) and [evaluation inventory](pressure21_transition.md). The formal sensitivity run is `r106_v2`. Its `manifest.json` records the sampled families and execution-source fingerprints through the linked subset and source files. Use a fresh run name for a new reproduction. The three original reference models and their evaluation records are reused.

```powershell
$runId = "r106_sensitivity_reproduction"
uv run python scripts/train_r106.py --prepare --run-id $runId
foreach ($setting in "w64","w256","n30","n60") {
    foreach ($seed in 11002,11003,11004) {
        uv run python scripts/train_r106.py --group $setting --seed $seed --run-id $runId
        if ($LASTEXITCODE) { throw "Training failed for $setting, $seed" }
    }
}
uv run python scripts/run_r106_evaluations.py --run-id $runId --jobs 4
```

The three width settings share the teacher permutation stream obtained immediately after initializing the 128-wide reference model at each seed. Continuation starts with the original fresh, seed-specific permutation stream. Per-model training curves and the selected checkpoints are under `artifacts/runs/<run-id>/<setting>/il_<seed>/il_model/`. Each evaluated checkpoint has a separate 106-day cohort under `artifacts/runs/<run-id>/evaluation/`. The matched outcomes and uncertainty estimates are in `results/<run-id>/summary.csv`, `run_summary.csv`, and `paired_summary.csv`, with individual outcomes in `daily.csv` and `paired_daily.csv`.

## Charging intensity

Run `uv run python scripts/summarize_r106_charging_intensity.py` to reproduce the six-policy comparison from the 600 selected formal trajectories. The summaries are written to `outputs/charging_intensity/`. PPO uses seed 24003, episode 140. For each slot and EV, multiply the executed grid-side power by charging efficiency, then divide by battery capacity to obtain battery-side power per unit capacity in inverse hours. Weight each slot-EV value by its delivered battery energy. The comparison reports the mean of the 50 daily energy-weighted means and the maximum individual slot intensity over the 50 days, separately for each capacity and policy. This is a power-derived measure of charging exposure. Cell current, temperature, and lifetime were not measured.

The inputs and trajectory hashes, SoC dynamics, inactive slots, and aggregate purchased energy are checked before aggregation. The current six-policy table is `results/pressure21_v1/charging_intensity/combined_summary.csv`, produced by `scripts/summarize_current_ppo_intensity.py`. The earlier independent replay remains in `artifacts/runs/r106_rl_lr_20261107/charging_intensity_selected_ppo_verified/combined_summary.csv`; both use the same selected PPO checkpoint.

## DDQN and PPO learning-rate search

See [the RL learning-rate study](r106_rl_lr_search.md) for the matched three-rate search, validation rankings, formal-day comparison, and reproduction commands.
