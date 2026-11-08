# Causal-state group ablation

> **Release inputs.** The original expert cases and cache are included. The twelve selected ablation checkpoints and selection metadata are included; training is needed only to repeat model fitting, while evaluation generates the trajectories; the recorded per-run summaries are in `results/pressure21_v1/`.

Restore the matched 21-input reference trajectories with `uv run --locked python scripts/regenerate_core.py learned --dataset formal` and `uv run --locked python scripts/regenerate_core.py learned --dataset scale`.

The `pressure21_v1` study compares the full 21-dimensional policy with Local-only, Without A, Without B and Without C. All policies use the same 21-input, 128-by-128 network and the same sequential safety post-processing (SPP). Removed normalized coordinates are fixed to zero during both training and online inference by `evcharging.state_ablation.mask_state`; the network is retrained with the masked inputs. Local-only retains current price, clock and the five current-vehicle features at indices 0, 1 and 10–14. Without A zeros 2–8, Without B zeros 9–14, and Without C zeros 15–20. A's count feature and B's relative position are related, so removing one specified group does not remove every correlated quantity from the others. SPP uses physical fleet information for every variant; Local-only refers to the policy input.

The full policy and each ablation use three matched initialization seeds 11002–11004 and the same audited 360/90 expert trajectories. Training has a teacher-forced stage with Adam, batch 512, learning rate 0.001, at most 300 epochs and patience 30, selecting the smallest expert-validation MSE. A continuation stage then uses 20 epochs, Adam 0.0001 and self-generated preceding raw proposals at saved expert SoCs; the minimum autoregressive validation MSE among epochs 0–20 is selected. Local-only and Without A cannot observe index 7 and do not reconstruct it. Labels remain the original expert actions: continuation is not new expert optimization or closed-loop relabeling. Public entry points are `scripts/build_pressure21_cache.py`, `scripts/train_il_pressure21.py`, `scripts/train_il_pressure21_ablation.py` and `scripts/select_pressure21_ablation.py`. Frozen models and selection provenance are under `artifacts/runs/pressure21_v1/`; the formal train/evaluation workflow is documented in [the experiment protocol](experiment_protocol.md).

The 200-EV comparison uses the same 50 formal scenarios at each of 600 and 800 kW. The scale check uses three common scenarios at each of 1,000 EV/4,000 kW and 3,000 EV/12,000 kW. Values are scenario means within each training run followed by mean ± sample SD across the three runs. A vehicle meets its target at the original 1e-4 SoC tolerance; the gap is the sum of positive departure-energy deficits in kWh per scenario. Per-run results and source hashes are in `results/pressure21_v1/ablation_run_summary.csv` and `results/pressure21_v1/ablation_summary.json`.

## Results

| Fleet / limit | Policy | C | D | F | Scenarios met (%) | EVs met (%) | Total gap (kWh) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 / 600 kW | Ours | 1649.70 ± 1.95 | 53.23 ± 4.16 | 1170.75 ± 0.38 | 97.33 ± 1.15 | 99.71 ± 0.14 | 0.0123 ± 0.0026 |
| 200 / 600 kW | Local-only | 1652.54 ± 0.65 | 78.00 ± 2.12 | 1180.18 ± 0.20 | 68.67 ± 4.16 | 97.59 ± 0.59 | 0.4626 ± 0.0560 |
| 200 / 600 kW | Without A | 1649.76 ± 1.74 | 62.05 ± 4.59 | 1173.44 ± 0.26 | 83.33 ± 11.02 | 98.40 ± 1.12 | 0.1673 ± 0.0973 |
| 200 / 600 kW | Without B | 1746.66 ± 1.10 | 273.27 ± 3.11 | 1304.65 ± 1.46 | 82.00 ± 9.17 | 99.78 ± 0.10 | 0.3139 ± 0.1214 |
| 200 / 600 kW | Without C | 1652.58 ± 0.90 | 56.08 ± 1.54 | 1173.63 ± 0.56 | 84.67 ± 1.15 | 98.67 ± 0.35 | 0.0989 ± 0.0471 |
| 200 / 800 kW | Ours | 1566.19 ± 1.35 | 62.14 ± 2.79 | 1114.97 ± 0.37 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 200 / 800 kW | Local-only | 1597.35 ± 1.03 | 61.25 ± 1.14 | 1136.52 ± 0.39 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 200 / 800 kW | Without A | 1571.31 ± 7.18 | 66.19 ± 4.28 | 1119.78 ± 3.75 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 200 / 800 kW | Without B | 1672.66 ± 5.00 | 356.15 ± 7.31 | 1277.71 ± 5.17 | 94.00 ± 3.46 | 99.95 ± 0.05 | 0.0246 ± 0.0326 |
| 200 / 800 kW | Without C | 1581.82 ± 0.77 | 55.88 ± 0.20 | 1124.04 ± 0.50 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 1000 / 4000 kW | Ours | 7831.13 ± 8.03 | 302.27 ± 14.10 | 5572.47 ± 3.46 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 1000 / 4000 kW | Local-only | 7965.95 ± 6.63 | 285.06 ± 3.97 | 5661.68 ± 3.45 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 1000 / 4000 kW | Without A | 7840.74 ± 35.64 | 318.04 ± 24.78 | 5583.93 ± 17.52 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 1000 / 4000 kW | Without B | 8355.80 ± 28.07 | 1835.49 ± 51.51 | 6399.71 ± 33.28 | 77.78 ± 38.49 | 99.92 ± 0.13 | 0.2389 ± 0.4138 |
| 1000 / 4000 kW | Without C | 7901.19 ± 2.62 | 264.77 ± 1.80 | 5610.26 ± 1.36 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 3000 / 12000 kW | Ours | 23230.60 ± 24.84 | 954.00 ± 46.87 | 16547.62 ± 9.43 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 3000 / 12000 kW | Local-only | 23807.61 ± 29.79 | 910.04 ± 14.65 | 16938.34 ± 16.54 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 3000 / 12000 kW | Without A | 23333.70 ± 136.70 | 972.57 ± 71.36 | 16625.36 ± 74.34 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 3000 / 12000 kW | Without B | 25019.23 ± 116.30 | 5454.85 ± 145.21 | 19149.92 ± 119.02 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |
| 3000 / 12000 kW | Without C | 23518.78 ± 5.46 | 814.84 ± 9.08 | 16707.60 ± 4.53 | 100.00 ± 0.00 | 100.00 ± 0.00 | < 1e-10 |

The 1,000- and 3,000-EV entries are a limited scale check with three fixed scenarios per size, not a substitute for the 50-day formal comparison. Cost, unfairness, service and gap should be interpreted jointly. SPP can make the final allocation depend on fleet information even when the learned policy uses only local inputs.

## Reproduction

After preparing the audited expert assets and frozen model manifests, run `uv run python scripts/evaluate_pressure21_ablation.py --dataset formal` and `uv run python scripts/evaluate_pressure21_ablation.py --dataset scale`, followed by `uv run python scripts/summarize_pressure21_ablation.py`. Each evaluator writes to an independent output directory and refuses to overwrite an existing run. The original masked-model checkpoints can be regenerated with `scripts/train_il_pressure21_ablation.py` and the public cache/API; no private feature implementation is imported. See the current [Pressure21 training protocol](pressure21_training.md) for the complete preparation, selection and hash-verification sequence.
