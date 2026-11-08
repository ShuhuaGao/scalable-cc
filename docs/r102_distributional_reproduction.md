# Distributional policy features

> **Release inputs.** The three selected 27-input checkpoints and their training summaries are included; retrain only to repeat fitting. The released expert cases and cache support the training-data step. Formal and scale trajectories must be produced by the evaluation step before aggregation; the compact outcome tables are in `results/r102_distributional_v1/`.

Restore the matched 21-input reference trajectories with `uv run --locked python scripts/regenerate_core.py learned --dataset formal` and `uv run --locked python scripts/regenerate_core.py learned --dataset scale`.

The 27-input policy adds six statistics to the 21-input policy: for both preceding and pending vehicles, the shortest remaining parking time among vehicles still needing charge, and the maximum and population variance of bounded urgency. Both policies use the same expert demonstrations, two 128-unit hidden layers, two-stage training and checkpoint selection, training seeds 11002–11004, and safety post-processing.

Each trained model is evaluated on 50 common days at 200 EVs and three common days at each larger fleet size. Results average days within each run and then average the three runs. C is charging cost, D is fleet unfairness and F = 0.7 C + 0.3 D. A day meets all targets when every departure shortfall is at most 10⁻⁴ SoC. Gap is positive battery-energy shortfall per day.

| EVs / kW | Inputs | C | D | F | Daily max individual D | All-target days (%) | Gap (kWh/day) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 / 600 | 21 | 1649.70 | 53.23 | 1170.75 | 5.009 | 97.33 | 0.0123 |
| 200 / 600 | 27 | 1652.71 | 54.29 | 1173.18 | 5.088 | 84.67 | 0.0710 |
| 200 / 800 | 21 | 1566.19 | 62.14 | 1114.97 | 10.692 | 100 | <1e-10 |
| 200 / 800 | 27 | 1567.86 | 64.86 | 1116.96 | 10.110 | 100 | <1e-10 |
| 1000 / 4000 | 21 | 7831.13 | 302.27 | 5572.47 | 18.402 | 100 | <1e-10 |
| 1000 / 4000 | 27 | 7871.35 | 343.55 | 5613.01 | 16.370 | 100 | <1e-10 |
| 3000 / 12000 | 21 | 23230.60 | 954.00 | 16547.62 | 35.512 | 100 | <1e-10 |
| 3000 / 12000 | 27 | 23381.63 | 1081.46 | 16691.58 | 23.683 | 66.67 | 0.1241 |

The enhanced policy reduces maximum individual unfairness in three settings but increases fleet cost and unfairness in all four; the 21-input policy has a lower weighted objective for each training seed in each setting. At 200 EVs, paired 95% bootstrap intervals for the change in maximum individual unfairness (27 minus 21 inputs) are [-0.113, 0.256] at 600 kW and [-1.154, -0.068] at 800 kW. The large-fleet comparisons contain three days per setting.

To reproduce the matched study, prepare the expert archive, Pressure21 cache and evaluation inputs, then run from the repository root with a fresh shared run ID:

```sh
uv run python scripts/build_r102_distribution_cache.py --run-id r102_reproduce
uv run python scripts/train_r102_distribution.py --variant pressure --seed 11002 --run-id r102_reproduce
uv run python scripts/train_r102_distribution.py --variant pressure --seed 11003 --run-id r102_reproduce
uv run python scripts/train_r102_distribution.py --variant pressure --seed 11004 --run-id r102_reproduce
uv run python scripts/evaluate_r102_distribution.py --dataset formal --workers 4 --run-id r102_reproduce
uv run python scripts/summarize_distributional_ablation.py --run-id r102_reproduce
```
