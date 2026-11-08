# Heterogeneous charging-efficiency evaluation

> **Release inputs.** The daily records and checks are included in `results/pressure21_v1/r44/`. The per-vehicle draws and trajectories must be regenerated using [the protocol](r44_efficiency_protocol.md).

The prespecified protocol is in `docs/r44_efficiency_protocol.md`. This evaluation keeps the frozen Pressure21 seed-11002 model and the 50 original formal days, and changes only the per-EV efficiency. The 10,000 independent fixed efficiencies have observed mean 0.9503867051 and range [0.9200031424, 0.9799930129]. Both methods and capacities use the identical efficiency vector for each day. Every heterogeneous LP instance is feasible (50/50 at each capacity), and no vehicle is individually unreachable under full rated charging. Therefore the predefined all-day and LP-feasible subsets coincide.

The entries below are mean ± sample SD across the 50 paired days. Full means the frozen IL policy with all SPP steps. Cost and unfairness D are summed over each day; the percentages compare the two methods' mean values on the same days. These SDs describe day-to-day variation for one frozen training seed, not uncertainty across training runs.

| Capacity | Method | Cost, heterogeneous | D, heterogeneous | EVs meeting target | Days all met |
|---|---|---:|---:|---:|---:|
| 600 kW | Full | 1651.944 ± 60.616 | 49.122 ± 13.771 | 99.75% | 98% |
| 600 kW | UPA | 2116.412 ± 69.812 | 513.281 ± 90.557 | 100% | 100% |
| 800 kW | Full | 1567.924 ± 59.391 | 59.861 ± 14.867 | 100% | 100% |
| 800 kW | UPA | 2155.334 ± 82.192 | 704.071 ± 75.595 | 100% | 100% |

Under heterogeneous efficiencies Full reduces mean cost by 21.946% and 27.254%, and mean D by 90.430% and 91.498%, relative to UPA at 600 and 800 kW respectively. The paired Full-minus-UPA daily cost differences are -464.468 ± 54.581 and -587.411 ± 68.256, and the corresponding D differences are -464.160 ± 86.957 and -644.209 ± 62.902.

The nominal-to-heterogeneous Full mean cost changes are +0.0359 (+0.00217%) and +0.1698 (+0.01083%) at 600 and 800 kW, respectively; D changes are +0.2091 (+0.42746%) and +0.6806 (+1.15010%). Nominal Full cost was 1651.908 ± 60.725 and 1567.754 ± 59.416, and D was 48.913 ± 13.752 and 59.181 ± 14.289. UPA mean cost changes are -0.0324 (-0.00153%) and -0.0842 (-0.00391%); D changes are +2.7040 (+0.52959%) and +2.3931 (+0.34106%). Relative changes are calculated as the paired mean difference divided by the nominal mean, not as a mean of individual percentage changes.

At 600 kW Full leaves 0.0130515 ± 0.0922879 kWh of battery energy unmet per day, equal to a mean daily unmet fraction 0.000547755% of battery demand; UPA leaves zero. At 800 kW both methods have numerical-zero gaps. Mean grid demand is 2518.194 ± 74.250 kWh/day in both capacity settings. Actual purchased energy is 2518.181 ± 74.252 kWh/day for Full and 2518.194 ± 74.250 kWh/day for UPA at 600 kW, and 2518.194 ± 74.250 kWh/day for both at 800 kW. The largest physical residual across all nominal and heterogeneous trajectories is 3.41e-13. Target success uses a 1e-4 SoC tolerance; the gap metric still retains positive shortfalls below this threshold.

Detailed daily and summary data are in `results/pressure21_v1/r44/daily.csv` and `summary.csv`; `complete.json` records the LP counts, realized efficiency range, checks and hashes. The original run recorded per-vehicle efficiency draws and trajectory arrays in `artifacts/runs/pressure21_v1/r44_efficiency/`; these large arrays are not included in the compact release. The formal run reused the simulation helper, which is checked against the public simulator in `scripts/verify_r44_efficiency.py`: four heterogeneous trajectories and eight nominal scalar/vector trajectories agree to absolute tolerance 1e-12. The same verification checks 250 R26 input and nominal-trajectory hashes, all 50 non-efficiency inputs, all random draws, and per-vehicle individual reachability. This individual check is post-processing, saved separately as `individual_feasibility.csv` and `verification.json`; it did not select scenarios.
