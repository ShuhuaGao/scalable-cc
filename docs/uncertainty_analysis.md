# Evaluation uncertainty analysis

The scenario tables and bootstrap summaries are included. To repeat the 900-model-day trajectory audit, restore the three runs of each learned method with `uv run --locked python scripts/regenerate_core.py learned --dataset formal`. The small-fleet comparisons use the supplied daily records.

Ours is stored under the legacy key `IL` for reproduction compatibility.

## Current `pressure21_v1` analysis

From the repository root, run `uv run python scripts/analyze_pressure21_uncertainty.py` after the formal Pressure and retrained DDQN/PPO evaluations and the audited state-independent references are complete. The script validates 900 trajectory and input hashes and writes `results/pressure21_v1/training_repeats.csv` and `results/pressure21_v1/uncertainty_summary.json`. The current run contains three independent checkpoints per learned method, evaluated on the same 50 formal days at each of 600 and 800 kW. The main comparison uses Pressure seed 11002, DDQN seed 21002 and PPO seed 24003. The main PPO checkpoint is seed 24003, episode 140 at learning rate 0.00005, and is also the selected checkpoint for that training run; all selection occurs on the independent expert or RL validation sets, not these formal scenarios.

Scenario uncertainty uses 10,000 paired day-level bootstrap samples per capacity and generator seed 20261101; the 20 small-fleet days use a separate stream with seed 20261102. Each comparison applies the same resampled day indices to both methods. Estimates are Ours minus the comparator; negative cost, unfairness and objective differences favor Ours. The purchased-energy unit cost is the ratio of summed costs to summed grid energy, recalculated for each bootstrap sample. Scenario-failure rate is the percentage of days with at least one EV missing its target; vehicle-failure rate is the percentage of EVs missing their targets. The threshold is a `1e-4` SoC shortfall; positive energy gaps are retained even below that threshold. Percentile intervals are descriptive and are not adjusted for multiple comparisons. A point interval with no observed failures does not establish zero risk.

For training variation, average the 50 common days within each trained run, then compute the mean and sample standard deviation across the three independent runs. The 150 run–day records are not 150 independent days. Values below are the mean ± sample SD across the three run-level means; a displayed gap of zero is below the shown precision.

| Capacity (kW) | Method | Cost | Unfairness | All-met scenarios (%) | Shortfall (kWh/day) |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 800 | Ours | 1566.19 ± 1.35 | 62.14 ± 2.79 | 100.00 ± 0.00 | 0.000 ± 0.000 |
| 800 | DDQN | 2063.12 ± 0.41 | 350.87 ± 0.70 | 76.00 ± 3.46 | 1.347 ± 0.291 |
| 800 | PPO | 2003.99 ± 1.01 | 158.70 ± 15.66 | 100.00 ± 0.00 | 0.000 ± 0.000 |
| 600 | Ours | 1649.70 ± 1.95 | 53.23 ± 4.16 | 97.33 ± 1.15 | 0.012 ± 0.003 |
| 600 | DDQN | 1891.63 ± 4.94 | 588.61 ± 71.78 | 0.00 ± 0.00 | 191.818 ± 7.053 |
| 600 | PPO | 2003.36 ± 0.99 | 161.66 ± 16.17 | 100.00 ± 0.00 | 0.000 ± 0.000 |

The selected Pressure continuation epochs are 10, 12 and 10 for seeds 11002, 11003 and 11004. Model selection and checkpoint hashes are recorded in `artifacts/runs/pressure21_v1/il_selection.json`, `rl_selection.json` (DDQN), and `artifacts/runs/ppo_half_formal_20261107/selection.json` (current PPO). Each RDDPG run uses its independently frozen actor and its own state and safety bounds. `scripts/audit_pressure21_rddpg.py` verifies all 318 original RDDPG trajectories against their archived scenario and Li-criterion records, then compares the three new Pressure runs on the same days and fleet sizes; outputs are `results/pressure21_v1/li_criterion_scenarios.csv` and `li_criterion_summary.json`. RDDPG is not counted among the 900 IL/DDQN/PPO records.

The separate timing study uses one CPU thread and three fixed-day scenarios per fleet size. It stores per-slot wall times, compares timed and standard simulation arrays exactly, and reports their scenario means and maxima. Timing samples do not constitute independent training runs.
