# RL learning-rate sensitivity

> **Release inputs.** The archived training sources, original configurations, all twenty 1x checkpoints, validation records and formal scenario table are included. The half-rate and double-rate candidates are trained by the workflow below. The strict historical audit also checks reconstructed formal trajectories and metrics.

The learning-rate search uses one matched initialization per algorithm under `pressure21_v1`. DDQN uses the original seed 21002 and learning rates 0.000005, 0.00001 and 0.00002. PPO uses the original seed 24003 and learning rates 0.00005, 0.0001 and 0.0002. Each new candidate has the original budget of 1,000 or 200 episodes respectively. All other settings and training scenario sequences are fixed. The original 1x candidate reuses all ten historical checkpoints. The PPO comparison here is within seed 24003; the historical main selection had compared checkpoints from seeds 24003 and 24004.

Before training, `protocol.json` fixed the candidate settings, validation rules, and source hashes. The new candidates use the archived original training-time source code, verified against the original training plan and copied to an isolated engine. DDQN candidates are ranked across all ten checkpoints by failures, summed SoC gap and objective on seeds 73000–73008. PPO candidates are ranked across all ten checkpoints by failures, total energy gap when failures occur, and objective on seeds 80000–80019. Exact ties use factor order 0.5x, 1x, 2x and ascending episode. No formal scenario is used for selection.

| Algorithm | Adam learning rate | Best episode | Missed validation departures | Mean daily validation objective |
| --- | ---: | ---: | ---: | ---: |
| DDQN | 0.000005 | 100 | 21 | 67.934 |
| DDQN | **0.00001** | **100** | **16** | **68.512** |
| DDQN | 0.00002 | 100 | 18 | 68.358 |
| PPO | **0.00005** | **140** | **0** | **73.973** |
| PPO | 0.0001 | 80 | 0 | 74.639 |
| PPO | 0.0002 | 40 | 0 | 75.597 |

The single-seed search selected DDQN 1x at episode 100 and PPO 0.5x at episode 140. The subsequently selected formal main PPO is this same seed 24003, episode 140 checkpoint; its separate two-run selection is recorded in `artifacts/runs/ppo_half_formal_20261107/selection.json`. For all 20 original 1x checkpoints, the recalculated validation metrics exactly match the archived records. After `selection.json` was written, the changed PPO winner and its 1x control were evaluated using the same current simulator on the original 50 formal days at both 600 and 800 kW. Both attained all targets in all 50 days at each capacity, and all 200 trajectories passed the physical residual audit. The 100 recalculated 1x scenario metrics and input hashes agree exactly with the original formal archive. The two PPO policies use the same training seed; this comparison measures learning-rate sensitivity for that seed.

| Capacity | PPO policy | Cost | Unfairness | Weighted objective | All-met days |
|---|---|---:|---:|---:|---:|
| 600 kW | 1x | 2005.141 | 178.074 | 1457.021 | 50/50 |
| 600 kW | 0.5x | 2003.042 | 143.127 | 1445.067 | 50/50 |
| 800 kW | 1x | 2005.535 | 175.569 | 1456.545 | 50/50 |
| 800 kW | 0.5x | 2003.897 | 140.715 | 1444.942 | 50/50 |

The supplementary paired analysis compares this validation-selected PPO with the original IL seed 11002 on the same inputs. It uses the paired-day bootstrap procedure, 10,000 resamples and seed 20261101; intervals quantify scenario variation for these fixed models. The complete differences, intervals and percentage calculations are in `paired_analysis.json`. The experiment and analysis results are under `artifacts/runs/r106_rl_lr_20261107/`.

The main comparisons use the selected half-rate checkpoint. The charging-intensity calculation replays its 100 formal model-days, verifies the executed schedules against the archived daily metrics, and combines these results with the five other policies. At 600 and 800 kW, respectively, the mean energy-weighted intensities are 0.0749 and 0.0751 per hour and the peaks are 0.2964 and 0.3029 per hour. The reported six-method table and physical checks are in `charging_intensity_selected_ppo_verified/combined_summary.csv` and its accompanying `manifest.json` under the completed run.

The 1x search reuses the included ten original DDQN checkpoints (episodes 100–1000) and ten original PPO checkpoints (episodes 20–200). Generate the original-path formal trajectories first with `uv run python scripts/regenerate_core.py formal`; the recorded hashes and metrics remain the audit reference. Then run the search from the repository root with a new output directory:

```sh
uv run python scripts/run_r106_rl_lr_search.py --phase prepare --output artifacts/runs/r106_rl_lr_reproduction
uv run python scripts/run_r106_rl_lr_search.py --phase train --output artifacts/runs/r106_rl_lr_reproduction
uv run python scripts/run_r106_rl_lr_search.py --phase select --output artifacts/runs/r106_rl_lr_reproduction
uv run python scripts/evaluate_r106_rl_lr_search.py --output artifacts/runs/r106_rl_lr_reproduction --workers 3
uv run python scripts/audit_r106_rl_lr_search.py --output artifacts/runs/r106_rl_lr_reproduction
uv run python scripts/analyze_r106_rl_lr_search.py --output artifacts/runs/r106_rl_lr_reproduction
```

To reproduce six-policy charging intensity after regenerating the required trajectories, run `uv run python scripts/summarize_r106_charging_intensity.py`. To analyze the PPO checkpoint from a new learning-rate search, use:

```sh
uv run python scripts/summarize_r106_selected_ppo_intensity.py --run artifacts/runs/r106_rl_lr_reproduction --output artifacts/runs/r106_rl_lr_reproduction/charging_intensity_selected_ppo_verified --workers 4
```

The learning-rate audit and paired analysis use the 200-record formal cohort for the selected PPO checkpoint and its original-rate control. Exact execution-time copies of the training launcher, selection launcher and formal evaluator are `execution_driver.py`, `selection_driver.py` and `formal_driver.py` in the output directory; their SHA-256 hashes are recorded in `driver_provenance.json`. The output directory contains the frozen protocol, per-checkpoint validation metrics, selected model hashes, and paired formal-day records.
