# Evaluation data and figures

Repository paths are relative to the checkout root. The fixed main policies are IL 11002, DDQN 21002, PPO 24003 (episode 140, learning rate 0.00005), and RDDPG 33001. `IL` is the data key for Ours. UC and UPA are deterministic rules. IDEAL is the rolling-horizon expert used for the small-fleet reference.

| Result | Scientific inputs and outputs |
| --- | --- |
| Synthetic example, tariff and expert weight sweep (Figs. 4–7) | `scripts/plot_beta_ideal10h.py` reads 13 expert solutions under `artifacts/runs/ideal10h_v1/production/beta/`; it generates the example inputs from the recorded seeds. |
| Training loss (Fig. 8) | The seed-11002 teacher and continuation curves under `artifacts/runs/pressure21_v1/il_11002/il_model/`. |
| Actions and slot-level cost and unfairness (Figs. 9–10) | The 10-EV day 76000 at 40 kW. `results/figure_data/pressure21_plot_arrays.npz` contains the plotted action, SoC, cost and unfairness arrays for six methods. |
| Method comparison and per-vehicle values (Figs. 11–13) | The compact example-day arrays and 20 matched-day scenario tables under `artifacts/runs/pressure21_v1/evaluation/`, `artifacts/runs/ideal10h_v1/small_evaluation/`, `artifacts/runs/li2023_r105/paper_33001/`, and `artifacts/runs/ppo_half_formal_20261107/evaluation/small/`. The last table contains the selected PPO results. |
| Scheduling time and memory (Fig. 14) | `artifacts/pressure21_resources_20261106/results.json`, with three days per size. See the [measurement protocol](r22_resource_benchmark.md). |
| Fleet-size comparisons (Fig. 15) | Scenario tables under `artifacts/runs/pressure21_v1/evaluation/scale/`, `artifacts/runs/ideal10h_v1/scalability/`, `artifacts/runs/li2023_r105/paper_33001/` and `results/li2023_r105/` contain three matched days at each size. Each point averages cost or unfairness per configured vehicle. |
| Main service, cost and unfairness table | `results/pressure21_v1/editorial_values.json` (`formal_table`) contains the 12 method/capacity rows across 600 method-days. `results/pressure21_v1/main_table_rows.tex` stores the displayed rounding. |
| Matched comparisons and ablations | `results/pressure21_v1/` contains the [training and scenario uncertainty](uncertainty_analysis.md) and [state-group ablation](state_ablation.md) summaries. |

To render the ten result panels for Figs. 8–15 from numerical data and training curves, run:

```sh
uv run --locked python scripts/plot_pressure21.py --dest outputs/main_figures
```

`results/figure_data/pressure21_full_validation.json` records the original full-archive validation and the compact array digest. Rendering checks the compact arrays and the recorded scenario summaries; it does not revalidate omitted full trajectories. The generated figure manifest lists the inputs actually read. The example-day bar plot, the two per-vehicle panels and the 20-day comparison all use the same frozen PPO. The installed paper figure digests in `results/pressure21_v1/installed_figures_manifest.json` identify the manuscript PDFs. A local render can have different PDF metadata and fonts. Diagrams of the method and network are manuscript artwork.

The following commands render the synthetic inputs and expert weight sweep, the distribution/SPP diagnostic, and the correlated-arrival comparison. Each writes to a subdirectory of `outputs/`.

```sh
uv run --locked python scripts/plot_beta_ideal10h.py
uv run --locked python scripts/plot_r26_diagnostics.py
uv run --locked python scripts/plot_r107_correlated.py
```

The R2.6 figure reads `results/pressure21_v1/r26/b_slot_daily.csv` and `b_daily.csv`. The R10.7 figure reads `results/figure_data/r107_arrival_departure.npz`; `r107_sources.json` records the original input hashes and compact array digest.

The release inventory in `artifacts/SHA256SUMS` gives SHA-256 digests for the included data and result files. An execution-source fingerprint in a study record identifies the version used for that experiment. The inventory describes the files in this checkout, including manifests with portable paths.

Full formal trajectories can be regenerated locally using `uv run --locked python scripts/regenerate_core.py inputs`, followed by `uv run --locked python scripts/regenerate_core.py formal --output outputs/six_method_formal`. This replays the six formal methods and checks their metrics against the recorded scenario tables; it does not regenerate the 10-EV expert solutions. Generate new expert trajectories with `scripts/expert_ideal10h.py` and an Artelys Knitro license; see [the experiment protocol](experiment_protocol.md).

To evaluate the three selected imitation networks on both formal station limits, use `uv run python scripts/evaluate_pressure21.py --dataset formal --workers 3 --output outputs/pressure21_evaluation`. The evaluator checks checkpoint identity, physical constraints and SoC dynamics and writes a new scenario table and trajectories. Use a fresh output path for another run.
