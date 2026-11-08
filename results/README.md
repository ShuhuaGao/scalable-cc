# Compact results inventory

Use these tables to inspect the manuscript results or redraw its figures. The frozen policies, expert demonstrations and evaluation inputs are under `artifacts/`. To regenerate omitted per-day trajectories, follow the [data guide](../artifacts/README.md); the [experiment protocol](../docs/experiment_protocol.md) links to instructions for each supplementary study.

- `pressure21_v1/`: main comparison, ablations, uncertainty, timing and R23/R26/R44/R77/R107 summaries.
- `r102_distributional_v1/` and `r106_v2/`: matched feature and sensitivity results, including daily records.
- `li2023_r105/`: RDDPG comparison and three-run summaries.
- `ideal10h_v1/`, `ppo_half_formal_20261107/`, `r106_v1/` and `training_repeats/`: additional summaries and figure inputs.
- `artifacts/runs/pressure21_v1/order_sensitivity/` contains the compact order-study records; its fixed permutation input is `artifacts/runs/order_sensitivity/orders.npz`.
- `artifacts/pressure21_resources_20261106/results.json` and `artifacts/pressure21_linear_audit_final_20261106/audit.json` retain resource and equivalence audit summaries.

Use fresh output directories when running the documented workflows; many drivers intentionally refuse to overwrite existing results. The published values are observations from the recorded runs, while regenerated runs may differ with hardware, software and solver behavior.
