# Sensitivity to vehicle processing order

The fixed permutations and 505 scenario records are included. Before running the order experiment, restore the development reference trajectories with `uv run --locked python scripts/regenerate_core.py learned --dataset development`.

This development-set experiment repeats the original order-sensitivity design using the selected 21-dimensional Pressure policy, training seed 11002, at 200 EV and 800 kW. It uses the same five existing days 64000–64004 and exactly the original `artifacts/runs/order_sensitivity/orders.npz`: 100 complete full-day permutations per day from `default_rng(20261103)`, plus the identity order. The original order arrays were checked element by element against the recorded RNG procedure. No model is retrained and no extra scenarios are introduced; the development days and this fixed training seed do not provide an independent test of a broader fleet or training distribution.

For each run, the complete permutation maps processing position to original EV index and remains fixed for the day; each slot processes the active subsequence of that ordering. All six vehicle attributes are permuted together, and the price series is unchanged. The versioned simulator runs the same sequential decision logic and SPP with the public `pressure21_v1` state API. Because a full permutation changes both network decision ordering and the corresponding ordered SPP processing, the result describes the deployed controller as a whole. It cannot isolate the effect of the 21-dimensional state representation from that of SPP.

## Results

All 505 runs met every target at the original 1e-4 SoC tolerance. For the 500 random orders, the overall range of F relative to the corresponding day's identity order is **−0.623332% to +1.404633%**, where F = 0.7 C + 0.3 D. The within-day F coefficient of variation is **0.082676%–0.248474%**. The daily mean over 100 random orders differs from identity by **−0.254090% to +0.401082%**. These quantities use each day as the relevant context, not 500 independent days.

C and D must be considered separately. Across day-specific random-order extrema, C ranges from **−0.535385% to +1.107630%** relative to identity and its daily CV spans **0.090998%–0.214059%**. D ranges from **−11.152556% to +16.698092%** and its daily CV spans **1.509098%–3.983313%**. Thus the smaller F variation should not be presented as the same bound for unfairness.

| Day | Identity F | Random mean F | Random SD F | F CV (%) | Maximum absolute F difference from identity (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64000 | 1070.365 | 1067.646 | 2.244 | 0.210 | 0.623 |
| 64001 | 1132.357 | 1131.548 | 1.010 | 0.089 | 0.377 |
| 64002 | 1151.980 | 1151.337 | 0.952 | 0.083 | 0.210 |
| 64003 | 1107.432 | 1104.783 | 1.472 | 0.133 | 0.450 |
| 64004 | 1144.877 | 1149.469 | 2.856 | 0.248 | 1.405 |

For day 64004 the identity F was lower than all 100 random-order F values. For the other four days, the random-order mean was below identity. These observations do not demonstrate a consistent improvement direction across days or establish that no optimal ordering rule exists. Five-day means ± sample SD were 1121.402 ± 33.189 for identity F and 1120.956 ± 35.184 for the within-day random-order means.

## Validation and stored data

The five identity runs exactly match all seven stored `pressure21_v1` development arrays (`soc`, `raw`, `clip`, `scale`, `final`, `cost`, and `unfair`). The stored development references do not contain `states` or `records`, so identity state/record equality against those references is not claimed. The new identity runs save these arrays and are reused as the five experiment identity records; no separate baseline simulation was added.

Each trajectory stores those seven arrays mapped back to original EV order. `states` rows remain in the actual sequential decision order, while `records[:, 1]` contains the associated original EV index; `order` maps each processing position to the original EV index. To recover the processing position for each decision, use `np.argsort(order)[records[:, 1]]`. Do not sort decision-state rows by original EV ID when interpreting A/B/C. Each run's F, C, D, vehicle and scenario completion, positive energy gap, trajectory hash and physical residuals are saved in `64000.csv` through `64004.csv`. Per-day means, sample SD, extrema, relative differences, CV and maximum absolute relative deviations are in `scenario_summary.csv`; `summary.json` additionally reports statistics across the five days.

The maximum positive departure-energy gap among all 505 trajectories was 1.872e-14 kWh. The maximum physical or objective-reconstruction residual was 5.684e-13, below the 1e-8 audit threshold. `promotion_manifest.json` records the per-file hash-checked reuse of all 505 trajectories without rerunning the study. The formal `manifest.json` freezes the original input, order, model, feature-builder, configuration and simulator hashes, as well as the execution environment; `execution/` preserves both the original execution script/configuration and a snapshot of the new public reproduction entry point/configuration. The separate independent audit in `audit/` reconstructs all 505 trajectory metrics and checks 4,476 individual decision states against a scalar feature implementation and single-sample network inference. The retained independent audit is archived alongside the formal results.

## Reproduction

The compact formal results are in `artifacts/runs/pressure21_v1/order_sensitivity/`; the 505 trajectories and execution archive are not included. The original 505 trajectories were hash-verified and copied into that directory; all seven identity arrays exactly match the corresponding `pressure21_v1/evaluation/development/` trajectories, and all per-day summary values are unchanged. The prior experiment used the same Pressure state, inputs and orders; its execution script and independent audit were retained in the original run as provenance. A separate targeted check ran the public schema on one non-identity order and reproduced the saved physical, state and record arrays exactly.

In a separate working copy with the original order archive, shared development inputs, selected Pressure model and formal development trajectories available, but without the existing `artifacts/runs/pressure21_v1/order_sensitivity/` output directory, run from the repository root:

```sh
uv run python scripts/order_pressure21.py --preflight
uv run python scripts/order_pressure21.py --identities
uv run python scripts/order_pressure21.py
```

The public entry point uses at most three workers, an explicit `pressure21_v1` state schema and the unchanged SPP. It intentionally refuses to overwrite prior trajectories. The `--identities` stage performs the five identity runs that are then reused in the 505-run study.
