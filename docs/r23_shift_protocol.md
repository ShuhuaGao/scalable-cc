# Equipment and price shifts

> **Release inputs.** The compact daily and summary results are in `results/pressure21_v1/r23/`. Generate R26 diagnostics and trajectories first using the command in [the R26 protocol](r26_distribution_spp.md); the freeze step checks their original hashes.

The 50 original formal 200-EV days (65000–65049), limits 600 and 800 kW, frozen Pressure21 seed 11002 beta 0.3 and UPA are used. Three separate shifts are applied to each original day, without selection or resampling. Equipment uses independent streams `SeedSequence([day,2301,1])` and `SeedSequence([day,2301,2])` for capacity N(35,5) clipped to [20,40] and power N(7,1) clipped to [4,8]. Arrival, departure, SoCs, efficiency and price remain fixed. The second shift multiplies the original slot prices by 1.2; the third uses `np.roll(price, 8)` on the 56 slots. All other fields remain fixed. Raw draws and clipping are recorded, and all cases are retained. Individual unreachable vehicles are identified without modifying them.

Full and UPA use the original safety post-processing. For price shifts, nominal UPA actions and SoC are reused and costs recomputed under shifted prices; LP classifications remain applicable because physical inputs are unchanged. For the equipment shift, a fixed-capacity LP classifies every day and limit, distinguishing individual and station-level infeasibility. The 600 method-day-limit records comprise 400 new closed-loop simulations and 200 reused UPA trajectories. Results cover all 50 cases and, separately, the LP-feasible subset by limit; unknown LP status is retained in the full set only.

Daily mean and sample SD are reported for cost, D, per-EV and all-day target attainment at 1e-4 SoC tolerance, positive battery-side gap, its fraction of demand, and grid-side purchased energy and demand. Comparisons use paired Full-minus-UPA differences and percentage reductions calculated from group means. An independent audit checks physical residuals and recomputes metrics from saved trajectories. Price scaling mechanically changes monetary cost; the circular shift preserves the daily price multiset but changes its alignment with parking and creates an artificial day-boundary seam. The joint equipment shift also changes energy demand and capacity pressure.

## Reproduction

In a separate working copy with the existing R26 inputs and trajectories present but without `artifacts/runs/pressure21_v1/r23/` or `results/pressure21_v1/r23/`, run from the repository root:

```sh
uv run python scripts/run_r23_shift.py freeze
uv run python scripts/run_r23_shift.py run
uv run python scripts/audit_r23_shift.py
```

The freeze step checks the hashes of reused inputs and trajectories and records the current implementation sources; it does not require the published code to have the same hash as the original execution snapshot. The audit writes the daily and summary tables to the new results directory. See [the results](r23_shift_results.md).
