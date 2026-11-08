# Baseline timing

> **Release inputs.** The recorded compact timing summary is included in `results/pressure21_v1/r24_direct/`. A new timing run requires the frozen method checkpoints and benchmark inputs described below.

The timing study uses the nine existing Figure 14 inputs at 100, 1000 and 3000 configured EVs, three days per size, 56 slots per day and a 4 kW per configured EV station limit. It compares the frozen Pressure21 seed 11002 policy, the existing urgency-proportional allocation (UPA) implementation, and the Li et al. RDDPG charging adaptation using seed 33001, episode 6000. The input and checkpoint hashes are checked against their existing manifests; no scenarios, models or expert solutions are generated.

Run from the repository root with `uv run python scripts/benchmark_r24_baselines.py --output artifacts/runs/r24/timing_reproduction --summary results/pressure21_v1/r24_reproduction`. Both output directories must be absent. The driver runs 27 fresh processes serially, rotating the method order across cases, and fixes numerical libraries and PyTorch to one thread. Each neural policy receives ten zero-state forward passes before measurement. Each slot timer starts before the active mask and ends after the SoC update; model and input loading, warmup, post-loop metrics, writing and validation are excluded. Communication, state estimation and actuation delays are excluded. All slots are retained.

The measurement retains each method's in-memory action and physical-trajectory recording. Ours uses the incremental Pressure21 scheduler with PyTorch inference and SPP. UPA calls the existing urgency allocation once per slot, followed by SPP and the SoC update; the direct rule path avoids constructing the unused neural-policy state. RDDPG uses the existing sequential dynamic-energy-bound controller and NumPy actor inference, with its own constraint handling rather than SPP; its one-time initial boundary and risk calculations occur before the slot loop and are excluded. The methods' state representations, inference engines, constraint handling and recorded arrays differ, including additional neural states and decision records in Ours and RDDPG diagnostics, so the times describe these implementations rather than isolated algorithmic primitives or equal recording overhead.

Each process repeats the same simulation without slot profiling and requires exact equality of returned numeric trajectories and diagnostics. UPA also runs the original causal-state controller outside the timer and requires exact equality of raw, clipped, scaled and final actions, SoC, cost and unfairness. The case record includes physical residuals, cost, squared normalized-power unfairness, target service using the `1e-4` SoC tolerance, and hashes. Physical residuals must not exceed `1e-8`. The archive retains per-slot times and trajectories, source snapshots, model and input identities, execution order, hardware and software details. A completion marker is written only after all 27 runs succeed. Summary means are averages of the three daily mean slot times; the range is the minimum and maximum of those daily means, and the observed maximum is the largest individual slot across the three days. Three days do not establish a deployment tail-latency bound.

## Completed measurements

The final run used Windows 11 build 26300, Intel Core i9-14900KF, 68,541,505,536 bytes physical RAM, Python 3.14.6, NumPy 2.5.4 and PyTorch 2.14.1+cpu. Compact summaries are in `results/pressure21_v1/r24_direct/`.

After measurement, the driver received only a `Scenario` type annotation and explanatory function docstrings. These do not change execution behavior. The measured driver and its SHA-256 remain untouched in the final archive's `source_snapshot/` and `manifest.json`.

| EVs | Method | Mean slot (s) | Daily mean range (s) | Largest slot (s) |
| ---: | :--- | ---: | :--- | ---: |
| 100 | Ours | 0.003622 | 0.003425–0.003782 | 0.008382 |
| 100 | UPA | 0.0000741 | 0.0000720–0.0000755 | 0.000231 |
| 100 | RDDPG | 0.000658 | 0.000637–0.000675 | 0.001187 |
| 1000 | Ours | 0.034311 | 0.034102–0.034502 | 0.070113 |
| 1000 | UPA | 0.000452 | 0.000448–0.000457 | 0.000831 |
| 1000 | RDDPG | 0.006465 | 0.006447–0.006490 | 0.012435 |
| 3000 | Ours | 0.101326 | 0.099425–0.102289 | 0.197068 |
| 3000 | UPA | 0.001368 | 0.001326–0.001442 | 0.002504 |
| 3000 | RDDPG | 0.019393 | 0.019259–0.019581 | 0.034858 |

All nine direct UPA trajectories matched the original controller exactly on the listed arrays, and all 27 timed runs matched their unprofiled repetitions exactly. All vehicles met their targets at the stated tolerance; the largest checked physical residual was `1.53e-10`. Full precision and individual observations are in `results/pressure21_v1/r24_direct/summary.csv` and the original trajectory archive (not included in this compact release).
