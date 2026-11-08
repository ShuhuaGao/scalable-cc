# Equipment and price shift results

> **Release inputs.** The daily records and results for this study are included under `results/pressure21_v1/r23/`; regeneration requires the R26 trajectories described in [the protocol](r23_shift_protocol.md).

The frozen protocol is `r23_shift_protocol.md`. All 50 cases per shift and limit are individually reachable and LP-feasible, so the predefined feasible subset equals the full set. The 600 records comprise 400 new closed-loop trajectories and 200 UPA trajectories reused under the changed prices. Each daily observation is paired by original day.

| Shift | Limit (kW) | Cost Full / UPA | D Full / UPA | Full EV met (%) | Full all-day met (%) | Full gap (kWh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| equipment | 600 | 1956.384 ± 78.540 / 2362.585 ± 77.376 | 71.441 ± 27.416 / 311.420 ± 81.759 | 95.270 ± 8.623 | 70.000 ± 46.291 | 1.100 ± 2.241 |
| equipment | 800 | 1850.740 ± 72.265 / 2481.401 ± 95.958 | 51.759 ± 14.529 / 590.987 ± 76.148 | 99.990 ± 0.071 | 98.000 ± 14.142 | 0.001 ± 0.009 |
| price_level | 600 | 2022.842 ± 82.198 / 2539.733 ± 84.089 | 174.530 ± 16.092 / 510.577 ± 88.711 | 97.230 ± 9.097 | 58.000 ± 49.857 | 1.001 ± 3.014 |
| price_level | 800 | 1907.689 ± 78.383 / 2586.502 ± 99.104 | 146.761 ± 16.088 / 701.678 ± 74.171 | 100.000 ± 0.000 | 100.000 ± 0.000 | 0.000 ± 0.000 |
| price_roll | 600 | 1683.317 ± 56.518 / 2059.097 ± 79.353 | 66.246 ± 17.675 / 510.577 ± 88.711 | 99.720 ± 1.079 | 88.000 ± 32.826 | 0.008 ± 0.043 |
| price_roll | 800 | 1673.895 ± 60.558 / 2016.686 ± 78.781 | 87.361 ± 19.114 / 701.678 ± 74.171 | 99.900 ± 0.452 | 92.000 ± 27.405 | 0.002 ± 0.010 |

Values are daily means ± sample SD; all-day met is a 0/100 indicator. UPA met all EV targets on every day for all six shift/limit combinations. Detailed paired differences, ratio-of-means reductions, demand, purchased energy, gap fractions, and both subsets are in `results/pressure21_v1/r23/summary.csv`; daily records and input moments are alongside it.

The equipment draws have across-day mean capacity 34.513 kWh and rated power 6.910 kW. Lower/upper clipping rates are 0.17% / 14.97% for capacity and 0.18% / 15.19% for power. These are joint equipment changes, with altered energy demand and station pressure. Price scaling mechanically changes the monetary scale. The two-hour circular price shift preserves each day’s price multiset, while moving its alignment to arrivals and introducing an artificial boundary seam.

Independent saved-trajectory audit: maximum physical or saved-metric residual 4.55e-13; UPA action and SoC reuse maximum difference 0. Full trajectory, input, and source identities are in the final manifest.
