# Sensitivity to the objective weight

> **Release inputs.** Compact daily and derived results are included in `results/pressure21_v1/r77/`. The beta 0.1 and 0.5 expert and trajectory archives are regenerated using [the protocol](r77_beta_protocol.md).

The matched study uses one trained policy for each weight and 79 scenarios per policy. See [the experiment protocol](r77_beta_protocol.md) for the design.

## Statistical conventions

The matching key is dataset, input seed, fleet size and station capacity; the
input hashes must agree across beta values. Each size/capacity cohort reports
means of daily C, D, C/N and D/N; whole-day attainment is the fraction of
scenarios with no vehicle exceeding the 1e-4 SoC gap tolerance, and vehicle
attainment pools vehicles within that cohort. Positive battery-energy gaps use
all positive departure deficits, including values below that tolerance. Mean
gap is per day; demand share is total positive gap divided by total initial
battery demand. The 200-EV, 600-kW cohort is reported separately. Each weight
uses one training seed, so these results do not estimate training-run variance.

Regenerate the summary from saved
scenario records without solving or training:

```powershell
uv run python scripts/summarize_r77_beta.py --output-dir <new-empty-directory>
```

Run from the repository root. This command verifies 79 rows per weight and matching input
hashes, and refuses to replace existing summary files. Use a new directory for a
re-summary. Evaluation similarly refuses an existing beta output directory.
## Results

C and D are daily means. Positive energy gaps include floating-point residuals.

| EV | kW | beta | days | C | D | C/N | D/N | vehicle met % | day met % | gap kWh/day | gap/demand % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 40 | 0.1 | 20 | 75.0192 | 2.98253 | 7.50192 | 0.298253 | 100 | 100 | 3.33323e-16 | 2.87819e-16 |
| 10 | 40 | 0.3 | 20 | 75.7555 | 1.58636 | 7.57555 | 0.158636 | 100 | 100 | 1.50801e-16 | 1.30214e-16 |
| 10 | 40 | 0.5 | 20 | 76.3901 | 1.0797 | 7.63901 | 0.10797 | 100 | 100 | 1.50801e-16 | 1.30214e-16 |
| 100 | 400 | 0.1 | 3 | 780.022 | 48.6523 | 7.80022 | 0.486523 | 100 | 100 | 1.37169e-15 | 1.15181e-16 |
| 100 | 400 | 0.3 | 3 | 791.528 | 29.2096 | 7.91528 | 0.292096 | 100 | 100 | 0 | 0 |
| 100 | 400 | 0.5 | 3 | 799.571 | 21.4843 | 7.99571 | 0.214843 | 100 | 100 | 0 | 0 |
| 200 | 600 | 0.1 | 50 | 1630.37 | 96.726 | 8.15185 | 0.48363 | 97.89 | 78 | 0.188329 | 0.00787195 |
| 200 | 600 | 0.3 | 50 | 1651.91 | 48.9126 | 8.25954 | 0.244563 | 99.75 | 98 | 0.0135284 | 0.000565471 |
| 200 | 600 | 0.5 | 50 | 1668.4 | 39.3746 | 8.34198 | 0.196873 | 99.7 | 98 | 0.0242559 | 0.00101387 |
| 1000 | 4000 | 0.1 | 3 | 7712.24 | 518.272 | 7.71224 | 0.518272 | 100 | 100 | 1.7628e-14 | 1.46498e-16 |
| 1000 | 4000 | 0.3 | 3 | 7840.16 | 287.481 | 7.84016 | 0.287481 | 100 | 100 | 0 | 0 |
| 1000 | 4000 | 0.5 | 3 | 7923.77 | 210.617 | 7.92377 | 0.210617 | 100 | 100 | 2.21473e-15 | 1.84055e-17 |
| 3000 | 12000 | 0.1 | 3 | 22879.6 | 1579.96 | 7.62654 | 0.526654 | 100 | 100 | 1.20683e-13 | 3.34149e-16 |
| 3000 | 12000 | 0.3 | 3 | 23258.5 | 903.08 | 7.75285 | 0.301027 | 100 | 100 | 1.9525e-14 | 5.4061e-17 |
| 3000 | 12000 | 0.5 | 3 | 23470.9 | 639.61 | 7.82362 | 0.213203 | 100 | 100 | 9.18089e-15 | 2.54201e-17 |

Against beta 0.3, across the four 4 kW/EV cohorts beta 0.1 reduces cost by 0.972–1.632% and increases D by 66.563–88.011%; beta 0.5 increases cost by 0.838–1.066% and reduces D by 26.448–31.939%. At 200 EV and 600 kW, beta 0.1 reduces cost by 1.304% and increases D by 97.753%; beta 0.5 increases cost by 0.998% and reduces D by 19.500%. These are comparisons of one trained model per weight.

## Common objective and adjacent trade-offs

The derived analysis reads the saved `results.csv` only; it does not simulate,
train or solve. It computes the common reference `F_ref = 0.7 C + 0.3 D` for
each model and its own-weight objective `F_beta = (1-beta) C + beta D`. The
latter uses a different objective for each beta and must not be ranked across
weights. The common reference is a fixed comparison scale.

| EV | kW | F_ref at 0.1 | F_ref at 0.3 | F_ref at 0.5 | 0.3 above minimum % | D reduction per added C, 0.1→0.3 | D reduction per added C, 0.3→0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 40 | 53.4082104 | 53.5047288 | 53.7969580 | 0.180718 | 1.896347 | 0.798386 |
| 100 | 400 | 560.611274 | 562.832798 | 566.144929 | 0.396268 | 1.689759 | 0.960562 |
| 200 | 600 | 1170.276426 | 1171.009606 | 1179.689433 | 0.062650 | 2.219867 | 0.578499 |
| 1000 | 4000 | 5554.052740 | 5574.354400 | 5609.822952 | 0.365529 | 1.804284 | 0.919302 |
| 3000 | 12000 | 16489.712736 | 16551.908472 | 16621.483063 | 0.377179 | 1.786236 | 1.240981 |

Beta 0.1 has the smallest common objective in all five cohorts. The observed
`(D_before - D_after)/(C_after - C_before)` decreases in the second interval in
all five; these two discrete intervals do not establish a continuous curve.
At 200 EV and 600 kW, beta 0.1→0.3 changes vehicle and whole-day attainment
by +1.86 and +20 percentage points and reduces mean gap by 0.174801 kWh/day;
0.3→0.5 changes them by -0.05 and 0 percentage points and increases mean gap
by 0.0107275 kWh/day. Thus beta 0.3 is an observed compromise, not the
minimum of the common reference objective.

Full-precision objectives and changes are in `derived_objectives.csv`,
`derived_tradeoffs.csv` and `derived_analysis.json` under
`results/pressure21_v1/r77/`. Their SHA-256 hashes in that order are
`e19caf8b9966a3bc74588bafedcaae0d81671965b9963fb5b498d39f7adf2927`,
`0094cf3c7bf9ac9ae0a034d6ba114fdefd71c17b6b0c23214e62797293846eb5`, and
`e86fbbe6ed753b29194f42af7a8e047513e815501d0c0f40ff302e57a49fc01e`.
The unchanged `results.csv` source has SHA-256
`5ece853b1b55e9989cf9b30e20fad5c1c46b64c6266a7a6af384ffad2822b668`.
From the repository root, regenerate the derived files into a new directory with:

```powershell
uv run python scripts/analyze_r77_objectives.py --output-dir <new-empty-directory>
```

This command refuses to replace existing derived outputs. A separate-directory
regeneration matched all three files byte for byte; the three original
79-row scenario CSV hashes match the independent audit, and all 237 input and
trajectory hashes match their recorded values.
