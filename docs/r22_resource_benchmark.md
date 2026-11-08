# Pressure21 paired resource benchmark

> **Release inputs.** The recorded per-case results are included as `artifacts/pressure21_resources_20261106/results.json`. Rerunning also requires the 21 timing input files listed by the timing manifest and the frozen model.

The current resource study uses the frozen Pressure21 seed 11002 beta 0.3 model and the 21 input files listed in `artifacts/runs/pressure21_v1/evaluation/timing/scenarios.csv`. The input hashes and model hash are checked before every run. The cases cover 100, 500, 1000, 1500, 2000, 2500 and 3000 configured EVs, three fixed days each, 56 slots per day and 4 kW per configured EV. The experiment does not generate scenarios, train a policy or solve an expert problem.

From the repository root, run `uv run python scripts/benchmark_pressure21_resources.py --output artifacts/<new-directory>` on Windows. The directory must not exist. The driver fixes BLAS/OpenMP and PyTorch compute threads to one, executes one process at a time and starts a fresh process for each case and each method. The method order alternates deterministically across cases. Each process loads the scenario and checkpoint, records its working set, evaluates 10 fixed zero-state network forwards, and records the pre-loop and post-loop process memory. All slots are retained. The timer starts before the active mask and urgency computation and ends after the SoC update. It includes full-day adapter scans, state construction, policy inference, safety correction, and in-memory states and records appends. It excludes model and input loading, memory sampling, post-loop metrics and array conversion, output serialization, and validation. Communication, state estimation and actuation delays are excluded.

The optional `simulate` profiling interface uses the same scheduler as standard simulation. After the timed run, the same process performs a standard untimed simulation of the same method and demands exact equality of the saved trajectories and metrics. The two independently measured methods are also compared at tolerance `1e-6`; physical residuals must not exceed `1e-8`. Per-case records include cost, unfairness, service and residuals. Each NPZ contains all output arrays and the 56 slot times. `manifest.json` records deterministic execution order, source hashes and snapshots, software and machine information; `complete.json` is written only when all 42 processes and paired checks finish. No time measurements are filtered.

Memory is obtained from Windows `GetProcessMemoryInfo` and `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize`, in bytes. See [Microsoft's function documentation](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo) and [structure definition](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters). This is the process peak accumulated since process creation through the end of scheduling, including runtime, model, inputs, full-day arrays and accumulated trajectory buffers. It is not auxiliary algorithm memory, and a prior process peak cannot be separated from the scheduling peak. Load, warmup and pre-loop current and peak values are retained for interpretation. Memory reads occur outside the timers; post-loop memory precedes post-processing and validation.

For each scale, the plot reports the mean of three 56-slot mean times, the largest observed individual slot across all three days, and the mean peak working set with a range across the three processes. It shows 500–3000 EVs; 100-EV results remain in the archive. These are observations from three scenarios, not deployment tail-latency estimates or theoretical bounds.

## Completed results

The compact measurements are in `artifacts/pressure21_resources_20261106/results.json`; the complete trajectory archive is not included. It was measured on Windows 11 build 26300, Intel Core i9-14900KF, 68,541,505,536 bytes physical RAM, Python 3.14.6, NumPy 2.5.4 and PyTorch 2.14.1+cpu. Each process used one compute thread. The following table gives the linear implementation, with peak values in MiB (1 MiB = 2^20 bytes). Load baseline ranges refer to the process peak at load completion.

| EVs | Mean slot (s) | Largest slot (s) | Peak mean [min, max] (MiB) | Load peak range (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.00385080 | 0.00797920 | 237.880 [237.328, 238.324] | 233.734–234.137 |
| 500 | 0.01803713 | 0.05142340 | 244.168 [243.895, 244.422] | 233.664–234.477 |
| 1000 | 0.03537284 | 0.06767990 | 252.436 [252.312, 252.641] | 234.129–234.457 |
| 1500 | 0.05363630 | 0.10857050 | 259.844 [259.613, 259.969] | 233.852–234.129 |
| 2000 | 0.07166927 | 0.13537030 | 268.314 [267.957, 268.637] | 233.859–234.203 |
| 2500 | 0.09075313 | 0.16597580 | 275.582 [274.965, 275.891] | 233.785–234.328 |
| 3000 | 0.10580077 | 0.19049400 | 283.047 [282.320, 283.492] | 233.672–234.375 |

Across all 21 paired inputs, the largest state difference was `8.44e-15`; raw and corrected actions, SoC, cost, unfairness, decision records and service rates were identical. Each timed run was exactly identical to the same method without profiling. The maximum physical residual across all 42 runs was `7.28e-12`. All vehicles met the target under the recorded `1e-4` SoC tolerance. The released `results.json` contains reference measurements, pairwise errors and the case-level timing and memory statistics used to plot the figure. Running the benchmark in a fresh directory also produces slot trajectories.

