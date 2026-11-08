# Scalable coordinated charging of large EV fleets

This repository contains the companion code, pretrained policies, and evaluation data for the manuscript *Scalable Coordinated Charging of Large EV Fleets via Iterative Context-Aware Imitation Learning*.

A single shared policy proposes charging actions for connected vehicles in sequence. A fleet-wide safety post-processing (SPP) step enforces charging and station limits. The 21-feature policy uses the current slot price and connected vehicles' states. Its scheduler takes linear time in the number of connected vehicles per slot.

## Installation

Install uv to manage Python and the project dependencies. On Windows, run `winget install --id=astral-sh.uv -e`. On macOS or Linux, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Open a terminal in the repository root (the directory containing `uv.lock`). The following commands install Python 3.14 and create a local virtual environment using the locked dependency versions.

```sh
uv python install 3.14
uv sync --locked
```

### Knitro for expert optimization

The examples below use the supplied checkpoints and run without a solver license. To generate new expert trajectories or run the IDEAL optimizer, install Artelys Knitro 16 and configure a valid license. Artelys offers a [one-month full trial license](https://www.artelys.com/solvers/knitro/programs/).

Set `ARTELYS_LICENSE` to the path of your license file and check the installation from the repository root. The [solver guide](PYSOLVER/README.md) gives the expert-generation entry point.

```sh
uv run --locked python -m PYSOLVER.check_knitro
```

## Run one published example

```sh
uv run --locked python scripts/example_day.py
```

The example runs the frozen imitation policy on 10 vehicles at 40 kW over 56 slots. It checks the checkpoint digest, daily cost, unfairness and departure service against the archived record. The expected output is:

```text
Day 76000 | 10 EVs | 40 kW | 56 slots
Cost 74.222497 | Unfairness 0.777534 | Targets met 10/10
```

To replay the six online methods on the same 200-EV test day, run:

```sh
uv run --locked python scripts/compare_day.py --seed 65000 --capacity 600
```

This command checks the common input digest and compares cost, unfairness and departure counts against all six archived results. Choose seeds 65000–65049 and station capacities of 600 or 800 kW.

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/evcharging/` | Scenario generation, charging dynamics, policy networks, and safety post-processing. |
| `PYSOLVER/` | Knitro interface for generating expert charging trajectories. |
| `configs/` | Settings for training, checkpoint selection, and evaluation. |
| `scripts/` | Executable examples and scripts for training, evaluation, and plotting. |
| `artifacts/` | Selected scenario inputs, expert data, trained checkpoints, and reference records for the examples. |
| `results/` | Numerical summaries for the manuscript tables, figures, and supplementary studies. |
| `docs/` | Experiment-specific protocols, reproduction commands, and descriptions of the archived data. |

## Experiment guides

Start with the shared protocol in `docs/`, then follow the guide for the experiment you want to reproduce.

The examples use the data included in the checkout. Full trajectory archives and intermediate training outputs are generated locally. The [data guide](artifacts/README.md) identifies the inputs and commands for regenerating them. The [results guide](results/README.md) maps the reported tables and figures to their numerical summaries.

- [Experiment protocol](docs/experiment_protocol.md) — shared scenarios, solver settings, comparison methods and metrics.
- [Training and checkpoint selection](docs/pressure21_training.md) — imitation learning, DDQN and PPO. The selected PPO uses learning rate 0.00005 and episode 140.
- [Evaluation data and figures](docs/pressure21_transition.md) — main table, small-fleet examples, scalability and figure generation.
- [RDDPG charging adaptation](docs/li2023_r105.md) — dynamic power bounds, reward, training and evaluation.
- [Training and scenario uncertainty](docs/uncertainty_analysis.md), [network and dataset sensitivity](docs/r106_sensitivity.md), and [runtime and memory](docs/r22_resource_benchmark.md).
