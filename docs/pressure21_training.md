# Training and checkpoint selection

The shared `pressure21_v1` state has 21 inputs: two for price and clock, seven summarizing earlier vehicles, six for the current vehicle and six summarizing later vehicles. The state uses the current connected fleet and the raw power already proposed in the current slot. `src/evcharging/state_schema.py` defines the representation, and `src/evcharging/simulation_schema.py` performs sequential inference and fleet-wide SPP. `configs/pressure21_v1.json` specifies the main model paths.

## Imitation policy

The expert split comprises 120 training and 30 validation fleet/price families, each evaluated at three capacities. The 360 training and 90 validation trajectories yield 150,174 and 49,308 decisions. The precomputed arrays and their SHA-256 digests are under `artifacts/runs/pressure21_v1/expert_cache/`. The generator, expert labels and split are described in the [experiment protocol](experiment_protocol.md).

The network has two 128-unit GELU hidden layers and a scalar sigmoid output. Teacher-forced training uses Adam, batch size 512, learning rate 0.001, at most 300 epochs and patience 30. Minimum validation MSE selects the teacher checkpoint. Continuation starts from that checkpoint with fresh Adam at 0.0001. At each of 20 epochs, the policy reconstructs preceding raw power at the saved expert SoCs, with the original expert actions as labels. Minimum autoregressive validation MSE across epochs 0–20 selects the deployed weights.

| Initialization seed | Selected teacher epoch | Selected continuation epoch |
| ---: | ---: | ---: |
| 11002 (main) | 137 | 10 |
| 11003 | 144 | 12 |
| 11004 | 141 | 10 |

## Learned baselines

DDQN uses six discrete actions, a replay capacity of 5,000, a batch size of 64 and Adam at 0.00001. It trains for 1,000 episodes and saves a checkpoint every 100 episodes. The main run uses seed 21002, episode 100. Seeds 21002–21004 contribute to the training-variability analysis.

PPO uses a beta-distribution actor, Adam at **0.00005**, and 200 training episodes per seed. Ten checkpoints per seed are compared on the same 20 validation days. Ranking first minimizes missed departures, then total energy gap when any departures are missed, then mean daily objective. Main-model selection compares the 20 checkpoints from seeds 24003 and 24004. It selects **seed 24003, episode 140**. Variability uses an independent selection for each of seeds 24003, 24004 and 24005. The validation record is `artifacts/runs/ppo_half_formal_20261107/selection.json`. The [learning-rate study](r106_rl_lr_search.md) gives results at the other tested rates.

The main RDDPG comparison uses the final actor from the prespecified first run, seed 33001, after 6,000 episodes. Its training objective, power bounds and validation protocol are described in the [RDDPG adaptation](li2023_r105.md).

## Train the main seeds

These are full training commands. Use fresh run IDs. Each command reads the registered configuration or expert cache and writes checkpoints beneath `artifacts/runs/` with the supplied ID. Training is separate from the short [example](../README.md).

```sh
uv run python scripts/train_il_pressure21.py --variant pressure --seed 11002 --run-id reproduce_il
uv run python scripts/train_ddqn_pressure21.py --config configs/pressure21_v1.json --run-id reproduce_ddqn
uv run python scripts/train_ppo_pressure21.py --config configs/pressure21_v1.json --run-id reproduce_ppo
```

For PPO, the end of the training budget and the selected deployment episode are recorded separately. Use the selection record when evaluating the published model. See [evaluation data and figures](pressure21_transition.md) for the archived cohorts and outputs.

## Reproduce the three runs per RL method

The driver reads the six seed-specific configurations, sets the selected PPO learning rate on each PPO run and writes to a fresh `artifacts/runs/public_rl/` directory. Selection and evaluation use those new checkpoints:

```sh
uv run python scripts/run_pressure21_rl.py --workers 3
uv run python scripts/select_pressure21_rl.py
uv run python scripts/evaluate_pressure21_rl.py --dataset small
```

Use `--dataset formal` for the full 200-EV cohort. To replay the published main checkpoints on a single formal day, use `scripts/compare_day.py` as shown in the [README](../README.md).

## Retained models and evaluation trajectories

The repository includes the three selected IL, DDQN and PPO checkpoints, the three trained RDDPG actors, and the audited expert cache. The retained formal-day records support the comparison example. To regenerate the six main-method formal trajectories, use `uv run --locked python scripts/regenerate_core.py formal`; it restores missing files at their original paths, skips verified files and stops on a digest mismatch. For a new three-run IL evaluation, use `uv run --locked python scripts/evaluate_pressure21.py --dataset formal`. The latter writes into `outputs/pressure21_evaluation/formal/`. For the published three-run RL checkpoints, use `uv run --locked python scripts/evaluate_pressure21_rl.py --dataset formal --selection artifacts/runs/pressure21_v1/published_rl_selection.json`. The RL reproduction workflow above instead creates a selection file from newly trained checkpoints. See [core artifacts](../artifacts/README.md) for inputs and output details.

To restore published trajectories for all three training seeds without retraining, run `uv run --locked python scripts/regenerate_core.py learned --dataset formal`. Change `formal` to `small`, `development`, or `scale` for the runs present in each recorded dataset; `--seed` and `--capacity` restrict the cases. The command verifies historical trajectory digests and supports resumption. See [core artifacts](../artifacts/README.md) for details.
