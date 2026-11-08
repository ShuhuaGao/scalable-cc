# Experiment protocol

## Common operating conditions

Each synthetic day has 56 fifteen-minute slots between 07:00 and 21:00. All methods receive the same fleet, price realization, station limit and charging efficiency (0.95) on a given day. The primary comparison evaluates 200 vehicles on 50 days at each of 600 and 800 kW. A separate reference experiment evaluates 10 vehicles on 20 days at 40 kW.

The generator uses NumPy's `default_rng` with one seed per family. Rounded arrival and departure slots have normal distributions with means 8 and 40, standard deviation 4, and clipping intervals [0, 16] and [32, 56]. Arrival and target state of charge (SoC) have distributions N(0.4, 0.1²) and N(0.8, 0.1²), clipped to [0.2, 0.6 − 10⁻⁶] and [0.6 + 10⁻⁶, 1]. Battery capacity is N(30, 5²) kWh clipped to [20, 40], and rated power is N(6, 1²) kW clipped to [4, 8]. Arrival and departure are jointly redrawn until the individual demand fits the parking interval. Each slot's price is a normal draw centered on the nominal hourly tariff with a standard deviation of 10% of that tariff. See `src/evcharging/core.py` for the tariff and generator.

| Use | Scenario seeds |
| --- | --- |
| Expert training and validation | 62000–62119 and 63000–63029, at 4, 3.5 and 3 kW per EV |
| Development and formal evaluation | 64000–64019 and 65000–65049, each at both station limits |
| Small-fleet reference | 76000–76019 |
| DDQN training and validation | 70000–70999 and 73000–73008 |
| PPO training and run validation | 74000–74199 and 75000–75011 for 24003 and 24005. Run 24004 uses 78000–78199 and 79000–79011 |
| Common PPO checkpoint selection | 80000–80019 |
| Scalability | `81000 + 10 * fleet_size + i`, for `i = 0, 1, 2`, at 4 kW per EV |

## Methods and scoring

Ours (stored as `IL`), DDQN and PPO use the same 21-component causal state, sequential processing order and SPP procedure. UC requests full power when connected. UPA allocates power in proportion to charging urgency subject to individual bounds. Both rule-based policies use SPP. RDDPG uses the charging adaptation's dynamic energy boundary and sequential power limits. The [RDDPG protocol](li2023_r105.md) describes its nine features and distinct training reward.

Daily grid cost is the sum of delivered grid energy multiplied by slot prices. Unfairness sums the squared deviations of normalized charging actions from the allocation implied by each vehicle's share of fleet urgency and the total normalized action in that slot. Both directions of deviation contribute. The evaluation objective is `0.7 * cost + 0.3 * unfairness`. A vehicle meets its departure target within a SoC tolerance of 0.0001. Whole-day attainment counts days on which every vehicle meets its target. The shortfall columns in the main table average over vehicles missing their targets. Unit cost uses purchased grid energy.

## Perfect-information expert (IDEAL)

At every slot the expert solves a 40-slot nonlinear problem using the realized prices and known arrivals in the window, then executes the first action. The last windows use 39 additional prices drawn from the final nominal tariff with the same relative noise and a separate random stream seeded by `SeedSequence([scenario_seed, 104010])`. No extra vehicles enter after the operating day. The online learned policies receive current information only.

The first window starts from zero power. Subsequent windows shift the previous solution one slot forward and initialize new actions to zero. Knitro 16 runs four local multistarts, one thread, and a window budget of 30 seconds per 10 daily vehicles. The iteration limit is 200,000, the feasibility tolerances are 10⁻⁸, and relative and absolute optimality tolerances are 0.05 and 10. The expert checks physical and terminal-reference residuals against 10⁻⁴. All 450 training and validation trajectories passed the departure audit. Four windows returned nonoptimal statuses with checked residuals at most 4.81 × 10⁻⁹. IDEAL is a perfect-information reference with local solves.

The following is a **solver run** for a single expert training case. It requires the licensed Knitro runtime. Use a fresh output location.

```sh
uv run python scripts/expert_ideal10h.py --seed 62000 --vehicles 10 --capacity-per-ev 4 --starts 4 --seconds 30 --output outputs/expert_62000
```

## Study protocols

| Question | Protocol and numerical evidence |
| --- | --- |
| How were the policies selected? | [Training](pressure21_training.md), [RL learning rates](r106_rl_lr_search.md) |
| Which inputs and figures support the comparisons? | [Evaluation and figures](pressure21_transition.md), [uncertainty](uncertainty_analysis.md) |
| How do features and processing order affect performance? | [State groups](state_ablation.md), [distributional features](r102_distributional_reproduction.md), [processing order](order_sensitivity.md) |
| How do scenarios and controller settings affect outcomes? | [Operating shifts](r23_shift_protocol.md), [shift results](r23_shift_results.md), [SPP diagnostics](r26_distribution_spp.md), [efficiency](r44_efficiency_protocol.md), [correlated arrivals](r107_correlated_arrivals.md), [objective weight](r77_beta_protocol.md) |
| How do training budget, timing and memory scale? | [Width and family count](r106_sensitivity.md), [incremental scheduling](r22_incremental_validation.md), [resource benchmark](r22_resource_benchmark.md), [baseline timing](r24_baseline_timing.md) |

## Data availability and regeneration

The 450 expert runs, their audited cache, scenario inputs and selected checkpoints are included. Large daily trajectory archives can be regenerated as described in [core artifacts](../artifacts/README.md); the full expert optimization requires Knitro 16. The retained expert archive avoids rerunning the optimizer for training and feature-cache studies.
