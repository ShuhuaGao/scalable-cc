"""Train and independently validate a DDQN policy on three pressure levels."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evcharging.core import generate
from evcharging.models import network
from evcharging.simulation_schema import simulate


def evaluate(model: torch.nn.Module, config: dict, rates: list[float], tolerance: float) -> dict:
    """Evaluate deterministic actions on independent validation scenarios.

    :param model: Candidate action-value network.
    :param config: DDQN configuration.
    :param rates: Capacity per vehicle.
    :param tolerance: Target SoC tolerance.
    :return: Mean reward, target outcomes, and weighted objective.
    """
    outcomes = []
    for i in range(config["validation_count"]):
        s = generate(config["validation_seed_start"] + i, config["vehicles"], fixed_day=True)
        limit = rates[i % len(rates)] * config["vehicles"]

        def policy(x: np.ndarray, k: int, urgency: np.ndarray) -> float:
            """Select the highest-valued action.

            :param x: State features.
            :param k: Vehicle index.
            :param urgency: Vehicle urgency vector.
            :return: Normalized action.
            """
            del k, urgency
            with torch.no_grad():
                return int(model(torch.tensor(x, dtype=torch.float32)[None, :]).argmax()) / (config["actions"] - 1)

        trajectory = simulate(s, limit, policy, state_schema=config["state_schema"])
        gap = np.maximum(s.target - trajectory["soc"][s.td, np.arange(len(s.ta))], 0)
        objective = (1 - s.beta) * trajectory["cost"].sum() + s.beta * trajectory["unfair"].sum()
        outcomes.append({"seed": config["validation_seed_start"] + i, "limit": limit, "failed": int((gap > tolerance).sum()), "gap": float(gap.sum()), "objective": float(objective)})
    return {"mean_reward": -float(np.mean([x["objective"] for x in outcomes])), "failed": sum(x["failed"] for x in outcomes), "gap": sum(x["gap"] for x in outcomes), "objective": float(np.mean([x["objective"] for x in outcomes])), "scenarios": outcomes}


def main() -> None:
    """Train from scratch or restore optimizer, replay buffer, and RNG state.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pressure21_v1.json")
    parser.add_argument("--run-id", default="pressure21_v1/unspecified")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    all_config = json.loads((root / args.config).read_text(encoding="utf-8-sig"))
    cfg = all_config["ddqn"]
    rates = all_config["expert"]["capacity_per_ev_kw"]
    dest = root / "artifacts" / "runs" / args.run_id / "ddqn_model"
    dest.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = network(cfg["actions"])
    target = network(cfg["actions"])
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    replay = []
    history = []
    start = 0
    budget = cfg["episodes"]
    if args.resume:
        checkpoint = torch.load(dest / "latest.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"] != cfg or checkpoint["rates"] != rates:
            raise ValueError("Training configuration changed")
        model.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint["target"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        replay = checkpoint["replay"]
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        start = checkpoint["episode"]
        history = checkpoint["history"]
        budget = checkpoint["budget"]
    for episode in range(start, cfg["max_episodes"]):
        if episode >= budget:
            break
        s = generate(cfg["train_seed_start"] + episode, cfg["vehicles"], fixed_day=True)
        limit = rates[episode % len(rates)] * cfg["vehicles"]
        actions = []

        def policy(x: np.ndarray, k: int, urgency: np.ndarray) -> float:
            """Sample an epsilon-greedy action.

            :param x: State features.
            :param k: Vehicle index.
            :param urgency: Vehicle urgency vector.
            :return: Normalized action.
            """
            del k, urgency
            epsilon = max(cfg["epsilon_min"], cfg["epsilon_decay"] ** episode)
            with torch.no_grad():
                action = int(rng.integers(cfg["actions"])) if rng.random() < epsilon else int(model(torch.tensor(x, dtype=torch.float32)[None, :]).argmax())
            actions.append(action)
            return action / (cfg["actions"] - 1)

        trajectory = simulate(s, limit, policy, state_schema=cfg["state_schema"])
        x = trajectory["states"].astype(np.float32)
        records = trajectory["records"]
        reward = -((1 - s.beta) * trajectory["cost"][records[:, 0], records[:, 1]] + s.beta * trajectory["unfair"][records[:, 0], records[:, 1]])
        next_x = np.concatenate([x[1:], np.zeros_like(x[:1])])
        done = np.zeros(len(x), dtype=np.float32)
        done[-1] = 1
        replay.extend(zip(x, actions, reward, next_x, done))
        replay = replay[-cfg["replay_capacity"]:]
        for _ in range(cfg["gradient_updates_per_episode"]):
            batch = [replay[i] for i in rng.integers(0, len(replay), cfg["batch_size"])]
            bx = torch.tensor(np.array([z[0] for z in batch]), dtype=torch.float32)
            ba = torch.tensor([z[1] for z in batch])
            br = torch.tensor([z[2] for z in batch], dtype=torch.float32)
            bn = torch.tensor(np.array([z[3] for z in batch]), dtype=torch.float32)
            bd = torch.tensor([z[4] for z in batch], dtype=torch.float32)
            with torch.no_grad():
                expected = br + cfg["gamma"] * (1 - bd) * target(bn).gather(1, model(bn).argmax(dim=1, keepdim=True)).squeeze(1)
            loss = ((model(bx).gather(1, ba[:, None]).squeeze(1) - expected) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (episode + 1) % cfg["target_update_episodes"] == 0:
            target.load_state_dict(model.state_dict())
        if (episode + 1) % cfg["checkpoint_interval"] == 0:
            metrics = evaluate(model, cfg, rates, all_config["evaluation"]["soc_tolerance"])
            metrics["episode"] = episode + 1
            history.append(metrics)
            if episode + 1 == cfg["episodes"] and len(history) >= 3:
                rewards = [item["mean_reward"] for item in history[-3:]]
                if all(rewards[i+1] - rewards[i] >= cfg["extension_improvement"] * abs(rewards[i]) for i in range(2)):
                    budget = cfg["max_episodes"]
            checkpoint = {"episode": episode + 1, "model": model.state_dict(), "target": target.state_dict(), "optimizer": optimizer.state_dict(), "replay": replay, "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "history": history, "budget": budget, "config": cfg, "rates": rates}
            torch.save(checkpoint, dest / f"episode_{episode+1}.pt")
            torch.save(checkpoint, dest / "latest.pt")
            best = min(history, key=lambda x: (x["failed"], x["gap"], x["objective"]))
            (dest / "validation.json").write_text(json.dumps(history, indent=2), encoding="utf8")
            (dest / "selected.json").write_text(json.dumps(best, indent=2), encoding="utf8")
            print("episode", episode + 1, "validation", metrics["failed"], metrics["objective"], flush=True)


if __name__ == "__main__":
    main()
