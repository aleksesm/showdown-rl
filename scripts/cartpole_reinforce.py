"""REINFORCE on CartPole-v1.

Runs `compute_returns` + `reinforce_update` from
`showdown_rl.train_reinforce`, unchanged, on gymnasium's CartPole. The
same `FlatPolicy` class is used with feat_dim=4 and n_actions=2; the
species-embedding slot is fed a constant 0 so it just adds a bias.

No Showdown server needed. A few hundred episodes take seconds.

    .venv/bin/python scripts/cartpole_reinforce.py
    .venv/bin/python scripts/cartpole_reinforce.py --episodes 3000 --batch 10 --lr 1e-3

Stops when the mean return over the last 50 episodes is >= 450 (max is 500).
Writes results/reinforce_cartpole_curve.csv and results/reinforce_cartpole_curve.png.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import gymnasium as gym
import numpy as np
import torch

from showdown_rl.agent import Transition
from showdown_rl.policy import FlatPolicy
from showdown_rl.state import Observation
from showdown_rl.train_reinforce import reinforce_update

OBS_DIM = 4
N_ACTIONS = 2
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def make_policy(hidden: int) -> FlatPolicy:
    return FlatPolicy(
        feat_dim=OBS_DIM,
        n_species_slots=1,
        species_emb_dim=1,
        n_species=1,
        hidden=hidden,
        n_actions=N_ACTIONS,
    )


def run_episode(env: gym.Env, policy: FlatPolicy, seed: int | None) -> list[Transition]:
    obs, _ = env.reset(seed=seed)
    mask = np.ones(N_ACTIONS, dtype=bool)
    species = np.zeros(1, dtype=np.int64)
    episode: list[Transition] = []
    done = False
    while not done:
        o = Observation(features=np.asarray(obs, dtype=np.float32), species_ids=species)
        action, log_prob, value = policy.act(o, mask)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append(
            Transition(
                obs=o,
                mask=mask,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=float(reward),
            )
        )
    return episode


def train(
    *,
    n_episodes: int,
    batch: int,
    lr: float,
    gamma: float,
    entropy_coef: float,
    hidden: int,
    seed: int,
) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    policy = make_policy(hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    device = torch.device("cpu")

    history: list[dict] = []
    ep_idx = 0
    print(
        f"REINFORCE on CartPole-v1: {n_episodes} episodes, batch={batch}, lr={lr}, "
        f"gamma={gamma}, entropy_coef={entropy_coef}"
    )
    print(f"{'episode':>8} {'return':>7} {'avg50':>7} {'loss':>9} {'H':>6}")

    while ep_idx < n_episodes:
        episodes = [run_episode(env, policy, seed=seed + ep_idx + k) for k in range(batch)]
        stats = reinforce_update(
            policy, optimizer, episodes, gamma, device, entropy_coef=entropy_coef
        )
        for ep in episodes:
            ep_idx += 1
            ret = sum(t.reward for t in ep)
            history.append({"episode": ep_idx, "return": ret, **stats})
        avg50 = float(np.mean([h["return"] for h in history[-50:]]))
        last = history[-1]
        print(
            f"{ep_idx:>8} {last['return']:>7.0f} {avg50:>7.1f} {stats['loss']:>+9.4f} {stats['entropy']:>6.3f}"
        )
        if len(history) >= 50 and avg50 >= 450:
            print(f"Cleared: avg50={avg50:.1f} at episode {ep_idx}")
            break

    env.close()
    return history


def save(history: list[dict], stem: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / f"{stem}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [h["episode"] for h in history]
    rets = [h["return"] for h in history]
    avg = np.convolve(rets, np.ones(50) / 50, mode="valid") if len(rets) >= 50 else []
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps, rets, alpha=0.3, label="episode return")
    if len(avg):
        ax.plot(eps[49:], avg, label="50-episode mean")
    ax.axhline(450, ls="--", c="gray", label="cleared threshold")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("REINFORCE on CartPole-v1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}.png", dpi=120)
    print(f"wrote {csv_path} and {RESULTS / f'{stem}.png'}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--batch", type=int, default=10, help="episodes per gradient step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    history = train(
        n_episodes=args.episodes,
        batch=args.batch,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        hidden=args.hidden,
        seed=args.seed,
    )
    save(history, "reinforce_cartpole_curve")


if __name__ == "__main__":
    main()
