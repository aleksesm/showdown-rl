"""DQN on CartPole-v1.

Runs `QNetwork`, `select_action`, `ReplayBuffer`, `epsilon_schedule`,
`dqn_loss`, `sync_target` from `showdown_rl.train_dqn`, unchanged, on
gymnasium's CartPole. Same trick as scripts/cartpole_reinforce.py: a 4-input,
2-action network with the species-embedding slot fed a constant 0.

No Showdown server needed.

    .venv/bin/python scripts/cartpole_dqn.py
    .venv/bin/python scripts/cartpole_dqn.py --episodes 1000 --lr 5e-4 --target-sync 1000 --batch 128

Stops when the mean return over the last 50 episodes is >= 450 (max is 500).
Writes results/dqn_cartpole_curve.csv and results/dqn_cartpole_curve.png.
Compare the episode count with results/reinforce_cartpole_lr1e-3.csv.
"""

from __future__ import annotations

import argparse
import copy
import csv
import pathlib

import gymnasium as gym
import numpy as np
import torch

from showdown_rl.state import Observation
from showdown_rl.train_dqn import (
    DQNTransition,
    QNetwork,
    ReplayBuffer,
    dqn_update,
    epsilon_schedule,
    select_action,
    sync_target,
)

OBS_DIM = 4
N_ACTIONS = 2
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"


def make_qnet(hidden: int) -> QNetwork:
    return QNetwork(feat_dim=OBS_DIM, n_species_slots=1, species_emb_dim=1, n_species=1,
                    hidden=hidden, n_actions=N_ACTIONS)


def train(*, n_episodes: int, lr: float, gamma: float, hidden: int, seed: int, buffer_size: int,
          batch_size: int, eps_start: float, eps_end: float, eps_decay_steps: int, target_sync: int,
          learning_starts: int) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    qnet = make_qnet(hidden)
    target = copy.deepcopy(qnet)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)
    device = torch.device("cpu")
    mask = np.ones(N_ACTIONS, dtype=bool)
    species = np.zeros(1, dtype=np.int64)

    history: list[dict] = []
    env_steps = 0
    grad_steps = 0
    print(f"DQN on CartPole-v1: {n_episodes} episodes, lr={lr}, gamma={gamma}, buffer={buffer_size}, "
          f"batch={batch_size}, target_sync={target_sync}")
    print(f"{'episode':>8} {'return':>7} {'avg50':>7} {'loss':>9} {'Q':>8} {'eps':>5}")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ret = 0.0
        stats_acc = {"loss": 0.0, "q_mean": 0.0, "target_mean": 0.0}
        n_upd = 0
        while not done:
            eps = epsilon_schedule(env_steps, eps_start, eps_end, eps_decay_steps)
            o = Observation(features=np.asarray(obs, dtype=np.float32), species_ids=species)
            a = select_action(qnet, o, mask, eps, rng, device)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            # gymnasium's time-limit truncation is not a real terminal state: bootstrap through it
            buffer.push(DQNTransition(
                features=o.features, species_ids=species, mask=mask, action=a, reward=float(r),
                next_features=np.asarray(obs2, dtype=np.float32), next_species_ids=species,
                next_mask=mask, done=bool(terminated),
            ))
            obs = obs2
            ret += float(r)
            env_steps += 1
            if len(buffer) >= learning_starts:
                s = dqn_update(qnet, target, optimizer, buffer, batch_size=batch_size, gamma=gamma,
                               rng=rng, device=device)
                grad_steps += 1
                n_upd += 1
                for k in stats_acc:
                    stats_acc[k] += s[k]
                if grad_steps % target_sync == 0:
                    sync_target(qnet, target)
        if n_upd:
            for k in stats_acc:
                stats_acc[k] /= n_upd
        history.append({"episode": ep, "return": ret, "epsilon": eps, "env_steps": env_steps, **stats_acc})
        avg50 = float(np.mean([h["return"] for h in history[-50:]]))
        print(f"{ep:>8} {ret:>7.0f} {avg50:>7.1f} {stats_acc['loss']:>+9.4f} {stats_acc['q_mean']:>+8.2f} {eps:>5.2f}")
        if len(history) >= 50 and avg50 >= 450:
            print(f"Cleared: avg50={avg50:.1f} at episode {ep} ({env_steps} env steps)")
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

    eps_ = [h["episode"] for h in history]
    rets = [h["return"] for h in history]
    avg = np.convolve(rets, np.ones(50) / 50, mode="valid") if len(rets) >= 50 else []
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax.plot(eps_, rets, alpha=0.3, label="episode return")
    if len(avg):
        ax.plot(eps_[49:], avg, label="50-episode mean")
    ax.axhline(450, ls="--", c="gray", label="cleared threshold")
    ax.set_ylabel("return")
    ax.set_title("DQN on CartPole-v1")
    ax.legend()
    ax2.plot(eps_, [h["q_mean"] for h in history], label="mean Q(s,a) on batch")
    ax2.plot(eps_, [h["epsilon"] * 100 for h in history], c="tab:orange", alpha=0.6, label="ε × 100")
    ax2.set_xlabel("episode")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}.png", dpi=120)
    print(f"wrote {csv_path} and {RESULTS / f'{stem}.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=5_000)
    p.add_argument("--target-sync", type=int, default=500, help="gradient steps between hard syncs")
    p.add_argument("--learning-starts", type=int, default=500)
    p.add_argument("--stem", default="dqn_cartpole_curve")
    a = p.parse_args()
    history = train(n_episodes=a.episodes, lr=a.lr, gamma=a.gamma, hidden=a.hidden, seed=a.seed,
                    buffer_size=a.buffer, batch_size=a.batch, eps_start=a.eps_start, eps_end=a.eps_end,
                    eps_decay_steps=a.eps_decay_steps, target_sync=a.target_sync,
                    learning_starts=a.learning_starts)
    save(history, a.stem)


if __name__ == "__main__":
    main()
