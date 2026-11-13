"""REINFORCE (policy gradient) on Pokémon Showdown.

The algorithm is in `compute_returns` and `reinforce_update`; the rest of the
file is the training loop, logging and CLI.

Tests (no server needed):
    .venv/bin/python -m pytest tests/test_train_reinforce.py -v

Training on Showdown needs the local server in another terminal:
    node external/pokemon-showdown/pokemon-showdown start --no-security
    .venv/bin/python -m showdown_rl.train_reinforce --iters 30 --battles-per-iter 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import pathlib
import time

import numpy as np
import torch
import torch.optim as optim

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from .agent import PolicyAgent, Transition
from .policy import FlatPolicy


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """Discounted reward-to-go for one episode.

    G_t = Σ_{k≥0} γ^k r_{t+k},  equivalently  G_t = r_t + γ·G_{t+1}  with
    G_T = 0 past the end of the episode.

    Args:
        rewards: [r_0, r_1, …, r_{T-1}] in chronological order.
        gamma: discount factor in [0, 1].

    Returns:
        [G_0, G_1, …, G_{T-1}], same length as `rewards`. Empty in, empty out.

    Sanity checks (these are the unit tests):
        compute_returns([0, 0, 0, 1.0], gamma=1.0) → [1.0, 1.0, 1.0, 1.0]
        compute_returns([0, 0, 0, 1.0], gamma=0.5) → [0.125, 0.25, 0.5, 1.0]
        compute_returns([1, 1, 1],     gamma=0.0)  → [1.0, 1.0, 1.0]
    """
    g_t = 0
    disc_rewards = []
    for r_t in reversed(rewards):
        g_t = r_t + gamma * g_t
        disc_rewards.insert(0, g_t)

    return disc_rewards


def reinforce_update(
    policy: FlatPolicy,
    optimizer: optim.Optimizer,
    episodes: list[list[Transition]],
    gamma: float,
    device: torch.device,
    entropy_coef: float = 0.01,
) -> dict[str, float]:
    """One REINFORCE gradient step over a batch of episodes.

    Args:
        policy: the FlatPolicy being trained (policy.py). Its
            `evaluate_actions(features, species, masks, actions)` returns
            `(log_probs, entropies, values)`, each shape (N,), connected to
            the weights by the autograd graph. `values` is not used by REINFORCE.
        optimizer: Adam over `policy.parameters()`.
        episodes: batch of completed episodes. Each is a list of Transition
            (a dataclass in agent.py: obs, mask, action, log_prob, value,
            reward) in turn order. reward is 0 except on the last step.
        gamma: discount factor.
        device: where tensors live (cpu / mps / cuda).
        entropy_coef: β, weight on the entropy bonus.

    Returns:
        dict with keys "loss", "pg_loss", "entropy" (Python floats) and
        "n_steps" (int, total time steps in the batch).

    Effect: exactly one gradient step on the policy's weights.

    Math, with N = total steps in the batch:
        Â_t     = (G_t − mean_t G_t) / (std_t G_t + ε)
        pg_loss = − (1/N) Σ_t  log π_θ(a_t|s_t) · Â_t
        loss    = pg_loss − β · (1/N) Σ_t H(π_θ(·|s_t))

    Notes:
        - An empty batch returns zeros and leaves the optimizer untouched
          (a mean over zero elements would be NaN).
        - Returns are baselined with the batch mean and divided by the batch
          std: subtracting the mean cuts variance, dividing by the std makes
          the update size independent of the reward scale.
        - log π_θ(a_t|s_t) and the entropies are re-evaluated under the
          current weights, because the log-probs stored during play carry no
          autograd graph.
        - The advantages are data, not a function of θ. `.detach()` is a
          no-op here because the returns are built outside the graph.
        - The entropy term is subtracted: higher entropy is rewarded and the
          loss is minimised.
        - The gradient norm is clipped at 1 to bound the step a single bad
          batch can take.
    """

    if not episodes:
        return {"loss": 0, "pg_loss": 0, "entropy": 0, "n_steps": 0}

    features, species, masks, actions, returns = _stack_transitions(episodes, gamma, device)

    advantages = (returns - returns.mean())/(returns.std() + 1e-8)

    log_probs, entropies, values = policy.evaluate_actions(features, species, masks, actions)

    N = returns.shape[0]
    pg_loss = - 1/N * torch.inner(log_probs, advantages.detach())

    entropy = torch.mean(entropies)
    loss = pg_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss.item(), 
        "pg_loss": pg_loss.item(), 
        "entropy": entropy.item(), 
        "n_steps": N
        }


# ---------------------------------------------------------------------------
# Training loop, logging, CLI
# ---------------------------------------------------------------------------


def _stack_transitions(
    episodes: list[list[Transition]], gamma: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a batch of episodes into stacked tensors + per-step returns.

    Returns:
        (features, species, masks, actions, returns)
        with shapes:
            features: (N, FEATURES_DIM)   float32
            species:  (N, SPECIES_IDS_DIM) int64
            masks:    (N, ACTION_DIM)      bool
            actions:  (N,)                 int64
            returns:  (N,)                 float32
        where N = total transitions across all episodes.
    """
    all_features, all_species, all_masks, all_actions, all_returns = [], [], [], [], []
    for ep in episodes:
        rewards = [t.reward for t in ep]
        returns = compute_returns(rewards, gamma)
        for t, G in zip(ep, returns, strict=True):
            all_features.append(torch.from_numpy(t.obs.features))
            all_species.append(torch.from_numpy(t.obs.species_ids))
            all_masks.append(torch.from_numpy(t.mask))
            all_actions.append(t.action)
            all_returns.append(G)
    features = torch.stack(all_features).to(device)
    species = torch.stack(all_species).to(device)
    masks = torch.stack(all_masks).to(device)
    actions = torch.tensor(all_actions, dtype=torch.long, device=device)
    returns = torch.tensor(all_returns, dtype=torch.float32, device=device)
    return features, species, masks, actions, returns


RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"


def _write_csv(history: list[dict], stem: str) -> pathlib.Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{stem}_curve.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    return path


def _write_plot(history: list[dict], stem: str, battles_per_iter: int) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = [h["iter"] for h in history]
    wr = [h["win_rate"] for h in history]
    ent = [h["entropy"] for h in history]
    k = 10
    wr_smooth = np.convolve(wr, np.ones(k) / k, mode="valid") if len(wr) >= k else []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(its, wr, alpha=0.3, label=f"win rate per iter ({battles_per_iter} battles)")
    if len(wr_smooth):
        ax1.plot(its[k - 1 :], wr_smooth, label=f"{k}-iter mean")
    ax1.axhline(0.5, ls="--", c="gray", label="random play")
    ax1.set_ylabel("win rate vs training opponent")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax2.plot(its, ent)
    ax2.axhline(np.log(13), ls="--", c="gray", label="max entropy log 13")
    ax2.set_ylabel("policy entropy")
    ax2.set_xlabel("iteration")
    ax2.legend()
    fig.suptitle("REINFORCE on gen9randombattle")
    fig.tight_layout()
    path = RESULTS / f"{stem}_curve.png"
    fig.savefig(path, dpi=120)
    return path


OPPONENTS = {
    "random": RandomPlayer,
    "maxbasepower": MaxBasePowerPlayer,
    "simpleheuristics": SimpleHeuristicsPlayer,
}


async def train(
    *,
    n_iterations: int = 30,
    battles_per_iter: int = 8,
    lr: float = 1e-3,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    seed: int = 0,
    device: str = "cpu",
    stem: str,
    opponent_name: str = "random",
    init_path: str,
    save_every_k: int = 200
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    policy = FlatPolicy().to(dev)
    if init_path:
        policy.load_state_dict(torch.load(RESULTS / f"{init_path}", map_location=dev))
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    suffix = str(int(time.time()) % 100000)
    agent = PolicyAgent(
        policy=policy,
        device=dev,
        account_configuration=AccountConfiguration(f"RnfBot{seed}_{suffix}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )
    opponent = OPPONENTS[opponent_name](
        account_configuration=AccountConfiguration(f"RnfOpp{seed}_{suffix}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )

    history = []
    print(f"REINFORCE: {n_iterations} iters × {battles_per_iter} battles, "
          f"lr={lr}, gamma={gamma}, device={device}")
    print(f"{'iter':>4} {'win%':>7} {'loss':>9} {'pg':>9} {'H':>7} {'steps':>6}")

    for it in range(1, n_iterations + 1):
        await agent.battle_against(opponent, n_battles=battles_per_iter)
        episodes = agent.flush_completed_episodes()
        stats = reinforce_update(
            policy, optimizer, episodes, gamma, dev, entropy_coef=entropy_coef
        )
        wr, _n = agent.flush_recent_win_rate()
        history.append({"iter": it, "win_rate": wr, **stats})
        print(f"{it:>4} {wr*100:>6.1f}% {stats['loss']:>+9.4f} {stats['pg_loss']:>+9.4f} "
              f"{stats['entropy']:>7.3f} {stats['n_steps']:>6d}")
        _write_csv(history, stem)  # every iteration, so a crash keeps the curve so far
        if it % save_every_k == 0:
            RESULTS.mkdir(exist_ok=True)
            ckpt = RESULTS / f"{stem}_policy_{it}iters.pt"
            torch.save(policy.state_dict(), ckpt)
            png = _write_plot(history, stem, battles_per_iter)
            print(f"wrote {RESULTS / f'{stem}_curve:{it}iters.csv'}, {png}, {ckpt}")


    RESULTS.mkdir(exist_ok=True)
    ckpt = RESULTS / f"{stem}_policy.pt"
    torch.save(policy.state_dict(), ckpt)
    png = _write_plot(history, stem, battles_per_iter)
    print(f"wrote {RESULTS / f'{stem}_curve.csv'}, {png}, {ckpt}")
    return {"history": history, "policy": policy}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--battles-per-iter", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opponent", default="random", choices=list(OPPONENTS))
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--stem", required=True, help="results/<stem>_curve.{csv,png}, <stem>_policy.pt, should include C or S for curriculum or scratch")
    p.add_argument("--init", default=None, help="exact name of model's .pt file in /results")
    p.add_argument("--save", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(
        train(
            n_iterations=args.iters,
            battles_per_iter=args.battles_per_iter,
            lr=args.lr,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            device=args.device,
            stem=args.stem,
            opponent_name=args.opponent,
            init_path=args.init,
            save_every_k=args.save
        )
    )


if __name__ == "__main__":
    main()
