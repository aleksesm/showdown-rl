"""Double DQN on Pokémon Showdown.

The algorithm is in `QNetwork`, `select_action`, `ReplayBuffer`,
`epsilon_schedule`, `dqn_loss` and `sync_target`; the rest of the file is the
training loop, logging and CLI.

Tests (no server needed):
    .venv/bin/python -m pytest tests/test_train_dqn.py -v

CartPole (no server):
    .venv/bin/python scripts/cartpole_dqn.py

Showdown needs the local server in another terminal:
    node external/pokemon-showdown/pokemon-showdown start --no-security
    .venv/bin/python -m showdown_rl.train_dqn --stem dqn_v1 --iters 200 --battles-per-iter 16
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from .actions import ACTION_DIM
from .agent import PolicyAgent, Transition
from .state import FEATURES_DIM, SPECIES_IDS_DIM, Observation, num_species
from .train_reinforce import OPPONENTS, RESULTS, _write_csv

MASK_FILL = -1e9  # same trick as policy.py: illegal actions can never win a max


@dataclass
class DQNTransition:
    """One (s, a, r, s', done) sample plus the legal-action masks of s and s'.

    All arrays are numpy: features float32 (FEATURES_DIM,), species_ids int64
    (SPECIES_IDS_DIM,), masks bool (ACTION_DIM,). `done` is True iff this was
    the last step of the battle; then `next_*` are placeholders (copies of the
    current state) and must not contribute to the target.
    """

    features: np.ndarray
    species_ids: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_features: np.ndarray
    next_species_ids: np.ndarray
    next_mask: np.ndarray
    done: bool


# A sampled minibatch, as tensors on the training device, keyed by the
# DQNTransition field names:
#   features (B, FEATURES_DIM) float32     species_ids (B, SPECIES_IDS_DIM) int64
#   mask (B, ACTION_DIM) bool              action (B,) int64
#   reward (B,) float32                    done (B,) float32   (0.0 / 1.0)
#   next_features, next_species_ids, next_mask: same shapes as their s-versions
Batch = dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Q_θ(s, ·): one forward pass returns the Q-value of every action.

    Input wiring is the same as `FlatPolicy` (policy.py): species ids go
    through an `nn.Embedding` and are flattened, then concatenated with the
    float features; an MLP trunk follows. Output is a single head of size
    `n_actions` with **no** activation (Q-values are unbounded regressions).

    The sizes are keyword arguments (feat_dim, n_species_slots,
    species_emb_dim, hidden, n_actions, n_species) so that
    `scripts/cartpole_dqn.py` can build a 4-input / 2-action version.
    `n_species=None` means `num_species()` from state.py.

    forward(features (B, feat_dim) float32, species_ids (B, n_species_slots) int64)
        -> q (B, n_actions) float32

    `forward` does not mask; masking is done by the callers (select_action,
    dqn_loss), because the loss needs the raw values of the action taken.
    """

    def __init__(
        self,
        *,
        feat_dim: int = FEATURES_DIM,
        n_species_slots: int = SPECIES_IDS_DIM,
        species_emb_dim: int = 8,
        hidden: int = 256,
        n_actions: int = ACTION_DIM,
        n_species: int | None = None,
    ):
        super().__init__()
        if n_species is None:
            n_species = num_species()
        self.species_emb = nn.Embedding(n_species, species_emb_dim)
        input_dim = species_emb_dim * n_species_slots + feat_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.q_vals = nn.Linear(hidden, n_actions)


    def forward(self, features: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        emb = self.species_emb(species_ids).flatten(start_dim=-2)
        x = torch.cat([features, emb], dim=-1)
        h = self.trunk(x)
        return self.q_vals(h)



def select_action(
    qnet: QNetwork,
    obs: Observation,
    mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: str | torch.device = "cpu",
) -> int:
    """ε-greedy over *legal* actions.

    With probability `epsilon`: a uniformly random index among `mask == True`.
    Otherwise: `argmax_a Q_θ(s, a)` restricted to legal actions.

    Contract:
        - Returns a Python int in [0, ACTION_DIM) with `mask[action]` True,
          always — even when the unmasked argmax is illegal.
        - No gradient is needed here.
        - Randomness comes only from `rng` (so tests can seed it), never from
          `np.random` or `torch` global state.
        - `epsilon == 0.0` is fully greedy; `epsilon == 1.0` fully random.
    """
    var = rng.uniform(0, 1)
    legal_actions = np.flatnonzero(mask)
    if var < epsilon:
        return int(rng.choice(legal_actions))
    features = torch.from_numpy(obs.features).unsqueeze(0).to(device)
    species = torch.from_numpy(obs.species_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        q_vals = qnet(features, species)[0, :].cpu().numpy()
    q_vals_masked = np.where(mask, q_vals, -np.inf)
    return int(np.argmax(q_vals_masked))


class ReplayBuffer:
    """Fixed-capacity ring buffer of `DQNTransition`s with uniform sampling.

    Contract:
        - `push(t)` stores one transition. Once `capacity` is reached, the
          oldest transition is overwritten (index `i % capacity`).
        - `len(buffer)` is the number of stored transitions, ≤ capacity.
        - `sample(batch_size, rng, device)` draws `batch_size` stored
          transitions uniformly at random (with replacement is fine), using
          only `rng`, and returns a `Batch` dict with the keys and shapes
          listed above `Batch`. `dones` are float32 0/1, not bool, so they can
          multiply the bootstrap term. Raises `ValueError` if
          `batch_size > len(self)`.
        - Storage is a Python list of `DQNTransition`.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.transitions = []
        self.index = 0

    def __len__(self) -> int:
        return len(self.transitions)

    def push(self, t: DQNTransition) -> None:
        if len(self) < self.capacity:
            self.transitions.append(t)
        else:
            self.transitions[self.index%self.capacity] = t
        self.index += 1

    def sample(self, batch_size: int, rng: np.random.Generator, device: str | torch.device = "cpu") -> Batch:
        if batch_size > len(self):
            raise ValueError
        FIELDS = {
            "features": np.float32, "species_ids": np.int64, "mask": np.bool,
            "action": np.int64, "reward": np.float32, "done": np.float32,
            "next_features": np.float32, "next_species_ids": np.int64, "next_mask": np.bool,
        }
        indices = rng.integers(0, len(self), batch_size)

        batch = [self.transitions[i] for i in indices]

        out = {
            name: torch.from_numpy(np.stack([getattr(t, name) for t in batch], dtype=dtype)).to(device)
            for name, dtype in FIELDS.items()
        }

        return out


def epsilon_schedule(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Linear decay from `eps_start` at step 0 to `eps_end` at `step >= decay_steps`, constant after.

    `step` counts environment steps (turns), not iterations. Returns a float.
    Edge: `decay_steps == 0` means "always eps_end".
    """
    if decay_steps == 0:
        return eps_end
    decay_rate = (eps_end - eps_start) / decay_steps
    return max(eps_end, eps_start + step * decay_rate)


def dqn_loss(
    qnet: QNetwork, target_net: QNetwork, batch: Batch, gamma: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Double-DQN Huber loss on one minibatch.

    For each sample i:
        a*_i = argmax over LEGAL a' (next_masks) of  Q_θ(s'_i, a')       online selects
        y_i  = r_i + γ · (1 − done_i) · Q_θ⁻(s'_i, a*_i)               target evaluates
        loss = Huber( Q_θ(s_i, a_i) − y_i ), averaged over the batch

    Contract:
        - `y` carries no gradient (it is a constant for autograd).
        - The returned loss is a scalar tensor that `backward()` works on.
        - The dict has Python floats: "loss", "q_mean" (mean of Q_θ(s_i, a_i)),
          "target_mean" (mean of y_i). These are plotted.
        - Illegal next actions are never selected by the argmax, and
          terminal transitions have y_i == r_i exactly.
    """
    features = batch["features"]
    species_ids = batch["species_ids"]
    actions = batch["action"]
    next_features = batch["next_features"]
    next_species_ids = batch["next_species_ids"]
    next_mask = batch["next_mask"]
    with torch.no_grad():
        q_s_pr_a_pr = qnet(next_features, next_species_ids)    # Q_θ(s'_i, a')
        q_s_pr_a_pr = q_s_pr_a_pr.masked_fill(~next_mask, MASK_FILL)
        a_star = torch.argmax(q_s_pr_a_pr, dim=1)    #a*_i
        r = batch["reward"]
        q_s_pr_a_star_target = target_net(next_features, next_species_ids)
        q_s_pr_a_star_target = torch.gather(q_s_pr_a_star_target, dim=1, index=a_star.unsqueeze(1)).squeeze(1)    # Q_θ⁻(s'_i, a*_i)
        y = r + gamma * (1 - batch["done"]) * q_s_pr_a_star_target
    q_s_a = qnet(features, species_ids)
    q_s_a = torch.gather(q_s_a, dim=1, index=actions.unsqueeze(1)).squeeze(1)
    loss = torch.nn.functional.smooth_l1_loss(q_s_a, y)
    loss_scalar = loss.item()
    q_mean = torch.mean(q_s_a).item()
    target_mean = torch.mean(y).item()
    return loss, {"loss": loss_scalar, "q_mean": q_mean, "target_mean": target_mean}



def sync_target(qnet: QNetwork, target_net: QNetwork) -> None:
    """Hard update θ⁻ ← θ. In place; returns nothing."""
    d = qnet.state_dict()
    target_net.load_state_dict(d)



# ---------------------------------------------------------------------------
# Training loop, logging, CLI
# ---------------------------------------------------------------------------


def episode_to_transitions(episode: list[Transition]) -> list[DQNTransition]:
    """Turn one buffered episode (obs_t, mask_t, a_t, r_t) into (s, a, r, s', done) samples.

    s'_t is simply obs_{t+1}; the last step is terminal and reuses its own
    obs as a placeholder for s'.
    """
    out: list[DQNTransition] = []
    for t, tr in enumerate(episode):
        last = t == len(episode) - 1
        nxt = tr if last else episode[t + 1]
        out.append(
            DQNTransition(
                features=tr.obs.features,
                species_ids=tr.obs.species_ids,
                mask=tr.mask,
                action=tr.action,
                reward=float(tr.reward),
                next_features=nxt.obs.features,
                next_species_ids=nxt.obs.species_ids,
                next_mask=nxt.mask,
                done=last,
            )
        )
    return out


class DQNAgent(PolicyAgent):
    """PolicyAgent that acts ε-greedily on a QNetwork. `epsilon` is set by the trainer each iteration."""

    def __init__(self, *args, qnet: QNetwork, rng: np.random.Generator, device: str = "cpu", **kwargs):
        super().__init__(*args, policy=None, device=device, **kwargs)  # type: ignore[arg-type]
        self.qnet = qnet
        self.rng = rng
        self.epsilon = 1.0

    def choose_move(self, battle):
        from .actions import action_mask, to_poke_env
        from .state import encode

        mask = action_mask(battle)
        if not mask.any():
            return self.choose_random_move(battle)
        obs = encode(battle)
        action = select_action(self.qnet, obs, mask, self.epsilon, self.rng, self.device)
        self._current_episode.append(Transition(obs=obs, mask=mask, action=action, log_prob=0.0, value=0.0))
        try:
            return to_poke_env(action, battle)
        except (IndexError, ValueError):
            return self.choose_random_move(battle)


def dqn_update(
    qnet: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    *,
    batch_size: int,
    gamma: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    """One gradient step: sample → `dqn_loss` → backward → clip → step."""
    batch = buffer.sample(batch_size, rng, device)
    loss, stats = dqn_loss(qnet, target_net, batch, gamma)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
    optimizer.step()
    return stats


def _write_plot(history: list[dict], stem: str, battles_per_iter: int) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = [h["iter"] for h in history]
    wr = [h["win_rate"] for h in history]
    k = 10
    wr_smooth = np.convolve(wr, np.ones(k) / k, mode="valid") if len(wr) >= k else []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax1.plot(its, wr, alpha=0.3, label=f"win rate per iter ({battles_per_iter} battles, ε-greedy)")
    if len(wr_smooth):
        ax1.plot(its[k - 1 :], wr_smooth, label=f"{k}-iter mean")
    ax1.axhline(0.5, ls="--", c="gray", label="random play")
    ax1.set_ylabel("win rate vs training opponent")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax2.plot(its, [h["q_mean"] for h in history], label="mean Q(s, a) on batch")
    ax2.plot(its, [h["target_mean"] for h in history], label="mean target y")
    ax2.axhline(1, ls=":", c="gray")
    ax2.axhline(-1, ls=":", c="gray")
    ax2.set_ylabel("Q")
    ax2.legend()
    ax3.plot(its, [h["loss"] for h in history], label="Huber loss")
    ax3.set_ylabel("loss")
    ax3.set_xlabel("iteration")
    ax3b = ax3.twinx()
    ax3b.plot(its, [h["epsilon"] for h in history], c="tab:orange", alpha=0.6, label="ε")
    ax3b.set_ylabel("ε")
    ax3b.set_ylim(0, 1)
    ax3.legend(loc="upper left")
    ax3b.legend(loc="upper right")
    fig.suptitle("DQN on gen9randombattle")
    fig.tight_layout()
    path = RESULTS / f"{stem}_curve.png"
    fig.savefig(path, dpi=120)
    return path


async def train(
    *,
    n_iterations: int,
    battles_per_iter: int,
    lr: float,
    gamma: float,
    seed: int,
    device: str,
    stem: str,
    opponent_name: str,
    init_path: str | None,
    save_every_k: int,
    buffer_size: int,
    batch_size: int,
    eps_start: float,
    eps_end: float,
    eps_decay_steps: int,
    target_sync: int,
    learning_starts: int,
    updates_per_transition: float,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    dev = torch.device(device)
    qnet = QNetwork().to(dev)
    if init_path:
        qnet.load_state_dict(torch.load(RESULTS / init_path, map_location=dev))
    target_net = copy.deepcopy(qnet)
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)

    suffix = str(int(time.time()) % 100000)
    agent = DQNAgent(
        qnet=qnet,
        rng=rng,
        device=device,
        account_configuration=AccountConfiguration(f"DqnBot{seed}_{suffix}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )
    opponent = OPPONENTS[opponent_name](
        account_configuration=AccountConfiguration(f"DqnOpp{seed}_{suffix}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )

    history: list[dict] = []
    env_steps = 0
    grad_steps = 0
    print(f"DQN: {n_iterations} iters × {battles_per_iter} battles, lr={lr}, gamma={gamma}, "
          f"buffer={buffer_size}, batch={batch_size}, target_sync={target_sync}, device={device}")
    print(f"{'iter':>4} {'win%':>7} {'loss':>9} {'Q':>8} {'target':>8} {'eps':>5} {'steps':>6} {'buf':>7}")

    for it in range(1, n_iterations + 1):
        agent.epsilon = epsilon_schedule(env_steps, eps_start, eps_end, eps_decay_steps)
        await agent.battle_against(opponent, n_battles=battles_per_iter)
        episodes = agent.flush_completed_episodes()
        n_new = 0
        for ep in episodes:
            for tr in episode_to_transitions(ep):
                buffer.push(tr)
                n_new += 1
        env_steps += n_new

        stats_acc: dict[str, float] = {"loss": 0.0, "q_mean": 0.0, "target_mean": 0.0}
        n_updates = int(round(n_new * updates_per_transition)) if len(buffer) >= learning_starts else 0
        for _ in range(n_updates):
            s = dqn_update(qnet, target_net, optimizer, buffer,
                           batch_size=batch_size, gamma=gamma, rng=rng, device=dev)
            grad_steps += 1
            for k in stats_acc:
                stats_acc[k] += s[k]
            if grad_steps % target_sync == 0:
                sync_target(qnet, target_net)
        if n_updates:
            for k in stats_acc:
                stats_acc[k] /= n_updates

        wr, _n = agent.flush_recent_win_rate()
        history.append({"iter": it, "win_rate": wr, "epsilon": agent.epsilon, "env_steps": env_steps,
                        "grad_steps": grad_steps, "n_updates": n_updates, **stats_acc})
        print(f"{it:>4} {wr*100:>6.1f}% {stats_acc['loss']:>+9.4f} {stats_acc['q_mean']:>+8.3f} "
              f"{stats_acc['target_mean']:>+8.3f} {agent.epsilon:>5.2f} {n_new:>6d} {len(buffer):>7d}")
        _write_csv(history, stem)
        if it % save_every_k == 0:
            RESULTS.mkdir(exist_ok=True)
            torch.save(qnet.state_dict(), RESULTS / f"{stem}_qnet_{it}iters.pt")
            _write_plot(history, stem, battles_per_iter)

    RESULTS.mkdir(exist_ok=True)
    ckpt = RESULTS / f"{stem}_qnet.pt"
    torch.save(qnet.state_dict(), ckpt)
    png = _write_plot(history, stem, battles_per_iter)
    print(f"wrote {RESULTS / f'{stem}_curve.csv'}, {png}, {ckpt}")
    return {"history": history, "qnet": qnet}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--battles-per-iter", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opponent", default="random", choices=list(OPPONENTS))
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--stem", required=True, help="results/<stem>_curve.{csv,png}, <stem>_qnet.pt")
    p.add_argument("--init", default=None, help="name of a <stem>_qnet.pt in results/ to start from")
    p.add_argument("--save", type=int, default=200, help="checkpoint every K iterations")
    p.add_argument("--buffer", type=int, default=50_000, help="replay capacity (transitions)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=20_000, help="env steps (turns) to reach eps-end")
    p.add_argument("--target-sync", type=int, default=500, help="gradient steps between hard syncs")
    p.add_argument("--learning-starts", type=int, default=1_000, help="transitions in buffer before updates")
    p.add_argument("--updates-per-transition", type=float, default=0.25,
                   help="gradient steps per newly collected transition")
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    asyncio.run(
        train(
            n_iterations=a.iters, battles_per_iter=a.battles_per_iter, lr=a.lr, gamma=a.gamma,
            seed=a.seed, device=a.device, stem=a.stem, opponent_name=a.opponent,
            init_path=a.init, save_every_k=a.save, buffer_size=a.buffer, batch_size=a.batch,
            eps_start=a.eps_start, eps_end=a.eps_end, eps_decay_steps=a.eps_decay_steps,
            target_sync=a.target_sync, learning_starts=a.learning_starts,
            updates_per_transition=a.updates_per_transition,
        )
    )


if __name__ == "__main__":
    main()
