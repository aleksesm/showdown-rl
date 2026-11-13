"""poke-env Player subclass that drives a torch policy.

Used during both training and evaluation. The agent's job per turn:
  1. Encode the battle into an `Observation`.
  2. Compute the action mask.
  3. Sample an action from the policy under the mask.
  4. Buffer the transition (obs, mask, action) for the trainer to consume.
  5. Convert the action to a poke-env `BattleOrder`.

Rewards are filled in by `_battle_finished_callback` at the end of each
battle. v1 uses sparse terminal rewards only (+1 win / −1 loss). Dense
reward shaping is the subject of Phase 5.

The training loop calls `flush_completed_episodes()` after each batch
of battles to consume buffered episodes; it then re-evaluates actions
through the policy *with* gradients enabled to compute the PG loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .actions import action_mask, to_poke_env
from .state import Observation, encode

if TYPE_CHECKING:
    from poke_env.player import Player as _PlayerBase  # for type hints only
    from .policy import FlatPolicy

from poke_env.player import Player  # noqa: E402


@dataclass
class Transition:
    obs: Observation
    mask: np.ndarray   # bool, shape (ACTION_DIM,)
    action: int        # in [0, ACTION_DIM)
    log_prob: float    # at time of sampling — for diagnostics only
    value: float       # estimated state-value at time of sampling
    reward: float = 0.0  # filled at battle end (terminal only in v1)


class PolicyAgent(Player):
    """Player that acts according to a torch policy and buffers transitions.

    `device` may be "cpu" / "mps" / "cuda". For poke-env (single battle at
    a time, MLP forward pass is tiny), CPU is usually fine; MPS shines for
    larger architectures in Phase 4.
    """

    def __init__(self, *args, policy: "FlatPolicy", device: str = "cpu", deterministic: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.device = device
        self.deterministic = deterministic
        self._current_episode: list[Transition] = []
        self._completed_episodes: list[list[Transition]] = []
        # Per-iteration win counters (caller flushes via recent_win_rate)
        self._win_count = 0
        self._loss_count = 0

    # -- Player override ---------------------------------------------------

    def choose_move(self, battle):
        mask = action_mask(battle)
        if not mask.any():
            # Should not happen — every state has at least one legal action —
            # but guard against poke-env quirks.
            return self.choose_random_move(battle)

        obs = encode(battle)
        action, log_prob, value = self.policy.act(
            obs, mask, device=self.device, deterministic=self.deterministic
        )

        self._current_episode.append(
            Transition(obs=obs, mask=mask, action=action, log_prob=log_prob, value=value)
        )

        try:
            return to_poke_env(action, battle)
        except (IndexError, ValueError):
            # The mask should prevent this, but if a quirk lets it through,
            # fall back to a random legal move rather than crashing training.
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle) -> None:
        # Terminal reward
        if getattr(battle, "won", False):
            self._win_count += 1
            r = 1.0
        elif getattr(battle, "lost", False):
            self._loss_count += 1
            r = -1.0
        else:
            # Tie or unfinished — give 0
            r = 0.0

        if self._current_episode:
            self._current_episode[-1].reward = r
            self._completed_episodes.append(self._current_episode)
        self._current_episode = []

    # -- Trainer-side API --------------------------------------------------

    def flush_completed_episodes(self) -> list[list[Transition]]:
        eps = self._completed_episodes
        self._completed_episodes = []
        return eps

    def flush_recent_win_rate(self) -> tuple[float, int]:
        """Return (win_rate, n_battles_this_window) and reset the counters."""
        n = self._win_count + self._loss_count
        rate = (self._win_count / n) if n else 0.0
        self._win_count = 0
        self._loss_count = 0
        return rate, n
