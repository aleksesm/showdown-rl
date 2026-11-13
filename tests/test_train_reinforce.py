"""Unit tests for the REINFORCE functions in src/showdown_rl/train_reinforce.py.

    .venv/bin/python -m pytest tests/test_train_reinforce.py -v

`compute_returns` is fully testable from unit tests.

`reinforce_update` is also tested here — the test runs it on a tiny
fabricated batch and checks it returns a dict of the expected shape,
that the loss is finite, and that the policy's parameters actually
changed after the step. It does NOT test that the algorithm is
*correct* — that requires watching win-rate over many iterations
against a real opponent. The right place for that check is running
the training script.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from showdown_rl.actions import ACTION_DIM
from showdown_rl.agent import Transition
from showdown_rl.policy import FlatPolicy
from showdown_rl.state import FEATURES_DIM, SPECIES_IDS_DIM, Observation


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------


def test_compute_returns_terminal_reward_no_discount():
    from showdown_rl.train_reinforce import compute_returns
    assert compute_returns([0.0, 0.0, 0.0, 1.0], gamma=1.0) == [1.0, 1.0, 1.0, 1.0]


def test_compute_returns_terminal_reward_with_discount():
    from showdown_rl.train_reinforce import compute_returns
    out = compute_returns([0.0, 0.0, 0.0, 1.0], gamma=0.5)
    # G_3 = 1.0; G_2 = 0.5; G_1 = 0.25; G_0 = 0.125
    assert out == pytest.approx([0.125, 0.25, 0.5, 1.0])


def test_compute_returns_empty_rewards():
    from showdown_rl.train_reinforce import compute_returns
    assert compute_returns([], gamma=0.99) == []


def test_compute_returns_zero_gamma_means_no_future():
    """γ = 0: the agent is myopic — G_t = r_t exactly, no propagation."""
    from showdown_rl.train_reinforce import compute_returns
    assert compute_returns([1.0, 2.0, 3.0], gamma=0.0) == [1.0, 2.0, 3.0]


def test_compute_returns_mixed_signs():
    from showdown_rl.train_reinforce import compute_returns
    # Two reward signals: +1 at step 1, -1 at step 3, γ=0.9
    # G_3 = -1.0
    # G_2 = 0 + 0.9·(-1.0) = -0.9
    # G_1 = 1.0 + 0.9·(-0.9) = 1.0 - 0.81 = 0.19
    # G_0 = 0 + 0.9·0.19 = 0.171
    out = compute_returns([0.0, 1.0, 0.0, -1.0], gamma=0.9)
    assert out == pytest.approx([0.171, 0.19, -0.9, -1.0], abs=1e-6)


# ---------------------------------------------------------------------------
# reinforce_update — shape + smoke
# ---------------------------------------------------------------------------


def _make_fake_episode(length: int = 5, terminal_reward: float = 1.0) -> list[Transition]:
    """One fake episode with random obs and a terminal reward at the end."""
    eps: list[Transition] = []
    rng = np.random.default_rng(0)
    for k in range(length):
        feats = rng.standard_normal(FEATURES_DIM).astype(np.float32)
        species = rng.integers(0, 100, size=SPECIES_IDS_DIM).astype(np.int64)
        mask = np.ones(ACTION_DIM, dtype=bool)
        action = int(rng.integers(0, ACTION_DIM))
        r = terminal_reward if k == length - 1 else 0.0
        eps.append(
            Transition(
                obs=Observation(features=feats, species_ids=species),
                mask=mask,
                action=action,
                log_prob=0.0,
                value=0.0,
                reward=r,
            )
        )
    return eps


def test_reinforce_update_returns_correct_dict_shape():
    from showdown_rl.train_reinforce import reinforce_update

    policy = FlatPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    episodes = [_make_fake_episode(length=4, terminal_reward=1.0)]

    stats = reinforce_update(policy, optimizer, episodes, gamma=0.99, device=torch.device("cpu"))
    assert isinstance(stats, dict)
    for k in ("loss", "pg_loss", "entropy", "n_steps"):
        assert k in stats, f"missing key: {k}"
    assert isinstance(stats["loss"], float)
    assert isinstance(stats["n_steps"], int)
    assert stats["n_steps"] == 4


def test_reinforce_update_handles_empty_episodes():
    from showdown_rl.train_reinforce import reinforce_update

    policy = FlatPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    stats = reinforce_update(policy, optimizer, [], gamma=0.99, device=torch.device("cpu"))
    assert stats["n_steps"] == 0
    assert stats["loss"] == 0.0


def test_reinforce_update_actually_modifies_policy_weights():
    """Smoke test: after one update, at least one weight tensor should have changed."""
    from showdown_rl.train_reinforce import reinforce_update

    torch.manual_seed(0)
    policy = FlatPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    before = [p.detach().clone() for p in policy.parameters()]
    episodes = [_make_fake_episode(length=8, terminal_reward=1.0) for _ in range(4)]
    stats = reinforce_update(policy, optimizer, episodes, gamma=0.99, device=torch.device("cpu"))
    after = [p.detach().clone() for p in policy.parameters()]

    n_changed = sum(1 for a, b in zip(before, after, strict=True) if not torch.equal(a, b))
    assert n_changed > 0, "no policy parameter was updated by reinforce_update"
    assert np.isfinite(stats["loss"]), f"loss is not finite: {stats['loss']}"
