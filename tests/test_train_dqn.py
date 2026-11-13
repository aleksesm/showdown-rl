"""Unit tests for the DQN functions in src/showdown_rl/train_dqn.py.

    .venv/bin/python -m pytest tests/test_train_dqn.py -v

The tests check contracts (shapes, masking,
terminal handling, the ring buffer, the schedule, that a step changes the
weights). They do NOT check that the Double-DQN target is the *right*
target; that is what the CartPole run is for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from showdown_rl.actions import ACTION_DIM
from showdown_rl.agent import Transition
from showdown_rl.state import FEATURES_DIM, SPECIES_IDS_DIM, Observation
from showdown_rl.train_dqn import DQNTransition, episode_to_transitions

N_SPECIES = 50  # small embedding table so tests are fast; ids below are < 50


def _obs(rng: np.random.Generator) -> Observation:
    return Observation(
        features=rng.standard_normal(FEATURES_DIM).astype(np.float32),
        species_ids=rng.integers(0, N_SPECIES, size=SPECIES_IDS_DIM).astype(np.int64),
    )


def _qnet(seed: int = 0):
    from showdown_rl.train_dqn import QNetwork

    torch.manual_seed(seed)
    return QNetwork(n_species=N_SPECIES, hidden=32)


def _transition(rng: np.random.Generator, *, reward: float = 0.0, done: bool = False,
                mask: np.ndarray | None = None, next_mask: np.ndarray | None = None) -> DQNTransition:
    o, o2 = _obs(rng), _obs(rng)
    m = np.ones(ACTION_DIM, dtype=bool) if mask is None else mask
    m2 = np.ones(ACTION_DIM, dtype=bool) if next_mask is None else next_mask
    return DQNTransition(
        features=o.features, species_ids=o.species_ids, mask=m,
        action=int(rng.integers(0, ACTION_DIM)), reward=reward,
        next_features=o2.features, next_species_ids=o2.species_ids, next_mask=m2, done=done,
    )


def _filled_buffer(n: int, capacity: int = 1000, seed: int = 0, **kw):
    from showdown_rl.train_dqn import ReplayBuffer

    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(capacity)
    for _ in range(n):
        buf.push(_transition(rng, **kw))
    return buf, rng


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------


def test_qnetwork_output_shape():
    q = _qnet()
    rng = np.random.default_rng(0)
    obs = [_obs(rng) for _ in range(5)]
    feats = torch.stack([torch.from_numpy(o.features) for o in obs])
    species = torch.stack([torch.from_numpy(o.species_ids) for o in obs])
    out = q(feats, species)
    assert out.shape == (5, ACTION_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_qnetwork_cartpole_sized():
    from showdown_rl.train_dqn import QNetwork

    q = QNetwork(feat_dim=4, n_species_slots=1, species_emb_dim=1, n_species=1, hidden=16, n_actions=2)
    out = q(torch.zeros(3, 4), torch.zeros(3, 1, dtype=torch.long))
    assert out.shape == (3, 2)


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------


def test_select_action_greedy_respects_mask():
    from showdown_rl.train_dqn import select_action

    q = _qnet()
    rng = np.random.default_rng(1)
    for _ in range(20):
        obs = _obs(rng)
        mask = np.zeros(ACTION_DIM, dtype=bool)
        legal = rng.choice(ACTION_DIM, size=3, replace=False)
        mask[legal] = True
        a = select_action(q, obs, mask, epsilon=0.0, rng=rng)
        assert isinstance(a, int)
        assert mask[a], "greedy action must be legal"


def test_select_action_greedy_is_argmax_over_legal():
    from showdown_rl.train_dqn import select_action

    q = _qnet()
    rng = np.random.default_rng(2)
    obs = _obs(rng)
    mask = np.ones(ACTION_DIM, dtype=bool)
    mask[[0, 1, 2]] = False
    with torch.no_grad():
        qs = q(torch.from_numpy(obs.features)[None], torch.from_numpy(obs.species_ids)[None])[0].numpy()
    expected = int(np.argmax(np.where(mask, qs, -np.inf)))
    assert select_action(q, obs, mask, epsilon=0.0, rng=rng) == expected


def test_select_action_random_covers_all_legal_and_only_legal():
    from showdown_rl.train_dqn import select_action

    q = _qnet()
    rng = np.random.default_rng(3)
    obs = _obs(rng)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[[4, 7, 12]] = True
    seen = {select_action(q, obs, mask, epsilon=1.0, rng=rng) for _ in range(300)}
    assert seen == {4, 7, 12}


def test_select_action_uses_only_the_given_rng():
    from showdown_rl.train_dqn import select_action

    q = _qnet()
    obs = _obs(np.random.default_rng(0))
    mask = np.ones(ACTION_DIM, dtype=bool)
    a1 = [select_action(q, obs, mask, epsilon=1.0, rng=np.random.default_rng(7)) for _ in range(10)]
    a2 = [select_action(q, obs, mask, epsilon=1.0, rng=np.random.default_rng(7)) for _ in range(10)]
    assert a1 == a2, "same rng seed must give the same actions"


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


def test_buffer_len_grows_then_caps():
    from showdown_rl.train_dqn import ReplayBuffer

    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=5)
    assert len(buf) == 0
    for i in range(1, 8):
        buf.push(_transition(rng))
        assert len(buf) == min(i, 5)


def test_buffer_overwrites_oldest():
    """Capacity 3, push rewards 1..5: only 3, 4, 5 survive."""
    from showdown_rl.train_dqn import ReplayBuffer

    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=3)
    for r in range(1, 6):
        buf.push(_transition(rng, reward=float(r)))
    rng2 = np.random.default_rng(1)
    seen: set[float] = set()
    for _ in range(60):
        seen |= set(buf.sample(3, rng2)["reward"].tolist())
    assert seen == {3.0, 4.0, 5.0}, f"expected only the last three survivors, got {seen}"


def test_buffer_sample_shapes_and_dtypes():
    buf, rng = _filled_buffer(50)
    b = buf.sample(8, rng)
    assert set(b) >= {"features", "species_ids", "mask", "action", "reward", "done",
                      "next_features", "next_species_ids", "next_mask"}
    assert b["features"].shape == (8, FEATURES_DIM) and b["features"].dtype == torch.float32
    assert b["species_ids"].shape == (8, SPECIES_IDS_DIM) and b["species_ids"].dtype == torch.int64
    assert b["mask"].shape == (8, ACTION_DIM) and b["mask"].dtype == torch.bool
    assert b["action"].shape == (8,) and b["action"].dtype == torch.int64
    assert b["reward"].shape == (8,) and b["reward"].dtype == torch.float32
    assert b["done"].shape == (8,) and b["done"].dtype == torch.float32
    assert b["next_features"].shape == (8, FEATURES_DIM)
    assert b["next_species_ids"].shape == (8, SPECIES_IDS_DIM)
    assert b["next_mask"].shape == (8, ACTION_DIM) and b["next_mask"].dtype == torch.bool


def test_buffer_sample_too_large_raises():
    buf, rng = _filled_buffer(4)
    with pytest.raises(ValueError):
        buf.sample(5, rng)


def test_buffer_dones_are_zero_one_floats():
    buf, rng = _filled_buffer(20, done=True)
    b = buf.sample(10, rng)
    assert torch.equal(b["done"], torch.ones(10))


# ---------------------------------------------------------------------------
# epsilon_schedule
# ---------------------------------------------------------------------------


def test_epsilon_schedule_endpoints_and_linearity():
    from showdown_rl.train_dqn import epsilon_schedule

    assert epsilon_schedule(0, 1.0, 0.1, 100) == pytest.approx(1.0)
    assert epsilon_schedule(50, 1.0, 0.1, 100) == pytest.approx(0.55)
    assert epsilon_schedule(100, 1.0, 0.1, 100) == pytest.approx(0.1)
    assert epsilon_schedule(10_000, 1.0, 0.1, 100) == pytest.approx(0.1)
    assert isinstance(epsilon_schedule(3, 1.0, 0.1, 100), float)


def test_epsilon_schedule_zero_decay_steps():
    from showdown_rl.train_dqn import epsilon_schedule

    assert epsilon_schedule(0, 1.0, 0.05, 0) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# dqn_loss / sync_target
# ---------------------------------------------------------------------------


def test_dqn_loss_returns_scalar_with_grad_and_stats():
    from showdown_rl.train_dqn import dqn_loss

    q, tgt = _qnet(0), _qnet(1)
    buf, rng = _filled_buffer(64)
    loss, stats = dqn_loss(q, tgt, buf.sample(32, rng), gamma=0.99)
    assert loss.shape == () and loss.requires_grad
    assert torch.isfinite(loss)
    for k in ("loss", "q_mean", "target_mean"):
        assert isinstance(stats[k], float), k
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in q.parameters())


def test_dqn_loss_target_is_reward_when_done():
    """All transitions terminal with reward -1 → every target is exactly -1."""
    from showdown_rl.train_dqn import dqn_loss

    q, tgt = _qnet(0), _qnet(1)
    buf, rng = _filled_buffer(40, reward=-1.0, done=True)
    _, stats = dqn_loss(q, tgt, buf.sample(40, rng), gamma=0.99)
    assert stats["target_mean"] == pytest.approx(-1.0, abs=1e-6)


def test_dqn_loss_target_uses_target_net_on_the_only_legal_next_action():
    """next_mask allows one action only → y = r + γ·Q_target(s', that action), whatever the online net says."""
    from showdown_rl.train_dqn import dqn_loss

    q, tgt = _qnet(0), _qnet(1)
    next_mask = np.zeros(ACTION_DIM, dtype=bool)
    next_mask[9] = True
    buf, rng = _filled_buffer(16, reward=0.5, next_mask=next_mask)
    batch = buf.sample(16, rng)
    with torch.no_grad():
        q_next = tgt(batch["next_features"], batch["next_species_ids"])[:, 9]
    expected = (0.5 + 0.99 * q_next).mean().item()
    _, stats = dqn_loss(q, tgt, batch, gamma=0.99)
    assert stats["target_mean"] == pytest.approx(expected, abs=1e-5)


def test_dqn_loss_target_has_no_grad_into_target_net():
    from showdown_rl.train_dqn import dqn_loss

    q, tgt = _qnet(0), _qnet(1)
    buf, rng = _filled_buffer(32)
    loss, _ = dqn_loss(q, tgt, buf.sample(32, rng), gamma=0.99)
    loss.backward()
    assert all(p.grad is None for p in tgt.parameters()), "target network must not receive gradients"


def test_sync_target_copies_weights():
    from showdown_rl.train_dqn import sync_target

    q, tgt = _qnet(0), _qnet(1)
    assert any(not torch.equal(a, b) for a, b in zip(q.parameters(), tgt.parameters(), strict=True))
    sync_target(q, tgt)
    assert all(torch.equal(a, b) for a, b in zip(q.parameters(), tgt.parameters(), strict=True))
    # and they are still separate tensors: changing q must not change tgt
    with torch.no_grad():
        next(q.parameters()).add_(1.0)
    assert not torch.equal(next(q.parameters()), next(tgt.parameters()))


def test_dqn_update_changes_online_weights():
    from showdown_rl.train_dqn import dqn_update

    q, tgt = _qnet(0), _qnet(1)
    buf, rng = _filled_buffer(128, reward=1.0)
    opt = torch.optim.Adam(q.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in q.parameters()]
    stats = dqn_update(q, tgt, opt, buf, batch_size=32, gamma=0.99, rng=rng, device=torch.device("cpu"))
    assert np.isfinite(stats["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, q.parameters(), strict=True))


# ---------------------------------------------------------------------------
# Plumbing (green from the start)
# ---------------------------------------------------------------------------


def test_episode_to_transitions_shifts_next_state_and_marks_done():
    rng = np.random.default_rng(0)
    ep = []
    for k in range(4):
        m = np.zeros(ACTION_DIM, dtype=bool)
        m[k] = True
        ep.append(Transition(obs=_obs(rng), mask=m, action=k, log_prob=0.0, value=0.0,
                             reward=1.0 if k == 3 else 0.0))
    out = episode_to_transitions(ep)
    assert len(out) == 4
    for k in range(3):
        assert out[k].done is False
        assert np.array_equal(out[k].next_features, ep[k + 1].obs.features)
        assert np.array_equal(out[k].next_mask, ep[k + 1].mask)
        assert out[k].reward == 0.0
    assert out[3].done is True and out[3].reward == 1.0
