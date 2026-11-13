"""Unit tests for `wilson_interval` and `evaluate_matchup` in src/showdown_rl/evaluate.py.

    .venv/bin/python -m pytest tests/test_evaluate.py -v

`wilson_interval` is checked against hand-computed values and its edge
cases. `evaluate_matchup` is run with a fake `play_one` so no server is
needed; it checks ordering, counting, and that the interval comes from
`wilson_interval`. The plumbing tests are at the bottom (team generation
needs `node` and the Showdown checkout; it is skipped if they are missing).
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from showdown_rl.evaluate import (
    SHOWDOWN_DIR,
    ListTeambuilder,
    MatchupResult,
    evaluate_matchup,
    generate_team_pairs,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# wilson_interval
# ---------------------------------------------------------------------------


def test_wilson_known_value():
    # 45/100 at z=1.96: textbook value (0.3557, 0.5478)
    lo, hi = wilson_interval(45, 100)
    assert lo == pytest.approx(0.3557, abs=1e-3)
    assert hi == pytest.approx(0.5478, abs=1e-3)


def test_wilson_contains_point_estimate_and_is_in_unit_interval():
    for wins, n in [(0, 10), (3, 10), (5, 10), (10, 10), (150, 200), (1, 1)]:
        lo, hi = wilson_interval(wins, n)
        assert 0.0 <= lo <= wins / n <= hi <= 1.0


def test_wilson_edges_are_exact():
    assert wilson_interval(0, 20)[0] == 0.0
    assert wilson_interval(20, 20)[1] == 1.0


def test_wilson_is_symmetric():
    lo, hi = wilson_interval(30, 100)
    lo2, hi2 = wilson_interval(70, 100)
    assert lo == pytest.approx(1 - hi2)
    assert hi == pytest.approx(1 - lo2)


def test_wilson_narrows_with_n():
    w10 = wilson_interval(5, 10)
    w1000 = wilson_interval(500, 1000)
    assert (w1000[1] - w1000[0]) < (w10[1] - w10[0])


def test_wilson_empty():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_returns_python_floats():
    lo, hi = wilson_interval(7, 9)
    assert type(lo) is float and type(hi) is float


# ---------------------------------------------------------------------------
# evaluate_matchup (fake play_one, no server)
# ---------------------------------------------------------------------------


def _fake_play_one(outcomes: list[bool], seen: list):
    async def play_one(pair):
        seen.append(pair)
        return outcomes[len(seen) - 1]

    return play_one


def test_evaluate_matchup_counts_and_orders():
    pairs = [("own0", "opp0"), ("own1", "opp1"), ("own2", "opp2"), ("own3", "opp3")]
    seen: list = []
    r = asyncio.run(
        evaluate_matchup(
            _fake_play_one([True, False, True, True], seen), pairs, agent="a", opponent="b"
        )
    )
    assert isinstance(r, MatchupResult)
    assert seen == pairs  # every pair, once, in order
    assert (r.agent, r.opponent, r.wins, r.n) == ("a", "b", 3, 4)
    assert r.win_rate == pytest.approx(0.75)


def test_evaluate_matchup_uses_wilson():
    pairs = [("o", "p")] * 10
    r = asyncio.run(
        evaluate_matchup(_fake_play_one([True] * 6 + [False] * 4, []), pairs, agent="a", opponent="b")
    )
    assert (r.ci_lo, r.ci_hi) == pytest.approx(wilson_interval(6, 10))


def test_evaluate_matchup_is_sequential():
    # play_one must not be called for pair i+1 before pair i has resolved
    in_flight = 0
    max_in_flight = 0

    async def play_one(pair):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return True

    asyncio.run(evaluate_matchup(play_one, [("o", "p")] * 5, agent="a", opponent="b"))
    assert max_in_flight == 1


def test_evaluate_matchup_empty():
    r = asyncio.run(evaluate_matchup(_fake_play_one([], []), [], agent="a", opponent="b"))
    assert (r.wins, r.n, r.win_rate) == (0, 0, 0.0)


# ---------------------------------------------------------------------------
# plumbing (already green)
# ---------------------------------------------------------------------------


def test_list_teambuilder_yields_in_order():
    tb = ListTeambuilder(["t0", "t1", "t2"])
    assert [tb.yield_team() for _ in range(3)] == ["t0", "t1", "t2"]


@pytest.mark.skipif(
    shutil.which("node") is None or not SHOWDOWN_DIR.exists(),
    reason="needs node and the Showdown checkout in external/pokemon-showdown",
)
def test_generate_team_pairs_is_deterministic(tmp_path, monkeypatch):
    import showdown_rl.evaluate as ev

    monkeypatch.setattr(ev, "DATA", tmp_path)
    a = generate_team_pairs(3, seed=7)
    assert len(a) == 3 and all(len(p) == 2 for p in a)
    assert len({p[0] for p in a} | {p[1] for p in a}) == 6  # six distinct teams
    assert (tmp_path / "eval_teams_seed7_n3.json").exists()
    b = generate_team_pairs(3, seed=7)  # from cache
    assert a == b
    (tmp_path / "eval_teams_seed7_n3.json").unlink()
    assert generate_team_pairs(3, seed=7) == a  # regenerated, same seed, same teams


def test_load_qnet_roundtrip_and_eval_mode(tmp_path):
    import torch

    from showdown_rl.evaluate import load_qnet
    from showdown_rl.train_dqn import QNetwork

    src = QNetwork()
    ckpt = tmp_path / "x_qnet.pt"
    torch.save(src.state_dict(), ckpt)
    q = load_qnet(ckpt)
    assert not q.training
    for a, b in zip(src.state_dict().values(), q.state_dict().values(), strict=True):
        assert torch.equal(a, b)
