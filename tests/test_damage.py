"""Tests for src/showdown_rl/damage.py.

`damage.py` now wraps `poke_env.calc.calculate_damage` (the canonical
Gen-9 calc, a Python port of `@smogon/damage-calc`). The mechanics
themselves (STAB, type chart, abilities, items, weather, screens, …)
are tested by poke-env upstream; what we test here is *our wrapper*:

1. The pure-math `compute_damage_stats` layer that folds crit + roll
   variance into our `DamageStats` (min/max/EV/P(OHKO)/accuracy).
2. Stat estimation and opp-mon-state injection helpers
   (`populate_opp_stats`, `populate_opp_set_info`, `find_identifier`).
3. An optional end-to-end integration test that runs a real battle
   against a local Showdown server — skipped if the server isn't up.
"""

from __future__ import annotations

import pytest

from showdown_rl.damage import (
    DEFAULT_CRIT_RATE,
    DamageStats,
    compute_damage_stats,
    estimate_all_stats,
    estimate_stat,
    find_identifier,
    populate_opp_set_info,
    populate_opp_stats,
)
from showdown_rl.sets import Set


# ---------------------------------------------------------------------------
# Pure math layer
# ---------------------------------------------------------------------------


def test_compute_stats_basic_shape():
    s = compute_damage_stats(
        min_no_crit=10, max_no_crit=20, min_crit=15, max_crit=30,
        defender_hp=200, defender_hp_max=200, accuracy=1.0, p_crit=0.0,
    )
    assert s.min_pct == 10 / 200
    assert s.max_pct == 30 / 200  # max is always crit, regardless of p_crit
    assert s.ev_pct == pytest.approx(15 / 200)  # avg no-crit
    assert s.p_ohko == 0.0


def test_compute_stats_guaranteed_ohko_when_min_exceeds_hp():
    s = compute_damage_stats(
        min_no_crit=300, max_no_crit=400, min_crit=450, max_crit=600,
        defender_hp=100, defender_hp_max=200, accuracy=1.0,
    )
    assert s.p_ohko == 1.0


def test_compute_stats_zero_ohko_when_max_below_hp():
    s = compute_damage_stats(
        min_no_crit=10, max_no_crit=20, min_crit=25, max_crit=30,
        defender_hp=100, defender_hp_max=200, accuracy=1.0,
    )
    assert s.p_ohko == 0.0


def test_compute_stats_only_crit_ohkos():
    # No-crit can't OHKO; crit always OHKOs → p_ohko ≈ p_crit
    s = compute_damage_stats(
        min_no_crit=50, max_no_crit=99, min_crit=150, max_crit=200,
        defender_hp=100, defender_hp_max=200, accuracy=1.0,
        p_crit=DEFAULT_CRIT_RATE,
    )
    assert s.p_ohko == pytest.approx(DEFAULT_CRIT_RATE)


def test_compute_stats_crit_raises_ev_and_p_ohko():
    args = dict(
        min_no_crit=10, max_no_crit=20, min_crit=15, max_crit=30,
        defender_hp=200, defender_hp_max=200, accuracy=1.0,
    )
    no_crit = compute_damage_stats(**args, p_crit=0.0)
    all_crit = compute_damage_stats(**args, p_crit=1.0)
    assert all_crit.ev_pct > no_crit.ev_pct


def test_compute_stats_accuracy_propagated():
    s = compute_damage_stats(
        min_no_crit=10, max_no_crit=20, min_crit=15, max_crit=30,
        defender_hp=200, defender_hp_max=200, accuracy=0.7,
    )
    assert s.accuracy == 0.7


def test_compute_stats_handles_zero_hp_max():
    s = compute_damage_stats(
        min_no_crit=10, max_no_crit=20, min_crit=15, max_crit=30,
        defender_hp=0, defender_hp_max=0, accuracy=1.0,
    )
    assert s == DamageStats.zero(accuracy=1.0)


def test_p_ohko_partial_when_some_rolls_meet_threshold():
    # Range 100..115, threshold 110 → ~6/16 of rolls qualify
    s = compute_damage_stats(
        min_no_crit=100, max_no_crit=115, min_crit=150, max_crit=170,
        defender_hp=110, defender_hp_max=200, accuracy=1.0, p_crit=0.0,
    )
    # Rolls evenly spaced: 100, 101, 102, ..., 115. Six of them (110-115) qualify.
    assert s.p_ohko == pytest.approx(6 / 16, abs=0.05)


# ---------------------------------------------------------------------------
# Stat estimation
# ---------------------------------------------------------------------------


def test_estimate_stat_formula_100_base_level_100():
    # 100-base, level 100, 85 EV / 31 IV / neutral nature
    # → ((200 + 31 + 21) * 100/100 + 5) * 1.0 = 257
    assert estimate_stat(100, 100) == 257


def test_estimate_hp_formula_100_base_level_100():
    # 100-base HP, level 100, 85 EV / 31 IV
    # → (200 + 31 + 21) * 100/100 + 100 + 10 = 362
    assert estimate_stat(100, 100, is_hp=True) == 362


def test_estimate_all_stats_returns_six_stats():
    base = {"hp": 100, "atk": 80, "def": 90, "spa": 70, "spd": 110, "spe": 60}
    stats = estimate_all_stats(base, 75)
    assert set(stats.keys()) == set(base.keys())
    assert all(isinstance(v, int) and v > 0 for v in stats.values())


# ---------------------------------------------------------------------------
# Mutation helpers — populate_opp_stats / populate_opp_set_info / find_identifier
# ---------------------------------------------------------------------------


class FakePokemon:
    """Minimal Pokemon stand-in. Mirrors the underscore-attribute layout
    `damage.py` mutates, so tests don't need a real poke-env Pokemon.
    """

    def __init__(
        self,
        *,
        species: str = "fakemon",
        level: int = 75,
        base_stats: dict | None = None,
        stats: dict | None = None,
        max_hp: int = 0,
        hp: int = 0,
        ability: str | None = None,
        item: str | None = "",
    ):
        self.species = species
        self.level = level
        self.base_stats = base_stats or {
            "hp": 100, "atk": 80, "def": 90, "spa": 70, "spd": 110, "spe": 60,
        }
        self._stats = stats
        self._max_hp = max_hp
        self._current_hp = hp
        self._ability = ability
        self._item = item


def test_populate_opp_stats_fills_when_all_none():
    p = FakePokemon(stats={k: None for k in ("hp", "atk", "def", "spa", "spd", "spe")})
    populate_opp_stats(p)
    assert all(isinstance(v, int) and v > 0 for v in p._stats.values())
    assert p._max_hp == p._stats["hp"]
    assert p._current_hp == p._stats["hp"]


def test_populate_opp_stats_fills_when_stats_is_none():
    p = FakePokemon(stats=None)
    populate_opp_stats(p)
    assert isinstance(p._stats, dict)
    assert all(isinstance(v, int) for v in p._stats.values())


def test_populate_opp_stats_no_op_when_fully_populated():
    real = {"hp": 300, "atk": 200, "def": 180, "spa": 150, "spd": 170, "spe": 220}
    p = FakePokemon(stats=dict(real), max_hp=300, hp=300)
    populate_opp_stats(p)
    assert p._stats == real
    assert p._max_hp == 300


def test_populate_opp_set_info_fills_unknown_item_and_ability():
    p = FakePokemon(ability=None, item="unknown_item")
    s = Set(
        species="Garchomp", role="Setup Sweeper", level=74,
        ability="Rough Skin", item="Life Orb", tera_type="Ground",
        moves=("earthquake",),
    )
    populate_opp_set_info(p, s)
    assert p._ability == "roughskin"
    assert p._item == "lifeorb"


def test_populate_opp_set_info_respects_revealed_data():
    p = FakePokemon(ability="overcoat", item="leftovers")
    s = Set("Mon", "Role", 75, "Hustle", "Choice Band", "Normal", ("tackle",))
    populate_opp_set_info(p, s)
    assert p._ability == "overcoat"
    assert p._item == "leftovers"


def test_find_identifier_in_own_team():
    pkm = FakePokemon()

    class FakeBattle:
        team = {"p1: Garchomp": pkm}
        opponent_team: dict = {}

    assert find_identifier(FakeBattle(), pkm) == "p1: Garchomp"


def test_find_identifier_in_opp_team():
    pkm = FakePokemon()

    class FakeBattle:
        team: dict = {}
        opponent_team = {"p2: Heatran": pkm}

    assert find_identifier(FakeBattle(), pkm) == "p2: Heatran"


def test_find_identifier_raises_for_unknown_pokemon():
    class FakeBattle:
        team: dict = {}
        opponent_team: dict = {}

    with pytest.raises(ValueError):
        find_identifier(FakeBattle(), FakePokemon())


# ---------------------------------------------------------------------------
# Integration test — real battle. Auto-skipped if server isn't up.
# ---------------------------------------------------------------------------


def _server_up(host: str = "localhost", port: int = 8000) -> bool:
    import socket
    try:
        s = socket.create_connection((host, port), timeout=0.5)
        s.close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _server_up(),
    reason="local Showdown server not running on :8000 (start it to run this test)",
)
def test_damage_stats_against_real_battle():
    """Run one random battle and call damage_stats() on the first usable turn.

    Verifies the full poke_env.calc wrapper works end-to-end with realistic
    poke-env state (opp mons with missing .stats, unknown items, etc.).
    """
    import asyncio

    from poke_env import AccountConfiguration, LocalhostServerConfiguration
    from poke_env.player import RandomPlayer

    from showdown_rl.damage import damage_stats

    captured: list[DamageStats] = []

    class CapturingPlayer(RandomPlayer):
        def choose_move(self, battle):
            if (
                not captured
                and battle.active_pokemon
                and battle.opponent_active_pokemon
                and battle.available_moves
            ):
                move = battle.available_moves[0]
                stats = damage_stats(
                    battle.active_pokemon, battle.opponent_active_pokemon, move, battle
                )
                captured.append(stats)
            return super().choose_move(battle)

    async def run():
        p1 = CapturingPlayer(
            account_configuration=AccountConfiguration("DmgTest1", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
        )
        p2 = RandomPlayer(
            account_configuration=AccountConfiguration("DmgTest2", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
        )
        await p1.battle_against(p2, n_battles=1)

    asyncio.run(run())

    assert captured, "never reached a turn where damage_stats was callable"
    s = captured[0]
    assert 0.0 <= s.min_pct <= 5.0
    assert s.min_pct <= s.max_pct
    assert s.min_pct <= s.ev_pct <= s.max_pct
    assert 0.0 <= s.p_ohko <= 1.0
    assert 0.0 < s.accuracy <= 1.0
