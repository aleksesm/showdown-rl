"""Smoke tests for src/showdown_rl/actions.py.

Uses SimpleNamespace mocks of poke-env's Battle/Pokemon. The mocks expose
just the attributes our actions module touches (`active_pokemon`,
`team`, `available_moves`, `available_switches`, `can_tera`, and
Pokemon's `species` / `active` / `moves` / `.id`).

The integration test against a live battle (asserting that `to_poke_env`
returns orders the server actually accepts) will land alongside `state.py`
when we have the full obs+action loop.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from showdown_rl.actions import (
    ACTION_DIM,
    N_MOVES,
    N_SWITCHES,
    SWITCH_OFFSET,
    TERA_OFFSET,
    action_mask,
    stable_bench,
    stable_move_ids,
    to_poke_env,
)


def make_move(move_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=move_id)


def make_mon(species: str, move_ids: list[str] | None = None, active: bool = False) -> SimpleNamespace:
    moves = {mid: make_move(mid) for mid in (move_ids or [])}
    return SimpleNamespace(species=species, active=active, moves=moves)


def make_battle(
    active,
    bench,
    *,
    available_moves=None,
    available_switches=None,
    can_tera: bool = True,
) -> SimpleNamespace:
    """Build a fake Battle with the attributes our action code reads."""
    team = {f"p1: {active.species}": active}
    for m in bench:
        team[f"p1: {m.species}"] = m
    return SimpleNamespace(
        active_pokemon=active,
        team=team,
        available_moves=available_moves if available_moves is not None else list(active.moves.values()),
        available_switches=available_switches if available_switches is not None else bench,
        can_tera=can_tera,
    )


# --- Basic shape / dtype ----------------------------------------------------


def test_mask_shape_and_dtype():
    active = make_mon("garchomp", ["earthquake", "outrage", "swordsdance", "stoneedge"], active=True)
    bench = [make_mon(f"mon{i}", ["tackle"]) for i in range(5)]
    mask = action_mask(make_battle(active, bench))
    assert mask.shape == (ACTION_DIM,)
    assert mask.dtype == bool


def test_action_dim_is_thirteen():
    assert ACTION_DIM == 13
    assert N_MOVES + N_MOVES + N_SWITCHES == ACTION_DIM  # moves + tera-moves + switches
    assert TERA_OFFSET == N_MOVES
    assert SWITCH_OFFSET == N_MOVES + N_MOVES


# --- Stable ordering --------------------------------------------------------


def test_stable_move_order_is_alphabetical():
    active = make_mon("mon", ["stoneedge", "earthquake", "outrage", "swordsdance"])
    assert stable_move_ids(active) == ["earthquake", "outrage", "stoneedge", "swordsdance"]


def test_stable_move_order_with_no_active_returns_empty():
    assert stable_move_ids(None) == []


def test_stable_bench_is_alphabetical_by_species():
    active = make_mon("garchomp", ["eq"], active=True)
    bench_unordered = [
        make_mon("zapdos", ["thunderbolt"]),
        make_mon("alakazam", ["psychic"]),
        make_mon("mewtwo", ["psystrike"]),
    ]
    battle = make_battle(active, bench_unordered, available_switches=bench_unordered)
    assert [m.species for m in stable_bench(battle)] == ["alakazam", "mewtwo", "zapdos"]


def test_stable_ordering_unchanged_when_availability_shrinks():
    """The key invariant for RL: 'action 0' means the same move whether or not
    that move is currently available."""
    active = make_mon("mon", ["a", "b", "c", "d"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]

    full = make_battle(active, bench)
    restricted = make_battle(
        active, bench,
        available_moves=[active.moves["b"], active.moves["d"]],  # 'a' and 'c' disabled
    )

    # The order of move slots is the same regardless of which are available
    assert stable_move_ids(full.active_pokemon) == stable_move_ids(restricted.active_pokemon)


# --- Masking ----------------------------------------------------------------


def test_mask_all_legal_on_normal_turn():
    active = make_mon("mon", ["a", "b", "c", "d"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    mask = action_mask(make_battle(active, bench))
    assert mask.all()


def test_mask_disabled_move_blocked_with_tera_variant():
    active = make_mon("mon", ["a", "b", "c", "d"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    # 'b' is disabled (e.g., Choice locked away)
    available = [active.moves["a"], active.moves["c"], active.moves["d"]]
    mask = action_mask(make_battle(active, bench, available_moves=available))
    # Stable order: a, b, c, d → indices 0, 1, 2, 3
    assert mask[0] and not mask[1] and mask[2] and mask[3]
    # Tera variants follow the same pattern
    assert mask[4] and not mask[5] and mask[6] and mask[7]


def test_mask_force_switch_blocks_all_moves():
    """After a faint or pivot: available_moves is empty → all move bits off,
    switch bits unaffected."""
    active = make_mon("mon", ["a", "b", "c", "d"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    mask = action_mask(make_battle(active, bench, available_moves=[]))
    assert not mask[:SWITCH_OFFSET].any()
    assert mask[SWITCH_OFFSET:].all()


def test_mask_trapped_blocks_all_switches():
    active = make_mon("mon", ["a", "b", "c", "d"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    mask = action_mask(make_battle(active, bench, available_switches=[]))
    assert mask[:SWITCH_OFFSET].all()
    assert not mask[SWITCH_OFFSET:].any()


def test_mask_no_tera_blocks_tera_actions():
    active = make_mon("mon", ["a", "b", "c", "d"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    mask = action_mask(make_battle(active, bench, can_tera=False))
    assert mask[:N_MOVES].all()  # regular moves still OK
    assert not mask[TERA_OFFSET:SWITCH_OFFSET].any()
    assert mask[SWITCH_OFFSET:].all()


def test_mask_fainted_bench_mon_excluded():
    active = make_mon("mon", ["a", "b", "c", "d"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    # bench[2] (species "b2") is fainted → not in available_switches
    available = [bench[0], bench[1], bench[3], bench[4]]
    mask = action_mask(make_battle(active, bench, available_switches=available))
    # Stable bench: b0, b1, b2, b3, b4 → switch slots 8, 9, 10, 11, 12
    assert mask[8] and mask[9] and not mask[10] and mask[11] and mask[12]


def test_mask_no_active_returns_all_false():
    """Edge case: battle hasn't fully initialized yet."""
    battle = SimpleNamespace(
        active_pokemon=None,
        team={},
        available_moves=[],
        available_switches=[],
        can_tera=False,
    )
    assert not action_mask(battle).any()


# --- to_poke_env ------------------------------------------------------------


def test_to_poke_env_rejects_out_of_range_indices():
    active = make_mon("mon", ["a", "b", "c", "d"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    battle = make_battle(active, bench)
    with pytest.raises(ValueError):
        to_poke_env(-1, battle)
    with pytest.raises(ValueError):
        to_poke_env(13, battle)


def test_to_poke_env_move_action_creates_non_tera_order():
    active = make_mon("garchomp", ["earthquake", "outrage", "stoneedge", "swordsdance"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    battle = make_battle(active, bench)
    # action 0 = "earthquake" (alphabetically first)
    order = to_poke_env(0, battle)
    assert order is not None
    # The order's `terastallize` flag should be False
    assert order.terastallize is False
    # And it should reference the earthquake move
    assert order.order.id == "earthquake"


def test_to_poke_env_tera_action_sets_terastallize_flag():
    active = make_mon("garchomp", ["earthquake", "outrage", "stoneedge", "swordsdance"], active=True)
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    battle = make_battle(active, bench)
    # action 4 = "earthquake" with Tera
    order = to_poke_env(4, battle)
    assert order.terastallize is True
    assert order.order.id == "earthquake"


def test_to_poke_env_switch_action_creates_switch_order():
    active = make_mon("garchomp", ["earthquake"], active=True)
    bench_unordered = [
        make_mon("zapdos", ["t"]),
        make_mon("alakazam", ["t"]),
        make_mon("mewtwo", ["t"]),
        make_mon("dragonite", ["t"]),
        make_mon("nidoking", ["t"]),
    ]
    battle = make_battle(active, bench_unordered, available_switches=bench_unordered)
    # Stable bench: alakazam, dragonite, mewtwo, nidoking, zapdos
    # action 8 = alakazam (first alphabetically)
    order = to_poke_env(8, battle)
    assert order.order.species == "alakazam"
    # action 12 = zapdos (last alphabetically)
    order = to_poke_env(12, battle)
    assert order.order.species == "zapdos"


def test_to_poke_env_missing_move_slot_raises_index_error():
    """If the active mon has fewer than 4 moves and we ask for an absent slot."""
    active = make_mon("mon", ["a", "b"], active=True)  # only 2 moves
    bench = [make_mon(f"b{i}", ["t"]) for i in range(5)]
    battle = make_battle(active, bench)
    with pytest.raises(IndexError):
        to_poke_env(3, battle)  # asking for the 4th move slot
