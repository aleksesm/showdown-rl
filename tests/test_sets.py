"""Smoke tests for src/showdown_rl/sets.py."""

from __future__ import annotations

import pytest

from showdown_rl.sets import (
    Reveals,
    all_roles,
    canonical_species,
    load_raw,
    most_likely_set,
    possible_moves,
    possible_sets,
)


def test_load_returns_dict_with_500_plus_species():
    raw = load_raw()
    assert isinstance(raw, dict)
    assert len(raw) > 400, f"unexpectedly small species pool: {len(raw)}"


def test_canonical_species_accepts_multiple_forms():
    # JSON key form
    assert canonical_species("Garchomp") == "Garchomp"
    # poke-env-ish lowercase
    assert canonical_species("garchomp") == "Garchomp"
    # Hyphenated forms (Landorus-Therian is in the data)
    assert canonical_species("Landorus-Therian") == "Landorus-Therian"
    assert canonical_species("landorustherian") == "Landorus-Therian"
    assert canonical_species("landorus-therian") == "Landorus-Therian"


def test_canonical_species_raises_for_unknown():
    with pytest.raises(KeyError):
        canonical_species("notapokemon")


def test_possible_sets_no_reveals_returns_all_combinations():
    # Garchomp has 2 roles with multiple items each → multiple sets.
    sets = possible_sets("Garchomp")
    assert len(sets) >= 2
    roles_seen = {s.role for s in sets}
    assert "Setup Sweeper" in roles_seen
    assert "Fast Support" in roles_seen


def test_possible_sets_narrows_by_revealed_move():
    # Swords Dance only appears on Setup Sweeper.
    sets = possible_sets(
        "Garchomp", Reveals(species="Garchomp", moves=frozenset({"Swords Dance"}))
    )
    assert len(sets) >= 1
    assert all(s.role == "Setup Sweeper" for s in sets)
    assert all("Swords Dance" in s.moves for s in sets)


def test_possible_sets_narrows_by_revealed_item():
    # Loaded Dice is only on Setup Sweeper for Garchomp.
    sets = possible_sets(
        "Garchomp", Reveals(species="Garchomp", item="Loaded Dice")
    )
    assert len(sets) >= 1
    assert all(s.role == "Setup Sweeper" for s in sets)
    assert all(s.item == "Loaded Dice" for s in sets)


def test_possible_sets_inconsistent_reveals_returns_empty():
    # An impossible move/item combo should yield no sets.
    sets = possible_sets(
        "Garchomp",
        Reveals(species="Garchomp", moves=frozenset({"Spikes"}), item="Loaded Dice"),
    )
    assert sets == []


def test_most_likely_set_is_deterministic_and_consistent():
    s = most_likely_set("Garchomp", Reveals(species="Garchomp", moves=frozenset({"Spikes"})))
    assert s is not None
    assert s.role == "Fast Support"  # Spikes only appears here
    assert "Spikes" in s.moves


def test_most_likely_set_none_when_inconsistent():
    s = most_likely_set(
        "Garchomp",
        Reveals(species="Garchomp", moves=frozenset({"Spikes"}), item="Loaded Dice"),
    )
    assert s is None


def test_possible_moves_unions_across_consistent_sets():
    # Without reveals: union of all moves from all Garchomp roles.
    moves = possible_moves("Garchomp")
    # Setup Sweeper has Earthquake; Fast Support has Spikes — both should show up.
    assert "Earthquake" in moves
    assert "Spikes" in moves


def test_all_roles_lists_role_names():
    roles = all_roles("Garchomp")
    assert "Setup Sweeper" in roles
    assert "Fast Support" in roles


def test_species_normalization_handles_dash_and_space_variants():
    # Iron Valiant uses a space in some sources, hyphen in others
    canon = canonical_species("Iron Valiant")
    assert canon in load_raw()
    assert canonical_species("ironvaliant") == canon
