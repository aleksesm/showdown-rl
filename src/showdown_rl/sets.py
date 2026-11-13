"""Random battle set database for Gen 9.

Loaded from `data/gen9randombattle_sets.json` — Smogon's published set
distributions, refreshed via `scripts/refresh_sets.py`. This is the same
data the Showdown server uses to generate random battles.

The data file has one entry per species:

    {
      "Garchomp": {
        "level": 74,
        "abilities": [...],          # union across roles
        "items": [...],              # union across roles
        "roles": {
          "Setup Sweeper": {
            "abilities": [...],
            "items": [...],
            "teraTypes": [...],
            "moves": [...]           # pool (usually >4; battle picks 4)
          },
          ...
        }
      },
      ...
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "gen9randombattle_sets.json"


@dataclass(frozen=True)
class Set:
    """A concrete possible set for a Pokémon in gen9randombattle.

    `moves` is the role's full move pool, not exactly four moves — the actual
    in-battle moveset is a sample of size 4 from this pool. For damage-matrix
    purposes we treat any move in the pool as a possibility the opponent
    might use.
    """

    species: str
    role: str
    level: int
    ability: str
    item: str
    tera_type: str
    moves: tuple[str, ...]


@dataclass(frozen=True)
class Reveals:
    """What we've observed about an opponent's Pokémon so far.

    Used to narrow `possible_sets` to those consistent with what we've seen.
    """

    species: str
    moves: frozenset[str] = field(default_factory=frozenset)
    ability: str | None = None
    item: str | None = None
    tera_type: str | None = None


@cache
def load_raw() -> dict:
    """Load the raw JSON. Cached — file is read once per process."""
    return json.loads(DATA_PATH.read_text())


@cache
def _species_index() -> dict[str, str]:
    """Map normalized species name → canonical key in the JSON.

    poke-env may give us names in various forms ("garchomp",
    "landorustherian", "Landorus-Therian"); the JSON uses Smogon's
    capitalized hyphenated form. This index lets us look up either way.
    """
    raw = load_raw()
    index = {}
    for key in raw:
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        index[normalized] = key
        index[key] = key  # also accept the canonical form directly
    return index


def canonical_species(name: str) -> str:
    """Resolve any spelling/casing of a species name to the JSON's canonical key.

    Raises KeyError if the species is not in the gen9randombattle pool.
    """
    index = _species_index()
    if name in index:
        return index[name]
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    if normalized in index:
        return index[normalized]
    raise KeyError(f"Species not in gen9randombattle sets: {name!r}")


def species_data(species: str) -> dict:
    return load_raw()[canonical_species(species)]


def all_roles(species: str) -> list[str]:
    return list(species_data(species)["roles"].keys())


def possible_sets(species: str, reveals: Reveals | None = None) -> list[Set]:
    """Enumerate every Set for `species` consistent with `reveals`.

    Consistency rules:
      - Revealed moves must all appear in the role's move pool.
      - Revealed ability (if any) must be in the role's ability list.
      - Revealed item (if any) must be in the role's item list.
      - Revealed tera type (if any) must be in the role's tera-type list.

    Returns one Set per (role × ability × item × tera_type) combination that
    survives narrowing. Empty list means no consistent set — usually a stale
    data snapshot, since real battles always produce a valid set.
    """
    data = species_data(species)
    canonical = canonical_species(species)
    level = data["level"]
    rev = reveals or Reveals(species=species)

    sets: list[Set] = []
    for role_name, role in data["roles"].items():
        if rev.moves and not rev.moves.issubset(set(role["moves"])):
            continue

        abilities = role["abilities"]
        items = role["items"]
        teras = role["teraTypes"]

        if rev.ability is not None:
            if rev.ability not in abilities:
                continue
            abilities = [rev.ability]
        if rev.item is not None:
            if rev.item not in items:
                continue
            items = [rev.item]
        if rev.tera_type is not None:
            if rev.tera_type not in teras:
                continue
            teras = [rev.tera_type]

        for ability in abilities:
            for item in items:
                for tera in teras:
                    sets.append(
                        Set(
                            species=canonical,
                            role=role_name,
                            level=level,
                            ability=ability,
                            item=item,
                            tera_type=tera,
                            moves=tuple(role["moves"]),
                        )
                    )
    return sets


def most_likely_set(species: str, reveals: Reveals | None = None) -> Set | None:
    """V1 heuristic: return the first consistent Set, or None if none exist.

    The JSON has no probability data, so "most likely" is deterministic-but-
    arbitrary. A later version could replace this with proper
    probability-weighted selection / marginalization.
    """
    sets = possible_sets(species, reveals)
    return sets[0] if sets else None


def possible_moves(species: str, reveals: Reveals | None = None) -> set[str]:
    """Union of moves across all sets consistent with reveals.

    Used by the damage matrix to bound which opponent moves are "possible"
    when deciding what threats to encode.
    """
    moves: set[str] = set()
    for s in possible_sets(species, reveals):
        moves.update(s.moves)
    return moves
