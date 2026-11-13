"""Damage statistics for Phase 2 — thin wrapper around `poke_env.calc`.

`poke_env.calc.calculate_damage` is a 1869-line Python port of
[`@smogon/damage-calc`](https://github.com/smogon/damage-calc)'s
`gen789.ts` and is the canonical Gen-9 damage calculator. It handles all
the ability/item/weather/terrain edge cases (Levitate, Tinted Lens,
Multiscale, etc.) that an from-scratch port would miss.

This module wraps it to:

1. **Populate opponent stats** before calling (poke-env leaves `opp.stats`
   as all-`None` until inferred from in-battle damage). We fill with the
   standard random-battle estimates from a `Set`.
2. **Fold crit and roll variance** into the four scalar features we feed
   the policy: `min%, max%, EV%, P(OHKO|hit)`, plus `accuracy`. These are
   our own concept, not poke-env's.

Identifier lookup: `calculate_damage` requires the exact identifier
strings used by `battle.team` / `battle.opponent_team` (e.g.
``"p1: Iron Boulder"``). Use `find_identifier(battle, pokemon)` to look
them up.
"""

from __future__ import annotations

from dataclasses import dataclass

from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.calc import calculate_damage

from .sets import Set

# Random-battle stat estimation defaults
_DEFAULT_IV = 31
_DEFAULT_EV = 85
DEFAULT_CRIT_RATE = 1.0 / 24.0
N_ROLLS = 16  # Pokemon damage rolls: 16 evenly-spaced values from 85% to 100%


# ---------------------------------------------------------------------------
# Public output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DamageStats:
    """Damage statistics as fractions of defender's max HP, plus accuracy.

    All percentages in `[0.0, ∞)` — values > 1.0 mean overkill / guaranteed
    OHKO. Crit chance is folded into `ev_pct` and `p_ohko`; accuracy is kept
    separate so the agent can reason about miss risk independently.
    """

    min_pct: float
    max_pct: float
    ev_pct: float       # expected, including the crit branch
    p_ohko: float       # P(damage >= defender.hp | move hits), including crit
    accuracy: float     # in [0, 1]; 1.0 if move can't miss

    @classmethod
    def zero(cls, accuracy: float = 1.0) -> "DamageStats":
        return cls(0.0, 0.0, 0.0, 0.0, accuracy)


# ---------------------------------------------------------------------------
# Stat estimation + opp-mon population
# ---------------------------------------------------------------------------


def estimate_stat(base: int, level: int, *, is_hp: bool = False) -> int:
    """Standard Pokemon stat formula with random-battle defaults (31 IV, 85 EV,
    neutral nature). Used when we don't know the opponent's exact spread.
    """
    inner = (2 * base + _DEFAULT_IV + _DEFAULT_EV // 4) * level // 100
    if is_hp:
        return inner + level + 10
    return inner + 5  # neutral nature = ×1.0


def estimate_all_stats(base_stats: dict[str, int], level: int) -> dict[str, int]:
    return {
        "hp":  estimate_stat(base_stats["hp"],  level, is_hp=True),
        "atk": estimate_stat(base_stats["atk"], level),
        "def": estimate_stat(base_stats["def"], level),
        "spa": estimate_stat(base_stats["spa"], level),
        "spd": estimate_stat(base_stats["spd"], level),
        "spe": estimate_stat(base_stats["spe"], level),
    }


def _stats_unpopulated(pokemon: Pokemon) -> bool:
    """True if pokemon._stats is missing or has any None value."""
    s = getattr(pokemon, "_stats", None)
    if not s:
        return True
    return any(v is None for v in s.values())


def populate_opp_stats(pokemon: Pokemon) -> None:
    """Mutate `pokemon` to give it estimated stats if they aren't populated.

    For opp mons in random battles, poke-env knows level + base_stats but
    leaves `.stats` values as `None`. `calculate_damage` requires real
    numbers. We inject the standard random-battle estimates.
    """
    if not _stats_unpopulated(pokemon):
        return
    stats = estimate_all_stats(pokemon.base_stats, pokemon.level)
    pokemon._stats = stats  # noqa: SLF001 — intentional injection
    if not pokemon._max_hp:
        pokemon._max_hp = stats["hp"]
        pokemon._current_hp = stats["hp"]


def populate_opp_set_info(pokemon: Pokemon, opp_set: Set) -> None:
    """Inject revealed-or-inferred item/ability from a Set when the live
    battle data doesn't have them yet. No-op for revealed slots.
    """
    if not pokemon._ability:
        pokemon._ability = opp_set.ability.lower().replace(" ", "")
    item = (pokemon._item or "").lower()
    if item in ("", "unknown_item"):
        pokemon._item = opp_set.item.lower().replace(" ", "")


# ---------------------------------------------------------------------------
# Identifier lookup
# ---------------------------------------------------------------------------


def find_identifier(battle: AbstractBattle, pokemon: Pokemon) -> str:
    """Reverse-lookup a Pokemon's identifier in the battle's team dicts.

    `calculate_damage` requires the exact key form (e.g. ``"p1: Iron Boulder"``,
    with the space). We can't reconstruct it from `species` because poke-env
    capitalizes and may insert spaces.
    """
    for ident, p in battle.team.items():
        if p is pokemon:
            return ident
    for ident, p in battle.opponent_team.items():
        if p is pokemon:
            return ident
    raise ValueError(f"Pokemon {pokemon.species!r} not found in battle teams")


# ---------------------------------------------------------------------------
# Roll / crit folding (our own layer on top of poke-env's (min, max) output)
# ---------------------------------------------------------------------------


def _roll_average(min_dmg: int, max_dmg: int) -> float:
    """Average of the 16-roll distribution between min_dmg and max_dmg.

    Pokemon's 16 rolls are evenly spaced in damage (after floor), so the
    average is approximately (min + max) / 2.
    """
    return (min_dmg + max_dmg) / 2


def _p_at_least(min_dmg: int, max_dmg: int, threshold: int) -> float:
    """Fraction of the 16 evenly-spaced rolls that meet `threshold`."""
    if max_dmg < threshold:
        return 0.0
    if min_dmg >= threshold:
        return 1.0
    if max_dmg == min_dmg:
        return 1.0 if min_dmg >= threshold else 0.0
    n_meeting = sum(
        1
        for i in range(N_ROLLS)
        if min_dmg + i * (max_dmg - min_dmg) / (N_ROLLS - 1) >= threshold
    )
    return n_meeting / N_ROLLS


def _accuracy_to_fraction(raw_accuracy) -> float:
    """poke-env Move.accuracy is `True` (perfect) or an int 0-100."""
    if raw_accuracy is True:
        return 1.0
    if isinstance(raw_accuracy, (int, float)):
        a = float(raw_accuracy)
        return min(1.0, max(0.0, a / 100.0 if a > 1.0 else a))
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_damage_stats(
    *,
    min_no_crit: int,
    max_no_crit: int,
    min_crit: int,
    max_crit: int,
    defender_hp: int,
    defender_hp_max: int,
    accuracy: float,
    p_crit: float = DEFAULT_CRIT_RATE,
) -> DamageStats:
    """Fold crit + roll variance into one DamageStats. Pure math, no poke-env.

    Useful as a unit-testable seam below the poke-env wrapper.
    """
    if defender_hp_max <= 0:
        return DamageStats.zero(accuracy=accuracy)

    avg_no = _roll_average(min_no_crit, max_no_crit)
    avg_cr = _roll_average(min_crit, max_crit)
    ev = (1 - p_crit) * avg_no + p_crit * avg_cr

    p_ohko_no = _p_at_least(min_no_crit, max_no_crit, defender_hp)
    p_ohko_cr = _p_at_least(min_crit, max_crit, defender_hp)
    p_ohko = (1 - p_crit) * p_ohko_no + p_crit * p_ohko_cr

    return DamageStats(
        min_pct=min_no_crit / defender_hp_max,  # lowest is always no-crit, low roll
        max_pct=max_crit / defender_hp_max,     # highest is always crit, high roll
        ev_pct=ev / defender_hp_max,
        p_ohko=p_ohko,
        accuracy=accuracy,
    )


def damage_stats(
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    battle: AbstractBattle,
    *,
    attacker_set: Set | None = None,
    defender_set: Set | None = None,
    p_crit: float = DEFAULT_CRIT_RATE,
) -> DamageStats:
    """Compute `DamageStats` for `attacker` using `move` against `defender`.

    Side effects: if `attacker` or `defender` is an opp mon with missing
    `.stats`, this fills them with random-battle estimates (mutation of
    poke-env state). If `attacker_set` / `defender_set` is provided and the
    corresponding mon's item/ability isn't revealed, fills those too.
    """
    # Status / zero-BP moves never deal damage
    cat = getattr(move.category, "name", "").lower() if move.category else ""
    if "status" in cat or (move.base_power or 0) <= 0:
        return DamageStats.zero(accuracy=_accuracy_to_fraction(move.accuracy))

    # Inject missing state so calculate_damage's preconditions hold.
    populate_opp_stats(attacker)
    populate_opp_stats(defender)
    if attacker_set is not None:
        populate_opp_set_info(attacker, attacker_set)
    if defender_set is not None:
        populate_opp_set_info(defender, defender_set)

    attacker_id = find_identifier(battle, attacker)
    defender_id = find_identifier(battle, defender)

    min_no, max_no = calculate_damage(attacker_id, defender_id, move, battle, is_critical=False)
    min_cr, max_cr = calculate_damage(attacker_id, defender_id, move, battle, is_critical=True)

    hp_max = defender.max_hp or defender.stats["hp"] or 1
    hp = defender.current_hp if defender.current_hp is not None else hp_max

    return compute_damage_stats(
        min_no_crit=int(min_no),
        max_no_crit=int(max_no),
        min_crit=int(min_cr),
        max_crit=int(max_cr),
        defender_hp=int(hp),
        defender_hp_max=int(hp_max),
        accuracy=_accuracy_to_fraction(move.accuracy),
        p_crit=p_crit,
    )
