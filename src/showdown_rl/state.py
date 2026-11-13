"""Observation encoder for Phase 2.

Produces an `Observation` from a poke-env `AbstractBattle`:

- `.features` — fixed-shape `float32` array of length `FEATURES_DIM`.
- `.species_ids` — fixed-shape `int64` array of length `SPECIES_IDS_DIM`
  (12 = 2 active + 10 bench). Indices into a learned species embedding
  table that the policy network will own. We don't bake embeddings into
  the features tensor because (a) they'd duplicate parameters across
  feature offsets, and (b) decoupling lets us swap embedding dim per
  experiment without touching the encoder.

Block layout (species embeddings are kept out of the features tensor):

    [global (14)] [side own (12)] [side opp (12)]
    [active own (57)] [active opp (57)]
    [bench own (50)] [bench opp (50)]
    [damage matrix (127)]

Total features: 379. Plus 12 species IDs.

Determinism contract: every call returns the same shape regardless of
how much of the battle has been observed. Unknown values are filled with
zeros or "unknown" buckets, never omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from poke_env.battle import (
    AbstractBattle,
    Effect,
    Field,
    Move,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)

from .actions import stable_bench, stable_move_ids
from .damage import (
    DamageStats,
    damage_stats,
    populate_opp_set_info,
    populate_opp_stats,
)
from .sets import canonical_species, load_raw, most_likely_set, possible_moves

# ---------------------------------------------------------------------------
# Constants — feature counts per block
# ---------------------------------------------------------------------------

# Block 1: global
_WEATHER_KEYS = (Weather.RAINDANCE, Weather.SUNNYDAY, Weather.SANDSTORM, Weather.SNOWSCAPE, Weather.HAIL)
_TERRAIN_KEYS = (Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.PSYCHIC_TERRAIN, Field.MISTY_TERRAIN)
GLOBAL_DIM = len(_WEATHER_KEYS) + len(_TERRAIN_KEYS) + 1 + 1 + 1 + 2  # 5+4+1+1+1+2 = 14

# Block 2: per-side conditions (hazards + tailwind + 3 screens × {active, turns})
SIDE_DIM = 4 + 2 + 6  # hazards(4) + tailwind(2) + screens(6) = 12

# Block 3: active mon — HP + status + boosts + tera-type + tera-known + tera'd + volatiles + types
_STATUS_KEYS = (Status.BRN, Status.PAR, Status.PSN, Status.TOX, Status.SLP, Status.FRZ, Status.FNT)  # 7
_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")  # 7
_VOLATILE_KEYS = (Effect.SUBSTITUTE, Effect.LEECH_SEED, Effect.CONFUSION, Effect.TAUNT)  # 4
_ALL_TYPES = tuple(t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR"))  # 18
N_POKEMON_TYPES = len(_ALL_TYPES)  # 18
ACTIVE_DIM = 1 + len(_STATUS_KEYS) + len(_BOOST_KEYS) + N_POKEMON_TYPES + 1 + 1 + len(_VOLATILE_KEYS) + N_POKEMON_TYPES
# 1 + 7 + 7 + 18 (tera type one-hot) + 1 (tera-known flag) + 1 (is_terad) + 4 + 18 (own types) = 57

# Block 4: bench mon — HP + status + fainted + revealed
BENCH_DIM = 1 + len(_STATUS_KEYS) + 1 + 1  # 10
N_BENCH = 5  # per side

# Block 5: damage matrix
DAMAGE_DIM = 20 + 40 + 20 + 20 + 20 + 7  # = 127
N_MY_MOVES = 4
N_OPP_TOP_MOVES = 4

FEATURES_DIM = (
    GLOBAL_DIM
    + 2 * SIDE_DIM
    + 2 * ACTIVE_DIM
    + 2 * N_BENCH * BENCH_DIM
    + DAMAGE_DIM
)
SPECIES_IDS_DIM = 2 + 2 * N_BENCH  # 12

# ---------------------------------------------------------------------------
# Species → integer ID (stable, built once at import)
# ---------------------------------------------------------------------------

UNKNOWN_SPECIES_ID = 0


@cache
def _species_id_table() -> dict[str, int]:
    """Map canonical species name → int ID in [1, N]; 0 reserved for unknown."""
    raw = load_raw()
    table = {name: i + 1 for i, name in enumerate(sorted(raw.keys()))}
    return table


def species_id(species_name: str | None) -> int:
    if not species_name:
        return UNKNOWN_SPECIES_ID
    try:
        canon = canonical_species(species_name)
    except KeyError:
        return UNKNOWN_SPECIES_ID
    return _species_id_table().get(canon, UNKNOWN_SPECIES_ID)


def num_species() -> int:
    """Total species count including the unknown bucket (size of the embedding table)."""
    return len(_species_id_table()) + 1


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One turn's observation. Pass `.features` and `.species_ids` to the policy."""

    features: np.ndarray  # float32, shape (FEATURES_DIM,)
    species_ids: np.ndarray  # int64, shape (SPECIES_IDS_DIM,)


# ---------------------------------------------------------------------------
# Block encoders — each returns a 1D float32 array of fixed length
# ---------------------------------------------------------------------------


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _encode_global(battle: AbstractBattle) -> np.ndarray:
    out = _zeros(GLOBAL_DIM)
    i = 0
    # Weather one-hot (5)
    cur_weather = next(iter(getattr(battle, "weather", {}) or {}), None)
    for k in _WEATHER_KEYS:
        # PRIMORDIALSEA → bucket with RAINDANCE; DESOLATELAND → with SUNNYDAY; DELTASTREAM ignored
        if cur_weather == k:
            out[i] = 1.0
        elif k == Weather.RAINDANCE and cur_weather == Weather.PRIMORDIALSEA:
            out[i] = 1.0
        elif k == Weather.SUNNYDAY and cur_weather == Weather.DESOLATELAND:
            out[i] = 1.0
        i += 1
    # Terrain one-hot (4)
    fields = getattr(battle, "fields", {}) or {}
    for k in _TERRAIN_KEYS:
        out[i] = 1.0 if k in fields else 0.0
        i += 1
    # Trick Room, Gravity (2)
    out[i] = 1.0 if Field.TRICK_ROOM in fields else 0.0
    i += 1
    out[i] = 1.0 if Field.GRAVITY in fields else 0.0
    i += 1
    # Turn / 50 (1), capped
    out[i] = min(1.0, (getattr(battle, "turn", 0) or 0) / 50.0)
    i += 1
    # Tera-used own / opp (2)
    out[i] = 1.0 if _team_has_terad(battle, battle.team) else 0.0
    i += 1
    out[i] = 1.0 if _team_has_terad(battle, battle.opponent_team) else 0.0
    return out


def _team_has_terad(battle: AbstractBattle, team: dict) -> bool:
    return any(getattr(p, "is_terastallized", False) for p in team.values())


def _encode_side(battle: AbstractBattle, *, opp: bool) -> np.ndarray:
    """Hazards + tailwind + screens for one side."""
    out = _zeros(SIDE_DIM)
    cond = (
        getattr(battle, "opponent_side_conditions", {})
        if opp
        else getattr(battle, "side_conditions", {})
    )
    # Hazards (4)
    out[0] = 1.0 if SideCondition.STEALTH_ROCK in cond else 0.0
    out[1] = min(1.0, cond.get(SideCondition.SPIKES, 0) / 3.0)
    out[2] = min(1.0, cond.get(SideCondition.TOXIC_SPIKES, 0) / 2.0)
    out[3] = 1.0 if SideCondition.STICKY_WEB in cond else 0.0
    # Tailwind active + turns (2) — value in dict is turn started
    if SideCondition.TAILWIND in cond:
        out[4] = 1.0
        out[5] = _turns_remaining(cond[SideCondition.TAILWIND], battle.turn, total=4)
    # Screens (6) — Reflect, Light Screen, Aurora Veil, each {active, turns_remaining}
    for j, sc in enumerate(
        (SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL)
    ):
        if sc in cond:
            out[6 + 2 * j] = 1.0
            out[6 + 2 * j + 1] = _turns_remaining(cond[sc], battle.turn, total=5)
    return out


def _turns_remaining(turn_started: int, current_turn: int, *, total: int) -> float:
    if not isinstance(turn_started, int) or turn_started <= 0:
        return 0.0
    elapsed = max(0, current_turn - turn_started)
    return max(0.0, min(1.0, (total - elapsed) / total))


def _encode_active(battle: AbstractBattle, *, opp: bool) -> np.ndarray:
    out = _zeros(ACTIVE_DIM)
    mon = battle.opponent_active_pokemon if opp else battle.active_pokemon
    if mon is None:
        return out
    i = 0
    out[i] = mon.current_hp_fraction or 0.0
    i += 1
    # Status one-hot (7)
    for k in _STATUS_KEYS:
        out[i] = 1.0 if mon.status == k else 0.0
        i += 1
    # Boosts (7), scaled to [-1, 1]
    boosts = mon.boosts or {}
    for k in _BOOST_KEYS:
        out[i] = max(-1.0, min(1.0, boosts.get(k, 0) / 6.0))
        i += 1
    # Tera type one-hot (18) — for opp, all zero if not yet known
    tera = mon.tera_type
    tera_known = tera is not None and tera != PokemonType.THREE_QUESTION_MARKS
    if tera_known and not opp:
        # Our side: always known
        for j, t in enumerate(_ALL_TYPES):
            out[i + j] = 1.0 if tera == t else 0.0
    elif tera_known and opp and mon.is_terastallized:
        # Opp: only known after they tera
        for j, t in enumerate(_ALL_TYPES):
            out[i + j] = 1.0 if tera == t else 0.0
    i += N_POKEMON_TYPES
    # Tera-known flag (1)
    out[i] = 1.0 if (not opp or mon.is_terastallized) else 0.0
    i += 1
    # Is-terad (1)
    out[i] = 1.0 if mon.is_terastallized else 0.0
    i += 1
    # Volatile statuses (4)
    effects = getattr(mon, "effects", {}) or {}
    for k in _VOLATILE_KEYS:
        out[i] = 1.0 if k in effects else 0.0
        i += 1
    # Own types multi-hot (18)
    own_types = set(t for t in (mon.types or ()) if t is not None)
    for j, t in enumerate(_ALL_TYPES):
        out[i + j] = 1.0 if t in own_types else 0.0
    return out


def _encode_bench(battle: AbstractBattle, *, opp: bool) -> np.ndarray:
    """5 bench slots (10 features each). Unfilled slots = zeros."""
    out = _zeros(N_BENCH * BENCH_DIM)
    team = battle.opponent_team if opp else battle.team
    bench = sorted(
        (m for m in team.values() if not m.active),
        key=lambda m: m.species,
    )
    for slot in range(N_BENCH):
        off = slot * BENCH_DIM
        if slot >= len(bench):
            continue
        m = bench[slot]
        out[off] = m.current_hp_fraction if m.current_hp_fraction is not None else 1.0
        for j, k in enumerate(_STATUS_KEYS):
            out[off + 1 + j] = 1.0 if m.status == k else 0.0
        out[off + 1 + len(_STATUS_KEYS)] = 1.0 if m.fainted else 0.0
        # Revealed: for own side always 1; for opp 1 iff we've seen the species
        revealed = 1.0 if (not opp or m.species) else 0.0
        out[off + 1 + len(_STATUS_KEYS) + 1] = revealed
    return out


# ---------------------------------------------------------------------------
# Damage matrix
# ---------------------------------------------------------------------------


def _stat_vec_5(s: DamageStats) -> tuple[float, float, float, float, float]:
    return (s.min_pct, s.max_pct, s.ev_pct, s.p_ohko, s.accuracy)


def _stat_vec_4(s: DamageStats) -> tuple[float, float, float, float]:
    return (s.min_pct, s.max_pct, s.ev_pct, s.p_ohko)


def _stat_vec_2(s: DamageStats) -> tuple[float, float]:
    return (s.ev_pct, s.p_ohko)


_ZERO_STATS = DamageStats.zero(accuracy=0.0)


def _safe_damage(
    attacker: Pokemon,
    defender: Pokemon,
    move: Move | None,
    battle: AbstractBattle,
    *,
    attacker_set=None,
    defender_set=None,
) -> DamageStats:
    """damage_stats wrapper that returns DamageStats.zero on any failure.

    poke_env.calc has many preconditions; rather than gate on each, we treat
    failures (missing info, unhandled move, etc.) as "no damage info" and
    let the encoder produce zeros for that cell.
    """
    if move is None:
        return _ZERO_STATS
    try:
        return damage_stats(
            attacker, defender, move, battle,
            attacker_set=attacker_set, defender_set=defender_set,
        )
    except Exception:
        return _ZERO_STATS


def _moves_from_set_names(names: list[str], limit: int | None = None) -> list[Move]:
    out: list[Move] = []
    for name in names[:limit] if limit else names:
        try:
            mid = name.lower().replace(" ", "").replace("-", "").replace("'", "")
            out.append(Move(mid, gen=9))
        except Exception:
            continue
    return out


def _opp_set_for(mon: Pokemon):
    if mon is None or not mon.species:
        return None
    try:
        revealed_moves = frozenset(
            mon.moves.keys() if getattr(mon, "moves", None) else ()
        )
        from .sets import Reveals  # local import to avoid cycle at module load
        return most_likely_set(
            mon.species,
            Reveals(species=mon.species, moves=revealed_moves) if revealed_moves else None,
        )
    except KeyError:
        return None


def _own_stable_moves(battle: AbstractBattle) -> list[Move | None]:
    """Return our active mon's 4 move slots in stable order, padded with None."""
    active = battle.active_pokemon
    if active is None:
        return [None] * N_MY_MOVES
    move_ids = stable_move_ids(active)
    return [active.moves.get(mid) if mid else None for mid in (move_ids + [None] * N_MY_MOVES)[:N_MY_MOVES]]


def _top_n_opp_moves(
    attacker: Pokemon,
    defender: Pokemon,
    battle: AbstractBattle,
    attacker_set,
    n: int = N_OPP_TOP_MOVES,
) -> list[tuple[Move, DamageStats]]:
    """Pick the top N moves the opp could use against `defender`, by EV damage."""
    if attacker is None or defender is None or attacker_set is None:
        return []
    candidates = list(possible_moves(attacker.species))
    scored: list[tuple[float, Move, DamageStats]] = []
    for move in _moves_from_set_names(candidates):
        s = _safe_damage(attacker, defender, move, battle, attacker_set=attacker_set)
        scored.append((s.ev_pct, move, s))
    scored.sort(key=lambda x: -x[0])
    return [(m, s) for _, m, s in scored[:n]]


def _encode_damage_matrix(battle: AbstractBattle) -> np.ndarray:
    out = _zeros(DAMAGE_DIM)
    active_me = battle.active_pokemon
    active_opp = battle.opponent_active_pokemon
    if active_me is None or active_opp is None:
        return out

    opp_set = _opp_set_for(active_opp)
    my_bench = sorted((m for m in battle.team.values() if not m.active), key=lambda m: m.species)[:N_BENCH]
    opp_bench = sorted(
        (m for m in battle.opponent_team.values() if not m.active), key=lambda m: m.species
    )[:N_BENCH]
    my_moves = _own_stable_moves(battle)

    i = 0

    # Our 4 moves × opp active × 5 stats = 20
    for move in my_moves:
        s = _safe_damage(active_me, active_opp, move, battle, defender_set=opp_set)
        out[i : i + 5] = _stat_vec_5(s)
        i += 5

    # Our 4 moves × each opp bench × 2 stats = 40
    for slot in range(N_BENCH):
        defender = opp_bench[slot] if slot < len(opp_bench) else None
        d_set = _opp_set_for(defender) if defender else None
        for move in my_moves:
            if defender is None:
                out[i : i + 2] = (0.0, 0.0)
            else:
                s = _safe_damage(active_me, defender, move, battle, defender_set=d_set)
                out[i : i + 2] = _stat_vec_2(s)
            i += 2

    # Opp's top-4 moves × our active × 5 stats = 20
    top = _top_n_opp_moves(active_opp, active_me, battle, opp_set, n=N_OPP_TOP_MOVES)
    for k in range(N_OPP_TOP_MOVES):
        if k < len(top):
            out[i : i + 5] = _stat_vec_5(top[k][1])
        i += 5

    # Opp active's most-threatening move × each our bench × 4 stats = 20
    threat_move = top[0][0] if top else None
    for slot in range(N_BENCH):
        defender = my_bench[slot] if slot < len(my_bench) else None
        if defender is None or threat_move is None:
            out[i : i + 4] = (0.0, 0.0, 0.0, 0.0)
        else:
            s = _safe_damage(active_opp, defender, threat_move, battle, attacker_set=opp_set)
            out[i : i + 4] = _stat_vec_4(s)
        i += 4

    # Each opp bench × their most-threatening move × our active × 4 stats = 20
    for slot in range(N_BENCH):
        attacker = opp_bench[slot] if slot < len(opp_bench) else None
        if attacker is None:
            out[i : i + 4] = (0.0, 0.0, 0.0, 0.0)
            i += 4
            continue
        bench_set = _opp_set_for(attacker)
        bench_top = _top_n_opp_moves(attacker, active_me, battle, bench_set, n=1)
        if bench_top:
            out[i : i + 4] = _stat_vec_4(bench_top[0][1])
        i += 4

    # Speed comparison + priorities (7)
    my_spe = (active_me.stats or {}).get("spe") if active_me.stats else None
    opp_spe = (active_opp.stats or {}).get("spe") if active_opp.stats else None
    if my_spe is None and active_me.base_stats:
        from .damage import estimate_stat
        my_spe = estimate_stat(active_me.base_stats.get("spe", 50), active_me.level)
    if opp_spe is None and active_opp.base_stats:
        from .damage import estimate_stat
        opp_spe = estimate_stat(active_opp.base_stats.get("spe", 50), active_opp.level)
    # log-normalize speeds (typical range 50-400)
    out[i] = min(1.0, (my_spe or 0) / 400.0)
    i += 1
    out[i] = min(1.0, (opp_spe or 0) / 400.0)
    i += 1
    out[i] = 1.0 if (my_spe or 0) > (opp_spe or 0) else 0.0
    i += 1
    for move in my_moves:
        if move is not None:
            out[i] = max(-1.0, min(1.0, (move.priority or 0) / 5.0))
        i += 1

    return out


# ---------------------------------------------------------------------------
# Species IDs
# ---------------------------------------------------------------------------


def _encode_species_ids(battle: AbstractBattle) -> np.ndarray:
    out = np.zeros(SPECIES_IDS_DIM, dtype=np.int64)
    active_me = battle.active_pokemon
    active_opp = battle.opponent_active_pokemon
    if active_me is not None:
        out[0] = species_id(active_me.species)
    if active_opp is not None:
        out[1] = species_id(active_opp.species)
    my_bench = sorted((m for m in battle.team.values() if not m.active), key=lambda m: m.species)[:N_BENCH]
    opp_bench = sorted(
        (m for m in battle.opponent_team.values() if not m.active), key=lambda m: m.species
    )[:N_BENCH]
    for slot in range(N_BENCH):
        if slot < len(my_bench):
            out[2 + slot] = species_id(my_bench[slot].species)
        if slot < len(opp_bench):
            out[2 + N_BENCH + slot] = species_id(opp_bench[slot].species)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode(battle: AbstractBattle) -> Observation:
    """Encode one turn's battle state into an `Observation`.

    Side effect: mutates opp mons in place (injecting estimated stats and
    inferred item/ability from sets) — same mutation `damage.damage_stats`
    performs, just amortized across the multiple damage cells we compute.
    """
    # Pre-populate opp mon state once per turn to amortize across damage cells.
    for m in (battle.opponent_active_pokemon, *battle.opponent_team.values()):
        if m is None:
            continue
        populate_opp_stats(m)
        s = _opp_set_for(m)
        if s is not None:
            populate_opp_set_info(m, s)

    parts = [
        _encode_global(battle),
        _encode_side(battle, opp=False),
        _encode_side(battle, opp=True),
        _encode_active(battle, opp=False),
        _encode_active(battle, opp=True),
        _encode_bench(battle, opp=False),
        _encode_bench(battle, opp=True),
        _encode_damage_matrix(battle),
    ]
    features = np.concatenate(parts).astype(np.float32)
    assert features.shape == (FEATURES_DIM,), f"shape mismatch: {features.shape}"
    return Observation(features=features, species_ids=_encode_species_ids(battle))
