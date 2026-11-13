"""13-action discrete action space + masking for Phase 2.

Action indices:
    0-3  : use move 1-4 (no Tera)
    4-7  : use move 1-4 with Tera
    8-12 : switch to bench slot 1-5

**Stability matters for RL.** `battle.available_moves` and
`battle.available_switches` change order between turns (a move can drop out
because of Choice lock, Taunt, or PP exhaustion; a mon can drop out because
it fainted). If we mapped `action_idx 0` to "first element of
available_moves," then "action 0" would mean a different move on different
turns — the policy can't learn under that.

So we order moves and bench mons in a *stable* way (alphabetical) derived
from the full team data, not from the per-turn availability lists. The
availability lists are used only for masking, not for ordering.

All edge cases (force switch after faint or pivot, trapped, disabled moves,
fainted bench, Tera already used) are handled implicitly: the mask only
unsets bits that correspond to entries in `available_moves` /
`available_switches`, both of which poke-env keeps accurate.
"""

from __future__ import annotations

import numpy as np
from poke_env.battle import AbstractBattle, Pokemon
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

ACTION_DIM = 13
N_MOVES = 4
TERA_OFFSET = 4
SWITCH_OFFSET = 8
N_SWITCHES = 5


def stable_move_ids(active: Pokemon | None) -> list[str]:
    """Active mon's move IDs in stable alphabetical order.

    `Pokemon.moves` is a dict {move_id: Move}; ordering by key is stable
    across turns even when `available_moves` shrinks.
    """
    if active is None:
        return []
    return sorted(active.moves.keys())


def stable_bench(battle: AbstractBattle) -> list[Pokemon]:
    """Bench mons in stable alphabetical-by-species order."""
    bench = [m for m in battle.team.values() if not m.active]
    return sorted(bench, key=lambda m: m.species)


def action_mask(battle: AbstractBattle) -> np.ndarray:
    """Boolean mask of shape (13,). `mask[i] = True` iff action i is legal.

    Implementation contract: an action is "legal" iff submitting the
    corresponding `BattleOrder` would be accepted by the Showdown server.
    We rely on `battle.available_moves` / `battle.available_switches` to be
    the ground truth — poke-env maintains them through turn transitions.
    """
    mask = np.zeros(ACTION_DIM, dtype=bool)
    active = battle.active_pokemon
    if active is None:
        return mask

    available_move_ids = {m.id for m in battle.available_moves}
    can_tera = bool(battle.can_tera)
    for i, mid in enumerate(stable_move_ids(active)[:N_MOVES]):
        if mid in available_move_ids:
            mask[i] = True
            if can_tera:
                mask[TERA_OFFSET + i] = True

    available_switch_ids = {id(m) for m in battle.available_switches}
    for i, mon in enumerate(stable_bench(battle)[:N_SWITCHES]):
        if id(mon) in available_switch_ids:
            mask[SWITCH_OFFSET + i] = True

    return mask


def to_poke_env(action_idx: int, battle: AbstractBattle) -> BattleOrder:
    """Convert our integer action to a poke-env BattleOrder.

    Caller is responsible for verifying `action_mask(battle)[action_idx]`
    before calling — passing an illegal index will either raise here (if
    the corresponding slot is empty) or produce an order the server rejects
    (which ends the battle as a forfeit).
    """
    if not 0 <= action_idx < ACTION_DIM:
        raise ValueError(f"action_idx {action_idx} out of range [0, {ACTION_DIM})")

    active = battle.active_pokemon
    if action_idx < SWITCH_OFFSET:
        move_slot = action_idx if action_idx < N_MOVES else action_idx - TERA_OFFSET
        move_ids = stable_move_ids(active)
        if move_slot >= len(move_ids):
            raise IndexError(
                f"action {action_idx} maps to move slot {move_slot}, but active mon "
                f"only has {len(move_ids)} moves"
            )
        move = active.moves[move_ids[move_slot]]
        return Player.create_order(move, terastallize=action_idx >= TERA_OFFSET)

    bench_slot = action_idx - SWITCH_OFFSET
    bench = stable_bench(battle)
    if bench_slot >= len(bench):
        raise IndexError(
            f"action {action_idx} maps to bench slot {bench_slot}, but team has "
            f"only {len(bench)} bench mons"
        )
    return Player.create_order(bench[bench_slot])
