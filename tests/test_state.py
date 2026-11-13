"""Tests for src/showdown_rl/state.py and the full Phase 2 pipeline.

`state.py` weaves together sets, damage, and actions, so unit-mocking
every helper would mostly re-test wiring. Instead we rely on:

1. Pure-function unit tests for the species-ID mapping and shape contract.
2. A real-battle integration test (auto-skipped without server) that
   asserts the full pipeline (encode + action_mask + to_poke_env)
   produces a well-shaped observation, a non-empty action mask, and
   plays a complete battle without errors. This is the load-bearing
   end-to-end Phase 2 test.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from showdown_rl.state import (
    FEATURES_DIM,
    Observation,
    SPECIES_IDS_DIM,
    UNKNOWN_SPECIES_ID,
    num_species,
    species_id,
)


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------


def test_constants_sum_correctly():
    from showdown_rl.state import (
        ACTIVE_DIM,
        BENCH_DIM,
        DAMAGE_DIM,
        GLOBAL_DIM,
        N_BENCH,
        SIDE_DIM,
    )
    expected = GLOBAL_DIM + 2 * SIDE_DIM + 2 * ACTIVE_DIM + 2 * N_BENCH * BENCH_DIM + DAMAGE_DIM
    assert FEATURES_DIM == expected


def test_species_ids_dim_is_twelve():
    assert SPECIES_IDS_DIM == 12  # 2 active + 5+5 bench


def test_species_id_unknown_bucket_zero():
    assert species_id(None) == UNKNOWN_SPECIES_ID == 0
    assert species_id("") == 0
    assert species_id("notapokemon") == 0


def test_species_id_canonicalizes_case_and_punctuation():
    a = species_id("Garchomp")
    b = species_id("garchomp")
    assert a == b > 0


def test_species_id_handles_hyphenated_forms():
    a = species_id("Landorus-Therian")
    b = species_id("landorustherian")
    c = species_id("landorus-therian")
    assert a == b == c > 0


def test_num_species_covers_all_known():
    # 508 in the JSON + 1 unknown bucket
    assert num_species() == 509


def test_observation_is_namedtuple_like():
    arr_f = np.zeros(FEATURES_DIM, dtype=np.float32)
    arr_i = np.zeros(SPECIES_IDS_DIM, dtype=np.int64)
    obs = Observation(features=arr_f, species_ids=arr_i)
    assert obs.features.shape == (FEATURES_DIM,)
    assert obs.species_ids.shape == (SPECIES_IDS_DIM,)
    assert obs.features.dtype == np.float32
    assert obs.species_ids.dtype == np.int64


# ---------------------------------------------------------------------------
# Integration — real battle pipeline
# ---------------------------------------------------------------------------


def _server_up(host: str = "localhost", port: int = 8000) -> bool:
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
def test_full_phase_2_pipeline_on_real_battles():
    """End-to-end Phase 2 test.

    Play 3 random battles. On every turn:
      - encode(battle) returns Observation with correct shape, dtype, no NaNs.
      - species IDs are valid (0 = unknown bucket; ≥1 = known species).
      - action_mask(battle) is shape (13,) bool with at least one True bit
        (otherwise the agent would be stuck).
      - to_poke_env(chosen_action, battle) returns a valid order.

    This is the load-bearing test for Phase 2 — if it passes, the whole
    obs/action stack works together against the real simulator.
    """
    import asyncio

    from poke_env import AccountConfiguration, LocalhostServerConfiguration
    from poke_env.player import Player, RandomPlayer

    from showdown_rl.actions import ACTION_DIM, action_mask, to_poke_env
    from showdown_rl.state import encode

    inspection = {"turns_observed": 0, "errors": []}

    class EncodingPlayer(Player):
        """Encode + mask every turn; pick a uniform-random legal action."""

        def choose_move(self, battle):
            try:
                obs = encode(battle)
                # Shape / dtype invariants
                assert obs.features.shape == (FEATURES_DIM,), f"features shape: {obs.features.shape}"
                assert obs.features.dtype == np.float32
                assert obs.species_ids.shape == (SPECIES_IDS_DIM,)
                assert obs.species_ids.dtype == np.int64
                assert not np.any(np.isnan(obs.features)), "NaN in features"
                assert not np.any(np.isinf(obs.features)), "Inf in features"
                assert (obs.species_ids >= 0).all() and (obs.species_ids < num_species()).all()
                # Action mask invariants
                mask = action_mask(battle)
                assert mask.shape == (ACTION_DIM,)
                assert mask.dtype == bool
                assert mask.any(), f"no legal actions at turn {battle.turn}"
                inspection["turns_observed"] += 1
                # Pick a uniform-random legal action and convert
                legal = np.flatnonzero(mask)
                action_idx = int(np.random.choice(legal))
                return to_poke_env(action_idx, battle)
            except Exception as e:
                inspection["errors"].append(f"turn {battle.turn}: {type(e).__name__}: {e}")
                return self.choose_random_move(battle)

    async def play():
        p1 = EncodingPlayer(
            account_configuration=AccountConfiguration("EncTest1", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
        )
        p2 = RandomPlayer(
            account_configuration=AccountConfiguration("EncTest2", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format="gen9randombattle",
        )
        await p1.battle_against(p2, n_battles=3)

    asyncio.run(play())

    assert inspection["errors"] == [], (
        f"encountered {len(inspection['errors'])} errors during 3 battles:\n  "
        + "\n  ".join(inspection["errors"][:5])
    )
    assert inspection["turns_observed"] >= 5, (
        f"only {inspection['turns_observed']} turns observed — battles ended too fast"
    )
