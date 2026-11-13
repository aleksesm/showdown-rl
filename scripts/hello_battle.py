"""Phase 1 hello-world: two RandomPlayers battle on the local Showdown server.

Verifies the full pipe end-to-end before we layer any RL on top:
- poke-env is installed and the version we wrote against
- the local Showdown server is reachable at ws://localhost:8000
- a gen9randombattle can run to completion
- we can read the win/loss result programmatically

Run (with the local Showdown server already started):
    .venv/bin/python scripts/hello_battle.py [n_battles]
"""

from __future__ import annotations

import asyncio
import sys

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer


async def run(n_battles: int) -> None:
    p1 = RandomPlayer(
        account_configuration=AccountConfiguration("RandBot1", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )
    p2 = RandomPlayer(
        account_configuration=AccountConfiguration("RandBot2", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format="gen9randombattle",
    )

    await p1.battle_against(p2, n_battles=n_battles)

    print(f"Played {n_battles} battle(s) of gen9randombattle.")
    print(f"  RandBot1 wins: {p1.n_won_battles}")
    print(f"  RandBot2 wins: {p2.n_won_battles}")
    print(f"  ties:          {n_battles - p1.n_won_battles - p2.n_won_battles}")


def main() -> None:
    n_battles = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    asyncio.run(run(n_battles))


if __name__ == "__main__":
    main()
