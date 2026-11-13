"""Evaluation harness: win rates with Wilson intervals on paired team sets.

Protocol:

- Opponents: `RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`.
- N battles per matchup (default 200).
- The agent samples from its policy at evaluation time (``--greedy`` takes
  the argmax instead, for comparison).
- **Paired teams.** One list of N (own team, opponent team) pairs is
  generated once from a seed with Showdown's own random-battle generator,
  and *every* matchup in the table plays exactly that list, in order. Teams
  differ between battles, but battle i uses the same two teams in every
  row. ``gen9randombattle`` ignores supplied teams, so battles are played
  in ``gen9anythinggoes`` with the team validator switched off and Sleep
  Clause added (``EVAL_FORMAT``, ``EVAL_RULES``). The generated teams
  are random-battle teams, so the state distribution matches training.
  Battle randomness (damage rolls, crits, the opponent's own choices) is
  not fixed, only the teams.

  Why not ``gen9customgame`` (used until 2026-10-08): it is a *debug*
  format, and Showdown reports the opponent's exact HP in debug formats
  instead of a percentage. poke-env then stores the true HP stat as
  ``max_hp`` where training (a percentage format) stored 100, and the
  damage-calculator features, which divide predicted damage by ``max_hp``,
  shrink by 2-3x relative to what the networks were trained on. Measured
  cost for DQN v1 vs random: 0.78 in the training format vs 0.54-0.60 in
  ``gen9customgame``. Every non-debug Gen 9
  format reports percentages (sim/battle.ts ``reportExactHP``).

The statistics are in ``wilson_interval`` and ``evaluate_matchup``;
everything else is plumbing.

Run (server must be up):

    .venv/bin/python -m showdown_rl.evaluate --checkpoint results/reinforce_showdown_policy.pt --out results/eval_reinforce_v1.csv
    .venv/bin/python -m showdown_rl.evaluate --algo dqn --checkpoint results/dqn_showdown_v1_qnet.pt --out results/eval_dqn_v1.csv

Output: one CSV with one row per matchup, agent rows first, then
baseline-vs-baseline rows on the same team pairs.

``--algo dqn`` loads a ``<stem>_qnet.pt`` state_dict into a
``QNetwork`` and acts ε-greedily over legal actions with ``--eps``
(default 0.05, the training floor). The protocol row is ``dqn``;
``--eps 0`` gives the argmax row ``dqn_greedy``, the DQN counterpart of
``--greedy``. ``--greedy`` itself is ignored for DQN.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle
from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.teambuilder import Teambuilder

from .agent import PolicyAgent
from .policy import FlatPolicy
from .train_dqn import DQNAgent, QNetwork

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
SHOWDOWN_DIR = ROOT / "external" / "pokemon-showdown"
EVAL_FORMAT = "gen9anythinggoes"
# Custom rules appended to the challenge (``format@@@rules``). ``!Obtainable``
# disables the team validator (random-battle sets fail it, e.g. Zacian with
# Behemoth Blade); Sleep Clause Mod is in the random-battle ruleset.
EVAL_RULES = "!Obtainable,Sleep Clause Mod"

TeamPair = tuple[str, str]  # (own packed team, opponent packed team)
PlayOne = Callable[[TeamPair], Awaitable[bool]]  # plays one battle, returns "first player won"


@dataclass
class MatchupResult:
    agent: str
    opponent: str
    wins: int
    n: int
    win_rate: float
    ci_lo: float
    ci_hi: float


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    With p̂ = wins / n, the interval is the set of p for which the normal
    approximation of the binomial test does not reject at level z:

        centre = (p̂ + z²/(2n)) / (1 + z²/n)
        half   = z · sqrt( p̂(1−p̂)/n + z²/(4n²) ) / (1 + z²/n)
        return (centre − half, centre + half)

    Contract:
        - `wins`, `n` are integers with 0 ≤ wins ≤ n.
        - Returns `(lo, hi)` as Python floats with 0 ≤ lo ≤ p̂ ≤ hi ≤ 1.
        - Edge cases: `n == 0` has no proportion; return `(0.0, 1.0)`.
          `wins == 0` gives `lo == 0.0` exactly and `wins == n` gives
          `hi == 1.0` exactly. Algebraically the formula does this,
          numerically it does not, so these two cases are set explicitly.
        - Unlike the Wald interval p̂ ± z·sqrt(p̂(1−p̂)/n), this never
          produces an empty or out-of-range interval at p̂ ∈ {0, 1}, which
          is why it was chosen.
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = wins/n
    centre = (p_hat + z**2/(2*n)) / (1 + z**2/n)
    half = z * math.sqrt(p_hat * (1 - p_hat)/n + z**2/(4*n**2))/(1 + z**2/n)
    if wins == 0:
        lo = 0.0
    else:
        lo = centre - half
    if wins == n:
        hi = 1.0
    else:
        hi = centre + half
    return (lo, hi)


async def evaluate_matchup(
    play_one: PlayOne,
    team_pairs: Sequence[TeamPair],
    *,
    agent: str,
    opponent: str,
    z: float = 1.96,
) -> MatchupResult:
    """Play every team pair once, in order, and summarise.

    `play_one(pair)` is an awaitable that plays exactly one battle with
    `pair = (own team, opponent team)` and returns True iff the first
    player won. It is supplied by `make_play_one` (plumbing below) in the
    real run and by a fake in the tests.

    Contract:
        - Battles run sequentially (await each before starting the next):
          the pairing guarantee depends on order, and the server-side
          players are configured for one battle at a time.
        - `n` is `len(team_pairs)`; ties count as non-wins.
        - `win_rate` is `wins / n` (0.0 when n == 0).
        - `ci_lo`, `ci_hi` come from `wilson_interval(wins, n, z)`.
        - Returns a `MatchupResult` with `agent` and `opponent` passed through.
    """
    wins = 0
    n = len(team_pairs)
    for pair in team_pairs:
        if await play_one(pair):
            wins += 1
    if n == 0:
        win_rate = 0.0
    else:
        win_rate = wins/n
    return MatchupResult(agent, opponent, wins, n, win_rate, wilson_interval(wins, n, z)[0], wilson_interval(wins, n, z)[1])



# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

_GEN_JS = """
const {Teams} = require("./dist/sim/teams.js");
const [n, seed] = [+process.argv[1], +process.argv[2]];
for (let i = 0; i < n; i++) {
  const own = Teams.pack(Teams.generate("gen9randombattle", {seed: [seed, 0, 0, 2 * i]}));
  const opp = Teams.pack(Teams.generate("gen9randombattle", {seed: [seed, 0, 0, 2 * i + 1]}));
  console.log(JSON.stringify([own, opp]));
}
"""


def generate_team_pairs(n: int, seed: int) -> list[TeamPair]:
    """N deterministic (own, opp) random-battle team pairs from Showdown's generator.

    Cached at data/eval_teams_seed{seed}_n{n}.json so every evaluation run
    with the same seed and n plays the same teams, even after a Showdown
    update changes the generator.
    """
    cache = DATA / f"eval_teams_seed{seed}_n{n}.json"
    if cache.exists():
        return [tuple(p) for p in json.loads(cache.read_text())]
    out = subprocess.run(
        ["node", "-e", _GEN_JS, str(n), str(seed)],
        cwd=SHOWDOWN_DIR, capture_output=True, text=True, check=True,
    ).stdout
    pairs = [tuple(json.loads(line)) for line in out.splitlines() if line.strip()]
    assert len(pairs) == n, f"generator produced {len(pairs)} pairs, expected {n}"
    DATA.mkdir(exist_ok=True)
    cache.write_text(json.dumps(pairs))
    return pairs


class ListTeambuilder(Teambuilder):
    """Yields teams from a list, in order, once each."""

    def __init__(self, teams: Sequence[str]) -> None:
        self._teams = list(teams)
        self._i = 0

    def yield_team(self) -> str:
        team = self._teams[self._i]
        self._i += 1
        return team


class _EvalFormat:
    """Mixin for every evaluation player.

    - Team preview: lead with slot 1, like random battles do.
    - Challenges go out as ``EVAL_FORMAT@@@EVAL_RULES``. poke-env matches
      incoming challenges and battle rooms against ``self._format`` by string
      equality, and the server echoes the rules in the challenge but not in
      the room name, so ``_format`` stays the bare ``EVAL_FORMAT`` and the two
      challenge handlers compare on the part before ``@@@``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        orig = self.ps_client.challenge

        async def challenge(username: str, format_: str, packed_team: str | None) -> None:
            await orig(username, f"{format_}@@@{EVAL_RULES}", packed_team)

        self.ps_client.challenge = challenge

    def teampreview(self, battle: AbstractBattle) -> str:
        return "/team 123456"

    async def _handle_challenge_request(self, split_message: list[str]) -> None:
        challenger = split_message[2].strip()
        if challenger != self.username and len(split_message) >= 6:
            if split_message[5].split("@@@")[0] == self._format:
                await self._challenge_queue.put(challenger)

    async def _update_challenges(self, split_message: list[str]) -> None:
        for user, format_ in json.loads(split_message[2]).get("challengesFrom", {}).items():
            if format_.split("@@@")[0] == self._format:
                await self._challenge_queue.put(user)


class EvalRandomPlayer(_EvalFormat, RandomPlayer):
    pass


class EvalMaxBasePowerPlayer(_EvalFormat, MaxBasePowerPlayer):
    pass


class EvalSimpleHeuristicsPlayer(_EvalFormat, SimpleHeuristicsPlayer):
    pass


class EvalPolicyAgent(_EvalFormat, PolicyAgent):
    pass


class EvalDQNAgent(_EvalFormat, DQNAgent):
    pass


BASELINES: dict[str, type[Player]] = {
    "random": EvalRandomPlayer,
    "maxbasepower": EvalMaxBasePowerPlayer,
    "simpleheuristics": EvalSimpleHeuristicsPlayer,
}


def _account(name: str) -> AccountConfiguration:
    return AccountConfiguration(f"{name}{int(time.time()) % 100000}"[:18], None)


def make_baseline(name: str, tag: str) -> Player:
    return BASELINES[name](
        account_configuration=_account(tag),
        server_configuration=LocalhostServerConfiguration,
        battle_format=EVAL_FORMAT,
        max_concurrent_battles=1,
        log_level=30,
    )


def load_policy(checkpoint: Path, device: str = "cpu") -> FlatPolicy:
    policy = FlatPolicy()
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.to(device).eval()
    return policy


def make_agent(policy: FlatPolicy, *, greedy: bool, device: str = "cpu") -> PolicyAgent:
    return EvalPolicyAgent(
        policy=policy,
        device=device,
        deterministic=greedy,
        account_configuration=_account("EvalAgent"),
        server_configuration=LocalhostServerConfiguration,
        battle_format=EVAL_FORMAT,
        max_concurrent_battles=1,
        log_level=30,
    )


def load_qnet(checkpoint: Path, device: str = "cpu") -> QNetwork:
    qnet = QNetwork()
    qnet.load_state_dict(torch.load(checkpoint, map_location=device))
    qnet.to(device).eval()
    return qnet


def make_dqn_agent(qnet: QNetwork, *, epsilon: float, device: str = "cpu", seed: int = 0) -> DQNAgent:
    agent = EvalDQNAgent(
        qnet=qnet,
        rng=np.random.default_rng(seed),
        device=device,
        account_configuration=_account("EvalDQN"),
        server_configuration=LocalhostServerConfiguration,
        battle_format=EVAL_FORMAT,
        max_concurrent_battles=1,
        log_level=30,
    )
    agent.epsilon = epsilon
    return agent


def make_play_one(p1: Player, p2: Player) -> PlayOne:
    """Return a coroutine function that plays one battle between p1 and p2 on a team pair."""

    async def play_one(pair: TeamPair) -> bool:
        own, opp = pair
        p1.update_team(own)
        p2.update_team(opp)
        before = p1.n_won_battles
        await p1.battle_against(p2, n_battles=1)
        return p1.n_won_battles > before

    return play_one


async def run_table(
    checkpoint: Path, n: int, seed: int, *, algo: str, greedy: bool, eps: float, device: str, out: Path
) -> list[MatchupResult]:
    pairs = generate_team_pairs(n, seed)
    if algo == "dqn":
        agent_name = "dqn_greedy" if eps == 0.0 else "dqn"
        agent = make_dqn_agent(load_qnet(checkpoint, device), epsilon=eps, device=device, seed=seed)
    else:
        agent_name = "reinforce_greedy" if greedy else "reinforce"
        agent = make_agent(load_policy(checkpoint, device), greedy=greedy, device=device)
    baselines = {name: make_baseline(name, f"Ev{name[:6]}") for name in BASELINES}

    rows: list[MatchupResult] = []
    for name, opp in baselines.items():
        r = await evaluate_matchup(
            make_play_one(agent, opp), pairs, agent=agent_name, opponent=name
        )
        print(_fmt(r))
        rows.append(r)
        _write_csv(rows, out)

    # Same team pairs for baseline-vs-baseline, so the table is calibrated.
    names = list(BASELINES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            second = make_baseline(b, f"Ev{a[:3]}{b[:4]}")  # fresh account: a player can't fight itself
            r = await evaluate_matchup(
                make_play_one(baselines[a], second), pairs, agent=a, opponent=b
            )
            print(_fmt(r))
            rows.append(r)
            _write_csv(rows, out)
    return rows


def _fmt(r: MatchupResult) -> str:
    return (f"{r.agent:>18} vs {r.opponent:<16} {r.wins:>4}/{r.n:<4} "
            f"win={r.win_rate:.3f}  95% CI [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")


def _write_csv(rows: Sequence[MatchupResult], out: Path) -> None:
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="reinforce", choices=["reinforce", "dqn"])
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="default: results/reinforce_showdown_policy.pt or results/dqn_showdown_v1_qnet.pt")
    p.add_argument("--n", type=int, default=200, help="battles per matchup")
    p.add_argument("--seed", type=int, default=0, help="team-pair generator seed")
    p.add_argument("--greedy", action="store_true", help="reinforce: argmax instead of sampling")
    p.add_argument("--eps", type=float, default=0.05, help="dqn: ε-greedy at eval time; 0 = argmax row dqn_greedy")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.checkpoint is None:
        a.checkpoint = RESULTS / ("dqn_showdown_v1_qnet.pt" if a.algo == "dqn" else "reinforce_showdown_policy.pt")
    return a


if __name__ == "__main__":
    a = _parse_args()
    asyncio.run(run_table(a.checkpoint, a.n, a.seed, algo=a.algo, greedy=a.greedy, eps=a.eps, device=a.device, out=a.out))
