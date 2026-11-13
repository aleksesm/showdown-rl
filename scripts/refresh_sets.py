"""Re-download the gen9randombattle set distribution from pkmn.github.io.

Run when Smogon updates the random battle format:
    .venv/bin/python scripts/refresh_sets.py

This overwrites `data/gen9randombattle_sets.json`. Commit the diff so the
training data stays in sync with the current ladder format.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

URL = "https://pkmn.github.io/randbats/data/gen9randombattle.json"
DEST = Path(__file__).resolve().parent.parent / "data" / "gen9randombattle_sets.json"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=30) as resp:
        raw = resp.read()
    data = json.loads(raw)
    DEST.write_bytes(raw)
    print(f"Wrote {DEST} ({len(raw):,} bytes, {len(data)} species)")


if __name__ == "__main__":
    main()
