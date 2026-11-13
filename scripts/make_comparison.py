"""Assemble results/comparison.csv, comparison.png and the tables for comparison.md from results/eval2_*.csv.

Every eval2 file was produced by `showdown_rl.evaluate` in the fixed evaluation format on the same
200 seeded team pairs. Rows are the agent rows (first three lines of each file).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
OPPS = ["random", "maxbasepower", "simpleheuristics"]

# (file stem, label, algorithm, training battles, policy at eval, note)
ROWS = [
    # A: equal budget, behaviour policies
    ("eval2_reinforce_v1", "REINFORCE v1 (seed 0)", "REINFORCE", 3200, "sampled", "equal budget"),
    ("eval2_reinforce_v1s1", "REINFORCE v1 (seed 1)", "REINFORCE", 3200, "sampled", "equal budget"),
    ("eval2_dqn_v1_lr1e-3", "DQN v1 (seed 0, lr 1e-3)", "DQN", 3200, "eps 0.05", "equal budget"),
    # B: same checkpoints, deterministic
    ("eval2_reinforce_v1_greedy", "REINFORCE v1 (seed 0)", "REINFORCE", 3200, "argmax", "greedy row"),
    ("eval2_reinforce_v1s1_greedy", "REINFORCE v1 (seed 1)", "REINFORCE", 3200, "argmax", "greedy row"),
    ("eval2_dqn_v1_lr1e-3_greedy", "DQN v1 (seed 0, lr 1e-3)", "DQN", 3200, "eps 0", "greedy row"),
    # C: curriculum vs scratch, both algorithms, continued from the v1 checkpoints in A
    ("eval2_reinforce_c400", "REINFORCE c400 (v1 +200 it vs maxbasepower)", "REINFORCE", 6400, "sampled", "curriculum (5b)"),
    ("eval2_dqn_lr1e-3_c400", "DQN c400 (v1 +200 it vs maxbasepower, lr 2.5e-4)", "DQN", 6400, "eps 0.05", "curriculum (5b)"),
    ("eval2_reinforce_c600", "REINFORCE c600 (c400 +200 it vs simpleheuristics)", "REINFORCE", 9600, "sampled", "curriculum (5b)"),
    ("eval2_dqn_lr1e-3_c600", "DQN c600 (c400 +200 it vs simpleheuristics, lr 2.5e-4)", "DQN", 9600, "eps 0.05", "curriculum (5b)"),
    ("eval2_reinforce_smbp", "REINFORCE smbp (200 it vs maxbasepower, from scratch)", "REINFORCE", 3200, "sampled", "scratch (5b)"),
    ("eval2_reinforce_s200", "REINFORCE s200 (200 it vs simpleheuristics, from scratch)", "REINFORCE", 3200, "sampled", "scratch (5b)"),
    ("eval2_reinforce_s400", "REINFORCE s400 (400 it vs simpleheuristics, from scratch)", "REINFORCE", 6400, "sampled", "scratch (5b)"),
    ("eval2_reinforce_s600", "REINFORCE s600 (600 it vs simpleheuristics, from scratch)", "REINFORCE", 9600, "sampled", "scratch (5b)"),
    # D: DQN training parameters: the CartPole-derived learning rate vs lr 1e-3, same budget and seed
    ("eval2_dqn_v1", "DQN lr 2.5e-4 (CartPole settings)", "DQN", 3200, "eps 0.05", "lr check"),
    ("eval2_dqn_v1_lr1e-3", "DQN lr 1e-3 (= DQN v1 in A)", "DQN", 3200, "eps 0.05", "lr check"),
    ("eval2_dqn_v1_greedy", "DQN lr 2.5e-4 (CartPole settings)", "DQN", 3200, "eps 0", "lr check"),
    ("eval2_dqn_v1_lr1e-3_greedy", "DQN lr 1e-3 (= DQN v1 in A)", "DQN", 3200, "eps 0", "lr check"),
]
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the validated default palette


def load(stem: str) -> dict[str, dict[str, float]]:
    path = RESULTS / f"{stem}.csv"
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(path.open()):
        if r["agent"] in ("random", "maxbasepower", "simpleheuristics"):
            continue
        out[r["opponent"]] = {k: float(r[k]) for k in ("wins", "n", "win_rate", "ci_lo", "ci_hi")}
    return out


def fmt(cell: dict[str, float] | None) -> str:
    return "—" if cell is None else f"{cell['win_rate']:.3f} [{cell['ci_lo']:.2f}, {cell['ci_hi']:.2f}]"


def main() -> None:
    table = []
    for stem, label, algo, budget, pol, note in ROWS:
        d = load(stem)
        if not d:
            print(f"missing: {stem}.csv")
            continue
        table.append((stem, label, algo, budget, pol, note, d))

    with (RESULTS / "comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "algorithm", "training_battles", "eval_policy", "note", "opponent",
                    "wins", "n", "win_rate", "ci_lo", "ci_hi"])
        for stem, label, algo, budget, pol, note, d in table:
            for opp in OPPS:
                if opp in d:
                    c = d[opp]
                    w.writerow([stem, label, algo, budget, pol, note, opp,
                                int(c["wins"]), int(c["n"]), c["win_rate"], c["ci_lo"], c["ci_hi"]])

    def md_table(rows: list, title: str) -> str:
        lines = [f"**{title}**", "",
                 "| checkpoint | battles | eval policy | vs random | vs maxbasepower | vs simpleheuristics |",
                 "|---|---|---|---|---|---|"]
        for _stem, label, _algo, budget, pol, _note, d in rows:
            lines.append(f"| {label} | {budget} | {pol} | " + " | ".join(fmt(d.get(o)) for o in OPPS) + " |")
        return "\n".join(lines)

    groups = [
        ("A. Equal budget, behaviour policies (the headline comparison)", lambda n: n == "equal budget"),
        ("B. Same checkpoints, deterministic policies (argmax / ε = 0)", lambda n: n == "greedy row"),
        ("C. Curriculum vs scratch: both algorithms continued from their v1 checkpoint in A (REINFORCE rows are the checkpoints of the earlier curriculum experiment, re-evaluated in the fixed evaluation format)",
         lambda n: n in ("curriculum (5b)", "scratch (5b)")),
        ("D. DQN training parameters: CartPole-derived learning rate vs lr 1e-3 (same budget, same seed)", lambda n: n == "lr check"),
    ]
    md = "\n\n".join(md_table([r for r in table if sel(r[5])], title) for title, sel in groups)
    (RESULTS / "comparison_table.md").write_text(md + "\n")
    cmp_md = RESULTS / "comparison.md"
    if cmp_md.exists():
        text = cmp_md.read_text()
        start, end = "<!-- TABLES:START -->", "<!-- TABLES:END -->"
        if start in text and end in text:
            pre, rest = text.split(start, 1)
            _, post = rest.split(end, 1)
            cmp_md.write_text(pre + start + "\n" + md + "\n" + end + post)
    print(md)

    # Plot: equal-budget rows, one panel per opponent, one axis (win rate), CI as error bars.
    eq = [(label, algo, d) for _stem, label, algo, _budget, _pol, note, d in table if note == "equal budget"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, opp in zip(axes, OPPS, strict=True):
        ys = range(len(eq))
        for y, (_label, algo, d) in zip(ys, eq, strict=True):
            c = d.get(opp)
            if c is None:
                continue
            color = ORANGE if algo == "DQN" else BLUE
            ax.errorbar(c["win_rate"], y, xerr=[[c["win_rate"] - c["ci_lo"]], [c["ci_hi"] - c["win_rate"]]],
                        fmt="o", color=color, ecolor=color, elinewidth=2, capsize=3, markersize=7)
            ax.text(min(c["ci_hi"] + 0.02, 0.98), y, f"{c['win_rate']:.2f}", va="center", fontsize=8, color="#444")
        ax.axvline(0.5, color="#bbb", lw=1, ls="--")
        ax.set_xlim(0, 1)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([label for label, _, _ in eq], fontsize=8)
        ax.set_title(f"vs {opp}", fontsize=10)
        ax.set_xlabel("win rate (200 paired battles, Wilson 95% CI)", fontsize=8)
        ax.grid(axis="x", color="#eee")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker="o", color=BLUE, lw=0, label="REINFORCE (sampled)"),
               plt.Line2D([], [], marker="o", color=ORANGE, lw=0, label="DQN (ε = 0.05)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Equal training budget (3 200 battles vs RandomPlayer), same 200 team pairs", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=150, bbox_inches="tight")
    print("wrote", RESULTS / "comparison.csv", RESULTS / "comparison.png", RESULTS / "comparison_table.md")


if __name__ == "__main__":
    main()
