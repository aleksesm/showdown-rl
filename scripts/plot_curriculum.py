"""Figure: evaluation win rate vs training budget, curriculum vs scratch.

Reads results/eval_reinforce_{v1,c400,c600,s200,s400,s600,smbp}.csv (paired
teams, n=200, sampling) and writes
results/reinforce_showdown_curriculum.{csv,png}.
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
BATTLES_PER_ITER = 16
OPPONENTS = ["maxbasepower", "simpleheuristics"]
ARMS = {  # arm -> [(total training iters, eval tag)]
    "curriculum (random → maxbasepower → simpleheuristics)": [(200, "v1"), (400, "c400"), (600, "c600")],
    "scratch vs simpleheuristics": [(200, "s200"), (400, "s400"), (600, "s600")],
}
EXTRA = {"scratch vs maxbasepower": (200, "smbp")}
COLORS = {"curriculum": "#2a78d6", "scratch": "#eb6834"}  # validated categorical slots 1 and 2


def read_eval(tag: str) -> dict[str, dict[str, float]]:
    with (RESULTS / f"eval_reinforce_{tag}.csv").open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        key = r["opponent"] if r["agent"] == "reinforce" else f"{r['agent']}→{r['opponent']}"
        out[key] = {k: float(r[k]) for k in ("win_rate", "ci_lo", "ci_hi")}
    return out


def main() -> None:
    table: list[dict] = []
    for arm, points in {**ARMS, **{k: [v] for k, v in EXTRA.items()}}.items():
        for iters, tag in points:
            ev = read_eval(tag)
            for opp in ["random", *OPPONENTS]:
                table.append({"arm": arm, "checkpoint": tag, "train_iters": iters,
                              "train_battles": iters * BATTLES_PER_ITER, "opponent": opp, **ev[opp]})
    with (RESULTS / "reinforce_showdown_curriculum.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    baselines = read_eval("v1")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, opp in zip(axes, OPPONENTS, strict=True):
        for arm, points in ARMS.items():
            color = COLORS["curriculum" if arm.startswith("curriculum") else "scratch"]
            xs = [it * BATTLES_PER_ITER for it, _ in points]
            evs = [read_eval(tag)[opp] for _, tag in points]
            ys = [e["win_rate"] for e in evs]
            err = [[y - e["ci_lo"] for y, e in zip(ys, evs, strict=True)], [e["ci_hi"] - y for y, e in zip(ys, evs, strict=True)]]
            ax.errorbar(xs, ys, yerr=err, color=color, lw=2, marker="o", ms=7, capsize=3,
                        label=arm.split(" (")[0])
            ax.annotate(arm.split(" ")[0], (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color="#52514e")
        it, tag = EXTRA["scratch vs maxbasepower"]
        e = read_eval(tag)[opp]
        ax.errorbar([it * BATTLES_PER_ITER], [e["win_rate"]], yerr=[[e["win_rate"] - e["ci_lo"]], [e["ci_hi"] - e["win_rate"]]],
                    color=COLORS["scratch"], marker="s", ms=7, capsize=3, ls="none", label="scratch vs maxbasepower")
        ax.set_xlim(0, 600 * BATTLES_PER_ITER * 1.35)
        for name, key, style in [("random", f"random→{opp}", ":"), ("maxbasepower", f"maxbasepower→{opp}", "--")]:
            if key in baselines:
                ax.axhline(baselines[key]["win_rate"], color="#9a9893", lw=1, ls=style)
                ax.annotate(f"{name} baseline", (0, baselines[key]["win_rate"]), xytext=(4, -9),
                            textcoords="offset points", fontsize=8, color="#52514e", ha="left")
        ax.axhline(0.5, color="#52514e", lw=1, ls="-", alpha=0.4)
        ax.set_title(f"vs {opp}", fontsize=11)
        ax.set_xlabel("training battles (total)")
        ax.set_xticks([it * BATTLES_PER_ITER for it in (200, 400, 600)])
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("evaluation win rate")
    axes[0].set_ylim(0, 0.6)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("REINFORCE: curriculum vs from-scratch at equal battle budget  (Wilson 95% CI, n=200, paired teams)", fontsize=11)
    fig.tight_layout()
    out = RESULTS / "reinforce_showdown_curriculum.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
