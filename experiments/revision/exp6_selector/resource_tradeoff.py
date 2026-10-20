"""
resource_tradeoff.py -- Fig 8 redrawn as a resource trade: code size k vs shots kept and TVD.

Shots kept alone rewards weak detection: a larger [[k+2,k,2]] patch gives each logical qubit a
smaller share of its two detectors, so it discards fewer shots and lets more logical errors
through (on the 10-07 data the oracle choosing on shots is 2.10x shots but TVD +44%). This
figure puts accuracy beside it and shows what k buys: physical qubits.

Two stacked panels over code size k (evenly spaced; the tick shows the mean physical-qubit
saving, patches x n_code relative to naive k=2): (a) shots kept relative to naive k=2,
geometric mean; (b) mean TVD. Naive and AQORN static k at every k, the selector in its own
slot (it mixes k). Error bars: 95% bootstrap over circuits. Circuits whose TVD saturates (naive
k = 2 at >= 0.95 at every ratio: Ising, and under the logical-opt pass qugan_n39, swap_test_n25)
are dropped. k is evenly spaced because ceil(n/k) rounding on 25-40-qubit programs
makes the qubit axis uneven (k=8 often wastes more slots than k=10).

Also prints the selector re-ranked at several selection weights and the measured hindsight
frontier, on shots and TVD (the numbers behind the claim wording).

Reads Fig 8's standard CSV (every candidate simulated); no new simulation.

    PYTHONPATH=. venv/bin/python -m experiments.revision.exp6_selector.resource_tradeoff \
        [--source <resource_selection_headroom_standard[_vN].csv>] [--note "draft: ..."]
Writes results/revision/exp6/resource_tradeoff[_vN].csv and results/figures/
resource_tradeoff[_vN].{svg,pdf}.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("experiments/figures/resource_selection_headroom/resource_selection_headroom_standard.csv")
OUT = Path("results/revision/exp6")
K_VALUES = (2, 4, 6, 8, 10)
KEY = ["circuit", "ratio"]


#: Naive k = 2's TVD at or above this at every ratio means the proxy's ideal output is (near)
#: uniform and no affordable shot count resolves it, so the circuit says nothing about TVD.
SATURATED_TVD = 0.95


def load(path: Path):
    d = pd.read_csv(path)
    # Drop circuits whose TVD saturates: the Ising family always; under the logical-optimisation
    # pass also qugan_n39 and swap_test_n25, whose proxies keep Clifford-named rotations that make
    # the output uniform. Decided from the data, so a new source cannot hide one.
    naive = d[(d.series == "naive_k2")].groupby("circuit").tvd.min()
    saturated = sorted(naive[naive >= SATURATED_TVD].index)
    print(f"[resource_tradeoff] dropped (TVD saturated at every ratio): {', '.join(saturated)}")
    d = d[~d.circuit.isin(saturated)]
    d["phys"] = d.patches * d.n_code
    base = d[d.series == "naive_k2"].set_index(KEY)
    return d, base, d[d.series == "oracle_cand"]


def stats(df, base, rng, boots=2000):
    df = df.set_index(KEY).reindex(base.index)
    t = pd.DataFrame({"phys": np.log(df.phys / base.phys), "shots": np.log(df.lsr / base.lsr),
                      "tvd": df.tvd}, index=base.index)
    circuits = base.index.get_level_values(0).unique()
    draws = []                                       # resample circuits, not columns
    for _ in range(boots):
        b = pd.concat([t.loc[c] for c in rng.choice(circuits, len(circuits))])
        draws.append((np.exp(b.shots.mean()), b.tvd.mean()))
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return dict(phys=np.exp(t.phys.mean()), shots=np.exp(t.shots.mean()), tvd=t.tvd.mean(),
                shots_lo=lo[0], shots_hi=hi[0], tvd_lo=lo[1], tvd_hi=hi[1])


def table(d, base, cand) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k in K_VALUES:
        rows.append(dict(arm="naive", k=k, **stats(d[d.series == f"naive_k{k}"], base, rng)))
        aq = cand[(cand.source == "packer") & (cand.k == k)]
        rows.append(dict(arm="AQORN", k=k, **stats(aq, base, rng)))
    rows.append(dict(arm="selector", k=np.nan, **stats(d[d.series == "selector"], base, rng)))
    return pd.DataFrame(rows)


def claim_numbers(cand, base) -> pd.DataFrame:
    """Selector re-ranked at weight w, and the measured frontier (-ln LSR + lam ln TVD)."""
    groups = [cand.circuit, cand.ratio]

    def summary(label, picked):
        p = picked.set_index(KEY).reindex(base.index)
        return dict(curve=label, shots=np.exp(np.mean(np.log(p.lsr / base.lsr))),
                    tvd=p.tvd.mean(), tvd_change=p.tvd.mean() / base.tvd.mean() - 1,
                    k_picks=p.k.value_counts().to_dict())

    rows = []
    for w in (0, 10, 50, 100, 1000):
        score = cand.objective_detected + w * cand.objective_silent
        rows.append(summary(f"selector w={w}", cand.loc[score.groupby(groups).idxmin()]))
    for lam in (0, 0.25, 1, 4, 16):
        score = -np.log(cand.lsr) + lam * np.log(np.maximum(cand.tvd, 1e-3))
        rows.append(summary(f"frontier lam={lam}", cand.loc[score.groupby(groups).idxmin()]))
    return pd.DataFrame(rows)


def plot(tab: pd.DataFrame, note: str | None, out_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    sel_x = 12.2                                     # the selector's own slot (it mixes k)
    fig, (a, b) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.6), sharex=True)
    style = {"naive": dict(color="#9a9a9a", marker="v", dx=-0.18),
             "AQORN": dict(color="#0096ff", marker="o", dx=0.18)}
    for arm, st in style.items():
        s = tab[tab.arm == arm].sort_values("k")
        for ax, m in ((a, "shots"), (b, "tvd")):
            ax.errorbar(s.k + st["dx"], s[m], yerr=[s[m] - s[f"{m}_lo"], s[f"{m}_hi"] - s[m]],
                        color=st["color"], marker=st["marker"], markersize=4, capsize=1.5,
                        lw=1.2, label=f"{arm}, static $k$")
    sel = tab[tab.arm == "selector"].iloc[0]
    for ax, m in ((a, "shots"), (b, "tvd")):
        ax.errorbar([sel_x], [sel[m]], yerr=[[sel[m] - sel[f"{m}_lo"]], [sel[f"{m}_hi"] - sel[m]]],
                    color="#2b3a47", marker="*", markersize=9, capsize=1.5, ls="none",
                    label="AQORN selector (mixed $k$)", zorder=5)
        ax.axvline(11.1, color="#bbbbbb", lw=0.6, ls=":")
    a.set_ylabel("shots kept / naive $k$=2")
    a.set_yscale("log")
    a.set_yticks([1, 1.5, 2])
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    a.yaxis.set_minor_formatter(NullFormatter())
    a.axhline(1, color="#cccccc", lw=0.6, zorder=0)
    b.set_ylabel("mean TVD (lower is better)")
    phys = tab[tab.arm == "AQORN"].set_index("k").phys

    def saving(x):
        return "±0%" if abs(x - 1) < 1e-9 else f"{100 * (x - 1):+.0f}%"

    b.set_xticks(list(K_VALUES) + [sel_x])
    b.set_xticklabels([f"{k}\n{saving(phys[k])}" for k in K_VALUES] + [f"selector\n{saving(sel.phys)}"])
    b.set_xlim(1, 13.3)
    b.set_xlabel("code size $k$  /  physical qubits vs naive $k$=2")
    b.legend(fontsize=6.5, frameon=False, loc="upper left")
    if note:
        a.set_title(note, fontsize=6, color="#b00000")
    fig.tight_layout()
    paths = save_figure(fig, "resource_tradeoff", Path(__file__),
                        override=Path(out_dir) if out_dir else None, versioned=True)
    plt.close(fig)
    return paths


def plot_overlay(tab: pd.DataFrame, out_dir=None):
    """One panel (10-11): LSR against naive k=2 on the left axis and mean TVD on the right.
    Colour is the metric (LSR grey, TVD blue, matching their axes); line style is the arm
    (AQORN solid with filled markers, naive dashed with hollow ones). "Shots kept / naive" is
    the LSR ratio at a fixed raw budget, so the axis says LSR."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    sel_x = 12.2
    lsr_c, tvd_c = "#3b4c56", "#0097fd"
    fig, a = plt.subplots(figsize=(COLUMN_WIDTH, 2.55))
    b = a.twinx()
    b.spines["right"].set_visible(True)
    arms = {"AQORN": dict(ls="-", marker="o", filled=True, dx=0.16),
            "naive": dict(ls="--", marker="v", filled=False, dx=-0.16)}
    for arm, st in arms.items():
        s = tab[tab.arm == arm].sort_values("k")
        for ax, m, c in ((a, "shots", lsr_c), (b, "tvd", tvd_c)):
            ax.errorbar(s.k + st["dx"], s[m], yerr=[s[m] - s[f"{m}_lo"], s[f"{m}_hi"] - s[m]],
                        color=c, ls=st["ls"], marker=st["marker"], markersize=4, capsize=1.5,
                        lw=1.1, elinewidth=0.7, markeredgecolor=c, markeredgewidth=0.9,
                        markerfacecolor=c if st["filled"] else "white")
    sel = tab[tab.arm == "selector"].iloc[0]
    for ax, m, c, dx in ((a, "shots", lsr_c, -0.16), (b, "tvd", tvd_c, 0.16)):
        ax.errorbar([sel_x + dx], [sel[m]],
                    yerr=[[sel[m] - sel[f"{m}_lo"]], [sel[f"{m}_hi"] - sel[m]]],
                    color=c, marker="*", markersize=8, markeredgecolor=c, markeredgewidth=0.5,
                    capsize=1.5, ls="none", elinewidth=0.7, zorder=5)
    a.axvline(11.1, color="#bbbbbb", lw=0.6, ls=":")
    a.set_ylabel("LSR / naive $k$=2", color=lsr_c)
    a.tick_params(axis="y", colors=lsr_c)
    a.set_yscale("log")
    a.set_ylim(0.6, 2.4)
    a.set_yticks([0.75, 1, 1.5, 2])
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    a.yaxis.set_minor_formatter(NullFormatter())
    a.axhline(1, color="#cccccc", lw=0.6, zorder=0)
    b.set_ylabel("mean TVD", color=tvd_c)
    b.tick_params(axis="y", colors=tvd_c)
    b.spines["right"].set_color(tvd_c)
    a.spines["left"].set_color(lsr_c)
    b.set_ylim(0, 0.5)
    b.grid(False)
    phys = tab[tab.arm == "AQORN"].set_index("k").phys

    def saving(x):
        return "±0%" if abs(x - 1) < 1e-9 else f"{100 * (x - 1):+.0f}%"

    a.set_xticks(list(K_VALUES) + [sel_x])
    a.set_xticklabels([f"{k:g}\n{saving(phys[k])}" for k in K_VALUES]
                      + [f"selector\n{saving(sel.phys)}"])
    a.set_xlim(1, 13.3)
    a.set_xlabel("code size $k$  /  physical qubits vs naive $k$=2")
    handles = [Line2D([], [], color="black", ls="-", marker="o", markersize=4, label="AQORN, static $k$"),
               Line2D([], [], color="black", ls="--", marker="v", markersize=4,
                      markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.9,
                      label="naive, static $k$"),
               Line2D([], [], color="black", marker="*", markersize=7, ls="none",
                      label="AQORN selector")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8,
               handlelength=2.0, columnspacing=0.9, handletextpad=0.3, borderpad=0.0)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.92))
    paths = save_figure(fig, "resource_tradeoff_overlay", Path(__file__),
                        override=Path(out_dir) if out_dir else None, versioned=True)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--note", default=None, help="Red banner, e.g. which data this is.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overlay-from", default=None,
                        help="draw only the one-panel overlay from an aggregated table "
                             "(resource_tradeoff.csv / fig8_resource_tradeoff.csv)")
    args = parser.parse_args()
    if args.overlay_from:
        for out in plot_overlay(pd.read_csv(args.overlay_from), args.output_dir):
            print(f"[resource_tradeoff] saved {out}")
        return
    d, base, cand = load(Path(args.source))
    tab = table(d, base, cand)
    OUT.mkdir(parents=True, exist_ok=True)
    path, version = OUT / "resource_tradeoff.csv", 2
    while path.exists():
        path, version = OUT / f"resource_tradeoff_v{version}.csv", version + 1
    tab.to_csv(path, index=False)
    pd.set_option("display.width", 220)
    print(f"[resource_tradeoff] {args.source}: {base.shape[0]} columns; wrote {path}")
    print(tab.round(3).to_string(index=False))
    print(claim_numbers(cand, base).round(3).to_string(index=False))
    for out in plot(tab, args.note, args.output_dir):
        print(f"[resource_tradeoff] saved {out}")


if __name__ == "__main__":
    main()
