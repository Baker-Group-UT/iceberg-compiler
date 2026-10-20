"""
summarize.py -- turn every revision CSV into `REVISION_RESULTS.md`, with no one watching.

Runs last in the overnight queue. Every section reads its own CSV and is isolated: a
missing or failed experiment produces a "not run / failed" line with the log to read, never
a crash, so the report exists whatever happened in the night.

Numbers are computed here from the CSVs -- nothing is typed in by hand except the facts
established before the run (marked "established before the run") and the plan item
each section answers. Suggested revision sentences are suggestions, not paper text.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import json
import math
import subprocess
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.revision.common import REPO, RESULTS, git_state

REPORT = REPO / "REVISION_RESULTS.md"
#: Overridable for testing against smoke outputs (`--results`).
ROOT = RESULTS
LOGS = REPO / "logs" / "revision"

ARM_LABELS = {
    "aqorn": "AQORN (shipped, verify on)",
    "aqorn_raw": "AQORN search, verify off",
    "aqorn_seed_only": "AQORN signed-graph seed only",
    "aqorn_descent_from_naive": "AQORN descent from naive seed",
    "naive": "naive (program order)",
    "random_budget": "random search, AQORN's evaluation budget",
    "interaction_matching": "interaction-count matching (k=2, exact)",
    "interaction_partition": "interaction-count partition (same partitioner)",
    "interaction_partition_descent": "interaction-count partition + AQORN descent",
    "hypergraph_24": "[24] stage 1 (PBC hypergraph, km1, FM)",
    "greedy_pairs": "earlier pairwise greedy (archive)",
}


def label(arm: str) -> str:
    base, _, tail = arm.partition("+")
    text = ARM_LABELS.get(base, base)
    return f"{text} + safety net" if tail else text


def read(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    return frame if not frame.empty else None


def sidecar(path: Path) -> Dict:
    side = path.with_suffix(path.suffix + ".json")
    try:
        return json.loads(side.read_text())
    except Exception:  # noqa: BLE001
        return {}


def md(frame: pd.DataFrame, digits: int = 2, index: bool = True) -> str:
    if frame is None or frame.empty:
        return "_(no rows)_"
    frame = frame.copy()
    if index:
        frame = frame.reset_index()
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            if math.isnan(value):
                return "–"
            return f"{value:.{digits}f}"
        return str(value)
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def figures(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    return sorted(str(p.relative_to(REPO) if p.is_relative_to(REPO) else p) for p in directory.glob("*.pdf"))


def geomean(values) -> float:
    values = np.asarray([v for v in values if v == v and 0 < v < math.inf], dtype=float)
    return float(np.exp(np.log(values).mean())) if values.size else math.nan


def pct(x: float, digits: int = 1) -> str:
    return "–" if x != x else f"{x:.{digits}f}%"


class Report:
    def __init__(self) -> None:
        self.sections: List[str] = []
        self.risks: List[str] = []
        self.headlines: List[str] = []
        self._ranked: List[Tuple[int, int, str]] = []
        self.skipped: List[str] = []
        self.status: Dict[str, str] = {}

    def headline(self, priority: int, text: str) -> None:
        """Lower priority number = more important; ties keep insertion order."""
        self._ranked.append((priority, len(self._ranked), text))

    def ordered_headlines(self) -> List[str]:
        ranked = [t for _, _, t in sorted(self._ranked)]
        return ranked + [h for h in self.headlines if h and h not in ranked]

    def section(self, title: str, body: Callable[[], str]) -> None:
        try:
            text = body()
        except Exception:  # noqa: BLE001
            text = f"_Section failed while summarising:_\n\n```\n{traceback.format_exc()[-1500:]}\n```"
        self.sections.append(f"## {title}\n\n{text}\n")


# ----------------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------------

OBJ_NAMES = {"work_depth": "work+12·depth (verify objective)", "work": "weighted gates",
             "two_qubit_gates": "2q gates", "depth": "depth"}


def exp7a(rep: Report, name: str = "exhaustive_translate", title: str = "7a") -> str:
    path = ROOT / "exp7" / f"{name}.csv"
    frame = read(path)
    if frame is None:
        rep.skipped.append(f"Experiment{title}: no data ({path.name} missing; see logs/revision/)")
        return f"Not run or produced no rows (`{path}` missing)."
    out = []
    arms = ["naive", "aqorn", "aqorn_raw", "aqorn_seed_only", "interaction_partition",
            "interaction_matching", "hypergraph_24", "greedy_pairs", "random_budget"]
    block = frame[frame["arm"].isin(arms)]
    counts = frame[frame["arm"] == "aqorn"].groupby("config").size()
    out.append(f"Circuits per configuration: {counts.to_dict()} "
               f"(QASMBench originals + random structured programs). Space: complete tightest "
               f"patch count (every partition × patch order × lane map) — see caveats.")
    rows = []
    for (config, arm), g in block.groupby(["config", "arm"]):
        row = {"config": config, "arm": label(arm)}
        for o in ("work_depth", "depth", "two_qubit_gates"):
            row[f"{o} regret"] = g[f"{o}_regret_tight"].mean()
        row["P(optimal) work_depth"] = 100 * g["work_depth_optimal_tight"].mean()
        row["% space better (work_depth)"] = g["work_depth_pct_better_tight"].mean()
        if "depth_regret_full" in g:
            row["depth regret, full space"] = g["depth_regret_full"].mean()
        rows.append(row)
    table = pd.DataFrame(rows).set_index(["config", "arm"])
    out.append("Mean regret above the exhaustive optimum (%), tightest space unless noted:\n")
    out.append(md(table))
    aq = frame[frame["arm"] == "aqorn"]
    raw = frame[frame["arm"] == "aqorn_raw"]
    decomposition = pd.DataFrame({
        "AQORN work_depth regret": aq.groupby("config")["work_depth_regret_tight"].mean(),
        "partition gap": aq.groupby("config")["work_depth_partition_gap_pct"].mean(),
        "order/lane gap": aq.groupby("config")["work_depth_order_lane_gap_pct"].mean(),
        "search gap (model, raw)": raw.groupby("config")["search_gap_pct"].mean(),
        "model gap": aq.groupby("config")["model_gap_pct"].mean(),
        "Spearman(model, translated)": aq.groupby("config")["spearman_model_vs_work_depth"].mean(),
        "AQORN depth regret": aq.groupby("config")["depth_regret_tight"].mean(),
        "naive depth regret": frame[frame["arm"] == "naive"].groupby("config")["depth_regret_tight"].mean(),
    })
    out.append("\nWhere AQORN's regret comes from (means, %): the partition it picks vs the best "
               "partition (*partition gap*), and its patch order/lanes vs the best for that "
               "partition (*order/lane gap*); on the model, the search's distance from the "
               "model optimum (*search gap*) and the model optimum's translated distance from "
               "the true optimum (*model gap*).\n")
    out.append(md(decomposition))
    broken = int(frame.filter(like="_beats_space_tight").any(axis=1).sum()) \
        if any(c.endswith("_beats_space_tight") for c in frame.columns) else 0
    fails = int((frame.get("replication_matches_aqorn_raw") == False).sum()) \
        if "replication_matches_aqorn_raw" in frame else 0  # noqa: E712
    out.append(f"\nSanity: arms beating the exhaustive tightest optimum: {broken} (must be 0); "
               f"`_plan_under` replication mismatches: {fails}.")
    if title == "7a":
        dmean = aq["depth_regret_tight"].mean()
        wmean = aq["work_depth_regret_tight"].mean()
        rep.exp7a_numbers = (wmean, dmean, len(aq))
        if not (ROOT / "exp7" / "exhaustive_simulate.csv").exists():
          rep.headline(3, f"7a: AQORN is {pct(wmean, 2)} above the exhaustive optimum on its "
                             f"own objective but {pct(dmean)} on depth (mean over "
                             f"{len(aq)} circuits, 4–6 qubits, k=2)")
        if dmean > 5:
            rep.risks.append(
                f"**Depth optimality (7a).** On 4–6-qubit programs AQORN's mean *depth* regret "
                f"is {pct(dmean)} above the best placement in its own (tightest) space, against "
                f"{pct(wmean, 2)} on the work+depth objective it optimises. Depth is an outcome, not "
                f"its objective (`plan.measured_cost`), so claim near-optimality on work, not depth.")
        out.append(
            "\nCaveats: (1) the archived optimum (work regret 0.71/3.16/4.82% at 4/5/6 qubits, "
            "naive 6.84/10.28/9.89%; depth ~19–22%) was taken over the generator's canonical "
            "space, which the translator is **not** invariant under (patch order and lane "
            "alignment change gates and depth), so it was not an optimum; and it translated "
            "programs without measurements, unlike Fig 6. The first 20 random circuits per "
            "configuration are the archived ones. (2) Sparser patch counts are enumerated as "
            "canonical representatives only, so `full space` numbers are bounds. (3) 5 of the "
            "plan's 20 QASMBench names cannot be used: inverseqft_n4, qec_sm_n5 and shor_n5 "
            "need classical feed-forward or reset, vqe_uccsd_n4/n6 fail to parse.")
        out.append("\n*Suggested revision text:* report regret against an exhaustively "
                   "enumerated optimum on 4–6-qubit programs, on the objective the planner "
                   "optimises, and say plainly that depth is not optimised directly.")
    return "\n".join(out)


def exp7b(rep: Report) -> str:
    path = ROOT / "exp7" / "exhaustive_simulate.csv"
    frame = read(path)
    if frame is None:
        rep.skipped.append("Experiment7b: no data")
        return "Not run or produced no rows."
    out = []
    arms = ["naive", "aqorn", "aqorn_raw", "interaction_partition", "random_budget",
            "hypergraph_24", "greedy_pairs"]
    block = frame[frame["arm"].isin(arms) & frame["in_space"].astype(bool)]
    table = block.groupby(["config", "alpha_label", "arm"]).agg(
        circuits=("circuit_id", "nunique"),
        lsr_regret_pct=("lsr_regret_pct", "mean"),
        p_best_lsr=("is_best_lsr", lambda v: 100 * v.astype(bool).mean()),
        pct_space_better_lsr=("pct_better_lsr_20k", "mean"),
        tvd_regret_abs=("tvd_regret_abs", "mean"),
        lsr_200k=("lsr_200k", "mean"),
        tvd_floor=("tvd_floor_200k", "mean"),
    )
    table.index = table.index.set_levels([table.index.levels[0], table.index.levels[1],
                                          [label(a) for a in table.index.levels[2]]])
    out.append("Simulated (Stim, distance+idle) over the complete tightest space; every "
               "placement at 20k raw shots, the top 10 by LSR plus every arm again at 200k with "
               "fresh seeds; best decided at 200k. TVD against the exact ideal distribution.\n")
    out.append(md(table, 3))
    aq = frame[frame["arm"] == "aqorn"]
    hit = aq["best_is_in_sweep_top"].astype(bool).mean() * 100
    out.append(f"\nThe 200k winner was in the 20k top-10 for {hit:.0f}% of circuits. LSR regret "
               f"is relative (%) to the best LSR at 200k; TVD regret is absolute, against the "
               f"best TVD *among the 200k finalists* (the top 10 by LSR plus every arm), not the "
               f"whole space -- arm-to-arm TVD comparisons are exact pairings, the regret's "
               f"reference point is not. A 200k LSR has a standard error of "
               f"~{aq['lsr_se_200k'].mean():.4f}; the noiseless TVD floor at the finalists' "
               f"accepted counts is ~{aq['tvd_floor_200k'].mean():.3f}.")
    part_a = ""
    if getattr(rep, "exp7a_numbers", None):
        wmean, dmean, n7 = rep.exp7a_numbers
        part_a = (f"AQORN is {pct(wmean, 2)} above the exhaustive optimum on its own objective "
                  f"(search gap 0%: the rest is cost-model error; depth {pct(dmean)}); ")
    ip = block[block["arm"] == "interaction_partition"]
    rep.headline(3, f"7 (4–6 qubits, complete tightest space): {part_a}simulated, AQORN's LSR is "
                    f"{pct(aq['lsr_regret_pct'].mean(), 1)} below the best placement and a plain "
                    f"interaction-count partition's only {pct(ip['lsr_regret_pct'].mean(), 1)} -- "
                    f"at a worse TVD")
    # Does a conventional packer sit closer to the simulated optimum than AQORN?
    means = block.groupby("arm")[["lsr_regret_pct", "tvd_regret_abs"]].mean()
    if "aqorn" in means.index:
        conv = means.loc[[a for a in means.index if a in CONVENTIONAL]]
        if not conv.empty:
            best = conv["lsr_regret_pct"].idxmin()
            if conv.loc[best, "lsr_regret_pct"] < means.loc["aqorn", "lsr_regret_pct"]:
                rep.risks.append(
                    f"**A conventional packer is closer to the simulated optimum than AQORN "
                    f"(7b).** On 4–6-qubit programs simulated on Fig 6's machine, "
                    f"{label(best)} is {pct(conv.loc[best, 'lsr_regret_pct'], 2)} below the "
                    f"best placement's LSR against AQORN's "
                    f"{pct(means.loc['aqorn', 'lsr_regret_pct'], 2)} -- while its TVD regret is "
                    f"{conv.loc[best, 'tvd_regret_abs']:.3f} against AQORN's "
                    f"{means.loc['aqorn', 'tvd_regret_abs']:.3f}. Keeping more shots and being "
                    f"more accurate are different objectives here; say which one AQORN optimises.")
                out.append(
                    f"\nSurvival and accuracy disagree: {label(best)} keeps the most shots "
                    f"(LSR regret {pct(conv.loc[best, 'lsr_regret_pct'], 2)}) but its accepted "
                    f"output is further from the ideal distribution (TVD regret "
                    f"{conv.loc[best, 'tvd_regret_abs']:.3f} vs AQORN's "
                    f"{means.loc['aqorn', 'tvd_regret_abs']:.3f}). Co-locating interacting qubits "
                    f"cuts movement (fewer detected faults) but lets more faults land as "
                    f"undetectable weight-2 errors inside one patch -- worth checking per circuit.")
    return "\n".join(out)


def exp7c(rep: Report) -> str:
    path = ROOT / "exp7" / "compile_time.csv"
    frame = read(path)
    if frame is None:
        rep.skipped.append("Experiment7c: no data")
        return "Not run."
    out = []
    side = sidecar(path)
    for stage, g in frame.groupby("stage"):
        g = g[g["seconds_median"].notna()]
        slow = g.sort_values("seconds_median", ascending=False).head(5)
        out.append(f"**{stage}** ({len(g)} circuits): median {g['seconds_median'].median():.2f} s, "
                   f"mean {g['seconds_median'].mean():.2f} s, max {g['seconds_median'].max():.1f} s; "
                   f">10 s on {int((g['seconds_median'] > 10).sum())}, >60 s on "
                   f"{int((g['seconds_median'] > 60).sum())}. Slowest: " +
                   ", ".join(f"{r.name} ({int(r.num_qubits)}q) {r.seconds_median:.1f} s"
                             for r in slow.itertuples()))
    buckets = pd.cut(frame["num_qubits"], [0, 10, 25, 50, 100, 200])
    table = frame.pivot_table(index=buckets, columns="stage", values="seconds_median",
                              aggfunc="median", observed=True)
    out.append("\nMedian compile time (s) by program width:\n")
    out.append(md(table, 2))
    fig6 = frame[(frame["source"] == "fig6") & (frame["stage"] == "plan_packing")]
    if not fig6.empty and fig6["seconds_median"].max() > 30:
        worst = fig6.sort_values("seconds_median").iloc[-1]
        rep.risks.append(
            f"**\"Every benchmark compiles in seconds\" (V-C).** The Fig 6 packing call "
            f"(`plan_packing`, n=4, auto commutation, verify, Fig 6's noise profile) takes "
            f"{worst['seconds_median']:.0f} s on {worst['name']} and over 60 s on "
            f"{int((fig6['seconds_median'] > 60).sum())} of {len(fig6)} Fig 6 circuits "
            f"(single core, {side.get('extra', {}).get('cpu', '?')}). The descent re-prices "
            f"movement over all patch pairs for every candidate move.")
    out.append("\nSingle process; runs over 30 s were not repeated (see `repeats`). Machine: "
               f"{side.get('extra', {}).get('cpu', '?')}.")
    out.append("\nFigures: " + ", ".join(f"`{f}`" for f in figures(ROOT / "exp7")))
    return "\n".join(out)


def _reductions(frame: pd.DataFrame, metric: str, n_code: int) -> pd.Series:
    ok = frame[(frame["status"] == "ok") & (frame["n_code"] == n_code)]
    base = ok[ok["arm"] == "fig6_baseline"].set_index("name")[metric]
    out = {}
    for arm, g in ok.groupby("arm"):
        g = g.set_index("name")[metric]
        both = pd.concat([base, g], axis=1, keys=["b", "a"]).dropna()
        both = both[both["b"].abs() > 1e-12]  # e.g. no movement at all on one patch
        out[arm] = (100 * (both["b"] - both["a"]) / both["b"]).replace(
            [np.inf, -np.inf], np.nan).mean()
    return pd.Series(out)


def exp1a(rep: Report) -> str:
    path = ROOT / "exp1" / "packer_static_o1.csv"
    frame = read(path)
    if frame is None:
        rep.skipped.append("Experiment1a: no data")
        return "Not run."
    out = []
    for n_code in sorted(frame["n_code"].unique()):
        cols = {}
        for metric, name in (("depth", "depth"), ("two_qubit_gates", "2q"),
                             ("work", "weighted gates"), ("E_total", "expected errors"),
                             ("E_movement", "movement errors"), ("E_idle", "idle errors")):
            if metric in frame:
                cols[f"{name} red. %"] = _reductions(frame, metric, n_code)
        table = pd.DataFrame(cols)
        table.index = [label(a) if not a.startswith("fig6_") else a for a in table.index]
        out.append(f"\n**n_code = {n_code}** — mean % reduction against Fig 6's baseline "
                   f"(naive, both passes off); every arm under AQORN's commutation decision "
                   f"with merging on; expected errors from the proxy's schedule at NEUTRAL_ATOM "
                   f"rates (alpha = 1):\n")
        key = "expected errors red. %" if "expected errors red. %" in table else table.columns[0]
        out.append(md(table.sort_values(key, ascending=False)))
    e_red = _reductions(frame, "E_total", 4) if "E_total" in frame else pd.Series(dtype=float)
    d_red = _reductions(frame, "depth", 4)
    if "aqorn" in e_red:
        conventional = {a: v for a, v in e_red.items() if a.split("+")[0] in CONVENTIONAL}
        best = max(conventional, key=conventional.get) if conventional else None
        rep.packers_static = (e_red["aqorn"], e_red.get("naive", math.nan),
                              label(best) if best else "–", conventional.get(best, math.nan))
        if not (ROOT / "exp1" / "packer_simulated.csv").exists():
          rep.headline(2, f"1a: expected errors (movement/idle-aware) fall {pct(e_red['aqorn'])} with "
                        f"AQORN vs {pct(e_red.get('naive', math.nan))} with naive packing (both "
                        f"with H commutation + merging); best conventional packer "
                        f"{label(best) if best else '–'} {pct(conventional.get(best, math.nan))}"
                        f" (n=4, 71 Fig 6 circuits, proxies at NEUTRAL_ATOM rates)")
        if best is not None:
            ok4 = frame[(frame["status"] == "ok") & (frame["n_code"] == 4)]
            piv = ok4.pivot_table(index="name", columns="arm", values="E_total")
            a_, b_ = piv["aqorn"], piv[best]
            wins = int((a_ < b_ * (1 - 1e-9)).sum())
            losses = int((b_ < a_ * (1 - 1e-9)).sum())
            margin = e_red["aqorn"] - conventional[best]
            out.append(f"\nAQORN against the best conventional packer ({label(best)}) on expected "
                       f"errors, n=4, paired: AQORN lower on {wins} circuits, higher on {losses}, "
                       f"equal on {len(piv) - wins - losses}; mean margin {margin:.2f} points.")
            if margin < 1.0:
                rep.risks.append(
                    f"**AQORN's margin over conventional partitioners is small (1a).** On expected "
                    f"detectable errors (n=4, 71 circuits), {label(best)} reduces "
                    f"{pct(conventional[best])} against AQORN's {pct(e_red['aqorn'])} "
                    f"(+H+merge naive: {pct(e_red.get('naive', math.nan))}); AQORN is lower on "
                    f"{wins} circuits and higher on {losses}. The packing contribution is real but "
                    f"most of it is available to a good conventional partitioner.")
        if "random_budget" in d_red and d_red["random_budget"] > d_red.get("aqorn", math.inf):
            out.append(
                f"\nNote on depth: random search at AQORN's budget reduces depth more "
                f"({pct(d_red['random_budget'])} vs {pct(d_red['aqorn'])}) by scattering "
                f"interacting qubits, which raises movement error sharply; on expected errors it "
                f"reduces {pct(e_red.get('random_budget', math.nan))} against AQORN's "
                f"{pct(e_red['aqorn'])}. Depth alone is the wrong figure of merit here.")
    ok = frame[(frame["status"] == "ok") & (frame["n_code"] == 4)]
    one = ok.drop_duplicates("name")
    fb = int(one.get("note_aqorn_fell_back", pd.Series(dtype=bool)).astype(bool).sum())
    crit = one["verify_criterion"].value_counts().to_dict() if "verify_criterion" in one else {}
    out.append(f"\nAt n=4, AQORN's `verify` kept naive on {fb} of {len(one)} circuits. Verify "
               f"criterion per circuit: {crit} — `gates_depth_fallback` means the original "
               f"carries rz/rx, `measured_noise_split` raised inside `plan.measured_cost`, and "
               f"the check silently priced gates + depth instead of expected error.")
    if crit.get("gates_depth_fallback"):
        rep.risks.append(
            f"**Noise-aware verify is silently gate-counting on {crit['gates_depth_fallback']} of "
            f"{len(one)} Fig 6 circuits.** Fig 6 plans on the originals; any circuit with rz/rx "
            f"makes `measured_noise_split` raise (Stim's scanner rejects rotations), and "
            f"`plan.measured_cost` catches it and returns gates + 12·depth. So for those "
            f"circuits the commutation choice and the never-worse-than-naive check were made "
            f"on gate counts, not on the movement/idle-aware objective the code describes. "
            f"Library untouched; Experiment1/2 reproduce the pipeline as submitted.")
    # Sanity (plan section 5): AQORN never worse than naive on the criterion verify used.
    # Clifford-only originals: expected error (equal to the proxy's E_total, same rates and
    # scalars); originals with rotations: the silent fallback, weighted gates + 12 depth.
    violations = []
    for n_code, g in frame[frame["status"] == "ok"].groupby("n_code"):
        piv = g.pivot_table(index="name", columns="arm", values=["E_total", "work_depth"])
        crit = g.drop_duplicates("name").set_index("name")["verify_criterion"]
        for name in piv.index:
            column = "E_total" if crit.get(name) == "expected_error" else "work_depth"
            a_, n_ = piv[column].get("aqorn", {}).get(name), piv[column].get("naive", {}).get(name)
            if a_ is not None and n_ is not None and a_ > n_ * (1 + 1e-9):
                violations.append(f"{name} (n={n_code}, {column} {a_:.4g} > naive {n_:.4g})")
    out.append(f"\nSanity: AQORN worse than naive on its own verify criterion in "
               f"{len(violations)} (circuit, n) cases" + (": " + "; ".join(violations[:8])
                                                         if violations else " (must be 0)."))
    out.append("\nFigures/tables: per-arm placements are in the CSV (`placement`), reused by 1c.")
    out.append("\n*Suggested revision text:* a table of packer × {depth, 2q, expected error} "
               "under identical commutation and merging, including interaction-count "
               "partitioning, exact interaction matching, [24]'s hypergraph partition and a "
               "random search with AQORN's evaluation budget.")
    return "\n".join(out)


def exp1b(rep: Report) -> str:
    o1 = read(ROOT / "exp1" / "packer_static_o1.csv")
    o3 = read(ROOT / "exp1" / "packer_static_o3.csv")
    if o3 is None:
        rep.skipped.append("Experiment1b: no data")
        return "Not run."
    out = []
    rows = {}
    for label_, frame in (("O1 inputs (as submitted)", o1), ("O3 inputs", o3)):
        if frame is None:
            continue
        r = {}
        for metric in ("depth", "two_qubit_gates", "one_qubit_gates"):
            red = _reductions(frame, metric, 4)
            r[f"{metric}: +H"] = red.get("fig6_h_only", math.nan)
            r[f"{metric}: +H+merge"] = red.get("fig6_h_merge", math.nan)
            r[f"{metric}: AQORN"] = red.get("aqorn", math.nan)
        ok = frame[(frame["status"] == "ok") & (frame["n_code"] == 4)].drop_duplicates("name")
        r["logical depth (mean)"] = ok["logical_depth"].mean()
        r["logical gates (mean)"] = ok["logical_gates"].mean()
        rows[label_] = r
    table = pd.DataFrame(rows)
    out.append("Cumulative mean % reduction against the baseline translation of the *same* "
               "input (n=4). If AQORN's reductions survive O3 pre-optimisation, they are not "
               "logical-level savings a standard optimiser would have found.\n")
    out.append(md(table))
    if o1 is not None:
        a = _reductions(o1, "depth", 4).get("aqorn", math.nan)
        b = _reductions(o3, "depth", 4).get("aqorn", math.nan)
        rep.headline(4, f"1b: AQORN's depth reduction is {pct(a)} on O1 inputs and {pct(b)} "
                             f"on O3-preoptimised inputs")
    out.append("\n*Suggested revision text:* state that inputs were level-1 transpiled "
               "and report the reductions on level-3 inputs alongside.")
    return "\n".join(out)


#: Conventional packers -- what AQORN is compared against. The
#: hybrid (interaction seed + AQORN's descent) and AQORN's own ablations are reported but are
#: not "conventional", so they never trigger the claims-at-risk line on their own.
CONVENTIONAL = ("random_budget", "interaction_matching", "interaction_partition",
                "hypergraph_24", "greedy_pairs")
HYBRID = ("interaction_partition_descent",)


def exp1c(rep: Report) -> str:
    path = ROOT / "exp1" / "packer_simulated.csv"
    gate = ROOT / "exp1" / "_gate" / "GATE.txt"
    out = []
    if gate.exists():
        out.append(f"Sanity gate (Fig 6's baseline, +merge and AQORN samples re-drawn with Fig "
                   f"6's seeds): **{gate.read_text().splitlines()[0]}**")
    frame = read(path)
    if frame is None:
        rep.skipped.append("Experiment1c: no data" + (" (sanity gate failed)" if gate.exists()
                                                       and "FAIL" in gate.read_text() else ""))
        return "\n".join(out + ["No simulated rows."])
    side = sidecar(path)
    naive_rows = frame[frame["series"] == "naive"].set_index("name")
    resolved = set(naive_rows.index[(naive_rows["tvd_floor"] < 0.1) &
                                    (naive_rows["accepted_shots"] >= 1000) &
                                    (naive_rows["fig6_baseline_tvd"] <= 0.95)])
    aqorn = frame[frame["series"] == "aqorn"].set_index("name")
    same_as_naive = int((aqorn["placement"] == naive_rows["placement"].reindex(aqorn.index)).sum())
    rows = {}
    for series, g in frame.groupby("series"):
        g = g.set_index("name")
        both = pd.concat([aqorn["lsr"], g["lsr"], aqorn["raw_shots"]], axis=1,
                         keys=["a", "s", "n"]).dropna()
        se = np.sqrt((both["a"] * (1 - both["a"]) + both["s"] * (1 - both["s"])) / both["n"])
        r = g.loc[[n for n in g.index if n in resolved]]
        eq = 100 * (r["naive_tvd_at_matched_shots"] - r["tvd_eq_vs_naive"]) / \
            r["naive_tvd_at_matched_shots"]
        rows[series] = {
            "circuits": len(g),
            "geomean LSR / AQORN": geomean((both["s"] / both["a"]).values),
            "beats AQORN >2σ": int(((both["s"] - both["a"]) > 2 * se).sum()),
            "loses to AQORN >2σ": int(((both["a"] - both["s"]) > 2 * se).sum()),
            "LSR % vs naive": g["lsr_impr_vs_naive_pct"].mean(),
            "TVD red. % vs naive (resolved)": r["tvd_red_vs_naive_pct"].mean(),
            "TVD red. % vs naive, equal shots (resolved)": eq.mean(),
        }
    table = pd.DataFrame(rows).T
    table.index = [label(a) if not a.startswith("fig6_") else a for a in table.index]
    static = read(ROOT / "exp1" / "packer_static_o1.csv")
    fell = None
    if static is not None and "note_aqorn_fell_back" in static:
        fell = int(static[static["n_code"] == 4].drop_duplicates("name")[
            "note_aqorn_fell_back"].astype(bool).sum())
    why = (f"verify fell back on {fell}; on the other {same_as_naive - fell} the search itself "
           f"returned program order" if fell is not None else "mostly verify falling back")
    out.append(f"\n{frame['name'].nunique()} circuits at Fig 6's per-circuit alpha and shots; "
               f"AQORN's placement equals naive on {same_as_naive} of them ({why}). "
               f"TVD columns use the {len(resolved)} *resolved* circuits (naive's noiseless floor "
               f"< 0.1, ≥ 1k accepted shots, Fig 6 baseline TVD ≤ 0.95). Wins/losses count "
               f"circuits where the LSR difference exceeds twice its binomial standard error.\n")
    out.append(md(table.sort_values("geomean LSR / AQORN", ascending=False)))

    def best_of(names):
        keys = [k for k in rows if k.split("+")[0] in names]
        if not keys:
            return None, math.nan
        k = max(keys, key=lambda key: rows[key]["geomean LSR / AQORN"]
                if rows[key]["geomean LSR / AQORN"] == rows[key]["geomean LSR / AQORN"] else -1)
        return k, rows[k]["geomean LSR / AQORN"]

    best, value = best_of(CONVENTIONAL)
    hybrid, hvalue = best_of(HYBRID)
    if best is not None:
        static_part = ""
        if getattr(rep, "packers_static", None):
            ea, en, eb_label, eb = rep.packers_static
            static_part = (f"expected errors fall {ea:.1f}% with AQORN, {eb:.1f}% with the best "
                           f"conventional packer ({eb_label}), {en:.1f}% with naive; ")
        rep.headline(2, f"Packers (1a/1c, 71 Fig 6 circuits, same passes): {static_part}in "
                        f"simulation {label(best)} keeps {value:.3f}× AQORN's shots but its TVD is "
                        f"{rows[best]['TVD red. % vs naive (resolved)']:+.1f}% vs naive where "
                        f"AQORN's is {rows['aqorn']['TVD red. % vs naive (resolved)']:+.1f}% "
                        f"(positive = lower TVD)")
        if value >= 0.995:
            tvd_best = rows[best]["TVD red. % vs naive (resolved)"]
            tvd_aqorn = rows["aqorn"]["TVD red. % vs naive (resolved)"]
            rep.risks.append(
                f"**A conventional packer ties or beats AQORN on simulated LSR (1c) -- but not on "
                f"TVD.** {label(best)} keeps {value:.3f}× AQORN's shots on geometric mean over the "
                f"Fig 6 corpus, identical passes and noise (beats AQORN by >2σ on "
                f"{rows[best]['beats AQORN >2σ']} circuits, loses on "
                f"{rows[best]['loses to AQORN >2σ']}), yet its TVD on the {len(resolved)} resolved "
                f"circuits is {tvd_best:+.1f}% against naive where AQORN's is {tvd_aqorn:+.1f}% "
                f"(positive = lower TVD). Co-locating interacting qubits keeps shots and loses "
                f"accuracy (see 7b); the paper's LSR claim needs the TVD beside it, and \"packing "
                f"improves both\" holds for AQORN, not for interaction-count packing. AQORN's "
                f"placement equals naive on {same_as_naive} circuits.")
    if hybrid is not None and hvalue == hvalue and hvalue >= 0.995:
        rep.risks.append(
            f"**Interaction-count seed + AQORN's descent ties AQORN (1c):** {hvalue:.3f}× "
            f"AQORN's LSR — the signed-graph seed adds little beyond the descent.")
    out.append(f"\nStatus: {side.get('status', '?')}. Experiment4 statistics (bootstrap CI, "
               f"floor, equal-shot TVD) are per row in `{path}`.")
    return "\n".join(out)


def exp1d(rep: Report) -> str:
    s = read(ROOT / "exp1" / "packer_static_o1.csv")
    c = read(ROOT / "exp1" / "packer_simulated.csv")
    if s is None:
        return "Needs Experiment1a."
    ok = s[(s["status"] == "ok") & (s["n_code"] == 4)]
    piv = ok.pivot_table(index=["family", "name"], columns="arm", values="depth")
    fam = pd.DataFrame({
        "circuits": piv.groupby(level="family").size(),
        "H commutation % (depth)": (100 * (piv["fig6_baseline"] - piv["fig6_h_only"]) /
                                    piv["fig6_baseline"]).groupby(level="family").mean(),
        "+ merging % (depth)": (100 * (piv["fig6_h_only"] - piv["fig6_h_merge"]) /
                                piv["fig6_baseline"]).groupby(level="family").mean(),
        "+ packing % (depth)": (100 * (piv["fig6_h_merge"] - piv["aqorn"]) /
                                piv["fig6_baseline"]).groupby(level="family").mean(),
    })
    if c is not None:
        lsr = c.pivot_table(index=["family", "name"], columns="series", values="lsr")
        fam["+ packing, LSR % (sim)"] = (100 * (lsr["aqorn"] - lsr["naive"]) /
                                         lsr["naive"]).groupby(level="family").mean()
    out = ["Each pass's contribution by circuit family (n=4; depth contributions are "
           "percentage points of the baseline, so the three columns add up):\n",
           md(fam.sort_values("circuits", ascending=False))]
    return "\n".join(out)


def exp2a(rep: Report) -> str:
    path = ROOT / "exp2" / "fig6_calibration_table.csv"
    side = sidecar(path).get("extra", {})
    if not side:
        return "Not run."
    rep.risks.append(
        f"**Calibration target (established before the run).** Fig 6 calibrates the baseline "
        f"to LSR **0.06** (`BenchmarkSettings.target_selection_rate`; measured mean "
        f"{side['baseline_lsr_mean']:.3f}), but the paper text says **~0.2**."
        f" The per-circuit alpha spans {side['alpha_min']:.2f}–{side['alpha_max']:.1f} "
        f"(median {side['alpha_median']:.2f}), a {side['alpha_spread_factor']:.0f}× spread in "
        f"physical error rates (p2 {side['p2_min']:.1e}–{side['p2_max']:.1e}).")
    return (f"`{path}`: per circuit alpha and the physical rates it implies "
            f"(p1, p2, readout, move error ×alpha; T1, T2 ÷alpha), shots and baseline LSR. "
            f"alpha {side['alpha_min']:.2f}–{side['alpha_max']:.2f} (median {side['alpha_median']:.2f}, "
            f"mean {side['alpha_mean']:.2f}); {side['circuits_alpha_above_1']} circuits above the "
            f"NEUTRAL_ATOM machine's noise, {side['circuits_alpha_below_1']} below. Baseline LSR "
            f"{side['baseline_lsr_min']:.3f}–{side['baseline_lsr_max']:.3f}.\n\n*Suggested revision "
            f"text:* put this table in an appendix and correct \"~0.2\" to 0.06.")


def exp2b(rep: Report) -> str:
    path = ROOT / "exp2" / "fixed_noise_sweep.csv"
    frame = read(path)
    if frame is None:
        rep.skipped.append("Experiment2b: no data")
        return "Not run."
    out = []
    rows = []
    for alpha, g in frame.groupby("alpha"):
        p = g.pivot_table(index="name", columns="series",
                          values=["lsr", "accepted_shots", "tvd", "tvd_floor",
                                  "tvd_equal_vs_baseline_pair", "tvd_ci_lo", "tvd_ci_hi"])
        lsr, acc = p["lsr"], p["accepted_shots"]
        infeasible = acc["baseline"] < 1000
        resolved = (p["tvd_floor"]["baseline"] < 0.1) & ~infeasible & (acc["aqorn"] >= 1000)
        tvd = p["tvd"][resolved]
        eq = p["tvd_equal_vs_baseline_pair"][resolved] if "tvd_equal_vs_baseline_pair" in p else None
        ci_sep = ((p["tvd_ci_hi"]["aqorn"] < p["tvd_ci_lo"]["baseline"])[resolved]).sum()
        ci_worse = ((p["tvd_ci_lo"]["aqorn"] > p["tvd_ci_hi"]["baseline"])[resolved]).sum()
        row = {
            "alpha": alpha, "circuits": len(lsr),
            "baseline LSR (mean)": lsr["baseline"].mean(), "AQORN LSR (mean)": lsr["aqorn"].mean(),
            "geomean LSR ratio": geomean((lsr["aqorn"] / lsr["baseline"]).values),
            "% circuits AQORN ≥ baseline": 100 * (lsr["aqorn"] >= lsr["baseline"]).mean(),
            "baseline infeasible (<1k acc.)": int(infeasible.sum()),
            "rescued by AQORN": int((infeasible & (acc["aqorn"] >= 1000)).sum()),
            "TVD-resolved circuits": int(resolved.sum()),
            "TVD red. % (raw shots)": (100 * (tvd["baseline"] - tvd["aqorn"]) / tvd["baseline"]).mean(),
            "TVD red. % (equal accepted)": (100 * (eq["baseline"] - eq["aqorn"]) / eq["baseline"]).mean()
            if eq is not None else math.nan,
            "AQORN TVD CI wholly below baseline's": int(ci_sep),
            "AQORN TVD CI wholly above baseline's": int(ci_worse),
        }
        rows.append(row)
    table = pd.DataFrame(rows).set_index("alpha")
    out.append("One machine for every circuit (alpha the same everywhere; alpha = 1 is the "
               "NEUTRAL_ATOM model), 200k raw shots per point, Fig 6's proxies, AQORN's placement "
               "as Fig 6 computed it. *TVD-resolved*: the baseline's noiseless finite-sample floor "
               "is below 0.1 and both arms keep ≥ 1k shots.\n")
    out.append(md(table.T, 3, index=True))
    one = frame[frame["alpha"] == 1.0]
    if not one.empty:
        stack = one.groupby("series")[["lsr", "tvd", "accepted_shots"]].mean()
        out.append("\nalpha = 1, all four Fig 6 arms (means over circuits):\n")
        out.append(md(stack, 4))
        r = rows[[x["alpha"] for x in rows].index(1.0)] if 1.0 in [x["alpha"] for x in rows] else None
        if r:
            rep.headline(1,
                f"2b: on the NEUTRAL_ATOM machine (alpha=1) AQORN keeps {r['geomean LSR ratio']:.2f}× "
                f"the baseline's shots (geomean), ≥ baseline on {r['% circuits AQORN ≥ baseline']:.0f}% "
                f"of circuits; baseline infeasible on {r['baseline infeasible (<1k acc.)']}, AQORN "
                f"rescues {r['rescued by AQORN']}")
            if r["geomean LSR ratio"] < 1.05:
                rep.risks.append(f"**Fixed-noise LSR gain is small at alpha = 1** "
                                 f"({r['geomean LSR ratio']:.3f}× geomean).")
            if not math.isnan(r["TVD red. % (equal accepted)"]) and \
                    r["TVD red. % (equal accepted)"] < 0.5 * r["TVD red. % (raw shots)"]:
                rep.risks.append(
                    f"**TVD gain is mostly accepted-shot count.** At alpha = 1 the TVD "
                    f"reduction falls from {pct(r['TVD red. % (raw shots)'])} at a fixed raw budget "
                    f"to {pct(r['TVD red. % (equal accepted)'])} at equal accepted shots.")
    out.append("\nPer-point bootstrap 95% CIs, the noiseless floor at each accepted count, raw "
               "and accepted shots, and equal-accepted-shot TVDs are columns of "
               f"`{path}`; histograms (≤ 12 qubits) in "
               "`results/revision/exp2/fixed_noise_histograms.jsonl`.")
    out.append("\n*Suggested revision text:* show absolute LSR/TVD with CIs on one "
               "fixed machine, and the equal-accepted-shot TVD beside the fixed-raw-budget one.")
    return "\n".join(out)


def exp3a(rep: Report) -> str:
    frame = read(ROOT / "exp3" / "proxy_static.csv")
    if frame is None:
        rep.skipped.append("Experiment3a: no data")
        return "Not run."
    out = []
    rows = []
    for arm, g in frame.groupby("arm"):
        with_rot = g[g["rotations"] > 0]
        rows.append({
            "arm": arm, "circuits": len(g),
            "identical structure": int(g["identical_structure"].sum()),
            "proxy 2q discount % (circuits with rotations)":
                (100 * (with_rot["original_two_qubit_gates"] - with_rot["proxy_two_qubit_gates"]) /
                 with_rot["original_two_qubit_gates"]).mean(),
            "proxy depth change %": (100 * (g["proxy_depth"] - g["original_depth"]) /
                                     g["original_depth"]).mean(),
        })
    out.append(md(pd.DataFrame(rows).set_index("arm")))
    piv_o = frame.pivot_table(index="name", columns="arm", values="original_depth")
    piv_p = frame.pivot_table(index="name", columns="arm", values="proxy_depth")
    red_o = (100 * (piv_o["baseline"] - piv_o["aqorn"]) / piv_o["baseline"]).mean()
    red_p = (100 * (piv_p["baseline"] - piv_p["aqorn"]) / piv_p["baseline"]).mean()
    one = frame.drop_duplicates("name")
    same_pl = int(one["same_placement_when_planned_on_proxy"].sum())
    same_cm = int((one["commute_planned_on_original"] == one["commute_planned_on_proxy"]).sum())
    out.append(f"\nAQORN's mean depth reduction: {pct(red_o)} counted on originals, {pct(red_p)} "
               f"on proxies. Planning on the proxy instead: same placement on {same_pl}/{len(one)}, "
               f"same commutation on {same_cm}/{len(one)}.")
    with_rot = frame[(frame["rotations"] > 0) & (frame["arm"] == "aqorn")]
    disc = (100 * (with_rot["original_two_qubit_gates"] - with_rot["proxy_two_qubit_gates"]) /
            with_rot["original_two_qubit_gates"]).mean()
    rep.risks.append(
        f"**Proxies are cheaper circuits than the originals (3a).** An encoded Rz/Rx costs "
        f"(1q, 2q) = (1, 2) and the S that replaces it (2, 1) (`iceberg.costs`), so the proxy "
        f"Fig 6 simulates has on average {pct(disc)} fewer two-qubit gates than the original "
        f"whose gates Fig 6 counts ({len(with_rot)} circuits with rotations). Absolute "
        f"LSR/TVD in Fig 6 therefore describe a cheaper circuit; 3b measures whether the "
        f"*improvements* transfer.")
    return "\n".join(out)


def exp3b(rep: Report) -> str:
    from experiments.revision.exp3_proxy.proxy_simulated import improvements

    frame = read(ROOT / "exp3" / "proxy_simulated.csv")
    if frame is None:
        rep.skipped.append("Experiment3b: no data")
        return "Not run."
    from scipy.stats import spearmanr

    out = []
    imp = improvements(frame)
    aer = imp[imp["backend"] == "aer"]
    rows = []
    for alpha_label, g in aer.groupby("alpha_label"):
        wide = g.pivot_table(index="name", columns="version",
                             values=["lsr_impr_pct", "tvd_exact_red_pct",
                                     "lsr_impr_vs_hmerge_pct", "tvd_exact_red_vs_hmerge_pct"])
        for metric in ("lsr_impr_pct", "tvd_exact_red_pct", "lsr_impr_vs_hmerge_pct",
                       "tvd_exact_red_vs_hmerge_pct"):
            o, p = wide[metric]["original"], wide[metric]["proxy"]
            rows.append({"alpha": alpha_label, "metric": metric, "circuits": len(o),
                         "original mean %": o.mean(), "proxy mean %": p.mean(),
                         "sign agreement %": 100 * (np.sign(o) == np.sign(p)).mean(),
                         "Spearman": spearmanr(o, p).statistic if len(o) > 2 else math.nan})
    table = pd.DataFrame(rows).set_index(["alpha", "metric"])
    out.append("Aer density matrix (exact), gate noise only (no movement, no idle channel on "
               "Aer); 4-qubit originals with non-Clifford rotations plus random Clifford+Rz "
               "programs. Improvement of AQORN against the baseline (and against naive with "
               "both passes), measured on the original and on its proxy:\n")
    out.append(md(table))
    if ("calibrated", "tvd_exact_red_vs_hmerge_pct") in table.index:
        r = table.loc[("calibrated", "tvd_exact_red_vs_hmerge_pct")]
        out.append(f"\nThe full stack transfers from proxy to original; the packing-only increment "
                   f"(AQORN against naive with both passes) is small on 4-qubit circuits and "
                   f"transfers poorly: its TVD reduction is {r['original mean %']:.1f}% on originals "
                   f"vs {r['proxy mean %']:.1f}% on proxies, sign agreement "
                   f"{r['sign agreement %']:.0f}%, Spearman {r['Spearman']:.2f} at calibrated noise.")
    stim = frame[frame["backend"] == "stim"].set_index(["name", "alpha_label", "arm"])["tvd"]
    aerp = frame[(frame["backend"] == "aer") & (frame["version"] == "proxy")].set_index(
        ["name", "alpha_label", "arm"])["tvd"]
    both = pd.concat([stim.rename("stim"), aerp.rename("aer")], axis=1).dropna()
    if not both.empty:
        out.append(f"\nStim vs Aer on the proxy (distance and idle off): mean |ΔTVD| "
                   f"{(both['stim'] - both['aer']).abs().mean():.4f}, max "
                   f"{(both['stim'] - both['aer']).abs().max():.4f} ({len(both)} points).")
    calibrated = table.xs("calibrated", level="alpha") if "calibrated" in table.index.get_level_values(0) else None
    if calibrated is not None and "tvd_exact_red_pct" in calibrated.index:
        agree = calibrated.loc["tvd_exact_red_pct", "sign agreement %"]
        rep.headline(2, f"3b: AQORN-vs-baseline TVD improvements agree in sign between "
                             f"originals and proxies on {agree:.0f}% of small circuits")
        if agree < 80:
            rep.risks.append(f"**Proxy/original disagreement (3b).** The sign of AQORN's TVD "
                             f"improvement agrees between original and proxy on only "
                             f"{agree:.0f}% of the small circuits.")
    return "\n".join(out)


def exp6(rep: Report) -> str:
    out = []
    latest = None
    for suffix, title in (("", "6a — Fig 8's data, naive static ladder"),
                          ("_with_aqorn", "6b — with AQORN's planner at every static k")):
        path = ROOT / "exp6" / f"fig8_selector_comparisons{suffix}.csv"
        frame = read(path)
        if frame is None:
            out.append(f"**{title}:** not available.")
            continue
        frame = frame.rename(columns={frame.columns[0]: "comparison"}).set_index("comparison")
        out.append(f"**{title}** (selector LSR / comparator LSR, paired per (circuit, ratio), "
                   f"geometric mean with log-SE; share of rows the selector wins):\n")
        out.append(md(frame[["geomean_ratio", "log_se", "share_a_ge_b", "rows"]], 3))
        latest = frame
    if latest is not None:
        comparators = (
            ("selector_vs_naive_k_best_global", "best single k, naive"),
            ("selector_vs_naive_k_best_per_ratio", "best k per noise ratio, naive"),
            ("selector_vs_naive_k_best_per_point", "best k per (circuit, ratio), naive"),
            ("selector_vs_aqorn_k_best_global", "best single k, AQORN placement"),
            ("selector_vs_aqorn_k_best_per_ratio", "best k per noise ratio, AQORN placement"),
            ("selector_vs_aqorn_k_best_per_point", "best k per (circuit, ratio), AQORN placement"),
        )
        parts, failing = [], []
        for key, nice in comparators:
            if key not in latest.index:
                continue
            r = latest.loc[key]
            ratio, se = float(r["geomean_ratio"]), float(r["log_se"])
            # Significant win: the log ratio clears two standard errors.
            significant = ratio > 0 and math.log(ratio) > 2 * se
            parts.append(f"{nice} {ratio:.3f}× (±{se:.3f} log-SE; selector ahead on "
                         f"{100 * float(r['share_a_ge_b']):.0f}% of {int(r['rows'])} rows)")
            if not significant:
                failing.append(nice)
        if parts:
            out.append("\nIn one line: " + "; ".join(parts) + ".")
            aqorn_parts = [p for p in parts if "AQORN" in p]
            if not (ROOT / "exp6" / "compiled_selector.csv").exists():
                rep.headline(1, "6: selector vs best static code size chosen with hindsight — " +
                             "; ".join(aqorn_parts or parts[:3]))
            else:
                rep._fig8_parts = aqorn_parts or parts[:3]
        if failing:
            rep.risks.append(
                f"**Fig 8 claim.** \"Beats even the best static code chosen with "
                f"hindsight\" does not hold (no win beyond 2 log-SE) against: "
                f"{', '.join(failing)}. All comparators: " + "; ".join(parts) + ". Fig 8 as "
                f"submitted plots only the *worst* static k, against which the selector is "
                f"far ahead; the oracle-max comparators are biased toward the static arm "
                f"(max over noisy estimates), which makes this conservative for the selector.")
    comp = read(ROOT / "exp6" / "compiled_selector.csv")
    aqorn_csv = ROOT / "exp6" / "aqorn_static_k.csv"
    if comp is not None and aqorn_csv.exists():
        from experiments.revision.common import FIG8_CSV
        from experiments.revision.exp6_selector.best_static_replot import (
            load, paired, per_row, static_comparators,
        )

        frame = pd.concat([load(FIG8_CSV, aqorn_csv), comp], ignore_index=True, sort=False)
        rel = per_row(frame)
        series = {}
        series.update(static_comparators(rel, "aqorn"))
        series.update(static_comparators(rel, "naive"))
        rows = {}
        for key, nice in (("aqorn_k_best_global", "best single k, AQORN placement"),
                          ("aqorn_k_best_per_ratio", "best k per ratio, AQORN placement"),
                          ("aqorn_k_best_per_point", "best k per point, AQORN placement"),
                          ("naive_k_best_global", "best single k, naive"),
                          ("naive_k_worst", "worst static k, naive (Fig 8's comparator)")):
            if key in series:
                rows[f"compile_circuit vs {nice}"] = paired(rel["compile_circuit"], series[key])
        selector_key = "compile_circuit vs Fig 8's selector arm"
        rows[selector_key] = paired(rel["compile_circuit"], rel["selector"])
        table = pd.DataFrame(rows).T
        out.append("\n**6d — the shipped compiler (`compile_circuit`) as the selector arm** "
                   "(same sweep, alpha and shots; paired geometric-mean LSR ratio):\n")
        out.append(md(table[["geomean_ratio", "log_se", "share_a_ge_b", "rows"]], 3))
        sources = comp["source"].value_counts().to_dict()
        out.append(f"\n`compile_circuit` returned the planner's placement in "
                   f"{sources.get('packer', 0)} columns, the selector's in "
                   f"{sources.get('selector', 0)}, naive in {sources.get('naive', 0)}; chosen k: "
                   f"{comp['k'].value_counts().sort_index().to_dict()}.")
        key = "compile_circuit vs best single k, AQORN placement"
        if key in rows:
            r = rows[key]
            bare = latest.loc["selector_vs_aqorn_k_best_global"] \
                if latest is not None and "selector_vs_aqorn_k_best_global" in latest.index else None
            rep.headline(1, (f"6: against AQORN's planner at the best single static k chosen with "
                             f"hindsight, Fig 8's selector arm keeps "
                             f"{float(bare['geomean_ratio']):.3f}× the shots (loses) while the "
                             f"shipped `compile_circuit` keeps " if bare is not None else
                             "6: the shipped `compile_circuit` keeps ")
                         + f"{r['geomean_ratio']:.3f}× (±{r['log_se']:.3f}; a tie, without "
                           f"hindsight) and {rows[selector_key]['geomean_ratio']:.3f}× Fig 8's arm")
            mitigation = (
                f" **What survives (6d):** the shipped `compile_circuit` -- the selector's shape "
                f"with the planner's placement -- keeps {r['geomean_ratio']:.3f}× "
                f"(±{r['log_se']:.3f} log-SE) the shots of the best single static k chosen with "
                f"hindsight (AQORN placement), i.e. it matches it without hindsight, and "
                f"{rows[selector_key]['geomean_ratio']:.3f}× Fig 8's selector arm. Fig 8 plotted "
                f"`select_resources` alone, whose own placement is the weak part.")
            for index, text in enumerate(rep.risks):
                if text.startswith("**Fig 8 claim"):
                    rep.risks[index] = text + mitigation
            if r["geomean_ratio"] < 1 and math.log(1 / r["geomean_ratio"]) > 2 * r["log_se"]:
                rep.risks.append(
                    f"**Even the full compiler loses to a static code size (6d).** "
                    f"`compile_circuit` keeps {r['geomean_ratio']:.3f}× (±{r['log_se']:.3f} log-SE) "
                    f"the shots of AQORN's planner at the best single k chosen with hindsight.")
    mismatch = read(ROOT / "exp6" / "noise_mismatch.csv")
    if mismatch is not None:
        from experiments.revision.exp6_selector.noise_mismatch import summarize as mm

        table = mm(mismatch)
        out.append("\n**6c — selector told the wrong machine** (geomean LSR(r_model → r_true) / "
                   "LSR(matched), rows r_true, columns r_model):\n")
        out.append(md(table, 3))
    out.append("\nFigures: " + ", ".join(f"`{f}`" for f in figures(ROOT / "exp6")))
    return "\n".join(out)


def exp5_note(rep: Report) -> str:
    frame = read(ROOT / "exp1" / "packer_static_o1.csv")
    text = ("`simulator/attribution.py::measured_noise_split` attributes a translated circuit's "
            "expected detectable errors to gates, readout, idling (gate-, transport- and "
            "readout-induced) and movement (handling + flight) from the Stim schedule, "
            "statically — no shots. It is cheap (well under a second per circuit) but rejects "
            "rz/rx, so run it on proxies. Experiment1a already records it per arm (`E_*` "
            "columns). **Feasible tomorrow:** a movement/idle breakdown for Fig 6 is a replot of "
            "that CSV; a transport-scheduling sensitivity sweep needs only `distance_scalar` / "
            "`movement_speed` variations through the same function.")
    if frame is not None and "E_movement" in frame:
        ok = frame[(frame["status"] == "ok") & (frame["n_code"] == 4)]
        comps = [c for c in ("E_gates", "E_measurement", "E_idle_gates", "E_idle_transport",
                             "E_idle_readout", "E_move_handling", "E_flight_dephasing",
                             "E_movement", "E_idle", "E_total") if c in ok]
        piv = ok.pivot_table(index="name", columns="arm", values=comps)
        h, a = "fig6_h_merge", "aqorn"
        delta_total = piv["E_total"][a].sum() - piv["E_total"][h].sum()
        rows = {}
        for c in comps:
            H, A = piv[c][h], piv[c][a]
            rows[c] = {
                "sum, naive +H+merge": H.sum(), "sum, AQORN": A.sum(),
                "aggregate change %": 100 * (A.sum() - H.sum()) / H.sum() if H.sum() else math.nan,
                "share of total change %": 100 * (A.sum() - H.sum()) / delta_total
                if delta_total else math.nan,
                "circuits lower": int((A < H - 1e-12).sum()),
                "circuits higher": int((A > H + 1e-12).sum()),
            }
        table = pd.DataFrame(rows).T
        text += ("\n\nPreview: where AQORN's packing changes expected detectable errors, against "
                 "naive packing with the same passes (n=4, 71 circuits, proxies at NEUTRAL_ATOM "
                 "rates, summed over circuits):\n\n" + md(table, 3))
        grp = ok.drop_duplicates("name").set_index("name")["size_group"]
        by = []
        for g in ("small", "medium", "large"):
            m = grp.reindex(piv.index) == g
            if m.sum():
                ch = {c: 100 * (piv[c][a][m].sum() - piv[c][h][m].sum()) / piv[c][h][m].sum()
                      for c in ("E_movement", "E_idle", "E_gates", "E_total") if c in comps}
                by.append(f"{g} ({int(m.sum())}): " + ", ".join(
                    f"{k[2:]} {v:+.1f}%" for k, v in ch.items()))
        text += "\n\nBy size group: " + "; ".join(by) + "."
        mv = rows.get("E_movement", {}).get("aggregate change %", math.nan)
        it = rows.get("E_idle_transport", {}).get("share of total change %", math.nan)
        ga = rows.get("E_gates", {}).get("share of total change %", math.nan)
        if mv == mv and mv > -1.0:
            rep.risks.append(
                f"**\"Packing's LSR gains come from reduced movement and idling\" (V-B) is "
                f"half right.** Against naive packing with the same passes, AQORN cuts expected "
                f"errors {abs(rows['E_total']['aggregate change %']):.1f}% (aggregate, n=4, "
                f"NEUTRAL_ATOM rates): {it:.0f}% of that is idling *forced by transport* and "
                f"{ga:.0f}% is gate error, while direct movement error (handling + flight) "
                f"changes {mv:+.1f}%. Say \"less transport-induced idling\", not \"less "
                f"movement\".")
    return text


def exp8_versus_aqorn(frame: pd.DataFrame, total_circuits: int) -> pd.DataFrame:
    """Each silent weight against AQORN on the circuits Experiment8b simulated."""
    aq = frame[frame["series"] == "aqorn"].set_index("name")
    resolved = set(aq.index[(aq["tvd_floor"] < 0.1) & (aq["accepted_shots"] >= 1000)])
    if "fig6_baseline_tvd" in aq and frame["raw_shots"].nunique() > 1:
        resolved &= set(aq.index[aq["fig6_baseline_tvd"] <= 0.95])
    rows = {}
    for series, g in frame[frame["series"].str.startswith("silent_w")].groupby("series"):
        both = g.set_index("name").join(aq, rsuffix="_a", how="inner")
        same = both["duplicate_of"].fillna("") == "aqorn"
        n = both["raw_shots"]
        ratio = (both["lsr"] / both["lsr_a"]).where((both["lsr"] > 0) & (both["lsr_a"] > 0))
        se = np.sqrt(both["lsr"] * (1 - both["lsr"]) / n + both["lsr_a"] * (1 - both["lsr_a"]) / n)
        z_lsr = ((both["lsr"] - both["lsr_a"]) / se.replace(0, np.nan)).fillna(0)
        r = both.loc[[x for x in both.index if x in resolved]]
        r = r[(r["accepted_shots"] >= 1000)]
        z_tvd = ((r["tvd_a"] - r["tvd"]) / np.sqrt(r["tvd_boot_sd"] ** 2 + r["tvd_boot_sd_a"] ** 2)
                 .replace(0, np.nan)).fillna(0)
        eq = -100 * (r["tvd_eq_vs_aqorn"] - r["aqorn_tvd_at_matched_shots"]) / r["aqorn_tvd_at_matched_shots"]
        silent = (both["p_silent_given_kept"] / both["p_silent_given_kept_a"]).where(
            (both["p_silent_given_kept_a"] > 0) & (both["accepted_shots"] > 0))
        logs = np.log(ratio.dropna())
        rows[series] = {
            "weight": float(series[len("silent_w"):]),
            "circuits changed": int((~same).sum()),
            "shots kept vs AQORN, changed (geomean)": geomean(ratio.values),
            f"shots kept vs AQORN, all {total_circuits} (ties = 1)":
                float(np.exp(logs.sum() / total_circuits)) if len(logs) else math.nan,
            "LSR better / worse >2σ": f"{int((z_lsr > 2).sum())} / {int((z_lsr < -2).sum())}",
            "resolved": len(r),
            "TVD red. vs AQORN %": -100 * ((r["tvd"] - r["tvd_a"]) / r["tvd_a"]).mean(),
            "TVD red. vs AQORN %, equal shots": eq.mean(),
            "TVD better / worse >2σ": f"{int((z_tvd > 2).sum())} / {int((z_tvd < -2).sum())}",
            "silent per kept shot vs AQORN (geomean, frames)": geomean(silent.values),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("weight")


def exp8(rep: Report) -> str:
    gadgets = read(ROOT / "exp8" / "gadget_rates.csv")
    static = read(ROOT / "exp8" / "planner_static.csv")
    if gadgets is None or static is None:
        rep.skipped.append("Experiment8: not run")
        return "Not run (no `results/revision/exp8/` tables)."
    out = [
        "Question (from 1c): conventional packers keep more shots but give worse answers, and "
        "the share of two-qubit gates placed inside a patch predicts both. Does a cost model "
        "that prices *silent* (undetected) logical errors do better than AQORN's, which counts "
        "every error once?",
        "",
        "**Gadget rates** (`gadgets.py`): exact single-fault enumeration on the simulator's own "
        "noisy Stim circuit of each gadget, alpha = 1; silent logical error per execution in "
        "units of p2 (= 3e-3):",
        "",
    ]
    keep = ["CX_intra", "CZ_intra", "XCX_intra", "CX_cross", "XCX_cross", "CXT", "H_lane",
            "HH", "S", "RZ"]
    piv = gadgets.pivot_table(index="gadget", columns="n_code", values="logical_per_p2")
    out += [md(piv.reindex([k for k in keep if k in piv.index]).rename(
        columns=lambda c: f"n={c}"), 2), ""]
    out += [
        "Cross-patch gadgets, transversal macros and the whole-patch HH leak nothing at first "
        "order; every in-patch two-qubit gadget and the single-lane Hadamard turn 0.6–0.84 p2 "
        "of their faults into silent logical errors. The cost model adds `weight` × expected "
        "silent errors to AQORN's objective, seed graph and verify (weight 0 = AQORN).",
        "",
    ]

    base = static[static["silent_weight"] == 0.0].set_index("name")
    side = sidecar(ROOT / "exp8" / "planner_static.csv")
    mismatched = side.get("extra", {}).get("weight0_mismatched_shipped", None)
    checked = side.get("extra", {}).get("weight0_checked", None)
    if mismatched is not None:
        out.append(f"Weight 0 reproduces shipped AQORN's placement and commutation decision on "
                   f"{checked - len(mismatched)}/{checked} circuits"
                   + (f" (mismatched: {mismatched})" if mismatched else "") + ".")
        out.append("")
    rows = {}
    for weight, g in static.groupby("silent_weight"):
        g = g.set_index("name")
        b = base.reindex(g.index)
        same = (g["placement"] == b["placement"]) & (g["commute"] == b["commute"])
        rows[f"{weight:g}"] = {
            "changed": int((~same).sum()), "commutation changed": int((g["commute"] != b["commute"]).sum()),
            "verify fell back": int(g["fell_back"].sum()),
            "in-patch 2q share (proxy)": g["in_patch_share_proxy"].mean(),
            "predicted silent (proxy, sum)": g["silent_pred_proxy"].sum(),
            "expected errors E_total (proxy, sum)": g["E_total"].sum() if "E_total" in g else math.nan,
            "depth vs AQORN %": 100 * ((g["depth"] - b["depth"]) / b["depth"]).mean(),
            "2q gates vs AQORN %": 100 * ((g["two_qubit_gates"] - b["two_qubit_gates"])
                                          / b["two_qubit_gates"]).mean(),
        }
    out += [f"**Static** (8a, {static['name'].nunique()} Fig 6 circuits, n = 4, planned on "
            "originals; silent and E_total on the proxies at alpha = 1):", "",
            md(pd.DataFrame.from_dict(rows, orient="index").rename_axis("weight"), 3), ""]

    best = None
    for csv, title in (("planner_simulated.csv", "Fig 6's per-circuit calibration"),
                       ("planner_simulated_alpha1.csv", "one fixed machine, alpha = 1")):
        frame = read(ROOT / "exp8" / csv)
        if frame is None:
            out += [f"**Simulated, {title}:** not run.", ""]
            continue
        total = static["name"].nunique()
        table = exp8_versus_aqorn(frame, total)
        gate = (ROOT / "exp8" / "_gate" / "GATE.txt")
        gate_text = (" Sanity gate (Fig 6's naive and AQORN samples re-drawn): **"
                     + gate.read_text().split()[0] + "**.") if "alpha" not in csv and gate.exists() else ""
        out += [f"**Simulated, {title}** (8b; {frame['name'].nunique()} circuits where some "
                f"weight changed the plan — the rest are exact ties).{gate_text} TVD columns use "
                f"circuits where AQORN's noiseless floor < 0.1 and ≥ 1k shots were kept; "
                f"'silent per kept shot' is measured directly on each arm's noisy circuit "
                f"(`frames.py`) and covers every simulated circuit:", "",
                md(table.drop(columns=["weight"]).rename_axis("arm"), 3), ""]
        col = "TVD red. vs AQORN %, equal shots"
        if "alpha" in csv and not table.empty and table[col].astype(float).notna().any():
            best = table.loc[table[col].astype(float).idxmax()]
            best_fig6 = None
            fig6 = read(ROOT / "exp8" / "planner_simulated.csv")
            if fig6 is not None:
                t6 = exp8_versus_aqorn(fig6, total)
                if best.name in t6.index:
                    best_fig6 = t6.loc[best.name]
            rep.headline(
                2, f"8 (silent-error cost model): at weight {best['weight']:g}, on one machine "
                   f"(alpha = 1) TVD at equal shots falls {float(best[col]):.1f}% against "
                   f"AQORN's ({best['TVD better / worse >2σ']} circuits better / worse beyond "
                   f"2σ) for {float(best['shots kept vs AQORN, changed (geomean)']):.3f}× the shots "
                   f"on the {int(best['circuits changed'])} changed circuits"
                   + (f"; under Fig 6's calibration the same weight "
                      f"{'cuts' if float(best_fig6[col]) >= 0 else 'raises'} TVD "
                      f"{abs(float(best_fig6[col])):.1f}% for "
                      f"{float(best_fig6['shots kept vs AQORN, changed (geomean)']):.3f}× the shots"
                      if best_fig6 is not None else "") + ".")

    sim = read(ROOT / "exp8" / "planner_simulated_alpha1.csv")
    if sim is not None:
        pred = static.set_index(["name", "silent_weight"])["silent_pred_proxy"]
        agree = total_pairs = 0
        for name, g in sim.groupby("name"):
            a = g[g["series"] == "aqorn"]
            if a.empty or not (a["accepted_shots"].iloc[0] > 0):
                continue
            for row in g[g["series"].str.startswith("silent_w")].itertuples():
                if row.duplicate_of == "aqorn" or (name, row.silent_weight) not in pred.index:
                    continue
                predicted = pred[(name, row.silent_weight)] - pred[(name, 0.0)]
                measured = row.p_silent_given_kept - a["p_silent_given_kept"].iloc[0]
                if abs(predicted) < 1e-12:
                    continue
                total_pairs += 1
                agree += (predicted > 0) == (measured > 0)
        if total_pairs:
            out += [f"**Does the model predict the measurement?** At alpha = 1, where it "
                    f"predicted a weight's plan changes silent error relative to AQORN, the "
                    f"measured silent error per kept shot moved the same way in {agree} of "
                    f"{total_pairs} (circuit, weight) pairs.", ""]
    out.append("*Suggested revision text:* only if the α = 1 gain survives review: \"AQORN's "
               "objective counts every error once; pricing errors the code cannot detect "
               "(in-patch gadgets) trades a little survival for accuracy.\"")
    return "\n".join(out)


def make_figures(rep: Report) -> str:
    """A few house-style figures from the CSVs; each one isolated, versioned, never overwriting."""
    import matplotlib.pyplot as plt

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    target = ROOT / "figures"
    target.mkdir(parents=True, exist_ok=True)
    made: List[str] = []

    def save(fig, name):
        for path in save_figure(fig, name, Path(__file__), override=target, versioned=True):
            if path.suffix == ".pdf":
                made.append(str(path))
        plt.close(fig)

    def attempt(name, body):
        try:
            plt.rcdefaults()
            use_style()
            body()
        except Exception:  # noqa: BLE001
            made.append(f"{name}: failed ({traceback.format_exc().splitlines()[-1][:120]})")

    def fig7a():
        frame = read(ROOT / "exp7" / "exhaustive_translate.csv")
        if frame is None:
            return
        # AQORN in slot 0 (the house style's AQORN position), at most five series: the
        # style's cycle repeats after five and two bars would share a colour.
        arms = ["aqorn", "naive", "interaction_partition", "hypergraph_24", "random_budget"]
        short = {"aqorn": "AQORN", "naive": "naive", "interaction_partition": "interaction partition",
                 "hypergraph_24": "[24] stage 1", "random_budget": "random, equal budget"}
        configs = sorted(frame["config"].unique())
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.4), sharex=True)
        width = 0.8 / len(arms)
        for axis, objective, ylabel in ((axes[0], "work_depth", "regret, work (%)"),
                                        (axes[1], "depth", "regret, depth (%)")):
            for i, arm in enumerate(arms):
                block = frame[frame["arm"] == arm].groupby("config")[f"{objective}_regret_tight"]
                mean, sd = block.mean().reindex(configs), block.std().reindex(configs)
                axis.bar(np.arange(len(configs)) + (i - len(arms) / 2 + 0.5) * width, mean,
                         width, yerr=sd, label=short[arm],
                         error_kw={"elinewidth": 0.6, "capsize": 1.5})
            axis.set_ylabel(ylabel)
            axis.set_ylim(bottom=0)
        axes[1].set_xticks(np.arange(len(configs)))
        axes[1].set_xticklabels(configs)
        axes[0].legend(fontsize=6, ncol=2, frameon=False)
        axes[0].set_title("bars: mean over circuits, whiskers: ±1 SD", fontsize=7)
        save(fig, "exp7a_regret_vs_exhaustive")

    def fig1c():
        frame = read(ROOT / "exp1" / "packer_simulated.csv")
        if frame is None:
            return
        short = {"aqorn": "AQORN", "naive": "naive (program order)",
                 "random_budget": "random, AQORN's budget",
                 "interaction_matching": "interaction matching",
                 "interaction_partition": "interaction partition",
                 "hypergraph_24": "[24] stage 1",
                 "greedy_pairs": "pairwise greedy (earlier)",
                 "interaction_partition_descent": "interaction partition + descent",
                 "aqorn_seed_only": "AQORN seed only",
                 "aqorn_descent_from_naive": "AQORN descent from naive",
                 "aqorn_raw": "AQORN, verify off"}
        naive_rows = frame[frame["series"] == "naive"].set_index("name")
        resolved = set(naive_rows.index[(naive_rows["tvd_floor"] < 0.1) &
                                        (naive_rows["accepted_shots"] >= 1000) &
                                        (naive_rows["fig6_baseline_tvd"] <= 0.95)])
        aqorn = frame[frame["series"] == "aqorn"].set_index("name")["lsr"]
        names, lsr, tvd = [], [], []
        for series in short:
            g = frame[frame["series"] == series].set_index("name")
            if g.empty:
                continue
            both = pd.concat([aqorn, g["lsr"]], axis=1, keys=["a", "s"]).dropna()
            names.append(short[series])
            lsr.append(geomean((both["s"] / both["a"]).values))
            r = g.loc[[n for n in g.index if n in resolved]]
            tvd.append(r["tvd_red_vs_naive_pct"].mean())
        order = np.argsort(lsr)
        y = np.arange(len(names))
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.14, 2.9), sharey=True)
        left.plot(np.array(lsr)[order], y, linestyle="none", marker="o", markersize=4)
        left.axvline(1.0, color="#636363", linestyle="--", linewidth=1.0)
        left.set_xlabel("shots kept vs. AQORN (geomean)")
        right.plot(np.array(tvd)[order], y, linestyle="none", marker="s", markersize=4,
                   color="#3b4c56")
        right.axvline(0.0, color="#636363", linestyle="--", linewidth=1.0)
        right.set_xlabel(f"TVD reduction vs. naive, % ({len(resolved)} resolved circuits)")
        left.set_yticks(y)
        left.set_yticklabels(np.array(names)[order], fontsize=8)
        for axis in (left, right):
            axis.grid(axis="x", which="major")
        save(fig, "exp1c_packers_survival_vs_accuracy")

    def fig2b():
        frame = read(ROOT / "exp2" / "fixed_noise_sweep.csv")
        if frame is None:
            return
        fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH))
        dropped = 0
        for alpha, g in frame.groupby("alpha"):
            p = g.pivot_table(index="name", columns="series", values="lsr")
            if {"baseline", "aqorn"} <= set(p.columns):
                keep = (p["baseline"] > 0) & (p["aqorn"] > 0)
                dropped += int((~keep).sum())
                axis.plot(p.loc[keep, "baseline"], p.loc[keep, "aqorn"], linestyle="none",
                          markersize=3, label=f"$\\alpha$ = {alpha:g}")
        axis.plot([1e-5, 1], [1e-5, 1], color="#636363", linestyle="--", linewidth=1.0,
                  marker="none")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("baseline LSR (naive, both passes off)")
        axis.set_ylabel("AQORN LSR")
        axis.legend(fontsize=7, frameon=False, loc="upper left",
                    title=f"{dropped} points with zero accepted shots omitted",
                    title_fontsize=6)
        save(fig, "exp2b_fixed_noise_lsr")

    def fig3b():
        from experiments.revision.exp3_proxy.proxy_simulated import improvements

        frame = read(ROOT / "exp3" / "proxy_simulated.csv")
        if frame is None:
            return
        imp = improvements(frame)
        imp = imp[imp["backend"] == "aer"]
        fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH))
        for alpha_label, g in imp.groupby("alpha_label"):
            w = g.pivot_table(index="name", columns="version", values="tvd_exact_red_pct")
            if {"original", "proxy"} <= set(w.columns):
                axis.plot(w["original"], w["proxy"], linestyle="none", markersize=3,
                          label=str(alpha_label))
        low, high = axis.get_xlim()
        axis.plot([low, high], [low, high], color="#636363", linestyle="--", linewidth=1.0,
                  marker="none")
        axis.axhline(0, color="#bdbdbd", linewidth=0.6)
        axis.axvline(0, color="#bdbdbd", linewidth=0.6)
        axis.set_xlabel("AQORN TVD reduction, original (%)")
        axis.set_ylabel("AQORN TVD reduction, proxy (%)")
        axis.legend(fontsize=7, frameon=False)
        save(fig, "exp3b_original_vs_proxy")

    def fig8():
        static = read(ROOT / "exp8" / "planner_static.csv")
        panels = [(csv, title) for csv, title in
                  (("planner_simulated.csv", "Fig 6's per-circuit calibration"),
                   ("planner_simulated_alpha1.csv", r"one machine, $\alpha$ = 1"))
                  if read(ROOT / "exp8" / csv) is not None]
        if static is None or not panels:
            return
        total = static["name"].nunique()
        fig, axes = plt.subplots(1, len(panels), figsize=(7.14, 2.8), squeeze=False)
        for axis, (csv, title) in zip(axes[0], panels):
            table = exp8_versus_aqorn(read(ROOT / "exp8" / csv), total)
            x = table["shots kept vs AQORN, changed (geomean)"].astype(float)
            y = table["TVD red. vs AQORN %, equal shots"].astype(float)
            axis.plot([1.0], [0.0], linestyle="none", marker="*", markersize=9,
                      label="AQORN (weight 0)")
            axis.plot(x, y, linestyle="-", marker="o", markersize=4,
                      label="silent-aware, weight w")
            # Alternate label sides: neighbouring weights can land almost on top of each other.
            for index, (weight, xi, yi) in enumerate(zip(table["weight"], x, y)):
                axis.annotate(f"w={weight:g}", (xi, yi), textcoords="offset points",
                              xytext=(5, -9) if index % 2 else (5, 3), fontsize=6)
            axis.axhline(0, color="#bdbdbd", linewidth=0.6)
            axis.set_xlim(min(0.85, float(x.min()) - 0.02), 1.02)
            axis.set_title(title, fontsize=8)
            axis.set_xlabel("shots kept vs. AQORN (geomean)")
            axis.set_ylabel("TVD cut vs. AQORN (%)")
        axes[0][0].legend(fontsize=6, frameon=False, loc="lower left")
        fig.tight_layout()
        save(fig, "exp8_silent_cost_model")

    for name, body in (("7a", fig7a), ("1c", fig1c), ("2b", fig2b), ("3b", fig3b),
                       ("8", fig8)):
        attempt(name, body)
    for extra in sorted((ROOT / "exp6").glob("fig8_best_static_with_compiler*.pdf")) + \
            sorted((ROOT / "exp7").glob("compile_time*.pdf")):
        made.append(str(extra))
    if not made:
        return "No figures (no data yet)."
    return "\n".join(f"- `{m}`" for m in made)


def status_block() -> str:
    status = LOGS / "STATUS.txt"
    estimates = LOGS / "ESTIMATES.txt"
    out = []
    if status.exists():
        out.append("```\n" + status.read_text().strip() + "\n```")
    if estimates.exists():
        out.append("Estimates made before launch: `logs/revision/ESTIMATES.txt`.")
    return "\n\n".join(out) if out else "_No queue status file._"


def main() -> None:
    import argparse

    global ROOT, REPORT
    parser = argparse.ArgumentParser(description="Write REVISION_RESULTS.md from the CSVs.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    ROOT, REPORT = args.results, args.report
    rep = Report()
    rep.section("Experiment7a — optimality vs exhaustive search, by translation",
                lambda: exp7a(rep))
    rep.section("Experiment7d — exhaustive search at (5 qubits, k = 4), full space (stretch)",
                lambda: exp7a(rep, "exhaustive_translate_k4", "7d"))
    rep.section("Experiment7d — tightest space at (6, k = 4), (7, k = 2), (8, k = 2) (stretch)",
                lambda: exp7a(rep, "exhaustive_translate_tight", "7d"))
    rep.section("Experiment7b — optimality vs exhaustive search, simulated",
                lambda: exp7b(rep))
    rep.section("Experiment7c — compile time vs circuit size", lambda: exp7c(rep))
    rep.section("Experiment1a — packers under identical passes, static",
                lambda: exp1a(rep))
    rep.section("Experiment1b — level-3 pre-optimised inputs", lambda: exp1b(rep))
    rep.section("Experiment1c — packers simulated", lambda: exp1c(rep))
    rep.section("Experiment1d — contribution by circuit family", lambda: exp1d(rep))
    rep.section("Experiment2a — Fig 6 calibration table", lambda: exp2a(rep))
    rep.section("Experiment2b + Experiment4 — one fixed machine, TVD statistics "
                "", lambda: exp2b(rep))
    rep.section("Experiment3a — does the proxy change the compiled circuit?",
                lambda: exp3a(rep))
    rep.section("Experiment3b — non-Clifford originals vs proxies, simulated",
                lambda: exp3b(rep))
    rep.section("Experiment6 — Fig 8's best-static comparator, selector robustness "
                "", lambda: exp6(rep))
    rep.section("Experiment5 — feasibility note (not run tonight)", lambda: exp5_note(rep))
    rep.section("Experiment8 — a cost model that prices undetected errors (follow-up to 1c)",
                lambda: exp8(rep))
    rep.section("Figures drawn by this report", lambda: make_figures(rep))

    git = git_state()
    finished = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        "# v2 revision — overnight results",
        "",
        f"Generated {finished} by `experiments/revision/summarize.py` from the CSVs under "
        f"`results/revision/`. Code: branch `{git['branch']}` @ `{git['sha'][:10]}`; the "
        f"submitted tree is tag `v2-v4-submitted-tree` (`9f828aa`).",
        "",
        "## Summary",
        "",
    ]
    status_file = LOGS / "STATUS.txt"
    if status_file.exists():
        text = status_file.read_text()
        lines_ = text.splitlines()
        oks = [l.split()[3] for l in lines_ if "  ok  " in l]
        stopped = [l.split()[3] for l in lines_ if "  FAIL " in l and "exit 143" in l]
        fails = [l.split()[3] for l in lines_ if "  FAIL " in l and "exit 143" not in l]
        # This report is itself the queue's last item, so "finished" is also when the only
        # thing still running is this summary.
        finished = "queue finished" in text or (lines_ and lines_[-1].endswith("START  19_summarize"))
        lines.append(f"- Queue: {len(oks)} items ok, {len(fails)} failed"
                     + (f" ({', '.join(fails)})" if fails else "")
                     + (f", {len(stopped)} stopped on purpose for the time budget "
                        f"({', '.join(stopped)}; see Trimmed or skipped)" if stopped else "")
                     + ("; finished" if finished else "; still running") + ".")
    lines += [f"- {h}" for h in rep.ordered_headlines()[:5] if h] or ["- No headline numbers (see sections)."]
    lines += ["", "Queue status:", "", status_block(), "", "## Claims at risk", ""]
    def risk_rank(text: str) -> int:
        """Most decision-relevant first: Fig 8's claim, then packers and fixed noise."""
        order = (("Fig 8", 1), ("(1c)", 2), ("(1a)", 2), ("(7b)", 2), ("half right", 2),
                 ("Fixed-noise", 2),
                 ("TVD gain", 2), ("(3b)", 3), ("(3a)", 3), ("verify", 3), ("seconds", 3),
                 ("Calibration", 3), ("(7a)", 4))
        return min((rank for key, rank in order if key in text), default=5)

    ranked = sorted(rep.risks, key=risk_rank)
    lines += [f"{i}. {r}" for i, r in enumerate(ranked, 1)] or ["None detected."]
    lines += [""] + rep.sections
    lines += ["## Trimmed or skipped", ""]
    lines += [f"- {s}" for s in rep.skipped] or ["- Nothing reported missing."]
    trims = LOGS / "TRIMS.txt"
    if trims.exists():
        lines += ["", trims.read_text().strip()]
    lines += [
        "", "## Reproduce", "",
        f"Code SHA `{git['sha']}` (branch `{git['branch']}`). Queue: "
        "`caffeinate -dimsu nohup bash experiments/revision/run_overnight.sh > "
        "logs/revision/queue.log 2>&1 &`. Each item's exact command is in "
        "`experiments/revision/run_overnight.sh`; each CSV's JSON sidecar records settings, "
        "seeds, git SHA, cost-model version and wall time. Re-running a finished item resumes "
        "(and refuses a settings mismatch); pass `--out` to a new directory to recompute.",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
