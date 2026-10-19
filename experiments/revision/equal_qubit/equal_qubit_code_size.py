"""
equal_qubit_code_size.py -- which code size wins on a fixed device? Every k at an equal
physical-qubit budget, AQORN and naive, across idle/distance ratios (handoffs/
equal_qubit_code_size.md).

Every earlier code-size comparison puts each k at its tightest packing, ceil(n/k) patches, so
k = 2 also gets the most hardware (4 ceil(n/2) data qubits against 6 ceil(n/4) at k = 4). Here
each k gets the same budget Q of physical qubits and as many patches as fit:

    budgets     Q2 = 4 ceil(n/2), k = 2's tightest footprint (data qubits, the paper's and
                Fig 8's convention), and 1.25 Q2, 1.5 Q2; plus `Q2+anc`, the same comparison
                with the two parity ancillas of every patch counted (Q = 6 ceil(n/2), a patch
                costs n_code + 2) -- the emitted circuits carry them -- as a sensitivity check
    patches     P = floor(Q / cost per patch), clipped to [ceil(n/k), n]; a k whose ceil(n/k)
                patches do not fit is recorded as not fitting and skipped. `at_tightest` marks
                P = ceil(n/k) (no spare patches), `at_one_per_patch` marks P = n
    arms        AQORN = `plan_standard(program, n_code=k+2, patches=P, noise=profile)` with the
                ratio's planner profile (h_commutation "auto", the standard final check);
                naive = program order spread as evenly as possible over P patches (sizes
                differ by at most one, the larger first) under Fig 8's naive commutation rule
                (`commutation_pays`); at k = 2 that is `tight_trivial_packing(n, P)`
    noise       Fig 8's corpus, ratios and per-(circuit, ratio) alpha (read from Fig 8's
                standard CSV, not recalibrated), Fig 8's simulation settings and seeds
    measured    `evaluate(frames=)` exactly as Fig 8 does: LSR and TVD on 40k raw shots, and on
                100k Pauli frames the silent logical error per kept shot on occupied lanes
                only (`silent_per_kept_occupied`, the primary accuracy metric: at equal Q the
                large codes run mostly empty lanes) and on all lanes; static counts; the
                emitted circuit's qubit count; and the noiseless check on every arm
                (`noiseless_tvd` against naive at the tightest packing, `miscompiled`), whose
                value is also the TVD sampling floor

Identical (n_code, placement, commutation) configurations inside one (circuit, ratio) are
simulated once, and a (k, P) shared by several budgets is measured once and reported under each.

**The gate** (`--gate`): k = 2 at Q2 is k = 2's tightest packing, so the AQORN arm must
reproduce Fig 8's static k = 2 AQORN rung (`aqorn_fixed`) for every (circuit, ratio): the same
plan and alpha exactly, and LSR and silent error within sampling error -- Fig 8 ran unseeded, so
its samples cannot be repeated bit for bit (`gate`).

    PYTHONPATH=. venv/bin/python -m experiments.revision.equal_qubit.equal_qubit_code_size --gate --jobs 65
    ... --circuits adder_n28 wstate_n36 --jobs 50 --tag pilot      # time it
    ... --jobs 96                                                    # the full run
    ... --from-csv results/data/teddy_final_2026-10-11/equal_qubit_code_size.csv   # figure only
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

NAME = "equal_qubit_code_size"
FIG8 = Path("experiments/figures/resource_selection_headroom/resource_selection_headroom_standard.csv")
OUT = Path("results/equal_qubit")
K_VALUES = (2, 4, 6, 8, 10)
#: Data-qubit budgets as multiples of Q2; `Q2+anc` is handled apart (ancillas counted).
FACTORS = {"Q2": 1.0, "1.25Q2": 1.25, "1.5Q2": 1.5}
ANCILLAS = 2
#: Naive k = 2's TVD at or above this at every ratio: the proxy's output is near uniform and no
#: affordable shot count resolves it (Fig 8's rule, `resource_tradeoff.SATURATED_TVD`).
SATURATED_TVD = 0.95


def spread_packing(num_qubits: int, k: int, patches: int) -> List[Tuple[int, int]]:
    """Program order over exactly `patches` patches, sizes as equal as possible (the larger
    first), each at most k."""
    if not -(-num_qubits // k) <= patches <= num_qubits:
        raise ValueError(f"{num_qubits} qubits do not fit {patches} patches of {k}")
    base, extra = divmod(num_qubits, patches)
    placement, qubit = [], 0
    for patch in range(patches):
        for lane in range(base + (1 if patch < extra else 0)):
            placement.append((patch, lane))
            qubit += 1
    return placement


def budgets_for(num_qubits: int) -> Dict[str, Tuple[int, int]]:
    """Budget name -> (Q, cost per patch beyond n_code): data qubits, or data + ancillas."""
    q2 = 4 * -(-num_qubits // 2)
    out = {name: (int(math.floor(f * q2)), 0) for name, f in FACTORS.items()}
    out["Q2+anc"] = ((4 + ANCILLAS) * -(-num_qubits // 2), ANCILLAS)
    return out


def patches_for(num_qubits: int, k: int, budget: int, overhead: int) -> Optional[int]:
    tightest = -(-num_qubits // k)
    fit = budget // (k + 2 + overhead)
    if fit < tightest:
        return None
    return min(fit, num_qubits)


def _fig8_alphas(path: Path) -> Dict[Tuple[str, float], float]:
    d = pd.read_csv(path)
    return {(r.circuit, float(r.ratio)): float(r.alpha)
            for r in d.drop_duplicates(["circuit", "ratio"]).itertuples()}


def _run(job) -> List[Dict]:
    """One (circuit, ratio): every (k, P) it needs, both arms, deduplicated."""
    import warnings

    warnings.filterwarnings("ignore")
    from experiments.figures.resource_selection_headroom.resource_selection_headroom import (
        Settings, _arm_key,
    )
    from experiments.shared.simulation import (
        build_config, evaluate, miscompiled, noise_profile_for, noiseless_tvd,
    )
    from iceberg.gadget_options import transversal_intra
    from packing.plan import naive_packing
    from packing.passes import commutation_pays
    from packing.standard import plan_standard
    from translator.core import Translator

    name, circuit, ratio, alpha, points, gate_only = job
    settings = Settings(standard=True)
    sim = settings.sim.with_standard().with_noise_ratio(ratio)
    profile = noise_profile_for(sim)
    n = circuit.num_qubits
    commutes: Dict[int, bool] = {}

    def commute_for(n_code: int) -> bool:
        if n_code not in commutes:
            with transversal_intra(sim.transversal_merging):
                commutes[n_code] = commutation_pays(circuit, n_code)
        return commutes[n_code]

    clean_ref: Dict[Tuple[int, bool], float] = {}

    def measure(n_code: int, placement, commute: bool) -> Dict:
        key = (n_code, commute)
        if key not in clean_ref:
            clean_ref[key] = noiseless_tvd(circuit, naive_packing(n, n_code - 2), sim,
                                           n_code=n_code, h_commutation=commute)
        began = time.time()
        clean = noiseless_tvd(circuit, placement, sim, n_code=n_code, h_commutation=commute)
        bad = bool(miscompiled(clean_ref[key], clean))
        out = {"noiseless_tvd": clean, "noiseless_tvd_naive_tight": clean_ref[key],
               "miscompiled": bad}
        config = build_config(sim, n_code=n_code, alpha=alpha,
                              repetitions=settings.repetitions, h_commutation=commute,
                              merging=True)
        width = Translator(config=config.translator_config).translate(
            circuit, [tuple(p) for p in placement]).circuit.num_qubits
        out["emitted_qubits"] = int(width)
        if not bad:
            measured = evaluate(circuit, placement, config, frames=settings.frames,
                                frame_seed=sim.seed)
            if measured is not None:
                out.update(measured)
        out["sim_seconds"] = time.time() - began
        return out

    done: Dict[object, Dict] = {}
    rows: List[Dict] = []
    for k, P, budgets in points:
        n_code = k + 2
        arms = []
        began = time.time()
        plan = plan_standard(circuit, n_code, noise=profile, patches=P)
        arms.append(("aqorn", [tuple(int(v) for v in p) for p in plan.placement],
                     bool(plan.h_commutation),
                     {"plan_fell_back": bool(plan.fell_back), "plan_seconds": time.time() - began}))
        if not gate_only:
            arms.append(("naive", spread_packing(n, k, P), commute_for(n_code), {}))
        for arm, placement, commute, extra in arms:
            key = _arm_key(n_code, placement, commute)
            duplicate = key in done
            if not duplicate:
                done[key] = measure(n_code, placement, commute)
            m = done[key]
            for budget, Q, overhead in budgets:
                rows.append({
                    "circuit": name, "num_qubits": n, "ratio": ratio, "alpha": alpha,
                    "budget": budget, "budget_qubits": Q, "ancillas_counted": bool(overhead),
                    "k": k, "n_code": n_code, "patches": P, "arm": arm,
                    "at_tightest": P == -(-n // k), "at_one_per_patch": P == n,
                    "data_qubits": P * n_code, "qubits_with_ancillas": P * (n_code + ANCILLAS),
                    "occupancy": n / (P * k), "h_commutation": commute,
                    "placement": json.dumps([list(p) for p in placement]),
                    "duplicate": duplicate, **extra,
                    "lsr": m.get("lsr", math.nan), "tvd": m.get("tvd", math.nan),
                    "logical_error": m.get("silent_per_kept_occupied", math.nan),
                    "logical_error_all_lanes": m.get("silent_per_kept", math.nan),
                    "frame_lsr": m.get("frame_lsr", math.nan),
                    "one_qubit_gates": m.get("one_qubit_gates", math.nan),
                    "two_qubit_gates": m.get("two_qubit_gates", math.nan),
                    "circuit_depth": m.get("circuit_depth", math.nan),
                    "emitted_qubits": m["emitted_qubits"], "noiseless_tvd": m["noiseless_tvd"],
                    "noiseless_tvd_naive_tight": m["noiseless_tvd_naive_tight"],
                    "miscompiled": m["miscompiled"], "sim_seconds": m["sim_seconds"],
                })
    return rows


def jobs_for(circuits, ratios, alphas, gate_only: bool):
    jobs = []
    for name, circuit in circuits:
        n = circuit.num_qubits
        points: Dict[Tuple[int, int], List] = {}
        skipped = []
        budgets = budgets_for(n)
        for budget, (Q, overhead) in budgets.items():
            if gate_only and budget != "Q2":
                continue
            for k in ((2,) if gate_only else K_VALUES):
                P = patches_for(n, k, Q, overhead)
                if P is None:
                    skipped.append((budget, k))
                    continue
                points.setdefault((k, P), []).append((budget, Q, overhead))
        ordered = [(k, P, b) for (k, P), b in sorted(points.items())]
        for ratio in ratios:
            jobs.append((name, circuit, float(ratio), alphas[(name, float(ratio))], ordered,
                         gate_only))
        if skipped:
            print(f"[{NAME}] {name} (n = {n}): does not fit {skipped}")
    jobs.sort(key=lambda j: (-j[1].size(), -len(j[4])))
    return jobs


def gate(frame: pd.DataFrame, fig8: Path) -> bool:
    """
    k = 2 at Q2 against Fig 8's `aqorn_fixed` rung, column by column.

    Fig 8 ran unseeded (`SimSettings.seed` is None by design, and its frames used
    `frame_seed=sim.seed`), so no rerun can repeat its samples bit for bit. The gate is
    therefore exact where the code is deterministic -- the plan (gate counts, depth, patches,
    commutation, final-check fallback) and the alpha -- and statistical where it samples: per
    column the LSR difference over its binomial standard error (40k raw shots each) and the
    silent-error-per-kept-shot difference over its standard error (100k frames each). Both
    z sets must look like N(0, 1): |mean| < 3 / sqrt(columns), SD within [0.7, 1.4], chi-square
    p > 0.01.
    """
    from scipy import stats

    ref = pd.read_csv(fig8)
    ref = ref[ref.series == "aqorn_fixed"].set_index(["circuit", "ratio"])
    ours = frame[(frame.arm == "aqorn") & (frame.k == 2) & (frame.budget == "Q2")]
    j = ours.set_index(["circuit", "ratio"]).join(ref, rsuffix="_fig8", how="inner")
    exact = {"one_qubit_gates": "one_qubit_gates_fig8", "two_qubit_gates": "two_qubit_gates_fig8",
             "circuit_depth": "circuit_depth_fig8", "patches": "patches_fig8",
             "h_commutation": "h_commutation_fig8", "plan_fell_back": "plan_fell_back_fig8",
             "alpha": "alpha_fig8"}
    ok = len(j) == len(ref)
    print(f"[{NAME}] gate: {len(j)} of {len(ref)} Fig 8 columns")
    for mine, theirs in exact.items():
        left, right = j[mine], j[theirs]
        same = (left.astype(bool) == right.astype(bool)) if mine in ("h_commutation", "plan_fell_back") \
            else np.isclose(left.astype(float), right.astype(float), rtol=0, atol=1e-12)
        print(f"    exact  {mine:18s} equal on {int(same.sum())}/{len(j)}")
        ok &= bool(same.all())
    shots, frames = 40_000, 100_000
    p = (j.lsr + j.lsr_fig8) / 2
    z_lsr = (j.lsr - j.lsr_fig8) / np.sqrt(2 * p * (1 - p) / shots)
    kept = (j.frame_lsr + j.frame_lsr_fig8) / 2 * frames
    q = (j.logical_error_all_lanes + j.silent_per_kept) / 2
    z_err = (j.logical_error_all_lanes - j.silent_per_kept) / np.sqrt(2 * q * (1 - q) / kept)
    for label, z in (("LSR", z_lsr.dropna()), ("silent error per kept shot", z_err.dropna())):
        chi = 1 - stats.chi2.cdf(float((z ** 2).sum()), len(z))
        good = abs(z.mean()) < 3 / math.sqrt(len(z)) and 0.7 <= z.std() <= 1.4 and chi > 0.01
        print(f"    sampled {label}: z mean {z.mean():+.2f}, SD {z.std():.2f}, chi-square p "
              f"{chi:.3f} -> {'consistent' if good else 'INCONSISTENT'}")
        ok &= bool(good)
    return ok


# --- analysis and figure ------------------------------------------------------------------------


def saturated(frame: pd.DataFrame) -> List[str]:
    nk = frame[(frame.arm == "naive") & (frame.k == 2) & (frame.budget == "Q2")]
    low = nk.groupby("circuit").tvd.min()
    return sorted(low[low >= SATURATED_TVD].index)


def relative(frame: pd.DataFrame, budget: str, metric: str) -> pd.DataFrame:
    """Per (circuit, ratio, k, arm): metric / naive k = 2 at the same budget's reference Q2."""
    ref = frame[(frame.arm == "naive") & (frame.k == 2) & (frame.budget == "Q2")]
    ref = ref.set_index(["circuit", "ratio"])[metric]
    part = frame[frame.budget == budget].copy()
    part["ref"] = [ref.get((c, r), math.nan) for c, r in zip(part.circuit, part.ratio)]
    part["rel"] = part[metric] / part["ref"]
    return part


def geomean_ci(values: np.ndarray, rng, boots: int = 2000) -> Tuple[float, float, float]:
    v = np.log(values[np.isfinite(values) & (values > 0)])
    if not len(v):
        return math.nan, math.nan, math.nan
    draws = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(boots)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.exp(v.mean())), float(np.exp(lo)), float(np.exp(hi))


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    drop = saturated(frame)
    rows = []
    for budget in sorted(frame.budget.unique()):
        for metric in ("lsr", "logical_error", "tvd"):
            part = relative(frame, budget, metric)
            if metric == "tvd":
                part = part[~part.circuit.isin(drop)]
            for (ratio, k, arm), g in part.groupby(["ratio", "k", "arm"]):
                gm, lo, hi = geomean_ci(g.rel.to_numpy(), rng)
                rows.append(dict(budget=budget, metric=metric, ratio=ratio, k=k, arm=arm,
                                 gm=gm, lo=lo, hi=hi, n=int(g.rel.notna().sum()),
                                 at_tightest=bool(g.at_tightest.all())))
    return pd.DataFrame(rows)


def winners(frame: pd.DataFrame, budget: str = "Q2") -> pd.DataFrame:
    """Per ratio, AQORN arms: the best k on LSR (higher) and logical error (lower), the
    runner-up, and whether the margin clears one SE (paired log ratio across circuits)."""
    out = []
    a = frame[(frame.arm == "aqorn") & (frame.budget == budget)]
    for metric, higher in (("lsr", True), ("logical_error", False)):
        for ratio, g in a.groupby("ratio"):
            w = g.pivot_table(index="circuit", columns="k", values=metric)
            gm = np.exp(np.log(w).mean())
            order = gm.sort_values(ascending=not higher)
            best, second = order.index[0], order.index[1]
            d = np.log(w[best] / w[second]).dropna()
            se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else math.nan
            margin = float(np.exp(d.mean()))
            out.append(dict(budget=budget, metric=metric, ratio=ratio, best_k=int(best),
                            runner_up=int(second), margin=margin,
                            beyond_1se=bool(abs(d.mean()) > se), circuits=len(d),
                            best_wins_on=int((d > 0).sum() if higher else (d < 0).sum())))
    return pd.DataFrame(out)


def plot(frame: pd.DataFrame, out_dir: Path, note: Optional[str] = None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    from experiments.shared.figures import COLUMN_WIDTH, save_figure, use_style

    use_style()
    tab = summary(frame)
    q2 = tab[tab.budget == "Q2"]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.6), sharex=True)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    dx = {2: 0.88, 4: 0.94, 6: 1.0, 8: 1.06, 10: 1.12}
    for ax, metric, label in ((axes[0], "lsr", "shots kept / naive $k$=2"),
                              (axes[1], "logical_error", "logical error / naive $k$=2")):
        part = q2[q2.metric == metric]
        for i, k in enumerate(K_VALUES):
            s = part[(part.k == k) & (part.arm == "aqorn")].sort_values("ratio")
            if s.empty:
                continue
            ax.errorbar(s.ratio * dx[k], s.gm, yerr=[s.gm - s.lo, s.hi - s.gm], color=cycle[i],
                        marker="o", markersize=3.2, capsize=1.3, lw=1.1, linestyle="-",
                        label=f"AQORN $k$={k}")
        ax.axhline(1.0, color="#7a7a7a", lw=1.0, linestyle=(0, (4, 2)), zorder=0,
                   label="naive $k$=2")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}x"))
    axes[0].legend(fontsize=6.3, frameon=False, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, 1.30), handlelength=1.5, columnspacing=0.9)
    ax = axes[-1]
    ax.set_xscale("log", base=2)
    ticks = sorted(frame.ratio.unique())
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1/{1 / v:g}" if v < 1 else f"{v:g}"))
    ax.set_xlabel("idle / distance noise ratio ($\\leftarrow$ movement dominates)")
    if note:
        fig.suptitle(note, fontsize=6, color="#b00000")
    fig.tight_layout()
    paths = save_figure(fig, NAME, Path(__file__), override=out_dir)
    plt.close(fig)
    return paths


def main() -> None:
    from experiments.figures.resource_selection_headroom.resource_selection_headroom import (
        DEFAULT_RATIOS, Settings, load_circuits,
    )
    from experiments.shared.parallel import parallel_map
    from translator.logical_optimization import state

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--circuits", nargs="*", default=None)
    parser.add_argument("--ratios", nargs="*", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--fig8", default=str(FIG8))
    parser.add_argument("--gate", action="store_true", help="k = 2 at Q2, AQORN only, vs Fig 8")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--from-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        began = time.time()
        print(f"[{NAME}] logical optimisation: {state()}", flush=True)
        circuits = load_circuits(Settings(standard=True, circuits=args.circuits))
        alphas = _fig8_alphas(Path(args.fig8))
        jobs = jobs_for(circuits, args.ratios, alphas, args.gate)
        points = sum(len(j[4]) for j in jobs)
        print(f"[{NAME}] {len(circuits)} circuits x {len(args.ratios)} ratios = {len(jobs)} "
              f"columns, {points} (k, P) points x {1 if args.gate else 2} arms, "
              f"{args.jobs} workers", flush=True)
        produced = parallel_map(_run, jobs, jobs=args.jobs, label=NAME)
        frame = pd.DataFrame([row for rows in produced if rows for row in rows])
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = NAME + ("_gate" if args.gate else "") + (f"_{args.tag}" if args.tag else "")
        path = out / f"{stem}.csv"
        frame.to_csv(path, index=False)
        failed = [(j[0], j[2]) for j, r in zip(jobs, produced) if r is None]
        print(f"[{NAME}] {len(frame)} rows -> {path} in {(time.time() - began) / 60:.1f} min; "
              f"failed columns {failed}; miscompiled {int(frame.miscompiled.sum())}; "
              f"budget violations {int((frame.data_qubits + frame.ancillas_counted * ANCILLAS * frame.patches > frame.budget_qubits).sum())}",
              flush=True)
        if args.gate:
            ok = gate(frame, Path(args.fig8))
            print(f"[{NAME}] GATE {'PASSED' if ok else 'FAILED'}")
            raise SystemExit(0 if ok else 1)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    print(f"[{NAME}] TVD-saturated (dropped from TVD): {saturated(frame)}")
    print(winners(frame, "Q2").round(4).to_string(index=False))
    if args.output_dir:
        for p in plot(frame, Path(args.output_dir)):
            print(f"[{NAME}] saved {p}")


if __name__ == "__main__":
    main()
