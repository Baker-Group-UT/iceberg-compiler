"""
screen.py -- fast screening of compilation variants by measured silent error.

A variant is a plan source (which placement and commutation decision) plus translation
options. For every Fig 6 circuit, the baseline arm (naive, both passes off) and each
variant's arm are translated (the proxy, as Fig 6 simulates) and measured with
`stim.FlipSimulator` (`exp8_undetected.frames`) at Fig 6's per-circuit alpha and at alpha=1:

    p_detected            1 - LSR, to first approximation
    p_silent_given_kept   silent logical errors per kept shot; TVD ~ 0.95x this

From those, each variant's predicted Fig 6 effect against the baseline:

    lsr_impr   (1 - pd_variant) / (1 - pd_baseline) - 1
    tvd_red    1 - silent_kept_variant / silent_kept_baseline

Frame simulation sees only noise, so it cannot see a miscompile. Variants on circuits of
<= 20 qubits are therefore also simulated noiselessly through the pipeline and flagged if
their TVD exceeds the naive reference's (`experiments.shared.simulation.miscompiled`).

Usage:
    PYTHONPATH=. python -m experiments.tvd_explore.screen <variant> [<variant> ...] [--names ...]
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = Path("results/tvd_explore")
PLANS_CSV = Path("results/revision/exp8/planner_static.csv")
CHECK_MAX_QUBITS = 20

#: name -> plan source, translation switches (translator.explore_switches), translator
#: config overrides. Plan sources: "w<weight>" from Experiment8a, "naive", "sparse" (one
#: logical qubit per patch), each with the commutation decision of `commute_from`.
VARIANTS: Dict[str, Dict] = {
    "baseline": {"plan": "naive", "commute": False, "merging": False},
    "w0": {"plan": "w0"},
    "w10": {"plan": "w10"},
    "w30": {"plan": "w30"},
    "w1": {"plan": "w1"},
    "w3": {"plan": "w3"},
    "w10_hssx": {"plan": "w10", "switches": {"H_AS_S_SX_S": True}},
    "w3_hssx": {"plan": "w3", "switches": {"H_AS_S_SX_S": True}},
    "w30_hssx": {"plan": "w30", "switches": {"H_AS_S_SX_S": True}},
    "w10_after_merging": {"plan": "w10", "translator": {"h_commutation_stage": "after_merging"}},
    "w10_ft": {"plan": "w10", "translator": {"fault_tolerant": True}},
    # Transversal in-patch CZ/XCX (merging-stage macro): the gadget alone on weight 10's
    # plans, and the gadget with a planner that prices it (plan_variant w10T).
    "w10_T": {"plan": "w10", "switches": {"TRANSVERSAL_INTRA_TWO_QUBIT": True}},
    "w10T": {"plan": "csv:results/tvd_explore/plans_w10T.csv",
             "switches": {"TRANSVERSAL_INTRA_TWO_QUBIT": True}},
    "w10T_hssx": {"plan": "csv:results/tvd_explore/plans_w10T.csv",
                  "switches": {"TRANSVERSAL_INTRA_TWO_QUBIT": True, "H_AS_S_SX_S": True}},
    "w30T": {"plan": "csv:results/tvd_explore/plans_w30T.csv",
             "switches": {"TRANSVERSAL_INTRA_TWO_QUBIT": True}},
    "w100T": {"plan": "csv:results/tvd_explore/plans_w100T.csv",
              "switches": {"TRANSVERSAL_INTRA_TWO_QUBIT": True}},
    "sparse_w10commute": {"plan": "sparse", "commute_from": "w10"},
    "sparse_w10commute_hssx": {"plan": "sparse", "commute_from": "w10",
                               "switches": {"H_AS_S_SX_S": True}},
}


def plan_for(spec: Dict, name: str, n: int, plans: Dict[str, Tuple[list, bool]]):
    """(placement, commute, merging) for one variant on one circuit."""
    source = spec["plan"]
    merging = spec.get("merging", True)
    if source == "naive":
        placement = [(i // 2, i % 2) for i in range(n)]
        commute = spec.get("commute", plans["w0"][1])
    elif source == "sparse":
        placement = [(i, 0) for i in range(n)]
        commute = plans[spec.get("commute_from", "w10")][1]
    else:
        placement, commute = plans[source]
    return [tuple(int(v) for v in p) for p in placement], bool(spec.get("commute", commute)), merging


def _translate(program, placement, *, commute: bool, merging: bool, overrides: Dict):
    from pipeline.config import PipelineConfig
    from translator.core import Translator

    config = PipelineConfig(verbose=False, save=False)
    config.translator_config.n_code = 4
    config.translator_config.enable_h_commutation = commute
    config.translator_config.enable_merging = merging
    for key, value in overrides.items():
        setattr(config.translator_config, key, value)
    return Translator(config=config.translator_config).translate_batch(
        program, [list(placement)])[0].circuit


def _noiseless_tvd(program, placement, *, commute, merging, overrides, seed) -> float:
    from experiments.revision.common import NoiseSetup, sim_config, simulate_arm

    config = sim_config(NoiseSetup(alpha=0.0), n_code=4, commute=commute, merging=merging,
                        repetitions=4000, seed=seed)
    for key, value in overrides.items():
        setattr(config.translator_config, key, value)
    return float(simulate_arm(program, placement, config)["tvd"])


def point(job) -> List[Dict]:
    from experiments.shared.simulation import ensure_measurements, miscompiled
    from translator import explore_switches

    circuit, variant, spec, plans, alpha6 = job
    explore_switches.configure(**spec.get("switches", {}))
    proxy = ensure_measurements(circuit.proxy)
    n = proxy.num_qubits
    placement, commute, merging = plan_for(spec, circuit.name, n, plans)
    overrides = spec.get("translator", {})
    began = time.time()
    encoded = _translate(proxy, placement, commute=commute, merging=merging, overrides=overrides)
    row = {"name": circuit.name, "size_group": circuit.size_group, "num_qubits": n,
           "variant": variant, "commute": commute, "merging": merging,
           "patches": len({p for p, _ in placement}),
           "physical_qubits": encoded.num_qubits,
           "two_qubit_gates": sum(1 for i in encoded.data if len(i.qubits) == 2
                                  and i.operation.name not in ("measure", "barrier")),
           "depth": encoded.depth(), "placement": json.dumps([list(p) for p in placement])}
    from experiments.tvd_explore.fastframes import splits

    measured = splits(encoded, (alpha6, 1.0), shots=20_000, seed=11)
    for tag, alpha in (("fig6", alpha6), ("a1", 1.0)):
        split = measured[alpha]
        row.update({f"pd_{tag}": split["p_detected"], f"sk_{tag}": split["p_silent_given_kept"]})
    if n <= CHECK_MAX_QUBITS and variant != "baseline":
        naive = [(i // 2, i % 2) for i in range(n)]
        reference = _noiseless_tvd(proxy, naive, commute=commute, merging=merging,
                                   overrides={}, seed=5)
        clean = _noiseless_tvd(proxy, placement, commute=commute, merging=merging,
                               overrides=overrides, seed=5)
        row.update({"noiseless_tvd": clean, "noiseless_ref": reference,
                    "miscompiled": bool(miscompiled(reference, clean))})
    row["seconds"] = time.time() - began
    explore_switches.configure(**{k: False for k in spec.get("switches", {})})
    return [row]


def load_plans(name: str, frame: pd.DataFrame) -> Dict[str, Tuple[list, bool]]:
    block = frame[frame["name"] == name]
    return {f"w{w:g}": (json.loads(r.placement), bool(r.commute))
            for w, r in zip(block["silent_weight"], block.itertuples())}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted Fig 6 effects per variant against the baseline (Fig 6's TVD mask)."""
    from experiments.revision.common import fig6_reference

    ref = fig6_reference().set_index("name")
    base = frame[frame["variant"] == "baseline"].set_index("name")
    rows = {}
    for variant, block in frame[frame["variant"] != "baseline"].groupby("variant"):
        b = block.set_index("name").join(base, rsuffix="_b", how="inner")
        resolved = b.index[ref.reindex(b.index)["baseline_tvd"] <= 0.95]
        out = {"circuits": len(b)}
        for tag in ("fig6", "a1"):
            lsr = (1 - b[f"pd_{tag}"]) / (1 - b[f"pd_{tag}_b"]) - 1
            tvd = 1 - b[f"sk_{tag}"] / b[f"sk_{tag}_b"]
            out[f"LSR impr % ({tag})"] = 100 * lsr.mean()
            out[f"TVD red % ({tag})"] = 100 * tvd.loc[resolved].replace([np.inf, -np.inf], np.nan).mean()
        out["phys qubits x"] = float((b["physical_qubits"] / b["physical_qubits_b"]).mean())
        out["miscompiled"] = int(b.get("miscompiled", pd.Series(dtype=bool)).fillna(False).sum())
        rows[variant] = out
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    from experiments.revision.common import fig6_reference, load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--tag", default="screen")
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.tag}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["name", "variant"])
    have = set(zip(done["name"], done["variant"]))
    plans_frame = pd.read_csv(PLANS_CSV)
    ref = fig6_reference().set_index("name")
    circuits = load_fig6_circuits(names=args.names)
    variants = ["baseline"] + [v for v in args.variants if v != "baseline"]
    csv_sources = {VARIANTS[v]["plan"] for v in variants if VARIANTS[v]["plan"].startswith("csv:")}
    from_csv = {source: pd.read_csv(source[4:]).set_index("name") for source in csv_sources}

    def plans_for(name):
        plans = load_plans(name, plans_frame)
        for source, frame in from_csv.items():
            plans[source] = (json.loads(frame.loc[name, "placement"]), bool(frame.loc[name, "commute"]))
        return plans

    jobs = [(c, v, VARIANTS[v], plans_for(c.name), float(ref.loc[c.name, "alpha_final"]))
            for c in circuits for v in variants if (c.name, v) not in have]
    jobs.sort(key=lambda j: -j[0].num_qubits)
    print(f"[screen] {len(jobs)} (circuit, variant) points")
    results = parallel_map(point, jobs, jobs=args.jobs, label="screen")
    failed = [(j[0].name, j[1]) for j, r in zip(jobs, results) if r is None]
    rows = [row for r in results if r for row in r]
    frame = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    frame.to_csv(path, index=False)
    print(f"[screen] wrote {path}; failed: {failed}")
    print(summarize(frame).round(2).to_string())


if __name__ == "__main__":
    main()
