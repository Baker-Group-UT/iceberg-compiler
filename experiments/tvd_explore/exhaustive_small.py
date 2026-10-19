"""
exhaustive_small.py -- measured exhaustive placement for small programs.

Small circuits are where the model is weakest: Fig 6 calibrates them to alpha 6-21, where
pairs of faults whose syndromes cancel cause more than half the silent error, which no
per-gadget model sees. They are also the circuits where measuring every placement is cheap:
at k = 2 and n <= 10 the tightest packing has at most 945 partitions; above `--top-k`
the weight-10 model's best `top_k` per commutation setting are measured.

For every Fig 6 circuit with n <= `--max-qubits`: every partition of the program qubits into
ceil(n/2) patches of <= 2 (Fig 6's tightest packing), patches ordered by lowest qubit, lanes
by `packing.lanes.assign_lanes`, under both H-commutation settings -- plus the baseline arm.
Each candidate is translated (merging on) and measured with `stim.FlipSimulator` at the
circuit's Fig 6 alpha and at alpha = 1 (one conversion, rescaled: `fastframes`).

Every candidate's measurement is written, so selection rules are applied offline
(`select()`); the default is the lowest silent-per-kept at the target alpha among
candidates keeping at least the baseline's shots. A selected placement is a compile
decision -- its reported TVD has to come from an independent simulation (Fig 6's seeds),
not from the frames it was selected on.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd

OUT = Path("results/tvd_explore")
CHUNK = 24


def partitions(n: int) -> List[List[frozenset]]:
    """Every tightest partition (ceil(n/2) patches of <= 2), patches ordered by lowest qubit."""
    from experiments.revision.exp7_optimality.space import set_partitions

    tightest = math.ceil(n / 2)
    out = []
    for labels in set_partitions(n, 2):
        if max(labels) + 1 != tightest:
            continue
        groups = [frozenset(i for i, b in enumerate(labels) if b == block)
                  for block in range(tightest)]
        out.append(sorted(groups, key=min))
    return out


def prepare(job) -> List[tuple]:
    """One circuit's candidate chunks: all partitions, or the weight-10 model's top_k."""
    from experiments.revision.common import fig6_noise_profile
    from experiments.revision.exp8_undetected.cost_model import build_costs
    from experiments.revision.exp8_undetected.gadgets import load_rates
    from experiments.shared.simulation import ensure_measurements
    from packing.lanes import assign_lanes

    circuit, alpha6, shots, top_k, switches, rates_csv = job
    from experiments.revision.exp8_undetected.gadgets import UndetectedRates
    from translator import explore_switches

    explore_switches.configure(**{name: True for name in switches})
    rates = UndetectedRates.from_table(pd.read_csv(rates_csv), 4, 3e-3)
    proxy = ensure_measurements(circuit.proxy)
    n = proxy.num_qubits
    every = partitions(n)
    entries = [(-1, [(i // 2, i % 2) for i in range(n)], False, False)]  # the baseline arm
    index = 0
    for commute in (True, False):
        _, cost = build_costs(proxy, 4, commute=commute, noise=fig6_noise_profile(),
                              rates=rates, silent_weight=10.0)
        kept = sorted(every, key=cost.of)[:top_k] if len(every) > top_k else every
        for groups in kept:
            placement = [tuple(int(v) for v in p) for p in assign_lanes(groups, cost)]
            entries.append((index, placement, commute, True))
            index += 1
    return [(circuit, entries[start:start + CHUNK], alpha6, shots, switches)
            for start in range(0, len(entries), CHUNK)]


def point(job) -> List[Dict]:
    from experiments.revision.common import fig6_noise_profile
    from experiments.shared.simulation import ensure_measurements
    from experiments.tvd_explore.fastframes import splits
    from experiments.tvd_explore.screen import _translate

    circuit, chunk, alpha6, shots, switches = job
    from translator import explore_switches

    proxy = ensure_measurements(circuit.proxy)
    rows = []
    for index, placement, commute, merging in chunk:
        began = time.time()
        # Switches are merging-stage macros: on for candidates, off for the baseline arm.
        explore_switches.configure(**{name: bool(merging) for name in switches})
        encoded = _translate(proxy, placement, commute=commute, merging=merging, overrides={})
        measured = splits(encoded, (alpha6, 1.0), shots=shots, seed=13)
        rows.append({
            "name": circuit.name, "num_qubits": circuit.num_qubits, "candidate": index,
            "commute": commute, "merging": merging,
            "placement": json.dumps([list(p) for p in placement]),
            "two_qubit_gates": sum(1 for i in encoded.data if len(i.qubits) == 2
                                   and i.operation.name not in ("measure", "barrier")),
            "depth": encoded.depth(),
            "pd_fig6": measured[alpha6]["p_detected"], "sk_fig6": measured[alpha6]["p_silent_given_kept"],
            "pd_a1": measured[1.0]["p_detected"], "sk_a1": measured[1.0]["p_silent_given_kept"],
            "seconds": time.time() - began,
        })
    return rows


def select(frame: pd.DataFrame, tag: str = "fig6") -> pd.DataFrame:
    """Per circuit: the lowest silent-per-kept candidate keeping >= the baseline's shots."""
    rows = []
    for name, block in frame.groupby("name"):
        base = block[block["candidate"] == -1].iloc[0]
        pool = block[block["candidate"] >= 0]
        keeps = pool[(1 - pool[f"pd_{tag}"]) >= (1 - base[f"pd_{tag}"])]
        chosen = (keeps if not keeps.empty else pool).sort_values(f"sk_{tag}").iloc[0]
        rows.append({"name": name, "candidate": int(chosen["candidate"]),
                     "placement": chosen["placement"], "commute": bool(chosen["commute"]),
                     "lsr_impr": (1 - chosen["pd_fig6"]) / (1 - base["pd_fig6"]) - 1,
                     "tvd_red": 1 - chosen["sk_fig6"] / base["sk_fig6"],
                     "lsr_impr_a1": (1 - chosen["pd_a1"]) / (1 - base["pd_a1"]) - 1,
                     "tvd_red_a1": 1 - chosen["sk_a1"] / base["sk_a1"],
                     "constrained": not keeps.empty, "pool": len(pool)})
    return pd.DataFrame(rows)


def main() -> None:
    from experiments.revision.common import fig6_reference, load_fig6_circuits
    from experiments.shared.parallel import parallel_map

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--shots", type=int, default=50_000)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--switch", action="append", default=[])
    parser.add_argument("--rates", default="results/revision/exp8/gadget_rates.csv")
    parser.add_argument("--tag", default="exhaustive_small")
    parser.add_argument("--top-k", type=int, default=105,
                        help="Above this many partitions, measure only the weight-10 "
                             "model's best top-k per commutation setting.")
    args = parser.parse_args()
    ref = fig6_reference().set_index("name")
    names = args.names or [n for n in ref.index if int(ref.loc[n, "num_qubits"]) <= args.max_qubits]
    circuits = load_fig6_circuits(names=names)
    prepared = parallel_map(prepare, [(c, float(ref.loc[c.name, "alpha_final"]), args.shots,
                                       args.top_k, tuple(args.switch), args.rates)
                                      for c in circuits], jobs=args.jobs, label="prepare")
    jobs = [j for chunks in prepared if chunks for j in chunks]
    print(f"[exhaustive] {len(circuits)} circuits, {sum(len(j[1]) for j in jobs)} candidates "
          f"in {len(jobs)} jobs")
    results = parallel_map(point, jobs, jobs=args.jobs, label="exhaustive")
    frame = pd.DataFrame([row for r in results if r for row in r])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{args.tag}.csv", index=False)
    chosen = select(frame)
    chosen.to_csv(OUT / f"{args.tag}_selected.csv", index=False)
    print(f"[exhaustive] {sum(r is None for r in results)} jobs failed")
    print(chosen[["name", "pool", "constrained", "lsr_impr", "tvd_red"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
