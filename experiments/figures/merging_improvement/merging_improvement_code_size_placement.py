"""
merging_improvement_code_size_placement.py
------------------------------------------

The code-size sweep of ``merging_improvement_code_size.py``, on its own output paths.

The two scripts run the identical experiment -- same proxy circuits, same three
cumulative techniques (h commutation, + merging, + packing), same metrics, same CSV
schema, same plot -- so their CSVs are comparable row for row. Only the default
``--csv`` and ``--output`` differ, which is what lets a configuration A/B run without
clobbering the base sweep: point one at ``--partitioner swapped``, or at ``--packer
raw``, and diff the two summaries.

The naming is historical. When it was written, the base script called ``grouping``
(signed graph -> multilevel partition -> Hungarian slots -> hill-climb on the analytic
cost) and this variant substituted the then-new ``placement`` package. ``placement`` has
since been renamed ``packing``, ``grouping`` has been archived, and the base script
calls ``packing`` too -- so there is no substitution left to make and this is now a thin
CLI over the shared machinery. The committed
``merging_improvement_code_size_placement.csv`` is from that first ``placement`` run and
still says ``packer=placement`` in its own column; the ``grouping``-era CSV it was meant
to be diffed against went to ``archive/experiments/figures/merging_improvement/``, since
nothing live can regenerate it.

``--partitioner`` selects the seed partitioner inside ``packing``; ``packing/seed.py``
records what each arm measured, including the dropped ``legacy`` one that called
``grouping.multilevel.partition``.

Run it the same way as the base script and diff the summaries::

    python -m experiments.figures.merging_improvement.merging_improvement_code_size_placement
    python -m experiments.figures.merging_improvement.merging_improvement_code_size_placement \
        --groups small medium --k-values 2 4 6 8 10

Everything -- loading, metrics, CSV schema, summary and plotting -- comes from the base
module rather than being copied, so a change to the experiment cannot drift between the
two variants.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import experiments.figures.merging_improvement.merging_improvement_code_size as base
from packing.seed import DEFAULT_PARTITIONER, PARTITIONERS

Perm = base.Perm

DEFAULT_OUTPUT = Path(
    "experiments/figures/merging_improvement/"
    "merging_improvement_code_size_placement.svg"
)
DEFAULT_CSV = Path(
    "experiments/figures/merging_improvement/"
    "merging_improvement_code_size_placement.csv"
)


def parse_args() -> base.CodeSizeSettings:
    parser = argparse.ArgumentParser(
        description="Code-size sweep on the packing package."
    )
    parser.add_argument("--qasm-root", type=Path, default=base.DEFAULT_SETTINGS.qasm_root)
    parser.add_argument(
        "--groups", nargs="+", choices=base.SIZE_GROUPS,
        default=list(base.DEFAULT_SETTINGS.groups),
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=list(base.DEFAULT_K_VALUES),
        help="Logical qubits per patch. Code size is k + 2.",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=list(base.METRICS),
        default=list(base.DEFAULT_SETTINGS.metrics),
    )
    parser.add_argument("--max-gates", type=int, default=base.DEFAULT_SETTINGS.max_proxy_gates)
    parser.add_argument("--max-qubits", type=int, default=base.DEFAULT_SETTINGS.max_proxy_qubits)
    parser.add_argument(
        "--proxy-timeout", type=float, default=base.DEFAULT_SETTINGS.proxy_timeout_seconds,
    )
    parser.add_argument(
        "--packer", choices=["packing", "raw"], default="packing",
        help="packing = the full pipeline, with the two-translation verify against the "
        "naive placement (the default); raw = the same search with verify off.",
    )
    parser.add_argument(
        "--partitioner", choices=sorted(PARTITIONERS), default=DEFAULT_PARTITIONER,
        help="Seed partitioner inside packing. See packing/seed.py, which records what "
        "each arm measured.",
    )
    parser.add_argument(
        "--min-patches", type=int, default=base.DEFAULT_SETTINGS.min_patches,
        help="Restrict the aggregate to circuits needing at least this many patches, "
        "so single-patch points do not dilute the packing segment.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--from-csv", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    return base.CodeSizeSettings(
        qasm_root=args.qasm_root,
        groups=args.groups,
        k_values=args.k_values,
        metrics=args.metrics,
        max_proxy_gates=args.max_gates,
        max_proxy_qubits=args.max_qubits,
        proxy_timeout_seconds=args.proxy_timeout,
        packer=args.packer,
        partitioner=args.partitioner,
        min_patches=args.min_patches,
        output_path=args.output,
        csv_path=args.csv,
        from_csv=args.from_csv,
        show_plot=not args.no_show,
    )


def main() -> None:
    settings = parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if settings.from_csv is not None:
        frame = base.read_csv(settings.from_csv)
        metrics = base.active_metrics(frame, settings.metrics)
        base.print_summary(frame, metrics, settings)
        base.plot_results(frame, metrics, settings)
        return

    if settings.csv_path.exists():
        settings.csv_path.unlink()

    print(f"[packing] packer={settings.packer} partitioner={settings.partitioner}")
    for group in settings.groups:
        circuits = base.load_group_circuits(group, settings)
        if not circuits:
            print(f"[packing] no circuits loaded for {group}, skipping")
            continue
        low = min(c.num_qubits for _, c in circuits)
        high = max(c.num_qubits for _, c in circuits)
        print(f"[packing] {group}: {len(circuits)} circuits, {low}-{high} qubits")
        frame = base.run_experiment(circuits, group, settings)
        if not frame.empty:
            base.append_csv(frame, settings.csv_path)

    if not settings.csv_path.exists():
        raise RuntimeError(
            f"no data produced -- check --qasm-root ({settings.qasm_root}) and filters"
        )

    frame = base.read_csv(settings.csv_path)
    metrics = base.active_metrics(frame, settings.metrics)
    base.print_summary(frame, metrics, settings)
    base.plot_results(frame, metrics, settings)


if __name__ == "__main__":
    main()
