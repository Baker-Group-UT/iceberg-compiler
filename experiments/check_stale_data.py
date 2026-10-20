"""
check_stale_data.py -- which cached figure data predates the current cost model.

Every figure here caches hours of simulation to CSV, and a change to the planner makes
those numbers describe a compiler that no longer exists. Nothing detected that until
`packing.version` and the `cost_model_version` column existed, so this script exists to
answer one question before a deadline: *what do I have to rerun?*

Reports each cached CSV as CURRENT, STALE, or UNSTAMPED. Unstamped means the file predates
provenance entirely -- given when that landed, treat it as stale.

    PYTHONPATH=. python experiments/check_stale_data.py
    PYTHONPATH=. python experiments/check_stale_data.py --stale-only

Exit status is 1 when anything is stale, so it can gate a regeneration script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from experiments.shared.figures import PROVENANCE_COLUMN, project_root
from packing.version import COST_MODEL_VERSION

#: Where cached figure data lives, and the two roots are *different*. `experiments/figures`
#: is inside this code repo, which stands alone with its own remote; `results/data` is in the
#: paper project one level up, which is what `project_root` finds by looking for
#: `docs/figure-style.md`. Searching only the paper root silently missed every working cache.
SEARCH = (
    (Path(__file__).resolve().parents[1], "experiments/figures"),
    (None, "results/data"),
)


def scan(root: Path) -> List[Tuple[Path, str, object, int]]:
    rows: List[Tuple[Path, str, object, int]] = []
    for anchor, where in SEARCH:
        base = (anchor or root) / where
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except Exception as error:  # noqa: BLE001 - an unreadable cache is worth naming
                rows.append((path, "UNREADABLE", repr(error)[:40], 0))
                continue
            if PROVENANCE_COLUMN not in frame.columns:
                rows.append((path, "UNSTAMPED", None, len(frame)))
                continue
            found = sorted({int(v) for v in frame[PROVENANCE_COLUMN].dropna().unique()})
            if len(found) > 1:
                rows.append((path, "MIXED", found, len(frame)))
            elif found and found[0] == COST_MODEL_VERSION:
                rows.append((path, "CURRENT", found[0], len(frame)))
            else:
                rows.append((path, "STALE", found[0] if found else None, len(frame)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--stale-only", action="store_true",
                        help="Hide the files that are up to date.")
    args = parser.parse_args()

    root = project_root() or Path(__file__).resolve().parents[1]
    rows = scan(root)
    if not rows:
        print("no cached CSVs found")
        return 0

    print(f"current cost model version: {COST_MODEL_VERSION}\n")
    counts = {}
    for path, status, version, length in rows:
        counts[status] = counts.get(status, 0) + 1
        if args.stale_only and status == "CURRENT":
            continue
        shown = "-" if version is None else version
        try:
            name = path.relative_to(root)
        except ValueError:
            name = path
        print(f"  {status:<10s} v{str(shown):<5s} {length:>6d} rows  {name}")

    print("\n" + "  ".join(f"{status}: {count}" for status, count in sorted(counts.items())))
    outdated = sum(counts.get(status, 0) for status in ("STALE", "UNSTAMPED", "MIXED"))
    if outdated:
        print(f"\n{outdated} file(s) need regenerating before their numbers can be quoted.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
