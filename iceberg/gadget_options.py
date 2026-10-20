"""
gadget_options.py -- which gadget constructions a translation or a plan uses.

One option today, `transversal_intra_two_qubit`: an in-patch logical CZ / XCX at k = 2 (n = 4)
as a transversal single-qubit layer -- S (SX) on every data qubit, which on [[4,2,2]] is
CZbar . Zbar_1 Zbar_2 (XCXbar . Xbar_1 Xbar_2), plus the two Pauli corrections. No two-qubit
gate, so no silent fault at first order, against the 3-gate gadget's ~0.6 p2. It is part of the
standard configuration since 2026-10-07 (`packing.standard`, "w10T").

The constructions (`IcebergModule.CZ` / `XCX`) and the cost tables (`iceberg.costs.work_table`,
`depth_table`) read one process-wide flag, `translator.explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT`,
where the option was explored. `TranslatorConfig.transversal_intra_two_qubit` and the planners'
`transversal_intra_two_qubit` argument set that flag for the duration of a translation or a
plan (`transversal_intra`), so every table and construction inside the call agrees with the
config -- which is exactly how the measured w10T results were produced (the flag set for a
whole worker). A scope only ever turns the flag on: a script that set the global itself keeps
it on, so nothing that ran before this module existed changes behaviour.

Process-wide state is safe here because the code base parallelises with processes, not threads.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def transversal_intra_enabled() -> bool:
    """Whether in-patch CZ / XCX at k = 2 are built (and priced) transversally right now."""
    from translator import explore_switches

    return bool(explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT)


@contextmanager
def transversal_intra(on: bool) -> Iterator[None]:
    """Turn the transversal in-patch CZ / XCX on for the duration (no-op when `on` is False)."""
    from translator import explore_switches

    previous = explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT
    if on:
        explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT = True
    try:
        yield
    finally:
        explore_switches.TRANSVERSAL_INTRA_TWO_QUBIT = previous


@contextmanager
def solo_h_targeted(on: bool) -> Iterator[None]:
    """
    A Hadamard on a qubit alone in its patch as the targeted single-lane gadget, for the duration
    (no-op when `on` is False).

    By default such an H becomes a whole-patch HH -- the empty lanes are free to disturb -- in
    the Hadamard handler itself, so every arm gets it whether or not merging is on. That is
    single-qubit merging; `TranslatorConfig.solo_hh_promotion=False` scopes this on so a
    baseline can have none at all. Same switch as `explore_switches.SOLO_H_TARGETED`, and like
    `transversal_intra` a scope only ever turns it on.
    """
    from translator import explore_switches

    previous = explore_switches.SOLO_H_TARGETED
    if on:
        explore_switches.SOLO_H_TARGETED = True
    try:
        yield
    finally:
        explore_switches.SOLO_H_TARGETED = previous
