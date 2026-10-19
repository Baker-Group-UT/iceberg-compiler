"""
The two-level construction and its decoders (experiments/concatenation).

Pinned: noiseless, every check passes and the decoded distribution is the program's; spread is
what the layouts claim; and the decoders do what they say on single located faults -- one flipped
data bit in any inner patch is discarded by post-selection, and corrected by the erasure decoder
(the flagged inner patch's two qubits sit in different outer patches, one erasure each).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.concatenation import construct as C
from experiments.concatenation import decode as D


def _program() -> QuantumCircuit:
    program = QuantumCircuit(4)
    program.h(0); program.cx(0, 1); program.cx(1, 2); program.s(3); program.h(3)
    program.cx(2, 3); program.h(2); program.sx(1)
    return program


def _two(layout: str, merging: bool):
    return C.compile_two_level(_program(), [(0, 0), (0, 1), (1, 0), (1, 1)], outer_commute=False,
                               outer_merging=False, inner_layout=layout, inner_merging=merging,
                               inner_commute=merging, require_spread=layout != "index")


@pytest.mark.parametrize("layout", ["stride", "role_aligned", "index"])
def test_layouts_are_spread_as_claimed(layout):
    two = _two(layout, False)
    assert C.is_spread(two.inner_placement, two.flat.roles) == (layout != "index")


@pytest.mark.parametrize("merging", [False, True])
def test_noiseless_two_level_reproduces_the_program(merging):
    from experiments.revision.common import exact_ideal_distribution
    from simulator import stim_backend as sb
    from simulator.grid_constructor import positions_for_circuit

    two = _two("stride", merging)
    maps = D.maps_for(two)
    clean = sb.qiskit_to_stim(two.circuit, layout=positions_for_circuit(
        two.circuit, geometry=sb._geometry()), include_noise=False)
    out = D.decode(clean.compile_sampler(seed=5).sample(20_000), maps)
    assert out["postselect"].all() and out["erasure"].all()
    keys = ["".join("1" if b else "0" for b in row) for row in out["postselect_out"]]
    counts = Counter(keys)
    ideal = exact_ideal_distribution(_program())
    tvd = 0.5 * sum(abs(counts.get(k, 0) / 20_000 - ideal.get(k, 0)) for k in set(counts) | set(ideal))
    assert tvd < 0.03
    assert np.array_equal(out["postselect_out"], out["erasure_out"])


def test_single_flipped_data_bit_is_discarded_or_corrected():
    two = _two("stride", True)
    maps = D.maps_for(two)
    data_rows = [row for row, (name, _) in enumerate(maps.records) if name.startswith("m_data_")]
    flips = np.zeros((len(data_rows), len(maps.records)), dtype=bool)
    for shot, row in enumerate(data_rows):
        flips[shot, row] = True
    out = D.decode(flips, maps)
    assert not out["postselect"].any()
    assert out["erasure"].all()
    assert not out["erasure_out"].any()   # every output flip undone


def test_index_layout_cannot_correct_its_flagged_patch():
    two = _two("index", True)
    maps = D.maps_for(two)
    data_rows = [row for row, (name, _) in enumerate(maps.records) if name.startswith("m_data_")]
    flips = np.zeros((len(data_rows), len(maps.records)), dtype=bool)
    for shot, row in enumerate(data_rows):
        flips[shot, row] = True
    out = D.decode(flips, maps)
    # an inner patch holding two data qubits of one outer patch is two erasures there: discarded
    holds_two_data = []
    for shot, row in enumerate(data_rows):
        name = maps.records[row][0]
        patch = int(name.rsplit("_", 1)[1])
        members = [q for q, home in enumerate(maps.flat_home) if home == patch]
        outer_data = [q for qs in maps.outer_data.values() for q in qs]
        holds_two_data.append(sum(q in outer_data for q in members) == 2)
    assert not out["erasure"][np.array(holds_two_data)].any()
