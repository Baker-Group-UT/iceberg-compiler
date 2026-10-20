"""
The density sweep's shared simulations (`--share-identical`): an arm that compiles to exactly
an earlier arm's circuit at the same density point takes that arm's simulated metrics, so a
step that changes nothing contributes zero instead of seed noise. Pinned: the key is the
circuit (one key for one circuit, different keys for different ones, the same key every
time); a shared arm's metrics are its source's exactly and only identical circuits are
shared; without the flag nothing is; and the recount reproduces every arm, shared ones
included, from the frames the sweep drew.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figures.density_ablation_sweep import density_references as refs
from experiments.figures.density_ablation_sweep.density_ablation_sweep import (
    SPLIT_ARMS,
    DensityAblationSettings,
    _run_one_circuit_standard,
    simulated_circuit_key,
    tight_trivial_packing,
)

QUANTITIES = ("lsr", "tvd", "silent_per_kept", "p_silent", "frame_lsr")


def program() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cz(2, 3)
    qc.h(1)
    qc.s(3)
    qc.cx(1, 2)
    qc.h(3)
    qc.measure_all()
    return qc


def config(arm: str):
    return refs._arm_config(arm, alpha=0.05, seed=7, commute=True, split=True)


def test_shared_arms_reads_the_rows_field():
    assert refs.shared_arms({"shared_arms": "h_merge=h_merge_1q;packed=h_merge_1q"}) == {
        "h_merge": "h_merge_1q", "packed": "h_merge_1q"}
    assert refs.shared_arms({"shared_arms": ""}) == {}
    assert refs.shared_arms({"shared_arms": float("nan")}) == {}
    assert refs.shared_arms({}) == {}


def test_one_key_per_circuit():
    solo = tight_trivial_packing(4, 4)
    key = simulated_circuit_key(program(), solo, config("h_merge"))
    assert simulated_circuit_key(program(), solo, config("h_merge")) == key
    # Merging off keeps a solo qubit's H as the targeted gadget; on, it is a whole-patch HH.
    assert simulated_circuit_key(program(), solo, config("baseline")) != key
    # Another placement of the same arm is another circuit.
    other = [solo[1], solo[0], solo[2], solo[3]]
    assert simulated_circuit_key(program(), other, config("h_merge")) != key


def _job(share: bool):
    settings = DensityAblationSettings(
        standard=True, matched_patches=True, split_merging=True, share_identical=share,
        pilot_repetitions=400, target_kept_shots=400, min_repetitions=400,
        max_repetitions=4_000)
    circuit = program()
    return (settings, "small", 0, "shared_test_n4", circuit, circuit)


@pytest.fixture(scope="module")
def shared_rows():
    return _run_one_circuit_standard(_job(True))


def _points(rows):
    return [row for row in rows if row["component"] == "packing"]


def test_a_shared_arm_is_its_source_exactly(shared_rows):
    points = _points(shared_rows)
    assert all(row["share_identical"] for row in points)
    assert all(row["logical_optimization"] == "off" for row in points)  # conftest: pass off
    seen = 0
    for row in points:
        for arm, source in refs.shared_arms(row).items():
            assert SPLIT_ARMS.index(source) < SPLIT_ARMS.index(arm)
            assert source not in refs.shared_arms(row)  # sources are simulated arms
            for quantity in QUANTITIES:
                assert row[f"{arm}_{quantity}"] == row[f"{source}_{quantity}"]
            for quantity in ("one_qubit_gates", "two_qubit_gates", "depth"):
                assert row[f"{arm}_{quantity}"] == row[f"{source}_{quantity}"]
            seen += 1
    assert seen
    # One qubit per patch: the packed arm is the +merge arm's circuit by construction.
    solo = next(row for row in points if row["n_codes"] == 4)
    assert "packed" in refs.shared_arms(solo)


def test_only_identical_circuits_are_shared(shared_rows):
    """Arms left unshared at a point compile to different circuits (or kept their own seed)."""
    for row in _points(shared_rows):
        took = refs.shared_arms(row)
        simulated = [arm for arm in SPLIT_ARMS if arm not in took]
        keys = []
        for arm in simulated:
            permutation = refs._permutation(arm, row, 4)
            keys.append(simulated_circuit_key(program(), permutation, config(arm)))
        assert len(set(keys)) == len(keys)


def test_without_the_flag_nothing_is_shared():
    rows = _run_one_circuit_standard(_job(False))
    assert all(row["shared_arms"] == "" and not row["share_identical"] for row in rows)


def test_the_recount_reproduces_shared_arms(shared_rows, monkeypatch):
    matched = pd.DataFrame(shared_rows)
    points = matched[matched["component"] == "packing"].to_dict("records")

    def circuits(_):
        yield "small", 0, "shared_test_n4", program(), points

    monkeypatch.setattr(refs, "_circuits", circuits)
    copies, items = refs._recount_work(matched)
    recounted = [row for item in items for row in refs._recount_item(item)]
    assert recounted and all(row["matches"] for row in recounted)
    arms_recounted = {(row["n_codes"], row["arm"]) for row in recounted}
    for row in points:
        if 2 * row["n_codes"] == 4:
            continue
        for arm in refs.shared_arms(row):
            assert (row["n_codes"], arm) in arms_recounted
