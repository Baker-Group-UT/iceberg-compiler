"""
testexp.py — random Clifford round-trip probe across code sizes.

Generates a random Clifford program, simulates it as the reference, then translates
it onto a random permutation at k = 2, 4 and 6, printing the seed, the permutation,
the logical circuit and the translated circuit for each. Loops with the seed
increasing forever.

Everything here is assembled from existing pieces:

    experiments.shared.random_circuits.random_gate_circuit   the shared builder
    permutations.generator.random_permutation_from_circuit   the placement
    pipeline.translate_permutation                           the translation
    simulator.simulator.Simulator                            reference + noisy runs
    pipeline.analysis.postselection.postselect_against_reference

Those last two used to be one call to `pipeline.generate_random_circuit`, which offers
the sampler one patch per logical qubit. Splitting them is what `--packing` needs: see
PACKINGS for why the resulting sparse layout is a poor default for a correctness fuzz.
`--packing sparse` reproduces `generate_random_circuit` exactly, verified over 120
(seed, k) pairs, so nothing that used to be covered has stopped being covered.

The default is `--alpha 0`, which `ScaledNoiseModel` treats as explicitly noiseless.
That makes the comparison a correctness check rather than a noise measurement: the
postselected logical distribution should match the reference up to sampling error, so
LSR reads 1.000 and TVD sits near zero. A translation that does not preserve the
logical action shows up as a TVD jump on one k and not the others, next to the exact
seed and permutation that produced it. Raise `--alpha` to make it a noise probe
instead.

Both distributions are sampled, so the TVD floor is set by shot count and by how
widely the program spreads its support -- not by anything the compiler did. At the
defaults (4 qubits, 4000 shots) it runs 0.00-0.03; at 7 qubits it reaches 0.04, which
is close enough to the 0.05 threshold to produce false flags. Raise `--repetitions`
alongside `--qubits`, or raise `--tvd-threshold` to match.

Measured over seeds 0-14 at the defaults: 45 translations, fidelity 0.9995-1.0000,
LSR 1.0000 throughout, nothing flagged.

    PYTHONPATH=. venv/bin/python -m experiments.tests.testexp
    PYTHONPATH=. venv/bin/python -m experiments.tests.testexp --packing dense
    PYTHONPATH=. venv/bin/python -m experiments.tests.testexp --seed 40 --qubits 6
    PYTHONPATH=. venv/bin/python -m experiments.tests.testexp --qubits 7 --repetitions 20000
    PYTHONPATH=. venv/bin/python -m experiments.tests.testexp --stop-on-error
"""

from __future__ import annotations

import argparse
import itertools
import math
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import qiskit

from iceberg.utils import logical_capacity
from permutations.code_sets import generate_uniform_code_sets
from permutations.generator import random_permutation_from_circuit
from pipeline import translate_permutation
from analysis.postselection import postselect_against_reference
from pipeline.config import PipelineConfig
from simulator.config_presets import ScaledSimulator
from simulator.simulator import Simulator

from experiments.shared.random_circuits import random_gate_circuit

#: The translator's Clifford gate set, and `random_gate_circuit`'s own default.
CLIFFORD_GATES = ["h", "x", "z", "s", "cx", "cz"]

#: Logical qubits per patch to try, i.e. the [[k+2, k, 2]] codes.
K_VALUES = (2, 4, 6, 8, 10, 12)

#: How many patches the random permutation is allowed to scatter across.
#:
#: `sparse` offers one patch per logical qubit, which is what
#: `pipeline.generate_random_circuit` does. At k=6 a 4-qubit program then gets 4 patches
#: of 6 lanes -- 24 slots for 4 qubits -- so the sampled permutation usually spreads over
#: two or three near-empty patches. That exercises the sparse placement paths and little
#: else: more physical qubits to simulate, and the dense cases (full patches, whole-patch
#: macros, transversal alignment across matched lanes) barely come up.
#:
#: `dense` offers exactly `ceil(q/k)` patches, so the permutation has nowhere to spread
#: and every patch but the last is full.
#:
#: Both are worth running. The seed-32 bug lived in code that assumed a *full* patch, so
#: only the sparse layout could reach it; the whole-patch macros conversely need the dense
#: one to fire at all.
PACKINGS = ("dense", "sparse")


def build_config(
    *,
    n_code: int,
    seed: int,
    repetitions: int,
    alpha: float,
    sim_seed: Optional[int] = None,
    merging: bool = True,
) -> PipelineConfig:
    """
    A pipeline config for one code size.

    `PipelineConfig.seed` is the permutation seed -- it is what
    `generate_random_circuit` samples the placement from, and it is the seed printed
    per iteration. The *simulator* seed is separate, and `--sim-seed` sets it.

    It stays unset by default so that each iteration draws fresh shots, which keeps a
    borderline TVD from being borderline the same way every run. Seeding it is safe
    now: it used to cost 99% of the shots, because `Simulator` chunks a run into
    `repetitions / 100` pieces and one fixed `seed_simulator` made every chunk a replay
    of the first. That is fixed and pinned by `tests/test_simulator_seeding.py`.
    """
    config = PipelineConfig(
        sim_config=ScaledSimulator(repetitions=repetitions, seed=sim_seed, alpha=alpha),
        verbose=False,
        save=False,
        seed=seed,
    )
    config.translator_config.n_code = n_code
    config.translator_config.enable_merging = merging
    return config


def sample_permutation(
    program: qiskit.QuantumCircuit, n_code: int, seed: int, packing: str
) -> List[Tuple[int, int]]:
    """
    A random placement of `program` onto patches of size `n_code`.

    Same two calls `pipeline.generate_random_circuit` makes; the only difference is how
    many patches are offered, which is what `packing` selects. `dense` asks for the
    fewest that can hold the program, `sparse` for one per logical qubit -- see PACKINGS.
    """
    k = logical_capacity(n_code)
    patches = math.ceil(program.num_qubits / k) if packing == "dense" else program.num_qubits
    return random_permutation_from_circuit(
        circuit=program,
        code_sizes=generate_uniform_code_sets(max(1, patches), n=n_code),
        seed=seed,
    )


def tvd_floor(reference: Optional[Dict[str, float]], shots: int, margin: float = 3.0) -> float:
    """
    A TVD threshold derived from the sampling error, not guessed.

    Both distributions are sampled, so the TVD between two runs of the *same* circuit is
    not zero. For a distribution on `S` outcomes and `N` shots each, the expected value is
    about `sqrt(S / (pi N))` — it grows with the support and shrinks only as `1/sqrt(N)`.

    A fixed threshold therefore cannot work across circuit widths. At 4 qubits 0.05 was
    comfortable; at 10 qubits the floor is already ~0.04 and every honest run sits a hair
    under the line. The same mistake made `test_pipeline_roundtrip` flaky once, and it was
    fixed there the same way.

    `margin` is how many multiples of that floor to allow. 3 leaves ample room while
    staying far below the 0.5–0.75 a dropped gate produces.
    """
    support = len(reference) if reference else 1
    return max(1e-3, margin * math.sqrt(support / (math.pi * max(1, shots))))


def format_distribution(dist: Optional[Dict[str, float]], limit: int = 6) -> str:
    """The `limit` most likely outcomes, so a whole distribution stays one line."""
    if not dist:
        return "<empty>"
    ordered = sorted(dist.items(), key=lambda item: item[1], reverse=True)
    shown = ", ".join(f"{key}:{value:.4f}" for key, value in ordered[:limit])
    rest = len(ordered) - limit
    return shown + (f", (+{rest} more)" if rest > 0 else "")


def format_permutation(permutation: Sequence[Tuple[int, int]]) -> str:
    """`q0->(patch, slot)` per logical qubit, plus the patch count actually used."""
    body = "  ".join(
        f"q{qubit}->({patch},{slot})"
        for qubit, (patch, slot) in enumerate(permutation)
    )
    patches = sorted({patch for patch, _ in permutation})
    noun = "patch" if len(patches) == 1 else "patches"
    return f"{body}   [{len(patches)} {noun}: {patches}]"


def draw(circuit: qiskit.QuantumCircuit, fold: Optional[int]) -> str:
    return circuit.draw(output="text", fold=fold).single_string()


def run_case(
    program: qiskit.QuantumCircuit,
    reference,
    seed: int,
    n_code: int,
    packing: str,
    merging: bool,
    *,
    repetitions: int,
    alpha: float,
    fold: Optional[int],
    tvd_threshold: float,
    sim_seed: Optional[int],
) -> bool:
    """
    One (code size, packing, merging) combination. True when it looks right.

    Splitting merging into its own axis is what makes a failure self-diagnosing: a case
    that fails with merging on and passes with it off is a merging bug, and one that fails
    both ways is in placement or the gate layer. That distinction took a manual bisect
    every time before.
    """
    num_qubits = program.num_qubits
    k = logical_capacity(n_code)
    tightest = math.ceil(num_qubits / k)

    print("\n" + "-" * 100)
    print(
        f"seed {seed}   k={k}   [[{n_code},{k},2]] code   {packing} packing"
        + (f" ({tightest} patch{'es' if tightest != 1 else ''})"
           if packing == "dense" else f" (up to {num_qubits} patches)")
        + f"   merging {'on' if merging else 'off'}"
    )
    print("-" * 100)

    config = build_config(
        n_code=n_code, seed=seed, repetitions=repetitions, alpha=alpha,
        sim_seed=sim_seed, merging=merging,
    )
    permutation = sample_permutation(program, n_code, seed, packing)
    try:
        translated = translate_permutation(program, permutation, config=config)
    except Exception as error:  # noqa: BLE001 - the loop should survive one bad case
        print(f"  permutation: {format_permutation(permutation)}")
        print(f"  TRANSLATION FAILED: {type(error).__name__}: {error}")
        traceback.print_exc()
        return False

    print(f"  permutation: {format_permutation(translated.permutation)}")
    print(f"  physical:    {translated.circuit.num_qubits} qubits, "
          f"{translated.circuit.size()} gates, depth {translated.circuit.depth()}")
    if translated.fault_log:
        print(f"  fault log:   {len(translated.fault_log)} entries: "
              f"{translated.fault_log[:3]}")
    print("\n  translated circuit")
    if fold is not None:
        print(draw(translated.circuit, fold))

    noisy = Simulator(config.sim_config).noisy(translated.circuit)
    result = postselect_against_reference([translated], reference, [noisy])[0]
    ok = result.logical_total_variation_distance <= tvd_threshold
    print(
        f"\n  logical fidelity {result.logical_fidelity:.4f}   "
        f"TVD {result.logical_total_variation_distance:.4f}   "
        f"LSR {result.logical_selection_rate:.4f}"
        + ("" if ok else "   <-- MISMATCH")
    )
    print(f"  postselected: {format_distribution(result.postselected_noisy_dist)}")
    return ok


def run_seed(
    seed: int,
    *,
    num_qubits: int,
    num_gates: int,
    repetitions: int,
    alpha: float,
    fold: Optional[int],
    tvd_threshold: Optional[float],
    stop_on_error: bool,
    sim_seed: Optional[int] = None,
    packings: Sequence[str] = PACKINGS,
    mergings: Sequence[bool] = (True, False),
) -> bool:
    """
    One seed across every code size, packing density and merging setting.

    Returns False if something is worth stopping for.
    """
    program = random_gate_circuit(num_qubits, num_gates, CLIFFORD_GATES, seed=seed)
    logical_gates = program.size()
    logical_depth = program.depth()
    program.measure_all()

    print("\n" + "=" * 100)
    print(f"seed {seed}   |   {num_qubits} qubits, {logical_gates} gates, depth {logical_depth}")
    print("=" * 100)
    print("\nlogical circuit")
    print(draw(program, fold))

    # The reference is the logical program simulated without noise. One per seed,
    # shared across every k -- the code size does not change what the program means.
    reference_config = build_config(
        n_code=4, seed=seed, repetitions=repetitions, alpha=alpha, sim_seed=sim_seed
    )
    reference = Simulator(reference_config.sim_config).ideal(program)
    print(f"\nreference (ideal) distribution: {format_distribution(reference.ideal_dist)}")
    if tvd_threshold is None:
        tvd_threshold = tvd_floor(reference.ideal_dist, repetitions)
        print(f"reference support {len(reference.ideal_dist or ())} outcomes over "
              f"{repetitions} shots -> flagging TVD above {tvd_threshold:.4f}")

    clean = True
    for k in K_VALUES:
        for packing in packings:
            for merging in mergings:
                ok = run_case(
                    program, reference, seed, k + 2, packing, merging,
                    repetitions=repetitions, alpha=alpha, fold=fold,
                    tvd_threshold=tvd_threshold, sim_seed=sim_seed,
                )
                if not ok:
                    clean = False
                    if stop_on_error:
                        return False

    return clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=32, help="First seed; increments forever.")
    parser.add_argument("--qubits", type=int, default=10, help="Logical qubits in the program.")
    parser.add_argument("--gates", type=int, default=50, help="Program gates after cancellation.")
    parser.add_argument("--repetitions", type=int, default=25000, help="Shots per simulation.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Noise severity. 0 is noiseless, which turns the TVD line into a "
        "correctness check; 1 is the NEUTRAL_ATOM baseline.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Text-drawer width. -1 disables folding; omit to fold at terminal width.",
    )
    parser.add_argument(
        "--tvd-threshold",
        type=float,
        default=None,
        help="TVD above this is flagged MISMATCH. Omit to derive it from the reference's support and the shot count -- a fixed value cannot work across circuit widths, since the sampling floor grows with the support. Only meaningful at low --alpha.",
    )
    parser.add_argument(
        "--sim-seed",
        type=int,
        default=None,
        help="Seed the simulator too, making a run bit-reproducible. Unset by default "
        "so each iteration draws fresh shots.",
    )
    parser.add_argument(
        "--packing",
        choices=("both", "dense", "sparse"),
        default="both",
        help="How many patches the random permutation may spread across. 'dense' offers "
        "the fewest that hold the program, so patches come out full; 'sparse' offers one "
        "per logical qubit, which is what pipeline.generate_random_circuit does and which "
        "at k=6 gives a 4-qubit program 24 slots to scatter over. 'both' runs each in "
        "turn and is the default, since the two reach different code paths.",
    )
    parser.add_argument(
        "--merging",
        choices=("both", "on", "off"),
        default="on",
        help="Whether the merging pass runs. 'both' is the default and makes a failure "
        "self-diagnosing: failing with merging on but passing with it off is a merging "
        "bug, failing both ways is placement or the gate layer.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Halt at the first failed translation or flagged TVD instead of continuing.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Stop after this many seeds. Default is to run forever.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = (
        itertools.count(args.seed)
        if args.max_seeds is None
        else range(args.seed, args.seed + args.max_seeds)
    )
    flagged: List[int] = []

    for seed in seeds:
        clean = run_seed(
            seed,
            num_qubits=args.qubits,
            num_gates=args.gates,
            repetitions=args.repetitions,
            alpha=args.alpha,
            fold=args.fold,
            tvd_threshold=args.tvd_threshold,
            stop_on_error=args.stop_on_error,
            sim_seed=args.sim_seed,
            packings=PACKINGS if args.packing == "both" else (args.packing,),
            mergings=(True, False) if args.merging == "both"
            else (args.merging == "on",),
        )
        if not clean:
            flagged.append(seed)
            print(f"\n*** seed {seed} flagged. flagged so far: {flagged} ***")
            if args.stop_on_error:
                print("stopping (--stop-on-error)")
                return

    if flagged:
        print(f"\nflagged seeds: {flagged}")


if __name__ == "__main__":
    main()
