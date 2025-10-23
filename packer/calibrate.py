"""
packer.calibrate.py - For a set of standard or passed in gates will simulate 
the logical error rate distribution, logical selection rate distribution, respective stdevs for both
and mean and median for the distributions

It does this by:
1) Noiselessly encoding and running a random qiskit circuit in the module.
2) For the set single application gates (X, Z, RZ) we noisily simulate only their application in the circuit 
"""

# characterize_compose.py
from typing import Iterable, Optional, List, Tuple
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit import Instruction
import qiskit

# segment_translate.py
from typing import Callable, Optional, List, Tuple, Iterable
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit as qk_random_circuit, random_clifford_circuit
from qiskit_aer.noise import NoiseModel, depolarizing_error

from pipeline.analysis.postselection import apply_iceberg_postselection, postselect_against_reference
from pipeline.config import PipelineConfig
from simulator.config_presets import ScaledSimulator
from translator.core import TranslationContext, TranslationResult   # your class
from translator.gate_handlers import translate_and_apply

from simulator.simulator import Simulator

# ---------- Minimal reusable segment translator ----------
def translate_segment(
    program_circuit: QuantumCircuit,
    permutation: List[Tuple[int, int]],
    *,
    fault_tolerant: bool = False,
    verbose: bool = False,
    prelude: Optional[Callable[[TranslationContext], None]] = None,
    postlude: Optional[Callable[[TranslationContext], None]] = None,
    encode_override: bool = False,
) -> QuantumCircuit:
    """
    Translate ONLY the ops of `program_circuit` into an *encoded/physical* segment
    by reusing your gate-level translator. Optional prelude/postlude let you
    insert ENCODE or DECODE+MEASURE around the ops without duplicating logic.
    """
    print("encode override:", encode_override)
    ctx = TranslationContext.from_config(
        permutation, program_circuit, fault_tolerant=fault_tolerant, verbose=verbose, encode_override=encode_override
    )
    if prelude:
        prelude(ctx)

    # Replay program ops through the translator
    ops = list(program_circuit.data)
    skip_next = False
    for i, op in enumerate(ops):
        if skip_next:
            skip_next = False
            continue
        nxt = ops[i + 1] if i + 1 < len(ops) else None
        consumed = translate_and_apply(ctx, op, next_op=nxt)
        if consumed is not None:
            if consumed:
                skip_next = True
            else:
                raise RuntimeError("translate_segment: gate application failed")

    if postlude:
        postlude(ctx)

    return ctx.iceberg_set.circuit


# ---- Small logical char-circuits (one op) ----
def make_logical_char_prog(gate: str, targets: Iterable[int], theta: Optional[float] = None) -> QuantumCircuit:
    t = list(targets)
    n_logical = max(t) + 1 if t else 1
    prog = QuantumCircuit(n_logical)
    G = gate.upper()
    if G == "X":
        prog.x(t[0])
    elif G == "Z":
        prog.z(t[0])
    elif G == "RZ":
        if theta is None:
            raise ValueError("theta required for RZ")
        prog.rz(theta, t[0])
    elif G == "CX":
        if len(t) != 2: raise ValueError("CX needs [c,t]")
        prog.cx(t[0], t[1])
    elif G == "CZ":
        if len(t) != 2: raise ValueError("CZ needs [q0,q1]")
        prog.cz(t[0], t[1])
    elif G == "H":
        prog.h(t[0])
    else:
        raise NotImplementedError(gate)
    return prog

def build_full_composed_char_circuit(
    *,
    permutation: List[Tuple[int,int]],
    rand_depth: int = 20,
    rand_seed: Optional[int] = None,
    gate: str = "X",
    targets: Iterable[int] = (0,),
    theta: Optional[float] = None,
    char_label: str = "TEST_SEG",
    fault_tolerant: bool = False,
    verbose: bool = False,
) -> Tuple[QuantumCircuit, NoiseModel]:
    # A) ENCODE + random logical program (encoded)
    rand_prog_raw = qk_random_circuit(
        2, depth=rand_depth, seed=rand_seed, measure=False, num_operand_distribution={1: 0.5, 2: 0.5}
    )
    target_basis = ['x', 'z', 'h', 'rz', 'cx', 'cz', 'measure']
    rand_prog = qiskit.transpile(
        rand_prog_raw,
        basis_gates=target_basis,
        optimization_level=2,
    )
    segA = translate_segment(
        rand_prog, permutation,
        fault_tolerant=fault_tolerant, verbose=verbose,
        prelude=lambda ctx: ctx.iceberg_set.encode()
    )
    segA.barrier()

    # B) CHAR logical op → encoded segment
    char_prog = make_logical_char_prog(gate, targets, theta)
    segB = translate_segment(char_prog, permutation, 
        fault_tolerant=fault_tolerant, verbose=verbose, 
        encode_override=True
    )

    # Wrap B as a single named Instruction (so we can attach noise only here)
    char_inst: Instruction = segB.to_instruction(label=char_label)

    # C) DECODE + MEASURE
    empty = QuantumCircuit(char_prog.num_qubits)  # no ops; just to set width
    segC = translate_segment(
        empty, permutation,
        fault_tolerant=fault_tolerant, verbose=verbose,
        postlude=lambda ctx: (ctx.iceberg_set.decode(), ctx.iceberg_set.measure()),
    )

    # Compose A • [B] • C
    nq = segA.num_qubits
    full = QuantumCircuit(nq)
    full.compose(segA, qubits=full.qubits, inplace=True)      # A
    full.append(char_inst, qargs=full.qubits)                 # [B] as one op
    full.compose(segC, qubits=full.qubits, inplace=True)      # C

    # Noise ONLY on [B]
    noise = NoiseModel()
    oneq = depolarizing_error(0.01, 1)  # swap for your channel
    # Attach the 1q channel to each qubit position of CHAR_SEG
    for pos in range(nq):
        noise.add_quantum_error(oneq, [char_label], qubits=[pos])

    return full, noise, rand_prog_raw

# ---- Example run (swap in your Simulator/Repository flow) ----
def demo_run(permutation: List[Tuple[int,int]]):
    qc, noise, raw_qc = build_full_composed_char_circuit(
        permutation=permutation,
        rand_depth=4, rand_seed=6,
        gate="X", targets=(0,),
        char_label="TEST_SEG",
    )
    tr = TranslationResult(
        circuit=qc,
        permutation=permutation,
        context= TranslationContext.from_config(
            permutation, raw_qc, fault_tolerant=False, verbose=False
        ),
        contains_faults=False,
        fault_tolerant=False,
        fault_log=None,
        verbose=False,
    )
    target_basis = ['x', 'z', 'h', 'rz', 'cx', 'cz', 'measure']
    rand_prog = qiskit.transpile(
        raw_qc,
        basis_gates=target_basis,
        optimization_level=2,
    )
    rand_prog.measure_all()
    backend = AerSimulator()
    basis = backend.configuration().basis_gates
    qc_unrolled = qiskit.transpile(qc, backend=backend, basis_gates=basis, optimization_level=0)
    rand_prog_unrolled = qiskit.transpile(rand_prog, backend=backend, basis_gates=basis, optimization_level=0)

    sim_config=ScaledSimulator(alpha=1, repetitions=50_000).with_noise_model(noise)

    simulator = Simulator(sim_config)
    sim_res_raw = simulator.run(circuit=qc_unrolled)

    # simulation_results = postselect_against_reference([tr], ideal_sim_res, [noisy_sim_res])
    sim_results = apply_iceberg_postselection([tr], [sim_res_raw])

    return sim_results
    # sim = AerSimulator()
    # res = sim.run(qiskit.transpile(qc, sim), shots=50_000, noise_model=noise).result()
    # return res.get_counts()

print(demo_run([(0,0), (0,1)]))

