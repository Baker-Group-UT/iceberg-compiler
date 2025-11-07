# tcalibrate.py  (augmented)

from typing import Iterable, Optional, List, Tuple, Callable
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit as qk_random_circuit
from qiskit.circuit import Instruction
import qiskit

from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.library import SetDensityMatrix

from pipeline.analysis.postselection import apply_iceberg_postselection
from pipeline.config import PipelineConfig
from simulator.config_presets import ScaledSimulator, StandardSimulator
from translator.core import TranslationContext, TranslationResult
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
    print("encode override:", encode_override)
    ctx = TranslationContext.from_config(
        permutation, program_circuit, fault_tolerant=fault_tolerant, verbose=verbose, encode_override=encode_override
    )
    if prelude:
        prelude(ctx)

    for ci in program_circuit.data:
        flag = translate_and_apply(ctx, ci)
        if flag is False:
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
) -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit, QuantumCircuit]:
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

    # B) CHAR logical op → encoded segment (run NOISILY)
    char_prog = make_logical_char_prog(gate, targets, theta)
    segB = translate_segment(
        char_prog, permutation, 
        fault_tolerant=fault_tolerant, verbose=verbose, 
        encode_override=True
    )

    # C) DECODE + MEASURE (ideal)
    empty = QuantumCircuit(char_prog.num_qubits)
    segC = translate_segment(
        empty, permutation,
        fault_tolerant=fault_tolerant, verbose=verbose,
        postlude=lambda ctx: (ctx.iceberg_set.decode(), ctx.iceberg_set.measure()),
        encode_override=True
    )
    return segA, segB, segC, rand_prog_raw


def demo_run(permutation: List[Tuple[int,int]]):
    segA, segB, segC, raw_qc = build_full_composed_char_circuit(
        permutation=permutation,
        rand_depth=10, rand_seed=6,
        gate="CX", targets=(0,1),
        char_label="TEST_SEG",
    )

    # Preserve register names in the circuit we store in TranslationResult
    tr_circ = QuantumCircuit(*segA.qregs, *segA.cregs)
    tr_circ.compose(segA, inplace=True)
    tr_circ.compose(segB, inplace=True)
    tr_circ.compose(segC, inplace=True)

    tr = TranslationResult(
        circuit=tr_circ,
        permutation=permutation,
        context=TranslationContext.from_config(
            permutation, raw_qc, fault_tolerant=False, verbose=False
        ),
        contains_faults=False,
        fault_tolerant=False,
        fault_log=None,
        verbose=False,
    )

    # ----------------------
    # 1) PRE (ideal): run segA ideally to obtain psiA
    # ----------------------
    nq = segA.num_qubits
    if len(segA) == 0:
        psiA = Statevector.from_label('0' * nq)
    else:
        psiA = Statevector.from_instruction(segA)

    # ----------------------
    # 2) MIDDLE (noisy): run segB with noise, starting from psiA
    # ----------------------
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['x', 'z', 'h', 'rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ['cx', 'cz'])

    sim_mid = AerSimulator(method="density_matrix", noise_model=noise)
    segB_t = transpile(segB, sim_mid, optimization_level=0, layout_method="trivial")
    segB_t.save_density_matrix(label='rho_mid')

    res_mid = sim_mid.run(segB_t, initial_statevector=psiA).result()
    rhoB = res_mid.data(0)['rho_mid']

    # ----------------------
    # 3) POST (ideal): initialize from rhoB and execute segC ideally
    #    IMPORTANT: build the wrapper on segC's EXISTING registers to keep names.
    # ----------------------
    sim_post = AerSimulator(method="density_matrix")

    post_init = QuantumCircuit(*segC.qregs, *segC.cregs)  # re-use same qregs/cregs (names preserved)
    dm = DensityMatrix(rhoB)
    post_init.append(SetDensityMatrix(dm), post_init.qubits)
    post_init.compose(segC, inplace=True)

    post_t = transpile(post_init, sim_post, optimization_level=0, layout_method="trivial")
    res_post = sim_post.run(post_t).result()

    # Final execution through your Simulator to keep downstream identical.
    sim_config = StandardSimulator(repetitions=500_000)
    simulator = Simulator(sim_config)
    sim_res_raw = simulator.run(circuit=post_t)

    sim_results = apply_iceberg_postselection([tr], [sim_res_raw])
    return sim_results


if __name__ == "__main__":
    print(demo_run([(0,0), (0,1)]))
