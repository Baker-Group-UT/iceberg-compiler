from typing import Iterable


def compute_idle_times(
    qubits: Iterable[int],
    last_time: list[float],
    gate_time: float,
) -> list[tuple[int, float]]:
    """
    Coarse scheduling helper.
    Uses per-qubit last_time and aligns multi-qubit ops to the latest qubit time.
    Returns a list of (qubit, idle_time) and updates last_time in place.
    """
    qubits = list(qubits)
    if not qubits:
        return []
    start_time = max(last_time[q] for q in qubits)
    idle_times: list[tuple[int, float]] = []
    for q in qubits:
        t_idle = start_time - last_time[q]
        if t_idle > 0:
            idle_times.append((q, t_idle))
    end_time = start_time + gate_time
    for q in qubits:
        last_time[q] = end_time
    return idle_times
