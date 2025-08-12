"""main.py – quick entry‑point to benchmark CNOT‑type logical gates

Run:
    python -m main

It will:
    • run the gate simulations (N_states × shots noisy MC each),
    • save a bar‑chart PNG under ./results/, and
    • display the chart non‑blocking (so subsequent work continues).
"""
from gates import gate_AA, gate_AB, gate_ABp, gate_BBp
from executor import run_sim_groups

# -----------------------------------------------------------------------------
# Define simulation groups (extend as needed)
# -----------------------------------------------------------------------------
SIMS = [
    {
        "title": "CNOT logical gates (Iceberg)",
        "gates": [
            ("A–A′", gate_AA),
            ("A–B", gate_AB),
            # ("A′–B′", gate_ABp),
            # ("B–B′", gate_BBp),
        ],
        # optional per‑group extras
        "figsize": (6, 4),
        # "filename": "cnot_errors.png",
    },
    {
        "title": "CNOT logical gates (Iceberg) B reverse",
        "gates": [
            ("B–B′", gate_BBp),
            ("A′–B′", gate_ABp),
            # ("A–B", gate_AB),
            # ("A–A′", gate_AA),
        ],
        # optional per‑group extrasad
        "figsize": (6, 4),
        # "filename": "cnot_errors.png",
    }
]


def main() -> None:
    """Run the defined gate‑groups and plot/save results."""
    run_sim_groups(SIMS, N_states=1, shots=200, show=True)


if __name__ == "__main__":
    main()
