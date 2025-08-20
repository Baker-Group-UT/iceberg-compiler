import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from iceberg.IcebergCode import IcebergGrid

__all__ = ["gate_AA", "gate_AB", "gate_ABp", "gate_BBp"]

# ------------------------------------------------------------------
# Picklable logical-gate builders (must stay at top level!)
# ------------------------------------------------------------------


def gate_AA(grid: IcebergGrid):  # A–A'  (same patch 0→1)
    grid.CNOT(0, 1)


def gate_AB(grid: IcebergGrid):  # A–B   (patch-to-patch 0→0)
    grid.CNOT_k_Transversal(0, 1)


def gate_ABp(grid: IcebergGrid):  # A'–B' (patch-to-patch 0→1)
    grid.CNOT_k_Transversal(0, 1)  # same call, different initialisation


def gate_BBp(grid: IcebergGrid):  # B–B'  (same patch 1→1)
    grid.CNOT(2, 3)

