"""
iceberg — the [[n, n-2, 2]] Iceberg error-detecting code layer.

    layout              qubit roles and the logical-operator convention
    iceberg_module      one encoded patch and its logical gate set
    iceberg_set         a register of patches, plus cross-patch constructions
    iceberg_postselect  syndrome rejection and classical decoding
    costs               physical gate counts per logical operation, given n
    physical            physical gate emission and the flagged FT gadget

`layout.CodeLayout` is the single source of truth for the code convention; start
there when reading the rest.
"""

from .costs import gate_cost, work_table
from .iceberg_module import IcebergModule
from .iceberg_postselect import decode_data_register, iceberg_postselect
from .iceberg_set import IcebergSet
from .layout import CodeLayout
from .utils import SharedInt, logical_capacity

__all__ = [
    "CodeLayout",
    "IcebergModule",
    "IcebergSet",
    "SharedInt",
    "decode_data_register",
    "gate_cost",
    "iceberg_postselect",
    "logical_capacity",
    "work_table",
]
