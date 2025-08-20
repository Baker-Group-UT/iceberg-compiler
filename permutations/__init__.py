"""
This package generators the different unique permutations for 
a set of quantum states and types of codes used. 
Currently scoped to Iceberg codes.
"""
from .entrypoints import create_permutations
from .generator import generate_permutations, canonical_key
from .code_sets import (
    generate_all_code_sets,
    generate_economic_code_sets,
    generate_uniform_code_sets,
)

__all__ = [
    "create_permutations",
    "generate_permutations",
    "canonical_key",
    "generate_all_code_sets",
    "generate_economic_code_sets",
    "generate_uniform_code_sets",
]
