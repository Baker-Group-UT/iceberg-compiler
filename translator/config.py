"""
translator.config.py - Configuration object for Translator behavior.
"""

from dataclasses import dataclass


@dataclass
class TranslatorConfig:
    fault_tolerant: bool = False
    fault_tolerance_chance: float = -1.0
    verbose: bool = False
    n_code: int = 4  # Default [[4,2,2]] code size
    enable_merging: bool = True
    allow_transversal_swap_merges: bool = True
    encode_override: bool = False

    def to_dict(self) -> dict:
        return {
            "fault_tolerant": self.fault_tolerant,
            "verbose": self.verbose,
            "n_code": self.n_code,
            "enable_merging": self.enable_merging,
            "allow_transversal_swap_merges": self.allow_transversal_swap_merges,
            "encode_override": self.encode_override,
        }
