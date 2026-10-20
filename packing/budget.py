"""budget.py — what the machine can provide."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = ["ResourceBudget"]


@dataclass(frozen=True)
class ResourceBudget:
    """What the machine can provide. `None` means unconstrained."""

    max_physical_qubits: Optional[int] = None
    max_patches: Optional[int] = None

    def patch_ceiling(self, n_code: int) -> Optional[int]:
        """Most patches this budget allows at the given code size."""
        limits = []
        if self.max_patches is not None:
            limits.append(self.max_patches)
        if self.max_physical_qubits is not None:
            limits.append(self.max_physical_qubits // (n_code + 2))
        return min(limits) if limits else None

    def admits(self, patches: int, n_code: int) -> bool:
        ceiling = self.patch_ceiling(n_code)
        return ceiling is None or patches <= ceiling
