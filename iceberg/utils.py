"""
Utility functions for the Iceberg package.
"""

import math
import numpy as np


class SharedInt:
    __slots__ = ("value",)

    def __init__(self, value=0):
        self.value = value

    def __int__(self):
        return self.value

    # >>> add this: range(), slicing, and many CPython APIs use __index__, not __int__
    def __index__(self):
        return self.value

    # (optional, but helps with f-strings and logging)
    def __repr__(self):
        return f"SharedInt({self.value})"

    def __str__(self):
        return str(self.value)

    def __add__(self, other):
        return SharedInt(self.value + (other.value if isinstance(other, SharedInt) else other))

    def __sub__(self, other):
        return SharedInt(self.value - (other.value if isinstance(other, SharedInt) else other))

    def __iadd__(self, other):
        self.value += other.value if isinstance(other, SharedInt) else other
        return self

    def __isub__(self, other):
        self.value -= other.value if isinstance(other, SharedInt) else other
        return self


# Logical single-qubit Clifford angles for rotations
CLIFFORD_Z = [0, math.pi / 2, math.pi, -math.pi / 2]  # Rz: 0°, 90°, 180°, 270°
CLIFFORD_X = [0, math.pi]  # Rx: identity or π


def rand_clifford_angles() -> tuple[float, float]:
    """
    Sample (theta, phi) angles corresponding to a uniformly random single-qubit
    Clifford unitary using the decomposition H Rz(phi) H Rz(theta).

    Returns
    -------
    theta : float
        Angle around the X axis (after H)
    phi : float
        Angle around the Z axis (before H)
    """
    theta = np.random.choice(CLIFFORD_X)
    phi = np.random.choice(CLIFFORD_Z)
    return theta, phi


def logical_capacity(n_code: int) -> int:
    """
    returns the k qubits of a code. wildly simple.
    """
    return n_code - 2


# Here we need to add the extra functions for logical selection rate, etc.