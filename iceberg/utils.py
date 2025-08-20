"""
Utility functions for the Iceberg package.
"""
import math
import numpy as np

# Logical single-qubit Clifford angles for rotations
CLIFFORD_Z = [0, math.pi / 2, math.pi, -math.pi / 2]   # Rz: 0°, 90°, 180°, 270°
CLIFFORD_X = [0, math.pi]                               # Rx: identity or π

def rand_clifford_angles() -> tuple[float, float]:
    """
    Sample (θ, φ) angles corresponding to a uniformly random single-qubit
    Clifford unitary using the decomposition H Rz(φ) H Rz(θ).
    
    Returns
    -------
    θ : float
        Angle around the X axis (after H)
    φ : float
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
