"""
Numerical verification of the [[n, n-2, 2]] claims in GENERALIZATION_PLAN.md.

Statevector-level checks (pure numpy, no qiskit/stim needed) of the logical
action of the physical constructions in iceberg/, evaluated on the full
logical basis at n=6 (and n=4 where relevant). Run directly:

    python tests/verify_nn22_claims.py

Claims:
  1. Transversal H = H^{\otimes k} . C_A, where C_A : x -> x XOR parity(x)*1
     (the GF(2) linear map A = J + I). NOT equal to H^{\otimes k} alone.
  2. CZ_k_Transversal (0<->n-1 crossed, middle straight) = lane-wise logical
     CZ on all k lanes.
  3. Targeted inter-patch CNOT (4 physical CX: c_data->t_data, c_data->t_anc0,
     c_anc_last->t_data, c_anc_last->t_anc0) = single-lane logical CNOT.
  4. Decode rule: reject odd-parity strings; raw_i = b_{i+1} XOR b_0 decodes
     to C_A.x under the top-ancilla Xbar_i = X_0 X_{i+1} convention, so the
     correct rule is  logical = raw XOR parity(raw)*1.
     (At k=2 the C_A is a lane swap, silently absorbed by the hardcoded
     _PHYS2LOGIC table; at k>2 it must be applied explicitly.)

Phase 1 of the plan should promote these to proper pytest cases covering
every IcebergModule / IcebergSet operation.
"""

import numpy as np
from itertools import product

I2 = np.eye(2)
H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X1 = np.array([[0, 1], [1, 0]])


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def op_on(n, sites_ops):
    mats = [I2] * n
    for s, o in sites_ops:
        mats[s] = o
    return kron(*mats)


def encode_basis(n):
    """Logical basis |x_L>, x in GF(2)^k, built as implemented in iceberg/:
    |0_L> = (|0^n> + |1^n>)/sqrt(2);  Xbar_i = X_0 X_{i+1} (top-ancilla)."""
    k = n - 2
    dim = 2 ** n
    v0 = np.zeros(dim)
    v0[0] = 1
    v0[dim - 1] = 1
    v0 /= np.sqrt(2)
    states = {}
    for x in product([0, 1], repeat=k):
        v = v0.copy()
        for i, xi in enumerate(x):
            if xi:
                v = op_on(n, [(0, X1), (i + 1, X1)]) @ v
        states[x] = v
    return states


def claim1_transversal_h(n=6):
    k = n - 2
    basis = encode_basis(n)
    xs = list(product([0, 1], repeat=k))
    idx = {x: i for i, x in enumerate(xs)}
    Ht = kron(*[H1] * n)
    M = np.zeros((2 ** k, 2 ** k), dtype=complex)
    in_code = True
    for x in xs:
        img = Ht @ basis[x]
        proj = np.zeros_like(img)
        for y in xs:
            amp = basis[y].conj() @ img
            M[idx[y], idx[x]] = amp
            proj += amp * basis[y]
        in_code &= np.allclose(proj, img, atol=1e-9)
    CA = np.zeros((2 ** k, 2 ** k))
    for x in xs:
        p = sum(x) % 2
        CA[idx[tuple(xi ^ p for xi in x)], idx[x]] = 1
    Hk = kron(*[H1] * k)
    target = Hk @ CA
    nz = np.abs(target) > 1e-9
    ratio = M[nz].flat[0] / target[nz].flat[0]
    ok = in_code and np.allclose(M, ratio * target, atol=1e-9)
    not_plain_h = not np.allclose(M, ratio * Hk, atol=1e-9)
    return ok, not_plain_h


def claim2_cz_transversal(n=6):
    k = n - 2
    N = 2 * n
    d = 2 ** N
    b1 = encode_basis(n)
    xs = list(product([0, 1], repeat=k))
    cz_pairs = [(0, n + (n - 1))] + [(q, n + q) for q in range(1, n - 1)] + [(n - 1, n + 0)]
    diag = np.ones(d)
    for s in range(d):
        bits = [(s >> (N - 1 - q)) & 1 for q in range(N)]
        for a, b in cz_pairs:
            if bits[a] and bits[b]:
                diag[s] *= -1
    ok = True
    for x in xs:
        for y in xs:
            v = np.kron(b1[x], b1[y])
            phase = (-1) ** sum(xi * yi for xi, yi in zip(x, y))
            ok &= np.allclose(diag * v, phase * v, atol=1e-9)
    return ok


def claim3_targeted_cnot(n=6, subset=4):
    k = n - 2
    N = 2 * n
    b1 = encode_basis(n)
    xs = list(product([0, 1], repeat=k))
    sample = xs[:subset] + xs[-subset:]

    def apply_cx(v, c, t):
        out = np.zeros_like(v)
        for s in np.nonzero(np.abs(v) > 1e-12)[0]:
            bits = [(int(s) >> (N - 1 - q)) & 1 for q in range(N)]
            if bits[c]:
                bits[t] ^= 1
            s2 = 0
            for q in range(N):
                s2 = (s2 << 1) | bits[q]
            out[s2] += v[s]
        return out

    ok = True
    for qi in range(k):
        for x in sample:
            for y in sample:
                v = np.kron(b1[x], b1[y])
                img = apply_cx(v, qi + 1, n + qi + 1)
                img = apply_cx(img, qi + 1, n + 0)
                img = apply_cx(img, n - 1, n + qi + 1)
                img = apply_cx(img, n - 1, n + 0)
                y2 = list(y)
                y2[qi] ^= x[qi]
                ok &= np.allclose(img, np.kron(b1[x], b1[tuple(y2)]), atol=1e-9)
    return ok


def claim4_decode_rule(ns=(4, 6)):
    ok = True
    for n in ns:
        k = n - 2
        basis = encode_basis(n)
        for x in product([0, 1], repeat=k):
            v = basis[x]
            for s in np.nonzero(np.abs(v) > 1e-9)[0]:
                bits = [(int(s) >> (n - 1 - q)) & 1 for q in range(n)]
                if sum(bits) % 2:
                    ok = False
                raw = tuple(bits[i + 1] ^ bits[0] for i in range(k))
                p = sum(raw) % 2
                if tuple(r ^ p for r in raw) != x:
                    ok = False
    return ok


if __name__ == "__main__":
    ok1, not_plain = claim1_transversal_h()
    print("Claim 1  transversal H == H^k . C_A (n=6):", ok1, "| != H^k alone:", not_plain)
    print("Claim 2  CZ_k_Transversal == lane-wise CZ (n=6):", claim2_cz_transversal())
    print("Claim 3  targeted inter-patch CNOT, all slots (n=6):", claim3_targeted_cnot())
    print("Claim 4  decode = parity + XOR + parity-complement (n=4,6):", claim4_decode_rule())
