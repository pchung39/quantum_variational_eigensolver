import pennylane as qml
from pennylane import numpy as np
from qiskit.extensions import HamiltonianGate
import numpy as np

# a = np.matrix([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

# h = HamiltonianGate(a, 10000.0)

# print(h.decompositions)

import itertools
from itertools import product
from operator import matmul
import functools
import numpy as np
import pennylane as qml


# def decompose_hamiltonian(H):
#     N = int(np.log2(len(H)))
#     paulis = [qml.Identity(0) @ qml.Identity(1), 
#               qml.PauliX(0) @ qml.PauliX(1), 
#               qml.PauliY(0) @ qml.PauliY(1), 
#               qml.PauliZ(0) @ qml.PauliZ(1)
#               ]

#     obs = []
#     coeffs = []

#     for term in product(paulis, repeat=N):
#         matrices = [i._matrix() for i in term]
#         coeff = np.trace(functools.reduce(np.kron, matrices) @ H) / (2**N)

#         if not np.allclose(coeff, 0):
#             coeffs.append(coeff)

#             if not all(t is qml.Identity for t in term):
#                 obs.append(functools.reduce(matmul, [t(i) for i, t in enumerate(term) if t is not qml.Identity]))
#             else:
#                 obs.append(functools.reduce(matmul, [t(i) for i, t in enumerate(term)]))

#     return coeffs, obs


def decompose_hamiltonian(H):
    n = int(np.log2(len(H)))
    N = 2 ** n

    if H.shape != (N, N):
        raise ValueError(
            "The Hamiltonian should have shape (2**n, 2**n), for any qubit number n>=1"
        )

    if not np.allclose(H, H.conj().T):
        raise ValueError("The Hamiltonian is not Hermitian")

    paulis = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
    obs = []
    coeffs = []

    for term in itertools.product(paulis, repeat=n):
        matrices = [i._matrix() for i in term]
        coeff = np.trace(functools.reduce(np.kron, matrices) @ H) / N
        coeff = np.real_if_close(coeff).item()

        if not np.allclose(coeff, 0):
            coeffs.append(coeff)

            if not all(t is qml.Identity for t in term):
                obs.append(
                    functools.reduce(
                        matmul,
                        [t(i) for i, t in enumerate(term) if t is not qml.Identity],
                    )
                )
            else:
                obs.append(functools.reduce(matmul, [t(i) for i, t in enumerate(term)]))

    return coeffs, obs



a = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
coeffs, obs = decompose_hamiltonian(a)

print(coeffs, obs)