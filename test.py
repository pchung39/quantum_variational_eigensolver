# playing with Hamiltonians and testing out classical eigensolver algorithm

# TODO: cleanup imports

import numpy as np
from random import random
from scipy.optimize import minimize
import math
import pennylane as qml
from pennylane import numpy as np
from qiskit.extensions import HamiltonianGate
import numpy as np
import itertools
from itertools import product
from operator import matmul
import functools
from qiskit import *
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram

'''
Classical Model
'''

# def hamiltonian_operator(a, b, c, d):
#     """
#     Creates a*I + b*Z + c*X + d*Y pauli sum 
#     that will be our Hamiltonian operator.
    
#     """
#     pauli_dict = {
#         'paulis': [{"coeff": {"imag": 0.0, "real": a}, "label": "II"},
#                    {"coeff": {"imag": 0.0, "real": b}, "label": "ZZ"},
#                    {"coeff": {"imag": 0.0, "real": c}, "label": "XX"},
#                    {"coeff": {"imag": 0.0, "real": d}, "label": "YY"}
#                    ]
#     }
#     return WeightedPauliOperator.from_dict(pauli_dict)

# a, b, c, d = (0.5, 0.5, -0.5, -0.5)

# H = hamiltonian_operator(a, b, c, d)
# print("H: ", H.print_details())
# exact_result = NumPyEigensolver(H).run()
# reference_energy = min(np.real(exact_result.eigenvalues))
# print('The exact ground state energy is: {}'.format(reference_energy))


'''
Quantum Model
'''

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



# a = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
# a = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# coeffs, obs = decompose_hamiltonian(a)
# print(coeffs, obs)

def quantum_state_preparation(angle, circuit):
    #q = circuit.qregs[0] # q is the quantum register where the info about qubits is stored
    # circuit.rx(parameters[0], q[0]) # q[0] is our one and only qubit XD
    # circuit.ry(parameters[1], q[0])


    '''
    Ansatz
    '''
    # first parameter is RX angle
    # can use np format: np.array([np.pi, np.pi])
    circuit.rx(angle, 0)
    circuit.h(0)
    circuit.cx(0,1)
    circuit.h(1)
    return circuit



def vqe_circuit(angle, measure):
    """
    Creates a device ansatz circuit for optimization.
    :param parameters_array: list of parameters for constructing ansatz state that should be optimized.
    :param measure: measurement type. E.g. 'Z' stands for Z measurement.
    :return: quantum circuit.
    """
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)

    # quantum state preparation
    circuit = quantum_state_preparation(angle, circuit)

    # measurement
    # this is where you rotate either on the x or y axis and then measure 
    if measure == 'Z':
        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])
    elif measure == 'X':
        circuit.ry(-math.pi/2, q[0])
        circuit.ry(-math.pi/2, q[1])
        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])
    elif measure == 'Y':
        circuit.rx(math.pi/2, q[0])
        circuit.rx(math.pi/2, q[1])
        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])
    else:
        raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')

    return circuit

# q = QuantumRegister(2)
# c = ClassicalRegister(1)
# circuit = QuantumCircuit(q, c)

# test_circuit = quantum_state_preparation(circuit)
# test_circuit = vqe_circuit("X")
# test_circuit.draw(output="mpl")
# plt.show()


def quantum_module(angle, measure):
    # measure
    if measure == 'II':
        return 1
    elif measure == 'ZZ':
        #circuit = vqe_circuit(parameters, 'Z')
        circuit = vqe_circuit(angle,'Z')
    elif measure == 'XX':
        #circuit = vqe_circuit(parameters, 'X')
        circuit = vqe_circuit(angle,'X')
    elif measure == 'YY':
        #circuit = vqe_circuit(parameters, 'Y')
        circuit = vqe_circuit(angle,'Y')
    else:
        raise ValueError('Not valid input for measurement: input should be "I" or "X" or "Z" or "Y"')
    
    # TODO: tweak this to see if values change?
    shots = 1000
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()

    if counts.get('00'):
        counts_00 = counts['00']
    else:
        counts_00 = 0

    if counts.get('01'):
        counts_01 = counts['01']
    else:
        counts_01 = 0    

    if counts.get('10'):
        counts_10 = counts['10']
    else:
        counts_10 = 0

    if counts.get('11'):
        counts_11 = counts['11']
    else:
        counts_11 = 0


    print("COUNTS: ", counts)
    if measure == 'ZZ':
        expected_value = (counts_00 - counts_01 - counts_10 + counts_11) / shots
    elif measure == 'XX':
        expected_value = (counts_00 + counts_01 + counts_10 + counts_11) / shots
    elif measure == "YY":
        expected_value = (counts_00 + counts_01 + counts_10 - counts_11) / shots
    elif measure == "II":
        expected_value = (counts_00 + counts_01 + counts_10 + counts_11) / shots
    else:
        raise ValueError('Not valid input for measurement: input should be "I" or "X" or "Z" or "Y"')    

    print("Expectation value for pauli matrix ({}): {}".format(measure,expected_value))
    return expected_value

def pauli_operator_to_dict(pauli_operator):
    """
    from WeightedPauliOperator return a dict:
    {I: 0.7, X: 0.6, Z: 0.1, Y: 0.5}.
    :param pauli_operator: qiskit's WeightedPauliOperator
    :return: a dict in the desired form.
    """
    d = pauli_operator.to_dict()
    paulis = d['paulis']
    paulis_dict = {}

    for x in paulis:
        label = x['label']
        coeff = x['coeff']['real']
        paulis_dict[label] = coeff
    
    return paulis_dict


def hamiltonian_operator(a, b, c, d):
    """
    Creates a*I + b*Z + c*X + d*Y pauli sum 
    that will be our Hamiltonian operator.
    
    """
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": a}, "label": "II"},
                   {"coeff": {"imag": 0.0, "real": b}, "label": "ZZ"},
                   {"coeff": {"imag": 0.0, "real": c}, "label": "XX"},
                   {"coeff": {"imag": 0.0, "real": d}, "label": "YY"}
                   ]
    }
    return WeightedPauliOperator.from_dict(pauli_dict)

def vqe(angle):

    pauli_dict = pauli_operator_to_dict(H)        
    # quantum_modules
    print("MEASURE 'II'")
    quantum_module_I = pauli_dict['II'] * quantum_module(angle, 'II')
    print("MEASURE 'ZZ'")
    quantum_module_Z = pauli_dict['ZZ'] * quantum_module(angle, 'ZZ')
    print("MEASURE 'XX'")
    quantum_module_X = pauli_dict['XX'] * quantum_module(angle, 'XX')
    print("MEASURE 'YY'")
    quantum_module_Y = pauli_dict['YY'] * quantum_module(angle, 'YY')
    # summing the measurement results
    classical_adder = quantum_module_I + quantum_module_Z + quantum_module_X + quantum_module_Y
    print("Measured ground state: ", classical_adder)
    return classical_adder

a, b, c, d = (0.5, 0.5, -0.5, -0.5)

H = hamiltonian_operator(a, b, c, d)
print("H: ", H.print_details())
parameters_array = np.array([np.pi, np.pi])

angle_range = np.linspace(0, 2 * np.pi, 20)
print("range: ", angle_range)
data = []
for a in angle_range: 
    print("Angle: ", a)
    test_vqe = vqe(a)
    data.append(test_vqe)
    print("\n\n")

print("DATA: ", data)
import matplotlib.pyplot as plt
plt.xlabel('Angle [radians]')
plt.ylabel('Expectation value')
plt.plot(angle_range, data)
plt.show()
# tol = 1e-3 # tolerance for optimization precision.
# vqe_result = minimize(vqe, parameters_array, method="Powell", tol=tol)
# print('The exact ground state energy is: {}'.format(reference_energy))
# print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))