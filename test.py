# playing with Hamiltonians and testing out classical eigensolver algorithm

import numpy as np
from random import random
from scipy.optimize import minimize

from qiskit import *
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver
import matplotlib.pyplot as plt

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

def quantum_state_preparation(circuit):
    #q = circuit.qregs[0] # q is the quantum register where the info about qubits is stored
    # circuit.rx(parameters[0], q[0]) # q[0] is our one and only qubit XD
    # circuit.ry(parameters[1], q[0])


    '''
    Ansatz
    '''
    # first parameter is RX angle
    # can use np format: np.array([np.pi, np.pi])
    circuit.rx(0,0)
    circuit.rx(0,1)
    circuit.cx(0,1)
    circuit.cx(1,0)
    circuit.h(0)
    circuit.h(1)

    circuit.barrier()

    '''
    Hamiltonian decomposition rotations (rotations are regulated by the decomposition)
    '''
    circuit.rx(0,0)
    circuit.rx(0,0)

    circuit.ry(0,1)
    circuit.ry(0,1)
    
    circuit.barrier()
    return circuit



def vqe_circuit(parameters, measure):
    """
    Creates a device ansatz circuit for optimization.
    :param parameters_array: list of parameters for constructing ansatz state that should be optimized.
    :param measure: measurement type. E.g. 'Z' stands for Z measurement.
    :return: quantum circuit.
    """
    q = QuantumRegister(2)
    c = ClassicalRegister(1)
    circuit = QuantumCircuit(q, c)

    # quantum state preparation
    circuit = quantum_state_preparation(circuit, parameters)

    # measurement
    if measure == 'Z':
        circuit.measure(q[0], c[0])
    elif measure == 'X':
        circuit.u2(0, np.pi, q[0])
        circuit.measure(q[0], c[0])
    elif measure == 'Y':
        circuit.u2(0, np.pi/2, q[0])
        circuit.measure(q[0], c[0])
    else:
        raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')

    return circuit

q = QuantumRegister(2)
c = ClassicalRegister(1)
circuit = QuantumCircuit(q, c)

test_circuit = quantum_state_preparation(circuit)
test_circuit.draw(output="mpl")
plt.show()


def quantum_module(parameters, measure):
    # measure
    if measure == 'II':
        return 1
    elif measure == 'ZZ':
        circuit = vqe_circuit(parameters, 'Z')
    elif measure == 'XX':
        circuit = vqe_circuit(parameters, 'X')
    elif measure == 'YY':
        circuit = vqe_circuit(parameters, 'Y')
    else:
        raise ValueError('Not valid input for measurement: input should be "I" or "X" or "Z" or "Y"')
    
    shots = 8192
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # expectation value estimation from counts
    expectation_value = 0
    for measure_result in counts:
        sign = +1
        if measure_result == '1':
            sign = -1
        expectation_value += sign * counts[measure_result] / shots
        
    return expectation_value

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



def vqe(parameters):

    pauli_dict = pauli_operator_to_dict(H)        
    # quantum_modules
    quantum_module_I = pauli_dict['II'] * quantum_module(parameters, 'II')
    quantum_module_Z = pauli_dict['ZZ'] * quantum_module(parameters, 'ZZ')
    quantum_module_X = pauli_dict['XX'] * quantum_module(parameters, 'XX')
    quantum_module_Y = pauli_dict['YY'] * quantum_module(parameters, 'YY')
    
    # summing the measurement results
    classical_adder = quantum_module_I + quantum_module_Z + quantum_module_X + quantum_module_Y
    
    return classical_adder

# parameters_array = np.array([np.pi, np.pi])
# tol = 1e-3 # tolerance for optimization precision.

# vqe_result = minimize(vqe, parameters_array, method="Powell", tol=tol)
# print('The exact ground state energy is: {}'.format(reference_energy))
# print('The estimated ground state energy from VQE algorithm is: {}'.format(vqe_result.fun))