import numpy as np
from random import random
from scipy.optimize import minimize
import math
# import pennylane as qml
# from pennylane import numpy as np
from qiskit.extensions import HamiltonianGate
import numpy as np
import itertools
from itertools import product
from operator import matmul
import functools
from qiskit import *
# from qiskit.circuit.library.standard_gates import U2Gate
# from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.providers.aer import noise

class VQE(object): 

    def __init__(self, hamiltonian_operator, shots, noise=None, graph=False):
        self.graph = graph
        self.angle = 0
        self.shots = shots
        self.noise = noise

    @staticmethod
    def decompose_hamiltonian(self, hermitian_matrix):
        '''
        This method provides a utility to decompose any Hermitian matrix into constituent Pauli matrices.
        Helpful to know what coefficient values must be set for calculating the weights of each measurement axis. 

        INPUT
        -----
        - "matrix": takes a numpy array type of n*n dimension

        OUTPUT
        ------
        - "decomposed_hamiltonian": a dictionary where the key is the decomposed pauli matrix and the value is the coefficient.

        {
            "XX": -1,
            "IZ": 2.4,
            "YI": 0.9
        }  
        '''

        pauli_dictionary = {
            "I": np.array([[1,  0], [0,  1]], dtype=complex),
            "X": np.array([[0,  1], [1,  0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j,  0]], dtype=complex),
            "Z": np.array([[1,  0],[0, -1]], dtype=complex) 
        }

        pauli_coefficients = {}

        for matrix in ["I", "X", "Y", "Z"]:
            matrix_entry = pauli_dictionary[matrix]
            pauli_matrix_tensored = np.kron(matrix_entry, matrix_entry)
            pauli_coefficients["{}".format(matrix+matrix)] = 0.25 * np.trace(np.matmul(pauli_matrix_tensored, hermitian_matrix))

        return pauli_coefficients

    def generate_circuit(self, measurement_axis):

        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        circuit = QuantumCircuit(q, c)  

        '''
        Ansatz
        '''

        circuit.h(0)
        circuit.cx(0,1)
        circuit.rx(self.angle, 0)


        '''
        Pre-measurement rotations for measuring in X or Y axis
        '''
        if measurement_axis == 'X':
            circuit.ry(-np.pi/2, q[0])
            circuit.ry(-np.pi/2, q[1])
        elif measurement_axis == 'Y':
            circuit.rx(np.pi/2, q[0])
            circuit.rx(np.pi/2, q[1])
        else:
            continue

        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])

        return circuit  


    def execute_measurement(self, circuit):
        
        if self.noise:
            noise_model = self.noise_model()
            basis_gates = noise_model.basis_gates
            backend = Aer.get_backend('qasm_simulator')
            job = execute(circuit, backend, shots=self.shots, 
                          basis_gates=basis_gates, noise_model=noise_model)
        else:
            backend = Aer.get_backend('qasm_simulator')
            job = execute(circuit, backend, shots=self.shots)
            
        result = job.result()
        return result

    def calculate_expectation_values(self, measurement_results, axis):

        counts = measurement_results.get_counts()

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

        expected_value = (counts_00 - counts_01 - counts_10 + counts_11) / self.shots

        return expected_value

    def measure_in_axis(self, axis):

        circuit = self.generate_circuit(axis)
        measurement_results = self.execute_measurement(circuit)
        expectation_value = self.calculate_expectation_values(measurement_results, axis)

        return expectation_value

    def sum_decomposed_hamiltonian(self):

        x_measurement = -0.5 * (self.measure_in_axis("X"))
        y_measurement = -0.5 * (self.measure_in_axis("Y"))
        z_measurement = 0.5 * (self.measure_in_axis("Z"))
        i_measurement = 0.5 * 1

        ground_state = x_measurement + y_measurement + z_measurement + i_measurement
        
        return ground_state

    def noise_model(self):
        

        noise_model = noise.NoiseModel()

        for gate, error_info in self.noise.items():

            err_prob = error_info["error_prob"]
            if gate == "cx":
                error = noise.depolarizing_error(err_prob, 2)
            else:
                error = noise.depolarizing_error(err_prob, 1)

            if error_info["error_type"] == "general_error":
                noise_model.add_all_qubit_quantum_error(error, [gate])
            elif error_info["error_type"] == "non_local":
                noise_model.add_nonlocal_quantum_error(error, [gate])
            else:
                raise Exception("Error type must either be one of the following: ['general_error', 'non_local']")


        # TODO: compare these graphs - one qubit gate vs a two-qubit gate. 
        # rotation vs a hadamard
        # what can we infer about the nature of noise and how it decoheres and affects the accuracy of retrieving the ground state measurement?
        # Does noise to a 2 qubit gate affect the outcome MORE than a one qubit gate?

        return noise_model

    def find_ground_state(self):

        measurements = []

        angle_range = np.linspace(0, 2 * np.pi, 20)
        for angle in angle_range: 
            self.angle = angle
            measurement = self.sum_decomposed_hamiltonian()
            measurements.append(measurement)

        print("Ground state: {}".format(min(measurements)))

        if self.graph:
            plt.xlabel('Angle [radians]')
            plt.ylabel('Expectation value')
            plt.plot(angle_range, measurements)
            plt.show()

        
    
if __name__ == "__main__":
    noise_instructions = {
            "h": {
                "error_type": "general_error",
                "error_prob": 0.6
            }

        }
    vqe = VQE(hamiltonian_operator=1, shots=100, noise=noise_instructions, graph=True)
    vqe.find_ground_state()