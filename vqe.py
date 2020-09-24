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
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver
import matplotlib.pyplot as plt
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt

class VQE(object): 

    def __init__(self, hamiltonian_operator, shots):
        # ansatz is a list
        # self.ansatz = ansatz
        self.hamiltonian_operator = hamiltonian_operator
        self.angle = 0
        self.shots = shots

    def decompose_hamiltonian(self, matrix):
        '''
        This method provides a utility to decompose any Hermitian matrix into constituent Pauli matrices

        INPUT
        -----
        - "matrix": takes a numpy array type of n*n dimension

        OUTPUT
        ------
        - decomposed_hamiltonian: a dictionary where the key is the decomposed pauli matrix and the value is the coefficient.

        {
            "XX": -1,
            "IZ": 2.4,
            "YI": 1+2j
        }  
        '''
        pass

    def generate_circuit(self, measurement_axis):

        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        circuit = QuantumCircuit(q, c)  

        circuit.h(0)
        circuit.cx(0,1)
        circuit.rx(self.angle, 0)

        if measurement_axis == 'Z':
            circuit.measure(q[0], c[0])
            circuit.measure(q[1], c[1])
        elif measurement_axis == 'X':
            circuit.ry(-np.pi/2, q[0])
            circuit.ry(-np.pi/2, q[1])
            circuit.measure(q[0], c[0])
            circuit.measure(q[1], c[1])
        elif measurement_axis == 'Y':
            circuit.rx(np.pi/2, q[0])
            circuit.rx(np.pi/2, q[1])
            circuit.measure(q[0], c[0])
            circuit.measure(q[1], c[1])
        else:
            raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')

        return circuit  


    def execute_measurement(self, circuit):

        backend = BasicAer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=self.shots)
        result = job.result()

        return result

    def calculate_expectation_values(self, measurement_results):

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
        expectation_value = self.calculate_expectation_values(measurement_results)

        return expectation_value

    def sum_decomposed_hamiltonian(self):

        x_measurement = -0.5 * (self.measure_in_axis("X"))
        y_measurement = -0.5 * (self.measure_in_axis("Y"))
        z_measurement = 0.5 * (self.measure_in_axis("Z"))
        i_measurement = 0.5 * 1

        measurement_sum = x_measurement + y_measurement + z_measurement + i_measurement
        
        return measurement_sum

    def find_ground_state(self):

        # measure in remaining 3 bases
        measurements = []
        angle_range = np.linspace(0, 2 * np.pi, 20)
        for angle in angle_range: 
            self.angle = angle
            measurement = self.sum_decomposed_hamiltonian()
            measurements.append(measurement)

        print("Ground state: {}".format(min(measurements)))

        plt.xlabel('Angle [radians]')
        plt.ylabel('Expectation value')
        plt.plot(angle_range, measurements)
        plt.show()

        
    
if __name__ == "__main__":

    vqe = VQE(hamiltonian_operator=1, shots=1000)
    vqe.find_ground_state()
