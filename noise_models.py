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
from vqe import VQE


# probabilities = [0, 0.1, 0.3, 0.5, 0.7]
probabilities = [0]
angle_range = np.linspace(0, 2 * np.pi, 20)
m_list = []
for error_prob in probabilities:
    noise_instructions = {
            "y": {
                "error_type": "non_local",
                "error_prob": error_prob
            }

        }
    vqe = VQE(hamiltonian_operator=1, shots=1000, noise=noise_instructions, graph=False)
    final_measurements = vqe.find_ground_state()
    m_list.append(final_measurements)

# plt.plot( angle_range, m_list[0], color='skyblue')
# plt.plot( angle_range, m_list[1], color='red')
# plt.plot( angle_range, m_list[2], color='black')
# plt.plot( angle_range, m_list[3], color='blue')
# plt.plot( angle_range, m_list[4], color='green')
# plt.plot( angle_range, m_list[5], color='yellow')
# plt.plot( angle_range, m_list[6], color='purple')
#plt.legend()
# plt.show()
